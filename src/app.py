import streamlit as st
import tempfile, os, time
import numpy as np
import pandas as pd
import preprocess

from data import set_data as gwr_set_data
from GWR import run_GWR
from spatial import remove_outliers, heteroskedastic
from causal import causal_statistics
from GWR_utilities import build_dist_matrix, build_kernel_cache

from causal_forest import set_data as cf_set_data, train_causal_forest, evaluate as cf_evaluate
from propensity import set_data as ps_set_data, estimate_propensity, evaluate as ps_evaluate


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="GWR Causal Inference", layout="wide")
st.title("Causal inference model comparison")
st.caption("Upload a spatial dataset CSV, choose parameters, and compare GWR, causal forest, and propensity score models.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type="csv")

    st.header("Parameters")
    p = st.number_input("Number of covariates (p)", min_value=1, value=13, step=1,
                        help="Number of covariate columns to use, starting from column 4 onward.")
    n = st.number_input("Observations to use (0 = all)", min_value=0, value=0, step=100,
                        help="Set to 0 to use all rows in the dataset.")

    run_btn = st.button("Run all models", type="primary", disabled=uploaded is None)


# ── Session state ─────────────────────────────────────────────────────────────

for key in ("results", "gwr_model", "cf_model", "ps_model",
            "converted_path", "converted_df", "p_val", "n_use"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def run_gwr_pipeline(filepath, p, n):
    X_train, X_sub_train, Y_train, X_test, Y_test = gwr_set_data(filepath, p, n)
    n_train = len(Y_train)
    C_1 = run_GWR(X_train, X_sub_train, Y_train, p, n_train)
    new_X, new_Y = remove_outliers(X_train, X_sub_train, Y_train, p, n_train, C_1)
    new_sub_X = new_X[:, 2:(p + 3)]
    n2 = len(new_Y)
    C_temp = run_GWR(new_X, new_sub_X, new_Y, p, n2)
    C_2, adj_w = heteroskedastic(new_X, new_sub_X, new_Y, p, n2, C_temp)
    model = [new_X, new_sub_X, new_Y, p, n2, C_2, adj_w]
    ATE_error, PEHE = causal_statistics(filepath, model)
    return ATE_error, PEHE, model


def predict_gwr(model, X_new):
    new_X, new_sub_X, new_Y, p, n, C, adj_w = model
    dist_matrix = build_dist_matrix(new_X)
    cache = build_kernel_cache(new_X, new_sub_X, new_Y, C,
                               weights=adj_w, dist_matrix=dist_matrix)
    preds = []
    for row in X_new:
        dists = np.linalg.norm(row[:2] - new_X[:, :2], axis=1)
        nearest = int(np.argmin(dists))
        preds.append(row[2:] @ cache[nearest]['beta_hat'])
    return np.array(preds)


def run_cf_pipeline(filepath, p, n):
    (X_train, X_test, Y_train, Y_test,
     T_train, T_test, Y0_train, Y0_test,
     Y1_train, Y1_test) = cf_set_data(filepath, p, n)
    rf0, rf1 = train_causal_forest(X_train, Y_train, T_train)
    ATE_error, PEHE = cf_evaluate(rf0, rf1, X_test, Y0_test, Y1_test)
    return ATE_error, PEHE, (rf0, rf1)


def predict_cf(model, X_cov):
    rf0, rf1 = model
    return rf1.predict(X_cov) - rf0.predict(X_cov)


def run_ps_pipeline(filepath, p, n):
    (X_train, X_test, Y_train, Y_test,
     T_train, T_test, Y0_train, Y0_test,
     Y1_train, Y1_test) = ps_set_data(filepath, p, n)
    ps_train, ps_test = estimate_propensity(X_train, T_train, X_test)
    ATE_error, PEHE = ps_evaluate(
        Y_train, Y_test, T_train, T_test,
        ps_train, ps_test,
        Y0_train, Y0_test, Y1_train, Y1_test
    )
    return ATE_error, PEHE, {"X_train": X_train, "T_train": T_train}


def predict_ps(model, X_cov):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(model["X_train"])
    X_new_s = scaler.transform(X_cov)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_s, model["T_train"])
    return lr.predict_proba(X_new_s)[:, 1]


# ── Run models ────────────────────────────────────────────────────────────────

if run_btn and uploaded is not None:

    # ── Step 1: convert raw CSV → model-ready format ──────────────────────────
    with st.status("Preprocessing data…", expanded=True) as status:
        try:
            raw_df = pd.read_csv(uploaded)
            converted_df = preprocess.convert(raw_df)

            n_use = int(n) if int(n) > 0 else len(converted_df)
            p_val = int(p)
            converted_df = converted_df.iloc[:n_use].reset_index(drop=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w') as f:
                converted_df.to_csv(f, index=False)
                converted_path = f.name

            st.session_state.converted_path = converted_path
            st.session_state.converted_df   = converted_df
            st.session_state.p_val          = p_val
            st.session_state.n_use          = n_use

            st.write(f"{len(converted_df)} rows · {p_val} covariates")
            status.update(label="Preprocessing done", state="complete")

        except Exception as e:
            status.update(label="Preprocessing failed", state="error")
            st.error(str(e))
            st.stop()

    with st.expander("Preview converted data"):
        st.dataframe(converted_df.head(10), use_container_width=True)
        st.download_button("Download converted CSV",
                           converted_df.to_csv(index=False).encode(),
                           file_name="converted.csv", mime="text/csv")

    # ── Step 2: run models ────────────────────────────────────────────────────
    results = {}

    with st.status("Running models…", expanded=True) as status:

        st.write("Running GWR…")
        t0 = time.time()
        try:
            ate_err, pehe, gwr_model = run_gwr_pipeline(converted_path, p_val, n_use)
            st.session_state.gwr_model = gwr_model
            results["GWR"] = {"ATE Error": round(ate_err, 4), "PEHE": round(pehe, 4),
                               "Time (s)": round(time.time() - t0, 1), "error": None}
        except Exception as e:
            results["GWR"] = {"ATE Error": None, "PEHE": None,
                               "Time (s)": round(time.time() - t0, 1), "error": str(e)}
        st.write(f"GWR done ({results['GWR']['Time (s)']}s)")

        st.write("Running causal forest…")
        t0 = time.time()
        try:
            ate_err, pehe, cf_model = run_cf_pipeline(converted_path, p_val, n_use)
            st.session_state.cf_model = cf_model
            results["Causal forest"] = {"ATE Error": round(ate_err, 4), "PEHE": round(pehe, 4),
                                         "Time (s)": round(time.time() - t0, 1), "error": None}
        except Exception as e:
            results["Causal forest"] = {"ATE Error": None, "PEHE": None,
                                         "Time (s)": round(time.time() - t0, 1), "error": str(e)}
        st.write(f"Causal forest done ({results['Causal forest']['Time (s)']}s)")

        st.write("Running propensity score…")
        t0 = time.time()
        try:
            ate_err, pehe, ps_model = run_ps_pipeline(converted_path, p_val, n_use)
            st.session_state.ps_model = ps_model
            results["Propensity score"] = {"ATE Error": round(ate_err, 4), "PEHE": round(pehe, 4),
                                            "Time (s)": round(time.time() - t0, 1), "error": None}
        except Exception as e:
            results["Propensity score"] = {"ATE Error": None, "PEHE": None,
                                            "Time (s)": round(time.time() - t0, 1), "error": str(e)}
        st.write(f"Propensity score done ({results['Propensity score']['Time (s)']}s)")

        status.update(label="All models done", state="complete")

    st.session_state.results = results


# ── Results & Prediction ─────────────────────────────────────────────────────

if st.session_state.results:
    results = st.session_state.results
    p_val   = st.session_state.p_val
    conv_df = st.session_state.converted_df

    tab_results, tab_predict = st.tabs(["Results", "Predict new points"])

    # ── Tab: Results ──────────────────────────────────────────────────────────
    with tab_results:
        rows = []
        for name, r in results.items():
            rows.append({
                "Model":     name,
                "ATE Error": r["ATE Error"] if r["ATE Error"] is not None else "error",
                "PEHE":      r["PEHE"]      if r["PEHE"]      is not None else "error",
                "Time (s)":  r["Time (s)"],
                "Status":    "✓" if r["error"] is None else f"⚠ {r['error']}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        valid = {k: v for k, v in results.items() if v["PEHE"] is not None}
        if valid:
            best = min(valid, key=lambda k: valid[k]["PEHE"])
            st.caption(f"Lowest PEHE: **{best}** ({valid[best]['PEHE']})")

    # ── Tab: Predict ──────────────────────────────────────────────────────────
    with tab_predict:
        available = [name for name, r in results.items() if r["error"] is None]
        if not available:
            st.warning("No models completed successfully.")
            st.stop()

        selected = st.radio("Choose model", options=available, horizontal=True)

        cov_cols = [c for c in conv_df.columns
                    if c not in ("Y", "X", "Y_coord", "T", "Y0", "Y1")]

        input_tab1, input_tab2 = st.tabs(["Enter values manually", "Upload CSV"])

        # ── Manual entry ──────────────────────────────────────────────────────
        with input_tab1:
            with st.form("manual_predict"):
                if selected == "GWR":
                    c1, c2 = st.columns(2)
                    lon_in = c1.number_input("Longitude", value=float(conv_df["X"].mean()), format="%.4f")
                    lat_in = c2.number_input("Latitude",  value=float(conv_df["Y_coord"].mean()), format="%.4f")

                cov_vals = {}
                cols = st.columns(3)
                for i, feat in enumerate(cov_cols[:p_val]):
                    default = float(conv_df[feat].mean())
                    cov_vals[feat] = cols[i % 3].number_input(feat, value=default, format="%.4f")

                submitted = st.form_submit_button("Predict")

            if submitted:
                cov_row = np.array([cov_vals[f] for f in cov_cols[:p_val]]).reshape(1, -1)

                if selected == "GWR":
                    row = np.array([lon_in, lat_in, 1.0] + [cov_vals[f] for f in cov_cols[:p_val]])
                    pred = predict_gwr(st.session_state.gwr_model, row.reshape(1, -1))
                    st.metric("Predicted outcome (GWR)", f"{pred[0]:.4f}")

                elif selected == "Causal forest":
                    ite = predict_cf(st.session_state.cf_model, cov_row)
                    st.metric("Predicted ITE", f"{ite[0]:.4f}")

                elif selected == "Propensity score":
                    ps = predict_ps(st.session_state.ps_model, cov_row)
                    st.metric("Propensity score P(T=1|X)", f"{ps[0]:.4f}")

        # ── CSV upload ────────────────────────────────────────────────────────
        with input_tab2:
            st.caption(
                "Upload a CSV in the same format as your training data. "
                "It will be converted using the same DGP before predicting."
            )
            pred_file = st.file_uploader("Upload CSV", type="csv", key="pred_upload")

            if pred_file is not None:
                df_raw_new = pd.read_csv(pred_file)
                st.write(f"{len(df_raw_new)} rows loaded.")

                try:
                    df_pred_conv = preprocess.convert(df_raw_new)
                    st.dataframe(df_pred_conv.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
                    st.stop()

                if st.button("Run predictions"):
                    if selected == "GWR":
                        n_new = len(df_pred_conv)
                        X_new = np.ones((n_new, p_val + 3))
                        X_new[:, 0] = df_pred_conv["X"].to_numpy()
                        X_new[:, 1] = df_pred_conv["Y_coord"].to_numpy()
                        for i, col in enumerate(cov_cols[:p_val]):
                            X_new[:, i + 3] = df_pred_conv[col].to_numpy()
                        preds = predict_gwr(st.session_state.gwr_model, X_new)
                        df_pred_conv["GWR_prediction"] = preds

                    elif selected == "Causal forest":
                        X_cov = df_pred_conv[cov_cols[:p_val]].to_numpy()
                        df_pred_conv["CF_ITE"] = predict_cf(st.session_state.cf_model, X_cov)

                    elif selected == "Propensity score":
                        X_cov = df_pred_conv[cov_cols[:p_val]].to_numpy()
                        df_pred_conv["propensity_score"] = predict_ps(st.session_state.ps_model, X_cov)

                    st.dataframe(df_pred_conv, use_container_width=True)
                    st.download_button(
                        "Download predictions as CSV",
                        df_pred_conv.to_csv(index=False).encode(),
                        file_name=f"{selected.lower().replace(' ', '_')}_predictions.csv",
                        mime="text/csv"
                    )
