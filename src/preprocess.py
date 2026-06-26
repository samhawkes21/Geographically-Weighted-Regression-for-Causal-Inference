"""
preprocess.py
-------------
Converts a raw dataset (Crime format) into the model-ready format
using the exact data generating process from Section 6.1 of the paper.

Raw format columns:
    outcome, X(lon), Y(lat), cov1, cov2, ..., covP

Output format columns:
    Y, X(lon), Y_coord(lat), T, cov1..covP, Y0, Y1

DGP (Heiss [7], formulated by S. Hawkes):
    1. Standardise outcome, coordinates, and covariates.
    2. Confounder:
         U_i = 0.45*y_i + 0.20*sin(1.5*lat_i)*cos(1.2*lon_i) + eta_i
         eta_i ~ N(0, 0.6),  then U is standardised.
    3. Propensity score (logistic):
         logit(p_i) = alpha + 0.90*y_i + 0.35*x1 + 0.25*x2 - 0.20*x3
                      + 0.25*(x1*x2) + 0.35*sin(1.1*lat_i)
                      - 0.25*cos(1.0*lon_i) + 0.65*U_i
         logits clipped to [-5, 5]; T_i ~ Bernoulli(p_i)
    4. Baseline outcome:
         f_i = 1.00*y_i + 0.30*x1 + 0.20*x2^2 + 0.15*x3
               + 0.25*sin(1.2*lat_i)*cos(0.8*lon_i) + 0.35*U_i
         eta_i ~ N(0, 0.6), tau = 0.8
         Y0_i = f_i + eta_i
         Y1_i = f_i + tau + eta_i
         Y_i  = T_i*Y1_i + (1 - T_i)*Y0_i   (observed outcome)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Treatment intercept — controls overall treatment prevalence (~50% treated)
TREATMENT_INTERCEPT = 0.0

# Fixed noise std from the paper
NOISE_STD = 0.6

# Treatment effect
TAU = 0.8


def convert(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Convert a raw Crime-format DataFrame to model-ready format
    using the paper's DGP (Section 6.1).

    Parameters
    ----------
    df   : raw DataFrame — first col is outcome, second is lon (X),
           third is lat (Y), remaining are covariates
    seed : random seed

    Returns
    -------
    DataFrame with columns: Y, X, Y_coord, T, cov1..covP, Y0, Y1
    """
    rng = np.random.default_rng(seed)

    cols = list(df.columns)
    outcome_col = cols[0]
    lon_col     = cols[1]   # X
    lat_col     = cols[2]   # Y
    cov_cols    = cols[3:]

    # ── 1. Standardise everything ─────────────────────────────────────────────
    y_raw  = df[outcome_col].to_numpy().reshape(-1, 1)
    lon_raw = df[lon_col].to_numpy().reshape(-1, 1)
    lat_raw = df[lat_col].to_numpy().reshape(-1, 1)
    X_raw  = df[cov_cols].to_numpy()

    y_sc  = StandardScaler().fit_transform(y_raw).ravel()
    lon_s = StandardScaler().fit_transform(lon_raw).ravel()
    lat_s = StandardScaler().fit_transform(lat_raw).ravel()
    X_sc  = StandardScaler().fit_transform(X_raw)   # covariates kept standardised

    x1 = X_sc[:, 0]
    x2 = X_sc[:, 1] if X_sc.shape[1] > 1 else np.zeros(len(y_sc))
    x3 = X_sc[:, 2] if X_sc.shape[1] > 2 else np.zeros(len(y_sc))

    # ── 2. Confounder U ───────────────────────────────────────────────────────
    eta_u = rng.normal(0, NOISE_STD, size=len(y_sc))
    U_raw = (0.45 * y_sc
             + 0.20 * np.sin(1.5 * lat_s) * np.cos(1.2 * lon_s)
             + eta_u)
    U = StandardScaler().fit_transform(U_raw.reshape(-1, 1)).ravel()

    # ── 3. Treatment assignment ───────────────────────────────────────────────
    logit = (TREATMENT_INTERCEPT
             + 0.90 * y_sc
             + 0.35 * x1
             + 0.25 * x2
             - 0.20 * x3
             + 0.25 * (x1 * x2)
             + 0.35 * np.sin(1.1 * lat_s)
             - 0.25 * np.cos(1.0 * lon_s)
             + 0.65 * U)

    logit = np.clip(logit, -5, 5)
    p = 1.0 / (1.0 + np.exp(-logit))
    T = rng.binomial(1, p).astype(int)

    # ── 4. Potential outcomes ─────────────────────────────────────────────────
    eta_y = rng.normal(0, NOISE_STD, size=len(y_sc))
    f = (1.00 * y_sc
         + 0.30 * x1
         + 0.20 * (x2 ** 2)
         + 0.15 * x3
         + 0.25 * np.sin(1.2 * lat_s) * np.cos(0.8 * lon_s)
         + 0.35 * U)

    Y0 = f + eta_y
    Y1 = f + TAU + eta_y
    Y  = T * Y1 + (1 - T) * Y0     # observed outcome

    # ── 5. Assemble output ────────────────────────────────────────────────────
    # Use original (unstandardised) coordinates for the spatial kernel
    cov_df = pd.DataFrame(X_sc, columns=[f"cov{i+1}" for i in range(X_sc.shape[1])])

    out = pd.DataFrame({
        "Y":       Y,
        "X":       df[lon_col].to_numpy(),       # original lon
        "Y_coord": df[lat_col].to_numpy(),        # original lat
        "T":       T,
    })
    out = pd.concat([out, cov_df], axis=1)
    out["Y0"] = Y0
    out["Y1"] = Y1

    return out


def load_and_convert(filepath: str, seed: int = 42) -> pd.DataFrame:
    """Read a raw Crime-format CSV from disk and return the converted DataFrame."""
    df = pd.read_csv(filepath)
    _validate(df)
    return convert(df, seed=seed)


def _validate(df: pd.DataFrame):
    if df.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 columns (outcome, lon, lat, cov1…) "
            f"but got {df.shape[1]}. Check your CSV format."
        )
