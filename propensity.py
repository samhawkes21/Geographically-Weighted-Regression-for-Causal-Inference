import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_data(filename, p, n):
    df = pd.read_csv(filename, header=0)
    data = df.to_numpy()
    data = data[:n]

    Y = data[:, 0]
    T = data[:, 3]
    X = data[:, 4:4+p]

    Y0_true = df["Y0"].to_numpy()[:n]
    Y1_true = df["Y1"].to_numpy()[:n]

    (X_train, X_test,
     Y_train, Y_test,
     T_train, T_test,
     Y0_train, Y0_test,
     Y1_train, Y1_test) = train_test_split(
        X, Y, T, Y0_true, Y1_true,
        test_size=0.2, random_state=42
    )

    return (X_train, X_test,
            Y_train, Y_test,
            T_train, T_test,
            Y0_train, Y0_test,
            Y1_train, Y1_test)


def estimate_propensity(X_train, T_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, T_train)

    ps_train = model.predict_proba(X_train_s)[:, 1]
    ps_test  = model.predict_proba(X_test_s)[:, 1]

    return ps_train, ps_test


def ipw_ate(Y, T, ps):
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    treated   = np.sum((T == 1) * Y / ps) / np.sum(T == 1)
    control   = np.sum((T == 0) * Y / (1 - ps)) / np.sum(T == 0)
    return treated - control


def matching_ite(X_train, Y_train, T_train, X_test, Y_test, T_test,
                 ps_train, ps_test, caliper=0.05):
    ITE_hat  = []
    ITE_true_matched = []

    X_all  = np.vstack([X_train, X_test])
    Y_all  = np.concatenate([Y_train, Y_test])
    T_all  = np.concatenate([T_train, T_test])
    ps_all = np.concatenate([ps_train, ps_test])

    idx0 = np.where(T_all == 0)[0]
    idx1 = np.where(T_all == 1)[0]

    n_train = len(Y_train)
    test_indices = np.arange(n_train, n_train + len(Y_test))

    for i in test_indices:
        candidates = idx0 if T_all[i] == 1 else idx1
        diffs = np.abs(ps_all[i] - ps_all[candidates])
        nearest = candidates[np.argmin(diffs)]
        if diffs[np.argmin(diffs)] > caliper:
            continue

        if T_all[i] == 1:
            ite = Y_all[i] - Y_all[nearest]
        else:
            ite = Y_all[nearest] - Y_all[i]

        ITE_hat.append(ite)

    return np.array(ITE_hat)


def evaluate(Y_train, Y_test, T_train, T_test,
             ps_train, ps_test,
             Y0_train, Y0_test, Y1_train, Y1_test,
             caliper=0.05):

    Y_all  = np.concatenate([Y_train, Y_test])
    T_all  = np.concatenate([T_train, T_test])
    ps_all = np.concatenate([ps_train, ps_test])
    ATE_hat  = ipw_ate(Y_all, T_all, ps_all)
    ATE_true = np.mean(np.concatenate([Y1_train, Y1_test]) -
                       np.concatenate([Y0_train, Y0_test]))
    ATE_error = abs(ATE_hat - ATE_true)

    ITE_hat  = matching_ite(
        X_train=np.zeros((len(Y_train), 1)),   
        Y_train=Y_train, T_train=T_train,
        X_test=np.zeros((len(Y_test), 1)),
        Y_test=Y_test,  T_test=T_test,
        ps_train=ps_train, ps_test=ps_test,
        caliper=caliper
    )
    ITE_true = Y1_test - Y0_test

    min_len = min(len(ITE_hat), len(ITE_true))
    PEHE = np.mean((ITE_hat[:min_len] - ITE_true[:min_len]) ** 2)

    return ATE_error, PEHE


if __name__ == "__main__":
    filename = input("Name of csv file: ")
    p        = int(input("Number of parameters: "))
    n        = int(input("Number of observations: "))

    start = time.time()

    (X_train, X_test,
     Y_train, Y_test,
     T_train, T_test,
     Y0_train, Y0_test,
     Y1_train, Y1_test) = set_data(filename, p, n)

    ps_train, ps_test = estimate_propensity(X_train, T_train, X_test)

    ATE_error, PEHE = evaluate(
        Y_train, Y_test, T_train, T_test,
        ps_train, ps_test,
        Y0_train, Y0_test, Y1_train, Y1_test
    )

    elapsed = time.time() - start

    print(f"ATE Error : {ATE_error:.6f}")
    print(f"PEHE      : {PEHE:.6f}")
    print(f"Runtime   : {elapsed:.2f}s")