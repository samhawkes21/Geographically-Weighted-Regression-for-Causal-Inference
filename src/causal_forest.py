import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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

    return X_train, X_test, Y_train, Y_test, T_train, T_test, Y0_train, Y0_test, Y1_train, Y1_test


def train_causal_forest(X_train, Y_train, T_train,
                        n_estimators=500, min_samples_leaf=5):
    idx0 = T_train == 0
    idx1 = T_train == 1

    rf0 = RandomForestRegressor(n_estimators=n_estimators,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42, n_jobs=-1)
    rf1 = RandomForestRegressor(n_estimators=n_estimators,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42, n_jobs=-1)

    rf0.fit(X_train[idx0], Y_train[idx0])
    rf1.fit(X_train[idx1], Y_train[idx1])

    return rf0, rf1


def evaluate(rf0, rf1, X, Y0_true, Y1_true):
    mu0 = rf0.predict(X)
    mu1 = rf1.predict(X)

    ITE_hat = mu1 - mu0
    ITE_true = Y1_true - Y0_true

    ATE_true = np.mean(ITE_true)
    ATE_hat  = np.mean(ITE_hat)
    ATE_error = abs(ATE_hat - ATE_true)

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

    rf0, rf1 = train_causal_forest(X_train, Y_train, T_train)

    ATE_error, PEHE = evaluate(rf0, rf1, X_test, Y0_test, Y1_test)

    elapsed = time.time() - start

    print(f"ATE Error : {ATE_error:.6f}")
    print(f"PEHE      : {PEHE:.6f}")
    print(f"Runtime   : {elapsed:.2f}s")
