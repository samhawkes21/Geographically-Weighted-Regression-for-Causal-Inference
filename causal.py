import numpy as np
import pandas as pd
from GWR_utilities import build_dist_matrix, build_kernel_cache


def intervention(col, value, X_data, cache):
    """
    Apply do(X[col] = value) across all training points.
    O(n·p): reuses cached beta_hat, no weight recomputation.
    """
    results = np.empty(len(X_data))
    for i in range(len(X_data)):
        poi_sub = X_data[i, 2:].copy()
        poi_sub[col - 2] = value
        results[i] = poi_sub @ cache[i]['beta_hat']
    return results


def causal_statistics(filename, model):
    """
    Compute ATE error and PEHE.
    Cache built once, reused for both T=0 and T=1 interventions.
    """
    new_X, new_sub_X, new_Y, p, n, C, adj_w = model

    dist_matrix = build_dist_matrix(new_X)
    cache = build_kernel_cache(new_X, new_sub_X, new_Y, C,
                               weights=adj_w, dist_matrix=dist_matrix)

    df = pd.read_csv(filename, header=0)
    Y0 = df["Y0"].to_numpy()[:n]
    Y1 = df["Y1"].to_numpy()[:n]

    ATE = np.mean(Y1 - Y0)

    Yhat0 = intervention(3, 0, new_X, cache)
    Yhat1 = intervention(3, 1, new_X, cache)

    ATEhat = np.mean(Yhat1 - Yhat0)
    ATEerror = abs(ATEhat - ATE)

    T = Y1 - Y0
    That = Yhat1 - Yhat0
    min_len = min(len(T), len(That))
    PEHE = np.mean((That[:min_len] - T[:min_len]) ** 2)

    return ATEerror, PEHE