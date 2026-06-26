import numpy as np
from sklearn.neighbors import NearestNeighbors
from GWR_utilities import (build_dist_matrix, build_kernel_cache,
                            GCV_components_cached, Y_hat_cached)
from GWR import run_GWR


def remove_outliers(X_data, X_sub_data, Y_data, p, n, C, eval=5):
    """
    Remove spatial outliers based on studentised residuals.
    dist_matrix built once and passed into cache build.
    """
    dist_matrix = build_dist_matrix(X_data)
    cache = build_kernel_cache(X_data, X_sub_data, Y_data, C,
                               dist_matrix=dist_matrix)
    y_hats, s_diag = GCV_components_cached(cache)

    S = np.array(s_diag)
    v1 = np.sum(S)
    v2 = S @ S

    Q = (1 - S) ** 2
    E = Y_data - y_hats

    total_sq = np.sum(E ** 2)
    dof = (n - 1) - 2 * v1 + v2

    X_list, Y_list = [], []
    for i in range(n):
        if dof <= 0:
            continue
        sigma_i = (total_sq - E[i] ** 2) / dof
        denom = sigma_i * np.sqrt(Q[i])
        if denom == 0 or not np.isfinite(denom):
            continue
        r = E[i] / denom
        if np.abs(r) <= eval:
            X_list.append(X_data[i])
            Y_list.append(Y_data[i])

    return np.array(X_list), np.array(Y_list)


def heteroskedastic(X_data, X_sub_data, Y_data, p, n, C, k=30, max_iter=2, tol=1e-4):
    """
    Iterative heteroskedastic weight adjustment.
    dist_matrix built once and reused across all iterations.
    """
    coords = X_data[:, :2]
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, knn_indices = nbrs.kneighbors(coords)

    dist_matrix = build_dist_matrix(X_data)   # built once, reused every iter

    def smooth_residual_variance(residuals_sq):
        return np.mean(residuals_sq[knn_indices], axis=1)

    adjusted_weights = np.ones(n)

    cache = build_kernel_cache(X_data, X_sub_data, Y_data, C,
                               weights=adjusted_weights,
                               dist_matrix=dist_matrix)
    y_hats = np.array([Y_hat_cached(X_data[i, 2:], cache[i]) for i in range(n)])
    residuals = Y_data - y_hats

    for _ in range(max_iter):
        sigma_sq = smooth_residual_variance(residuals ** 2)
        adjusted_weights = 1.0 / np.sqrt(sigma_sq + 1e-8)

        C = run_GWR(X_data, X_sub_data, Y_data, p, n, adjusted_weights)

        cache = build_kernel_cache(X_data, X_sub_data, Y_data, C,
                                   weights=adjusted_weights,
                                   dist_matrix=dist_matrix)
        y_hats_new = np.array([Y_hat_cached(X_data[i, 2:], cache[i])
                                for i in range(n)])
        residuals_new = Y_data - y_hats_new

        if np.linalg.norm(residuals_new - residuals) < tol:
            break

        residuals = residuals_new

    return C, adjusted_weights