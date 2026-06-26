import numpy as np
from scipy.optimize import minimize_scalar
from GWR_utilities import build_dist_matrix, build_kernel_cache, GCV_components_cached


def run_GWR(X_data, X_sub_data, Y_data, p, n, weights=1):
    """
    GCV bandwidth selection.

    Key optimisation: dist_matrix is built ONCE before minimize_scalar runs.
    The optimiser typically calls GCV 15-40 times; previously each call
    recomputed all pairwise distances from scratch.
    """
    dist_matrix = build_dist_matrix(X_data)   # O(n²) — done exactly once

    def GCV(C):
        cache = build_kernel_cache(X_data, X_sub_data, Y_data, C,
                                   weights=weights, dist_matrix=dist_matrix)
        y_hats, s_diag = GCV_components_cached(cache)

        tr_S = np.sum(s_diag)
        denom = n - tr_S
        if abs(denom) < 1e-10:
            return np.inf
        numerator = n * np.sum((Y_data - y_hats) ** 2)
        return numerator / denom ** 2

    result = minimize_scalar(GCV, bounds=(1, n / 2), method='bounded')
    return result.x
