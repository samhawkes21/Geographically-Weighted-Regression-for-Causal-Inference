import numpy as np
from scipy.optimize import root_scalar


# ---------------------------------------------------------------------------
# Bandwidth helpers
# ---------------------------------------------------------------------------

def kernel_sum(bandwidth, distances, C):
    temp_weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    return np.sum(temp_weights) - C


def compute_bandwidth(distances, C):
    nonzero = distances[distances > 0]
    lower = np.min(nonzero) * 0.01 if len(nonzero) > 0 else 1e-6
    upper = np.max(distances)
    sol = root_scalar(kernel_sum, args=(distances, C),
                      method='brentq', bracket=[lower, upper])
    return sol.root


# ---------------------------------------------------------------------------
# Distance matrix  (computed once per dataset, passed everywhere)
# ---------------------------------------------------------------------------

def build_dist_matrix(X_data):
    """
    Compute the full (n, n) pairwise distance matrix from spatial coordinates.
    At n~800 this is ~5MB and trivially fast — computed once and reused for
    every bandwidth solve, GCV evaluation, and cache build.
    """
    coords = X_data[:, :2]
    # Broadcasting: equivalent to scipy.spatial.distance.cdist but no extra dep
    diff = coords[:, None, :] - coords[None, :, :]   # (n, n, 2)
    return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))  # (n, n)


# ---------------------------------------------------------------------------
# Cache build  (vectorised, no joblib)
# ---------------------------------------------------------------------------

def build_kernel_cache(X_data, X_sub_data, Y_data, C, weights=1,
                       dist_matrix=None):
    """
    Build per-point GWR cache: beta_hat, s_ii, y_hat for every training point.

    Strategy for n~1000:
      - Use pre-computed dist_matrix so distances are never recomputed.
      - Plain Python loop with numpy ops per point — no joblib spawning overhead.
      - np.diag(w) avoided by using element-wise multiply and einsum.
    """
    n = len(Y_data)
    p = X_sub_data.shape[1]

    if dist_matrix is None:
        dist_matrix = build_dist_matrix(X_data)

    scalar_weights = np.ndim(weights) == 0
    cache = []

    for i in range(n):
        distances = dist_matrix[i]                        # (n,) — free lookup
        bandwidth = compute_bandwidth(distances, C)
        w = np.exp(-0.5 * (distances / bandwidth) ** 2)  # (n,)

        if not scalar_weights:
            w = w * weights                               # per-obs weights

        # XTW = X_sub_data.T @ diag(w)  →  (X_sub_data * w[:, None]).T
        XTW = (X_sub_data * w[:, None]).T                 # (p, n)
        XTWX = XTW @ X_sub_data                           # (p, p)

        try:
            XTWX_inv = np.linalg.inv(XTWX)
        except np.linalg.LinAlgError:
            XTWX_inv = np.linalg.pinv(XTWX)

        beta_hat = XTWX_inv @ XTW @ Y_data               # (p,)
        y_hat_i = X_sub_data[i] @ beta_hat

        # s_ii = x_i^T (X^T W X)^{-1} x_i^T W  — only the i-th element
        hi = X_sub_data[i] @ XTWX_inv                    # (p,)
        s_ii = hi @ XTW[:, i]

        cache.append(dict(beta_hat=beta_hat, s_ii=s_ii, y_hat=y_hat_i))

    return cache


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def Y_hat_cached(poi_sub, cache_entry):
    return poi_sub @ cache_entry['beta_hat']


def _solve_beta(XTWX, XTW, Y_data):
    try:
        XTWX_inv = np.linalg.inv(XTWX)
    except np.linalg.LinAlgError:
        XTWX_inv = np.linalg.pinv(XTWX)
    return XTWX_inv @ XTW @ Y_data, XTWX_inv


def Y_hat(poi, X_data, X_sub_data, Y_data, C, weights=1, dist_matrix=None):
    """Single-point prediction. Uses dist_matrix row if available."""
    if dist_matrix is not None:
        # poi must be a row of X_data; find it by index match on coords
        idx = np.where((X_data[:, :2] == poi[:2]).all(axis=1))[0]
        if len(idx):
            distances = dist_matrix[idx[0]]
        else:
            distances = np.linalg.norm(poi[:2] - X_data[:, :2], axis=1)
    else:
        distances = np.linalg.norm(poi[:2] - X_data[:, :2], axis=1)

    bandwidth = compute_bandwidth(distances, C)
    w = np.exp(-0.5 * (distances / bandwidth) ** 2) * weights
    XTW = (X_sub_data * w).T
    XTWX = XTW @ X_sub_data
    beta_hat, _ = _solve_beta(XTWX, XTW, Y_data)
    return poi[2:] @ beta_hat


# ---------------------------------------------------------------------------
# GCV helpers
# ---------------------------------------------------------------------------

def GCV_components_cached(cache):
    y_hats = np.array([e['y_hat'] for e in cache])
    s_diag = np.array([e['s_ii'] for e in cache])
    return y_hats, s_diag


def GCV_components(X_data, X_sub_data, Y_data, n, C, weights=1):
    cache = build_kernel_cache(X_data, X_sub_data, Y_data, C, weights)
    return GCV_components_cached(cache)


def S_diag(X_data, X_sub_data, n, C, weights=1):
    raise NotImplementedError(
        "S_diag is superseded by build_kernel_cache; use cache['s_ii'] entries."
    )
