import numpy as np
from scipy.special import gammaincc, gamma

def gaussian_projection_ndis_delta_1d(D, index_sets, r, eps, reg_param1=0, reg_param2=0):
    """
    Compute δ_{X,Y}(ε) for Gaussian random projection D^T@G
    where G is a n by r Gaussian random matrix with i.i.d. standard normal entries.

    D is a column vector of size n.
    """

    D = np.asarray(D)
    assert D.ndim == 1, "D must be a column vector"
    assert eps >= 0, "eps must be >= 0"
    assert r > 0, "r must be a positive integer"
    index_sets = list([index_sets])
    assert len(index_sets) > 0, "index_sets must be a non-empty iterable"

    # Squared Frobenius norm of D
    normD2 = float(np.sum(D ** 2))
    if normD2 == 0:
        return 0 

    s = r / 2.0  # shape parameter for chi^2_r related gamma

    # Create D' by zeroing out the specified rows
    D_prime = D.copy()
    D_prime[index_sets] = 0.0
    normDp2 = float(np.sum(D_prime ** 2))
    if normDp2 <= 0:
        return 1

    rho = (normD2 + reg_param1**2) / (normDp2 + reg_param2**2)
    if np.isclose(rho, 1.0):
        return 0

    # t0 = 2 * (eps + (r/2) * log(rho)) / (rho - 1)
    t0 = 2.0 * (eps + 0.5 * r * np.log(rho)) / (rho - 1.0)

    # Upper incomplete gamma Γ(s, x):
    # Γ(s, x) = gammaincc(s, x) * gamma(s) in SciPy
    G_total = gamma(s)
    G_t0_over_2 = gammaincc(s, t0 / 2.0) * G_total
    G_rho_t0_over_2 = gammaincc(s, (rho * t0) / 2.0) * G_total

    delta = (G_t0_over_2 - np.exp(eps) * G_rho_t0_over_2) / G_total
    return max(0.0, min(1.0, float(delta)))


def gaussian_projection_ndis_delta(D, index, r, eps):
    """
    Compute δ_{X,Y}(ε) for Gaussian random projection D^T@G
    where G is a n by r Gaussian random matrix with i.i.d. standard normal entries.

    D is a n by d matrix.
    """

    D = np.asarray(D)
    assert D.ndim == 2, "D must be a matrix"
    assert eps >= 0, "eps must be >= 0"
    assert r > 0, "r must be a positive integer"
    assert index >= 0 and index < D.shape[0], "index must be a valid row index"
    d = D.shape[1]

    s = r / 2.0  # shape parameter for chi^2_r related gamma

    v = D[index].copy().reshape(-1, 1)
    M = D.T @ D
    inv_M = np.linalg.inv(M)
    pi = (v.T @ inv_M @ v).item()
    rho = 1 / (1 - pi)

    if np.isclose(rho, 1.0):
        return 0

    # t0 = 2 * (eps + (r/2) * log(rho)) / (rho - 1)
    t0 = 2.0 * (eps + 0.5 * r * np.log(rho)) / (rho - 1.0)

    # Upper incomplete gamma Γ(s, x):
    # Γ(s, x) = gammaincc(s, x) * gamma(s) in SciPy
    G_total = gamma(s)
    G_t0_over_2 = gammaincc(s, t0 / 2.0) * G_total
    G_rho_t0_over_2 = gammaincc(s, (rho * t0) / 2.0) * G_total

    delta = (G_t0_over_2 - np.exp(eps) * G_rho_t0_over_2) / G_total
    return max(0.0, min(1.0, float(delta)))


def rp_mean_cov_matrix(D, r, full_vec_cov=False):
    """
    Given D in R^{n x d} and G in R^{n x r} with i.i.d. N(0,1) entries,
    consider X = D^T G in R^{d x r}.

    Returns:
        mean:  d x r zero matrix (E[X])
        row_cov: d x d matrix D^T D  (Cov of each column of X)
        vec_cov (optional): (dr) x (dr) covariance of vec(X),
                            equal to kron(I_r, D^T D).
    """
    D = np.asarray(D)
    if D.ndim != 2:
        raise ValueError("D must be a 2D array of shape (n, d).")

    _, d = D.shape

    # Mean of X is zero
    mean = np.zeros((d, r))

    # Covariance of each column: Cov(X_{:,j}) = D^T D
    row_cov = D.T @ D  # shape (d, d)

    if not full_vec_cov:
        return mean, row_cov

    # Full covariance of vec(X): Cov(vec(X)) = I_r ⊗ (D^T D)
    vec_cov = np.kron(np.eye(r), row_cov)  # shape (d*r, d*r)
    return mean, row_cov, vec_cov
