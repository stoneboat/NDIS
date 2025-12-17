import numpy as np
from scipy.special import gammaincc, gamma
from scipy.optimize import brentq, fsolve
import math
import mpmath as mp


import mpmath as mp

def _compute_gamma_delta(s, t0, rho, eps, dps=80):
    """
    Compute δ = (Γ(s, t0/2) - exp(eps) * Γ(s, rho*t0/2)) / Γ(s),
    where Γ(s, x) is the UPPER incomplete gamma function.

    Returns
    -------
    float
        Delta value clamped to [0, 1].
    """
    mp.mp.dps = dps

    s = mp.mpf(s)
    t0 = mp.mpf(t0)
    rho = mp.mpf(rho)
    eps = mp.mpf(eps)

    x1 = t0 / 2
    x2 = rho * t0 / 2

    # Upper incomplete gamma Γ(s, x) = ∫_{x}^{∞} t^{s-1} e^{-t} dt
    # mpmath: gammainc(s, a, b) = ∫_{a}^{b} t^{s-1} e^{-t} dt
    G1 = mp.gammainc(s, x1, mp.inf)   # Γ(s, t0/2)
    G2 = mp.gammainc(s, x2, mp.inf)   # Γ(s, ρ t0/2)

    delta = (G1 - mp.e**eps * G2) / mp.gamma(s)

    # Clamp (also guards against tiny negative values from numerical error)
    if delta < 0:
        delta = mp.mpf('0')
    elif delta > 1:
        delta = mp.mpf('1')

    return float(delta)



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

    return _compute_gamma_delta(s, t0, rho, eps)


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

    return _compute_gamma_delta(s, t0, rho, eps)


def isotropic_gaussian_scaling(t, r, eps):
    """
    Compute δ_{X,Y}(ε) between Gaussian X and t*X, where X is a isotropic Gaussian random variable.
    
    t is between 0 and 1.
    """

    assert t >= 0 and t <= 1, "t must be between 0 and 1"
    assert eps >= 0, "eps must be >= 0"
    assert r > 0, "r must be a positive integer"

    s = r / 2.0  # shape parameter for chi^2_r related gamma
    rho = 1 / (t**2)

    if np.isclose(rho, 1.0):
        return 0

    # t0 = 2 * (eps + (r/2) * log(rho)) / (rho - 1)
    t0 = 2.0 * (eps + 0.5 * r * np.log(rho)) / (rho - 1.0)
    return _compute_gamma_delta(s, t0, rho, eps)



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

def t0_func(rho, eps, r):
    return 2 * (eps + r/2.0*np.log(rho)) / (rho-1)

def rp_find_rho_for_delta(delta_target, epsilon, r, rho_min=1+1e-6, rho_max=2):
    """
    Find the value of rho such that _compute_gamma_delta(r/2, t0(rho, epsilon, r), rho, epsilon) = delta_target.
    """
    s = r / 2.0
    
    def objective(rho):
        """Objective function: difference between computed delta and target delta."""
        if rho <= 1.0:
            return float('inf')  # rho must be > 1
        t0_val = t0_func(rho, epsilon, r)
        computed_delta = _compute_gamma_delta(s, t0_val, rho, epsilon)
        return computed_delta - delta_target
    
    # Use brentq for bracketed root finding (faster and more reliable)
    rho_solution = brentq(objective, rho_min, rho_max, xtol=1e-10, maxiter=100)
    return rho_solution



def smooth_leverage_upper_bound(X, ell, b, verbose=False):
    """
    Compute an upper bound on the b-smooth leverage S(D) for a database D
    represented by an n x d matrix X (rows are records).

    Assumptions:
      - X has full column rank (so A = X^T X is invertible).
      - All rows that can appear (including potential add/delete neighbors)
        have Euclidean norm <= ell.
      - Adjacency is add/delete.
      - Statistical leverage is defined via H = X (X^T X)^{-1} X^T.
    """

    X = np.asarray(X)
    assert ell > 0, "ell must be nonnegative."
    assert b > 1, "b must be > 1."

    A = X.T @ X
    eigvals = np.linalg.eigvalsh(A)
    lambda_min = float(np.min(eigvals))

    assert lambda_min > 0, "X^T X is not positive definite; leverage not well-defined."

    # Compute leverage scores: h_ii = x_i^T A^{-1} x_i
    # More stable than forming A^{-1}: solve A v = x_i for each row at once.
    # B = A^{-1} X^T  (d x n) via linear solve
    B = np.linalg.solve(A, X.T)          # shape (d, n)
    leverages = np.sum(X * B.T, axis=1)  # shape (n,)
    lev_D = float(np.max(leverages))
    if verbose:
        print(f"lev_D is {lev_D}")

    # k0 = ceil(lambda_min / ell^2)
    k0 = int(math.ceil(lambda_min / (ell ** 2)))
    S_upper = lev_D

    # For 1 <= k <= k0 - 1, use the bound: L_k(D) <= min{1, ell^2 / (lambda_min - k * ell^2)}
    # and consider b^{-k} * L_k(D).
    for k in range(1, max(k0, 1)):  # if k0=0, this loop doesn't run
        denom = lambda_min - k * (ell ** 2)
        if denom <= 0:
            Lk_bound = 1.0
        else:
            Lk_bound = min(1.0, (ell ** 2) / denom)
        term = (b ** (-k)) * Lk_bound
        if term > S_upper:
            S_upper = term

    # Tail k >= k0: L_k(D) <= 1, so supremum of b^{-k} is b^{-k0}
    if k0 > 0:
        tail = b ** (-k0)
        if tail > S_upper:
            S_upper = tail

    return S_upper

def f_smooth(a, b, t, ell, p):
    assert a > 1
    assert b > 1
    assert 0 < t < 1
    assert ell >= 0
    assert 0 < p < 1

    if ell == 0:
        return 0.0

    # k0 = smallest k with b^k p > 1/a (after which g_a becomes positive)
    if p > 1.0 / a:
        k = 0
    else:
        k = math.ceil(math.log(1.0 / (a * p), b))
        if k < 0:
            k = 0

    best = 0.0
    sqrt_a = math.sqrt(a)

    while True:
        x = (b ** k) * p
        val = 0.0
        if x > 1.0 / a:
            val = ell * (t ** k) * math.sqrt(a - 1.0 / x)
        if val > best:
            best = val

        # exact stopping rule using envelope ell*sqrt(a)*t^{k+1}
        if ell * sqrt_a * (t ** (k + 1)) <= best:
            return best

        k += 1