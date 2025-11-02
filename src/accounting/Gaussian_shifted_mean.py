# pip install dp-accounting; here I use the version 0.5.0
import dp_accounting as dpa
from dp_accounting import pld
import numpy as np
from scipy.special import erf, erfc
from scipy.signal import fftconvolve
from numpy.linalg import eigh

def delta_via_dp_accounting(eps: float, sigma: float, k: int, mu_norm: float = 1.0) -> float:
    """
    Returns the smallest delta for a fixed epsilon when releasing k i.i.d. samples
    from N(0, sigma^2 I) vs N(mu, sigma^2 I), where ||mu||_2 = mu_norm.
    """
    # In dp_accounting, GaussianDpEvent(noise_multiplier) expects sigma / sensitivity.
    noise_multiplier = sigma / mu_norm  # here mu_norm = 1 for your case
    
    # A single Gaussian mechanism invocation:
    g = dpa.GaussianDpEvent(noise_multiplier=noise_multiplier)
    
    # Compose k times (k i.i.d. draws):
    composed = dpa.SelfComposedDpEvent(g, k)
    
    # Use the PLD (privacy-loss distribution) accountant.
    acc = pld.PLDAccountant()
    acc.compose(composed)
    
    # Smallest delta achieving (eps, delta)-DP:
    return acc.get_delta(eps)

def delta_gaussian_shift_closed_form(eps: float, sigma: float, k: int, mu_norm: float = 1.0) -> float:
    Delta = mu_norm
    m = (Delta**2) / (2.0 * sigma**2)
    v = (Delta**2) / (sigma**2)
    sd = np.sqrt(k * v)

    def tail(z):
        return 0.5 * erfc(z / np.sqrt(2.0))

    termP = tail((eps - k*m) / sd)
    termQ = tail((eps + k*m) / sd)
    return float(np.maximum(0.0, termP - np.exp(eps) * termQ))


def delta_gaussian_same_covariance_composition(eps: float, Sigma: np.ndarray, mu: np.ndarray,
                             k: int = 1, null_tol: float = 1e-12) -> float:
    """
    δ_ε between N(0, Σ) and N(μ, Σ). Σ may be PSD. `k` = number of i.i.d. releases (composition).
    For a single comparison, use k=1.
    """
    lam, U = eigh(Sigma)
    pos = lam > null_tol
    if not np.any(pos):
        return 1.0 if np.linalg.norm(mu) > 0 else float(eps < 0)  # degenerate Σ

    mu_coords = U.T @ mu
    mu_null = mu_coords[~pos]
    if np.linalg.norm(mu_null) > null_tol:
        return 1.0

    # Mahalanobis norm: Δ^2 = μ^T Σ^+ μ = sum( mu_i^2 / λ_i ) over support
    Delta2 = np.sum((mu_coords[pos] ** 2) / lam[pos])
    Delta = float(np.sqrt(Delta2))

    return delta_gaussian_shift_closed_form(eps=eps, sigma=1.0, k=k, mu_norm=Delta)



def _normal_cdf(x, mean, std):
    z = (x - mean) / (std * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))

def _discretize_normal_to_grid(mean, std, L, h):
    """
    Discretize a 1D Normal(mean, std^2) to a probability mass function (pmf)
    on grid points x_i = -L + i*h (i=0..M-1), assigning each bin the probability
    mass over [x_i - h/2, x_i + h/2], truncated to [-L, L], then normalized.

    Mirrors DiscretizePRV (Alg. 2) idea from the paper: truncate and renormalize.
    """
    # Ensure an odd number of points so that x=0 is exactly the middle bin.
    M = int(round(2 * L / h)) + 1
    if M % 2 == 0:
        M += 1
    xs = -L + h * np.arange(M)

    # Bin edges (left/right), clamped to [-L, L]
    left_edges  = np.maximum(xs - 0.5 * h, -L)
    right_edges = np.minimum(xs + 0.5 * h,  L)

    # Probability per bin via CDF differences
    cdf_left  = np.vectorize(_normal_cdf)(left_edges,  mean, std)
    cdf_right = np.vectorize(_normal_cdf)(right_edges, mean, std)
    pmf = (cdf_right - cdf_left).astype(np.float64)

    s = pmf.sum()
    if s <= 0:
        raise ValueError("Grid too small: all mass truncated. Increase L or h.")
    pmf /= s  # renormalize after truncation
    return xs, pmf, M

# Linear convolution via FFT
def _fft_linear_convolve(a, b):
    """ Linear convolution via FFT """
    c = fftconvolve(a, b, mode="full")
    c = np.maximum(c, 0.0)
    c /= c.sum()

    return c

def _crop_center(c, M):
    """
    After convolving two M-length pmfs (on [-L, L]), we get length 2M-1,
    representing [-2L, 2L] with step h. Crop the center window back to [-L, L].
    """
    n = len(c)
    if n < M:
        raise ValueError("Cannot crop: result shorter than target length.")
    start = (n - M) // 2
    end = start + M
    r = c[start:end]
    s = r.sum()
    if s > 0:
        r /= s
    return r

def _pmf_power_fft(base_pmf, k, M):
    """
    Compose k i.i.d. copies via FFT using exponentiation by squaring + cropping.
    All pmfs are defined on the same fixed grid length M ([-L, L]).
    """
    # Dirac at 0 on the same grid: mass 1 at center bin.
    r = np.zeros_like(base_pmf)
    r[M // 2] = 1.0
    p = base_pmf.copy()

    kk = k
    while kk > 0:
        if kk & 1:
            r = _crop_center(_fft_linear_convolve(r, p), M)
        kk >>= 1
        if kk:
            p = _crop_center(_fft_linear_convolve(p, p), M)
    return r

def _tail_prob_from_pmf(xs, pmf, h, eps):
    """
    Compute Pr[Z > eps] from discretized pmf (bins centered at xs[i]),
    counting bins whose upper edge <= eps as '≤ eps'.
    """
    upper_edges = xs + 0.5 * h
    # number of bins fully at or below eps
    idx = np.searchsorted(upper_edges, eps, side='right')
    cdf_eps = pmf[:idx].sum() if idx > 0 else 0.0
    return max(0.0, 1.0 - cdf_eps)

def delta_via_fft_accounting(eps, sigma, k, mu_norm=1.0, h=None, L=None, target_M=1<<16):
    """
    Numerical (FFT-based) composition of Gaussian PRVs from 'Numerical Composition of DP'
    (Gopi, Lee, Wutschitz). Returns the smallest delta for a fixed epsilon.

    Parameters
    ----------
    eps : float
        Privacy parameter epsilon to evaluate.
    sigma : float
        Gaussian noise std (for N(0, sigma^2 I_d) vs N(mu, sigma^2 I_d)).
    k : int
        Number of i.i.d. compositions (independent releases).
    mu_norm : float, default 1.0
        L2-norm of the mean shift ||mu||_2 (sensitivity). Your case is 1.0.
    h : float or None
        Grid step. If None, chosen adaptively to keep grid size ~ target_M.
    L : float or None
        Truncation half-width. If None, choose heuristically from k, sigma, mu_norm, eps.
    target_M : int
        Target number of grid points (odd). Controls memory/time.

    Notes
    -----
    - Step (i) discretize PRVs (Alg. 2), (ii) compose via FFT (Alg. 1), (iii) compute
      δ(ε)=Pr[X_sum>ε] - e^{ε} Pr[Y_sum>ε]. See paper, pp. 9–10; Gaussian PRVs: Appx. B.1. 
    """
    # Base PRV parameters for Gaussian mechanism
    Delta = float(mu_norm)
    m = (Delta**2) / (2.0 * sigma**2)      # mean of X; Y has -m
    v = (Delta**2) / (sigma**2)            # variance of both X and Y
    std = np.sqrt(v)

    # Heuristic domain: capture most mass of k-fold sum for both X and Y
    mean_mag = abs(k * m)
    sd_sum = np.sqrt(k) * std
    if L is None:
        # 10 sd beyond the larger |mean| and epsilon; tweakable
        L = max(abs(eps) + 6.0 * sd_sum, mean_mag + 6.0 * sd_sum) + 1.0
    if L <= 0:
        raise ValueError("L must be positive.")

    # Choose h to hit target_M (odd)
    if h is None:
        h = max((2.0 * L) / (target_M - 1), 1e-5)
    M = int(round(2 * L / h)) + 1
    if M % 2 == 0:
        M += 1
    # Recompute h from finalized M to ensure xs exactly span [-L, L]
    h = 2.0 * L / (M - 1)

    # Discretize base PRVs X and Y
    xs, px, Mx = _discretize_normal_to_grid(+m, std, L, h)
    _,  py, My = _discretize_normal_to_grid(-m, std, L, h)
    assert Mx == M and My == M

    # k-fold composition via FFT with repeated squaring + cropping (Alg. 1 spirit)
    px_k = _pmf_power_fft(px, k, M)
    py_k = _pmf_power_fft(py, k, M)

    # Evaluate delta(eps) = P[X_sum > eps] - e^eps P[Y_sum > eps]
    tail_x = _tail_prob_from_pmf(xs, px_k, h, eps)
    tail_y = _tail_prob_from_pmf(xs, py_k, h, eps)
    delta = tail_x - np.exp(eps) * tail_y
    # Numerical stability
    return max(0.0, min(1.0, float(delta)))
