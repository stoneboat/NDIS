import numpy as np
from scipy.stats import norm

def two_sided_tail(a, t):
    """
    T(a, t) = P(|Z - a| >= t) for Z ~ N(0,1).
    Uses: T(a,t) = SF(a+t) + SF(t-a), where SF is 1 - CDF.
    Works with scalars or numpy arrays (broadcasting).
    """
    a = np.asarray(a)
    t = np.asarray(t)
    return norm.sf(a + t) + norm.sf(t - a)

def delta_gaussian_1d_closed_form(mu, sigma1, sigma2, eps):
    """
        Compute δ_{X̂,Ŷ}(ε) for 1D Gaussians with σ1 >= σ2 > 0,
        where X̂ ~ N(0, σ1^2) and Ŷ ~ N(mu, σ2^2).
    """
    # Input checks
    assert sigma1 >= sigma2 > 0, "sigma1 must be greater than sigma2 and positive."

    mu = np.asarray(mu, dtype=float)
    eps = np.asarray(eps, dtype=float)

    gamma = sigma1 / sigma2  # > 1
    gam2 = gamma * gamma

    # m and r as in the derivation
    m = mu / (sigma2 * (gam2 - 1.0))
    num = 2.0 * eps + 2.0 * np.log(gamma) + (mu * mu) / (sigma1 * sigma1)
    den = 1.0 - 1.0 / gam2  # = (gam2 - 1)/gam2 > 0
    r_sq = m * m + num / den
    # Numerical safety: r should be real and nonnegative
    r_sq = np.maximum(r_sq, 0.0)
    r = np.sqrt(r_sq)

    # T(γ m, r/γ) - e^ε T(m, r)
    term1 = two_sided_tail(gamma * m, r / gamma)
    term2 = two_sided_tail(m, r)
    delta = term1 - np.exp(eps) * term2

    # Optional: clamp tiny numerical drift back into [0,1]
    return np.clip(delta, 0.0, 1.0)


def delta_gaussian_1d_closed_form_reverse(mu, sigma1, sigma2, eps):
    """
        Compute δ_{Ŷ, X̂}(ε) for 1D Gaussians with σ1 >= σ2 > 0,
        where X̂ ~ N(0, σ1^2) and Ŷ ~ N(mu, σ2^2).
    """
    assert sigma1 >= sigma2 > 0
    mu  = np.asarray(mu,  dtype=float)
    eps = np.asarray(eps, dtype=float)
    if np.any(eps < 0): raise ValueError("eps must be ≥ 0")

    gamma = sigma1 / sigma2
    gam2  = gamma * gamma
    m = mu / (sigma2 * (gam2 - 1.0))
    num = 2.0 * np.log(gamma) + (mu * mu) / (sigma1 * sigma1) - 2.0 * eps
    den = 1.0 - 1.0 / gam2
    r2 = m*m + num/den
    r  = np.sqrt(np.maximum(r2, 0.0))

    delta = (1.0 - np.exp(eps)) + np.exp(eps) * two_sided_tail(gamma*m, r/gamma) - two_sided_tail(m, r)
    return np.clip(delta, 0.0, 1.0)


def delta_gaussian_shifted_covariance_closed_form(eps: float, sigma1: float, sigma2: float) -> float:
    """
        Compute δ_{σ1^2, σ2^2}(ε) for 1D zero-mean Gaussians with σ1 >= σ2 > 0,
        where X ~ N(0, σ1^2) and Y ~ N(0, σ2^2).
    """
    # Input checks
    assert sigma1 >= sigma2 > 0, "sigma1 must be greater than sigma2 and positive."
    gamma = float(sigma1) / float(sigma2)

    # Equal-variance case: δ = 0 for all ε
    if np.isclose(gamma, 1.0) or eps == 0:
        return 0.0

    # Compute s(ε)
    s_num = 2.0 * (eps + np.log(gamma))          # >= 0 since eps>=0 and gamma>=1
    s_den = gamma**2 - 1.0                       # > 0 because gamma>1
    s = np.sqrt(s_num / s_den)

    # Upper tail Φ̅(t) = norm.sf(t)
    phibar_s = norm.sf(s)
    phibar_gs = norm.sf(gamma * s)

    delta = 2.0 * phibar_s - 2.0 * np.exp(eps) * phibar_gs

    # Clip tiny negative due to roundoff
    return min(max(0.0, delta), 1.0)


def delta_gaussian_shifted_covariance_closed_form_reverse(eps: float, sigma1: float, sigma2: float) -> float:
    """
    Compute δ_{σ2^2, σ1^2}(ε) for 1D zero-mean Gaussians with σ1 >= σ2 > 0,
    where X ~ N(0, σ1^2) and Y ~ N(0, σ2^2). (Reverse order of your forward function.)
    """
    # Input checks to match your forward function's assumptions
    assert sigma1 >= sigma2 > 0, "sigma1 must be greater than sigma2 and positive."
    assert eps >= 0, "eps must be nonnegative."

    gamma = float(sigma1) / float(sigma2)

    # Equal-variance case: δ = 0 for all ε
    if np.isclose(gamma, 1.0) or eps == 0:
        return 0.0

    log_gamma = np.log(gamma)

    # If ε >= log γ, the active band disappears => δ = 0
    if eps >= log_gamma:
        return 0.0

    # s'(ε)
    sp_num = 2.0 * (log_gamma - eps)     # > 0 here
    sp_den = gamma**2 - 1.0              # > 0 since gamma > 1
    sp = np.sqrt(sp_num / sp_den)

    # Φ(t) = norm.cdf(t)
    phi_sp = norm.cdf(sp)
    phi_gsp = norm.cdf(gamma * sp)

    delta = 2.0 * phi_gsp - 2.0 * np.exp(eps) * phi_sp + (np.exp(eps) - 1.0)

    # Clip to [0, 1] for numerical safety
    return float(np.clip(delta, 0.0, 1.0))

