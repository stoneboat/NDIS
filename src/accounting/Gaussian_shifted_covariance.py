import numpy as np
from scipy.stats import norm

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

