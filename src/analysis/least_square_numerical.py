import numpy as np
from numpy.random import MT19937
import secrets
import warnings
from scipy.stats import qmc, norm
from scipy.special import gammaincc  # regularized upper incomplete gamma
import mpmath as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def radial_expectation_mc(
    lambda1: float,
    lambda2: float,
    lambda3: float,
    c: float,
    d: int,
    n_samples: int = 200_000,
    return_stderr: bool = True,
):
    """
    Monte Carlo estimate of
        E[(1 - exp( 0.5*R^2 * U^T (lambda1 e1 e1^T + lambda2 (I - e1 e1^T)) U
                      - lambda3 * R * e1^T U + c ))_+],
    where Z = RU ~ N(0, I_d).

    Using the decomposition Z=(X, Y) with X ~ N(0,1), Y ~ N(0, I_{d-1}),
    the exponent equals t = (lambda1/2)*X^2 + (lambda2/2)*||Y||^2 - lambda3*X + c.
    """
    seed = secrets.randbits(128)
    rng = np.random.Generator(MT19937(seed))

    # Using antithetic to reduce variance, we generate n//2 samples and append their negatives.
    m = n_samples
    m_half = (m + 1) // 2
    Z = rng.standard_normal(size=(m_half, d))
    Z = np.vstack([Z, -Z])  # pairs (Z, -Z)
    Z = Z[:m]               # trim to exactly n_samples

    X = Z[:, 0]
    if d > 1:
        W = np.sum(Z[:, 1:]**2, axis=1)  # chi^2_{d-1}
    else:
        W = np.zeros_like(X)             # d = 1 edge case

    # Exponent t = (lambda1/2)*X^2 + (lambda2/2)*W - lambda3*X + c
    t = 0.5 * lambda1 * (X**2) + 0.5 * lambda2 * W - lambda3 * X + c

    # We only need exp(t) when t < 0, to avoid overflow on large positive t.
    mask = t < 0.0
    contrib = np.zeros_like(t)
    contrib[mask] = 1.0 - np.exp(t[mask])  # positive part since t<0 => 1-exp(t) > 0
    # When t >= 0, (1 - exp(t))_+ = 0, already default.

    est = contrib.mean()
    if return_stderr:
        # Unbiased sample variance / n
        var = contrib.var(ddof=1)
        stderr = np.sqrt(var / m)
        return float(est), float(stderr)
    else:
        return float(est)

def radial_expectation_qmc(
    lambda1: float,
    lambda2: float,
    lambda3: float,
    c: float,
    d: int,
    n_points: int = 131072,    # power-of-two is best for Sobol
    antithetic: bool = True,   # reflect u -> 1 - u to pair points
    nbatches: int = 16,        # for batch-means stderr
    return_stderr: bool = True,
):
    """
    Quasi–Monte Carlo (Sobol) estimate of

        E[(1 - exp( 0.5*R^2 * U^T (lambda1 e1 e1^T + lambda2 (I - e1 e1^T)) U
                       - lambda3 * R * e1^T U + c ))_+],

    with Z = RU ~ N(0, I_d). Using Z=(X,Y), the exponent is:
        t = (lambda1/2)*X^2 + (lambda2/2)*||Y||^2 - lambda3*X + c.
    """

    if d < 1:
        raise ValueError("Dimension d must be >= 1.")
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")

    # Cryptographically strong seed for scrambling
    seed = secrets.randbits(128)
    sob = qmc.Sobol(d=d, scramble=True, seed=seed)

    # Generate Sobol points in [0,1)^d
    U = sob.random_base2(int(np.floor(np.log2(n_points))))
    n = U.shape[0]

    if antithetic:
        # Reflect: pairs (u, 1-u). Clip for numerical safety at boundaries.
        U_reflect = 1.0 - U
        U = np.vstack([U, U_reflect])
        # Trim to at most n_points if user gave a non-power-of-two
        if U.shape[0] > n_points:
            U = U[:n_points]
        n = U.shape[0]

    # Map to standard normals via inverse CDF
    # (norm.ppf handles vectorized inputs)
    Z = norm.ppf(U)

    X = Z[:, 0]
    if d > 1:
        W = np.sum(Z[:, 1:]**2, axis=1)  # chi^2_{d-1}
    else:
        W = np.zeros_like(X)

    # Exponent t = (lambda1/2)*X^2 + (lambda2/2)*W - lambda3*X + c
    t = 0.5 * lambda1 * (X**2) + 0.5 * lambda2 * W - lambda3 * X + c

    # Only compute exp() where needed (t<0), to avoid overflow.
    mask = t < 0.0
    contrib = np.zeros_like(t)
    contrib[mask] = 1.0 - np.exp(t[mask])

    mean = float(contrib.mean())

    if not return_stderr:
        return mean

    # Batch-means stderr (QMC lacks IID variance; this is a practical heuristic)
    # Ensure at least 2 batches and reasonable batch sizes
    B = max(2, min(nbatches, n))
    m = n // B
    if m < 2:
        # Too few points per batch; fall back to naive sample variance
        stderr = float(np.sqrt(contrib.var(ddof=1) / n))
        return mean, stderr

    trimsz = B * m
    batch_vals = contrib[:trimsz].reshape(B, m).mean(axis=1)
    stderr = float(np.sqrt(batch_vals.var(ddof=1) / B))
    return mean, stderr

def radial_expectation_qmc_reduced_right(
    lambda1: float,
    lambda2: float,
    lambda3: float,
    c: float,
    d: int,
    n_points: int = 131072,   # power of two preferred for Sobol
    antithetic: bool = True,  # reflect u -> 1-u for variance reduction
    nbatches: int = 16,       # for batch-means stderr
    return_stderr: bool = True,
):
    """
    QMC estimate of:
        E_X[ Q( (d-1)/2, s_*(X)/2 )
             - exp(a(X)) * (1 - lambda2)^(-(d-1)/2)
               * Q( (d-1)/2, (1 - lambda2)*s_*(X)/2 ) ],
    where X ~ N(0, 1),
          a(x) = c + 0.5*lambda1*x^2 - lambda3*x,
          s_*(x) = max(0, -2*a(x)/lambda2)  (assuming lambda2 < 0),
          Q(nu, z) = gammaincc(nu, z) (regularized upper incomplete gamma).
    
    This function evaluate the same function as 
        E[(1 - exp( 0.5*R^2 * U^T (lambda1 e1 e1^T + lambda2 (I - e1 e1^T)) U - lambda3 * R * e1^T U + c ))_+],
    """
    if d < 2:
        # The formula is derived for d >= 2 (nu = (d-1)/2 > 0), but works for d=1 with nu=0 as a limit.
        # We enforce d >= 1 for code robustness and warn if needed.
        pass

    # Owen-scrambled Sobol points in 1D (for X only)
    seed = secrets.randbits(64)
    sob = qmc.Sobol(d=1, scramble=True, seed=seed)

    # Generate Sobol points; best results when n_points is a power of two
    m_exp = int(np.floor(np.log2(max(2, n_points))))
    U = sob.random_base2(m_exp)[:, 0]  # shape (2**m_exp,)
    if antithetic:
        U_reflect = 1.0 - U
        U = np.concatenate([U, U_reflect], axis=0)

    # Trim to exactly n_points if requested smaller than generated
    if U.shape[0] > n_points:
        U = U[:n_points]
    n = U.shape[0]

    # Map to X ~ N(0,1)
    X = norm.ppf(U)

    # Precompute parameters
    nu = 0.5 * (d - 1)
    one_minus_l2 = 1.0 - lambda2
    weight = one_minus_l2 ** (-nu)  # (1 - lambda2)^(-nu)

    # a(X) and s_*(X)
    a = c + 0.5 * lambda1 * (X**2) - lambda3 * X
    # Since lambda2 < 0 in your setting, -2*a/lambda2 is nonnegative iff a >= 0
    s_star = np.maximum(0.0, -2.0 * a / lambda2)

    # Q(nu, z) = gammaincc(nu, z)
    term1 = gammaincc(nu, 0.5 * s_star)
    term2 = np.exp(a) * weight * gammaincc(nu, 0.5 * one_minus_l2 * s_star)

    contrib = term1 - term2
    mean = float(np.mean(contrib))

    if not return_stderr:
        return mean

    # Batch-means stderr (heuristic for QMC)
    B = max(2, min(nbatches, n))
    m = n // B
    if m < 2:
        stderr = float(np.sqrt(contrib.var(ddof=1) / n))
        return mean, stderr

    trimsz = B * m
    batch_vals = contrib[:trimsz].reshape(B, m).mean(axis=1)
    stderr = float(np.sqrt(batch_vals.var(ddof=1) / B))
    return mean, stderr

# ----- stable integrand, same as before -----
def _f_reduced_at_x_right(x, lambda1, lambda2, lambda3, c, d, base_dps):
    nu = mp.mpf(d - 1) / 2
    l1 = mp.mpf(lambda1); l2 = mp.mpf(lambda2); l3 = mp.mpf(lambda3); cc = mp.mpf(c)
    one_minus_l2 = 1 - l2
    with mp.workdps(base_dps + 40):
        a = cc + mp.mpf('0.5')*l1*(x*x) - l3*x
        s_star = mp.mpf('0') if a <= 0 else (-2*a/l2)  # l2<0 → s_star>0 when a>0
        Q1 = mp.gammainc(nu, mp.mpf('0.5')*s_star, mp.inf, regularized=True)
        log_scale = a - nu*mp.log(one_minus_l2)
        Q2 = mp.gammainc(nu, mp.mpf('0.5')*one_minus_l2*s_star, mp.inf, regularized=True)
        return Q1 - mp.e**(log_scale) * Q2

def _phi(x):
    return mp.e**(-x*x/2) / mp.sqrt(2*mp.pi)

def _piece_integral_task_right(args):
    """Worker: integrate g(x)=phi(x)f(x) on [a,b] with mpmath in its own process."""
    (a, b, lam1, lam2, lam3, c, d, dps) = args
    mp.mp.dps = dps
    def g(x):
        return _phi(x) * _f_reduced_at_x_right(x, lam1, lam2, lam3, c, d, dps)
    result = mp.quad(g, [a, b])
    # Extract real part and ensure we return a real number
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(f"Result has non-negligible imaginary part: {imag_part:.6e}, using real part only")
    return mp.re(result)

def radial_expectation_quad_adaptive_mp_right(
    lambda1: float,
    lambda2: float,
    lambda3: float,
    c: float,
    d: int,
    dps: int = 100,
    workers: int = 4,
    chunks_per_piece: int = 2,
    tail_sigma: float = 12.0,
    jitter: float = 0.0,
):
    """
    Parallel, deterministic integral of E[f(X)], X~N(0,1), splitting at kinks and
    distributing sub-intervals across processes.

    - workers: # of processes to use
    - chunks_per_piece: split each main piece into this many equal sub-intervals
    - tail_sigma: integrate only within [-L, L], with L = tail_sigma; outside mass is tiny
                  (phi tails beyond ~10σ are < 1e-23)
    """
    mp.mp.dps = dps

    # 1) Find kink locations (roots of a(x) = c + 0.5*λ1 x^2 - λ3 x)
    l1 = mp.mpf(lambda1); l3 = mp.mpf(lambda3); cc = mp.mpf(c)
    A = l1; B = -2*l3; C = 2*cc
    disc = B*B - 4*A*C

    # 2) Build main interval endpoints on a finite window [-L, L]
    L = mp.mpf(tail_sigma)
    pts = [-L]
    if disc > 0:
        sqrtD = mp.sqrt(disc)
        x_minus = (-B - sqrtD) / (2*A)
        x_plus  = (-B + sqrtD) / (2*A)
        # keep only kinks that lie within [-L, L]
        if -L < x_minus < L: pts.append(x_minus)
        if -L < x_plus  < L: pts.append(x_plus)
    elif disc == 0:
        x0 = (-B) / (2*A)
        if -L < x0 < L: pts.append(x0)
    pts.append(L)
    pts = sorted(pts)

    # 3) Optionally jitter away from exact kinks (helps some integrators)
    if jitter and jitter > 0:
        j = mp.mpf(jitter)
        pts = [pts[0]] + [p - j if i%2==1 else p + j for i,p in enumerate(pts[1:-1], start=1)] + [pts[-1]]
        pts = sorted(pts)

    # 4) Split each main piece into sub-intervals for more parallelism
    subints = []
    for i in range(len(pts)-1):
        a, b = pts[i], pts[i+1]
        if chunks_per_piece <= 1:
            subints.append((a, b))
        else:
            # equal splits in x-space (works fine here since phi decays in tails)
            for k in range(chunks_per_piece):
                t0 = mp.mpf(k) / chunks_per_piece
                t1 = mp.mpf(k+1) / chunks_per_piece
                subints.append((a + (b-a)*t0, a + (b-a)*t1))

    # 5) Launch processes
    tasks = [(a, b, lambda1, lambda2, lambda3, c, d, dps) for (a,b) in subints]

    if workers is None or workers <= 1:
        # serial fallback
        parts = [_piece_integral_task_right(t) for t in tasks]
    else:
        parts = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_piece_integral_task_right, t) for t in tasks]
            for f in as_completed(futs):
                parts.append(f.result())

    # 6) Sum contributions
    result = mp.fsum(parts)
    # Extract real part and ensure we return a real number
    # If imaginary part is significant, it indicates a numerical issue
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):  # Warn if imaginary part is significant
            warnings.warn(f"Result has non-negligible imaginary part: {imag_part:.6e}, using real part only")
    return mp.re(result)



# ---------- workers: one for a(x)>0, one for a(x)<=0 ----------
# The same as the _piece_integral_task function, but computed from the leverage (p) and augmented leverage (q)
def _piece_task_pos_right(args):
    """
    Integrate over [a,b] when a(x)>0 throughout the interval.
    g(x) = phi(x) * [ Q(ν, s*/2) - exp(a) (1-λ2)^(-ν) Q(ν, (1-λ2)s*/2) ],
    with s* = -2a/λ2 > 0.
    """
    (aL, bR, lam1, lam2, lam3, c, nu, one_minus_l2, dps) = args
    mp.mp.dps = dps

    half = mp.mpf('0.5')

    def f_pos(x):
        # a(x) strictly > 0 on this piece
        ax = c + half * lam1 * (x*x) - lam3 * x
        s_star = -2*ax/lam2                  # > 0 since λ2<0 and ax>0
        # Q regularized upper incomplete gamma
        Q1 = mp.gammainc(nu, half*s_star, mp.inf, regularized=True)
        Q2 = mp.gammainc(nu, half*one_minus_l2*s_star, mp.inf, regularized=True)
        # stable scaling
        return _phi(x) * (Q1 - mp.e**(ax - nu*mp.log(one_minus_l2)) * Q2)

    result = mp.quad(f_pos, [aL, bR])
    # Extract real part and ensure we return a real number
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(f"Result has non-negligible imaginary part: {imag_part:.6e}, using real part only")
    return mp.re(result)

def _piece_task_neg_right(args):
    """
    Integrate over [a,b] when a(x)<=0 throughout the interval.
    Here Q(ν,0)=1 for both terms, so
      g(x) = phi(x) * [ 1 - exp(a) (1-λ2)^(-ν) ].
    No gamma calls ⇒ faster and stabler.
    """
    (aL, bR, lam1, lam2, lam3, c, nu, one_minus_l2, dps) = args
    mp.mp.dps = dps

    half = mp.mpf('0.5')
    log_scale_const = -nu * mp.log(one_minus_l2)  # log((1-λ2)^(-ν))

    def f_neg(x):
        ax = c + half * lam1 * (x*x) - lam3 * x   # ≤ 0 on this piece
        return _phi(x) * (1 - mp.e**(ax + log_scale_const))

    result = mp.quad(f_neg, [aL, bR])
    # Extract real part and ensure we return a real number
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(f"Result has non-negligible imaginary part: {imag_part:.6e}, using real part only")
    return mp.re(result)

# ---------- main API ----------
def delta_pq_ls_right(p, q, d, r, eps, workers=32, sign_w=1, dps=60,
                chunks_per_piece=2, tail_sigma=12.0):
    """
    Fast/stable δ_{X,Y}(ε) computed directly from (p,q,d,r,eps), with
    sign-fixed pieces: no per-point sign checks in the integrand.
    """
    mp.mp.dps = int(dps)
    if not (0 < p < q < 1):
        raise ValueError("Require 0 < p < q < 1.")
    if r < 0:
        raise ValueError("Require r >= 0.")
    if sign_w not in (-1, 1):
        raise ValueError("sign_w should be ±1.")

    # ---- map (p,q,d,r,eps) → λ's, constants ----
    lam2 = -(q - p) / (1 - q)                                  # λ2 < 0
    lam1 = (2*p - p*p - q) / (1 - q)
    lam3 = sign_w * mp.sqrt(r) * (1 - p) * mp.sqrt(p) * mp.sqrt(q - p) / (1 - q)
    c = (eps
         + (d + 1) / 2 * mp.log(1 - p)
         - (d / 2) * mp.log(1 - q)
         - (r * p / 2) * ((q - p) / (1 - q)))
    one_minus_l2 = 1 - lam2
    nu = mp.mpf(d - 1) / 2

    # ---- determine kink locations: a(x)=0 roots ----
    # a(x) = c + 0.5*λ1 x^2 - λ3 x  =>  λ1 x^2 - 2λ3 x + 2c = 0
    A = mp.mpf(lam1)
    B = -2 * mp.mpf(lam3)
    C = 2 * mp.mpf(c)
    disc = B*B - 4*A*C

    L = mp.mpf(tail_sigma)
    pts = [-L]

    if A == 0:
        # a(x) linear: a(x) = c - λ3 x. One root at x = c/λ3 if λ3 != 0.
        if lam3 != 0:
            x0 = c / lam3
            if -L < x0 < L:
                pts.append(x0)
    else:
        if disc > 0:
            sqrtD = mp.sqrt(disc)
            x1 = (-B - sqrtD) / (2*A)
            x2 = (-B + sqrtD) / (2*A)
            if -L < x1 < L: pts.append(x1)
            if -L < x2 < L: pts.append(x2)
        elif disc == 0:
            x0 = (-B) / (2*A)
            if -L < x0 < L: pts.append(x0)

    pts.append(L)
    pts = sorted(pts)

    # ---- build sub-intervals, tagging each with sign(a) on the piece ----
    pieces = []
    half = mp.mpf('0.5')
    for i in range(len(pts)-1):
        aL, bR = pts[i], pts[i+1]
        # pick a midpoint safely inside (handle degenerate tiny intervals)
        xm = (aL + bR) / 2
        axm = c + half * lam1 * (xm*xm) - lam3 * xm
        is_pos = (axm > 0)
        # sub-split for parallelism
        if chunks_per_piece <= 1:
            pieces.append((aL, bR, is_pos))
        else:
            for k in range(chunks_per_piece):
                t0 = mp.mpf(k) / chunks_per_piece
                t1 = mp.mpf(k+1) / chunks_per_piece
                subL = aL + (bR - aL) * t0
                subR = aL + (bR - aL) * t1
                # keep the same sign tag for the whole piece (safe between roots)
                pieces.append((subL, subR, is_pos))

    # ---- launch workers with sign-specific kernels ----
    tasks_pos, tasks_neg = [], []
    for (aL, bR, is_pos) in pieces:
        payload = (aL, bR, lam1, lam2, lam3, c, nu, one_minus_l2, dps)
        if is_pos:
            tasks_pos.append(payload)
        else:
            tasks_neg.append(payload)

    parts = []

    # run serially if workers<=1 for reproducibility
    if not workers or workers <= 1:
        for t in tasks_pos:
            parts.append(_piece_task_pos_right(t))
        for t in tasks_neg:
            parts.append(_piece_task_neg_right(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = []
            for t in tasks_pos:
                futs.append(ex.submit(_piece_task_pos_right, t))
            for t in tasks_neg:
                futs.append(ex.submit(_piece_task_neg_right, t))
            for f in as_completed(futs):
                parts.append(f.result())

    result = mp.fsum(parts)
    # Extract real part and ensure we return a real number
    # If imaginary part is significant, it indicates a numerical issue
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):  # Warn if imaginary part is significant
            warnings.warn(f"Result has non-negligible imaginary part: {imag_part:.6e}, using real part only")
    return mp.re(result)

def _f_reduced_at_x_left(x, lambda1, lambda2, lambda3, c, d, base_dps):
    """
    Radial reduction integrand for the case 0 < lambda2 < 1.

    Implements:
        f(x) = [1 - Q(ν, s*/2)] - exp(a) (1-λ2)^(-ν) [1 - Q(ν, (1-λ2)s*/2)],
    where
        ν = (d-1)/2,
        a(x) = c + 0.5*λ1 x^2 - λ3 x,
        s* = max(0, -2 a(x)/λ2),
    and Q is the regularized upper incomplete gamma.
    """
    nu = mp.mpf(d - 1) / 2
    l1 = mp.mpf(lambda1)
    l2 = mp.mpf(lambda2)
    l3 = mp.mpf(lambda3)
    cc = mp.mpf(c)
    one_minus_l2 = 1 - l2

    with mp.workdps(base_dps + 40):
        a = cc + mp.mpf('0.5') * l1 * (x * x) - l3 * x

        # For 0 < λ2 < 1, the positive part is on [0, s*] with s* = max(0, -2a/λ2).
        if a >= 0:
            # Then s* <= 0, so the positive region has measure zero → contribution 0.
            return mp.mpf('0')

        s_star = -2 * a / l2   # > 0 since a < 0 and λ2 > 0

        # Q(ν, z) = regularized upper incomplete gamma
        Q1 = mp.gammainc(nu, mp.mpf('0.5') * s_star, mp.inf, regularized=True)
        log_scale = a - nu * mp.log(one_minus_l2)
        Q2 = mp.gammainc(nu, mp.mpf('0.5') * one_minus_l2 * s_star,
                         mp.inf, regularized=True)

        # (1 - Q1) - exp(a) (1-λ2)^(-ν) (1 - Q2)
        term1 = 1 - Q1
        term2 = mp.e**(log_scale) * (1 - Q2)
        return term1 - term2


def _piece_integral_task_left(args):
    """Worker: integrate g(x)=phi(x) f_pos(x) on [a,b] with mpmath in its own process."""
    (a, b, lam1, lam2, lam3, c, d, dps) = args
    mp.mp.dps = dps

    def g(x):
        return _phi(x) * _f_reduced_at_x_left(x, lam1, lam2, lam3, c, d, dps)

    result = mp.quad(g, [a, b])

    # Extract real part and ensure we return a real number
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(
                f"Result has non-negligible imaginary part: {imag_part:.6e}, "
                "using real part only"
            )
    return mp.re(result)


def radial_expectation_quad_adaptive_mp_left(
    lambda1: float,
    lambda2: float,
    lambda3: float,
    c: float,
    d: int,
    dps: int = 100,
    workers: int = 4,
    chunks_per_piece: int = 2,
    tail_sigma: float = 12.0,
    jitter: float = 0.0,
):
    """
    Parallel, deterministic integral of E[f(X)], X~N(0,1), for the case 0 < lambda2 < 1,
    splitting at the kinks of a(x) and distributing sub-intervals across processes.

    - workers: # of processes to use
    - chunks_per_piece: split each main piece into this many equal sub-intervals
    - tail_sigma: integrate only within [-L, L], with L = tail_sigma; outside mass is tiny
                  (phi tails beyond ~10σ are < 1e-23)
    """
    mp.mp.dps = dps

    # 1) Find kink locations (roots of a(x) = c + 0.5*λ1 x^2 - λ3 x)
    assert 0 < lambda2 < 1, "lambda2 must be between 0 and 1"
    l1 = mp.mpf(lambda1)
    l3 = mp.mpf(lambda3)
    cc = mp.mpf(c)
    A = l1
    B = -2 * l3
    C = 2 * cc
    disc = B * B - 4 * A * C

    # 2) Build main interval endpoints on a finite window [-L, L]
    L = mp.mpf(tail_sigma)
    pts = [-L]
    if disc > 0:
        sqrtD = mp.sqrt(disc)
        x_minus = (-B - sqrtD) / (2 * A)
        x_plus = (-B + sqrtD) / (2 * A)
        # keep only kinks that lie within [-L, L]
        if -L < x_minus < L:
            pts.append(x_minus)
        if -L < x_plus < L:
            pts.append(x_plus)
    elif disc == 0:
        x0 = (-B) / (2 * A)
        if -L < x0 < L:
            pts.append(x0)
    pts.append(L)
    pts = sorted(pts)

    # 3) Optionally jitter away from exact kinks (helps some integrators)
    if jitter and jitter > 0:
        j = mp.mpf(jitter)
        pts = (
            [pts[0]] +
            [p - j if i % 2 == 1 else p + j
             for i, p in enumerate(pts[1:-1], start=1)] +
            [pts[-1]]
        )
        pts = sorted(pts)

    # 4) Split each main piece into sub-intervals for more parallelism
    subints = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if chunks_per_piece <= 1:
            subints.append((a, b))
        else:
            # equal splits in x-space (works fine here since phi decays in tails)
            for k in range(chunks_per_piece):
                t0 = mp.mpf(k) / chunks_per_piece
                t1 = mp.mpf(k + 1) / chunks_per_piece
                subints.append((a + (b - a) * t0, a + (b - a) * t1))

    # 5) Launch processes
    tasks = [(a, b, lambda1, lambda2, lambda3, c, d, dps) for (a, b) in subints]

    if workers is None or workers <= 1:
        # serial fallback
        parts = [_piece_integral_task_left(t) for t in tasks]
    else:
        parts = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_piece_integral_task_left, t) for t in tasks]
            for f in as_completed(futs):
                parts.append(f.result())

    # 6) Sum contributions
    result = mp.fsum(parts)

    # Extract real part and ensure we return a real number
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(
                f"Result has non-negligible imaginary part: {imag_part:.6e}, "
                "using real part only"
            )
    return mp.re(result)
