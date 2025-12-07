import mpmath as mp
import warnings
from concurrent.futures import ProcessPoolExecutor
from analysis.least_square_numerical import _phi, delta_pq_ls_left

def _Q_reg_upper(nu: mp.mpf, z: mp.mpf) -> mp.mpf:
    """
    Regularized upper incomplete gamma Q(ν, z).
    mpmath's gammainc with `regularized=True` gives the upper tail.
    """
    if z <= 0:
        # Q(ν,0) = 1 (upper regularized)
        return mp.mpf('1')
    return mp.gammainc(nu, z, mp.inf, regularized=True)

def _w_nu(nu: mp.mpf, z: mp.mpf) -> mp.mpf:
    """
    w_ν(z) = z^{ν-1} e^{-z} / Γ(ν), which is -∂_z Q(ν,z).
    """
    if z <= 0:
        return mp.mpf('0')
    return z**(nu - 1) * mp.e**(-z) / mp.gamma(nu)

def _Phi(x: mp.mpf) -> mp.mpf:
    """
    Standard normal CDF Φ(x).
    """
    return mp.mpf('0.5') * (1 + mp.erf(x / mp.sqrt(2)))


def _x_from_u(u: mp.mpf) -> mp.mpf:
    """
    Inverse CDF: x = Φ^{-1}(u).
    """
    return mp.sqrt(2) * mp.erfinv(2*u - 1)

# ---------- term integrals (each worker computes one term) ----------

def _term1_integral(args) -> mp.mpf:
    """
    T1 = E[ w_ν(z₁(X)) ∂_p(-a(X)(1-p)/s)_+ ] over a(x)<0 pieces.
    Here a(x)<0 on all pieces.
    """
    (pieces_neg, lam1, lam3, c,
     lam1prime, lam3prime, cprime,
     nu, p_mp, s_mp, t, tprime, dps) = args

    mp.mp.dps = int(dps)
    half = mp.mpf('0.5')
    one = mp.mpf('1')

    def a_of_x(x):
        return c + half * lam1 * (x*x) - lam3 * x

    def a_p_of_x(x):
        return cprime + half * lam1prime * (x*x) - lam3prime * x

    def integrand(x):
        ax = a_of_x(x)
        # By construction ax < 0 on pieces_neg, but we keep a safety check.
        if ax >= 0:
            return mp.mpf('0')

        z1 = -ax * (one - p_mp) / s_mp   # (-a)*(1-p)/s
        w1 = _w_nu(nu, z1)
        ap = a_p_of_x(x)

        # d/dp[-a * (1-p)/s] = a/s - (1-p)*a_p/s
        d_z1_dp = ax / s_mp - (one - p_mp) * ap / s_mp
        return _phi(x) * w1 * d_z1_dp

    parts = []
    for (aL, bR) in pieces_neg:
        val = mp.quad(integrand, [aL, bR])
        parts.append(val)

    result = mp.fsum(parts)
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(
                f"T1: non-negligible imaginary part {imag_part:.6e}, using real part"
            )
    return mp.re(result)


def _term2_integral(args) -> mp.mpf:
    """
    T2 = -ν t^{ν-1} t' E[ e^{a(X)} (1 - Q(ν,z₂(X))) ] over a(x)<0 pieces.
    """
    (pieces_neg, lam1, lam3, c,
     lam1prime, lam3prime, cprime,
     nu, p_mp, s_mp, t, tprime, dps) = args

    mp.mp.dps = int(dps)
    half = mp.mpf('0.5')
    one = mp.mpf('1')

    coeff2 = -nu * t**(nu - 1) * tprime

    def a_of_x(x):
        return c + half * lam1 * (x*x) - lam3 * x

    def integrand(x):
        ax = a_of_x(x)
        if ax >= 0:
            return mp.mpf('0')

        z2 = -ax * (one - p_mp - s_mp) / s_mp
        Q2 = _Q_reg_upper(nu, z2)
        return _phi(x) * mp.e**(ax) * (1 - Q2)

    parts = []
    for (aL, bR) in pieces_neg:
        val = mp.quad(integrand, [aL, bR])
        parts.append(val)

    base = mp.fsum(parts)
    result = coeff2 * base
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(
                f"T2: non-negligible imaginary part {imag_part:.6e}, using real part"
            )
    return mp.re(result)


def _term3_integral(args) -> mp.mpf:
    """
    T3 = -t^ν E[ e^{a(X)} a_p(X) (1 - Q(ν,z₂(X))) ] over a(x)<0 pieces.
    """
    (pieces_neg, lam1, lam3, c,
     lam1prime, lam3prime, cprime,
     nu, p_mp, s_mp, t, tprime, dps) = args

    mp.mp.dps = int(dps)
    half = mp.mpf('0.5')
    one = mp.mpf('1')

    coeff3 = -t**nu

    def a_of_x(x):
        return c + half * lam1 * (x*x) - lam3 * x

    def a_p_of_x(x):
        return cprime + half * lam1prime * (x*x) - lam3prime * x

    def integrand(x):
        ax = a_of_x(x)
        if ax >= 0:
            return mp.mpf('0')

        ap = a_p_of_x(x)
        z2 = -ax * (one - p_mp - s_mp) / s_mp
        Q2 = _Q_reg_upper(nu, z2)
        return _phi(x) * mp.e**(ax) * ap * (1 - Q2)

    parts = []
    for (aL, bR) in pieces_neg:
        val = mp.quad(integrand, [aL, bR])
        parts.append(val)

    base = mp.fsum(parts)
    result = coeff3 * base
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(
                f"T3: non-negligible imaginary part {imag_part:.6e}, using real part"
            )
    return mp.re(result)


def _term4_integral(args) -> mp.mpf:
    """
    T4 = -t^ν E[ e^{a(X)} w_ν(z₂(X)) ∂_p(-a(X)(1-p-s)/s)_+ ] over a(x)<0 pieces.
    """
    (pieces_neg, lam1, lam3, c,
     lam1prime, lam3prime, cprime,
     nu, p_mp, s_mp, t, tprime, dps) = args

    mp.mp.dps = int(dps)
    half = mp.mpf('0.5')
    one = mp.mpf('1')

    coeff4 = -t**nu

    def a_of_x(x):
        return c + half * lam1 * (x*x) - lam3 * x

    def a_p_of_x(x):
        return cprime + half * lam1prime * (x*x) - lam3prime * x

    def integrand(x):
        ax = a_of_x(x)
        if ax >= 0:
            return mp.mpf('0')

        z2 = -ax * (one - p_mp - s_mp) / s_mp
        w2 = _w_nu(nu, z2)
        ap = a_p_of_x(x)

        # d/dp[-a * (1-p-s)/s] = a/s - (1-p-s)*a_p/s
        d_z2_dp = ax / s_mp - (one - p_mp - s_mp) * ap / s_mp
        return _phi(x) * mp.e**(ax) * w2 * d_z2_dp

    parts = []
    for (aL, bR) in pieces_neg:
        val = mp.quad(integrand, [aL, bR])
        parts.append(val)

    base = mp.fsum(parts)
    result = coeff4 * base
    if mp.im(result) != 0:
        imag_part = abs(mp.im(result))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(
                f"T4: non-negligible imaginary part {imag_part:.6e}, using real part"
            )
    return mp.re(result)


# ---------- main API ----------

def fprime_p(
    p: float,
    s: float,
    d: int,
    r: float,
    eps: float,
    workers: int = 1,
    dps: int = 60,
    tail_sigma: float = 12.0,
) -> mp.mpf:
    """
    Compute f'(p) for the fixed-residual-share setting:
      augmented leverage = p + s,
      r >= 1, d >= 2, eps >= 0, 0 < s < 1, 0 < p < 1 - s.

    The derivative is expressed as four Gaussian expectations, each implemented
    as an integral over the region where a(x) < 0 (active region, where the
    max(·)_+ terms are nonzero).

    Parameters
    ----------
    p : float
        Leverage ∈ (0, 1-s).
    s : float
        Residual share ∈ (0, 1).
    d : int
        Dimension, d >= 2.
    r : float
        Regularization parameter (scaled), r >= 1.
    eps : float
        ε >= 0.
    workers : int
        If <=1: run serial.
        If >1: spawn up to 4 workers, one per term T1–T4.
    dps : int
        mpmath decimal precision.
    tail_sigma : float
        Truncation parameter L: integrate over [-L, L].
    """
    # ---- basic checks ----
    if r < 1:
        raise ValueError("Require r >= 1.")
    if d < 2:
        raise ValueError("Require d >= 2.")
    if not (0 < s < 1):
        raise ValueError("Require 0 < s < 1.")
    if not (0 < p < 1 - s):
        raise ValueError("Require 0 < p < 1 - s.")
    if eps < 0:
        raise ValueError("Require eps >= 0.")
    if tail_sigma <= 0:
        raise ValueError("Require tail_sigma > 0.")

    mp.mp.dps = int(dps)
    one = mp.mpf('1')
    half = mp.mpf('0.5')

    p_mp = mp.mpf(p)
    s_mp = mp.mpf(s)
    r_mp = mp.mpf(r)
    eps_mp = mp.mpf(eps)
    d_mp = mp.mpf(d)

    # ---- parameters λ1, λ3, c and their derivatives ----
    # λ1(p) = (p^2 - p + s)/(1-p)^2
    lam1 = (p_mp**2 - p_mp + s_mp) / (one - p_mp)**2

    # λ3(p) = -sqrt(r p s) sqrt(1-p-s) / (1-p)^2
    lam3 = -mp.sqrt(r_mp * p_mp * s_mp) * mp.sqrt(one - p_mp - s_mp) / (one - p_mp)**2

    # c(p) = eps + 0.5 [ d ln(1-p-s) - (d+1) ln(1-p) ] - 0.5 r p s / (1-p)^2
    c = (
        eps_mp
        + half * (d_mp * mp.log(one - p_mp - s_mp) - (d_mp + 1) * mp.log(one - p_mp))
        - half * r_mp * p_mp * s_mp / (one - p_mp)**2
    )

    # derivatives:
    # λ1'(p) = (1 - p - 2s)/(1-p)^3
    lam1prime = (p_mp + 2 * s_mp - one) / (one - p_mp)**3

    # c'(p) = -d/(2(1-p-s)) + (d+1)/(2(1-p)) - 0.5 r s (1+p)/(1-p)^3
    cprime = (
        -d_mp / (2 * (one - p_mp - s_mp))
        + (d_mp + 1) / (2 * (one - p_mp))
        - half * r_mp * s_mp * (one + p_mp) / (one - p_mp)**3
    )

    # λ3'(p) = sqrt(r s) (2p^2 + 3ps - p + s - 1) / (2 sqrt(p) (1-p)^3 sqrt(1-p-s))
    lam3prime = (
        mp.sqrt(r_mp * s_mp)
        * (2 * p_mp**2 + 3 * p_mp * s_mp - p_mp + s_mp - one)
        / (2 * mp.sqrt(p_mp) * (one - p_mp)**3 * mp.sqrt(one - p_mp - s_mp))
    )

    # ν = (d-1)/2
    nu = (d_mp - 1) / 2

    # t(p) = (1-p)/(1-p-s),  t'(p) = s/(1-p-s)^2
    t = (one - p_mp) / (one - p_mp - s_mp)
    tprime = s_mp / (one - p_mp - s_mp)**2

    # ---- find region where a(x) < 0 (negative pieces) ----
    L = mp.mpf(tail_sigma)

    # a(x) = c + 0.5 λ1 x^2 - λ3 x
    # solve λ1 x^2 - 2 λ3 x + 2 c = 0
    A = lam1
    B = -2 * lam3
    C = 2 * c

    pts = [-L]

    if A == 0:
        # linear case: a(x) = c - λ3 x
        if lam3 != 0:
            x0 = c / lam3
            if -L < x0 < L:
                pts.append(x0)
    else:
        disc = B * B - 4 * A * C
        if disc > 0:
            sqrtD = mp.sqrt(disc)
            x1 = (-B - sqrtD) / (2 * A)
            x2 = (-B + sqrtD) / (2 * A)
            if -L < x1 < L:
                pts.append(x1)
            if -L < x2 < L:
                pts.append(x2)
        elif disc == 0:
            x0 = -B / (2 * A)
            if -L < x0 < L:
                pts.append(x0)

    pts.append(L)
    pts = sorted(pts)

    # Build negative-a pieces only: on each interval, test a(midpoint)
    pieces_neg = []
    for i in range(len(pts) - 1):
        aL, bR = pts[i], pts[i + 1]
        xm = (aL + bR) / 2
        axm = c + half * lam1 * (xm * xm) - lam3 * xm
        if axm < 0:
            pieces_neg.append((aL, bR))

    # If no negative region, f'(p) = 0.
    if not pieces_neg:
        return mp.mpf('0')

    # pack common args
    common_args = (
        pieces_neg,
        lam1, lam3, c,
        lam1prime, lam3prime, cprime,
        nu, p_mp, s_mp, t, tprime, dps,
    )

    # ---- serial or parallel evaluation of T1..T4 ----
    if workers is None or workers <= 1:
        T1 = _term1_integral(common_args)
        T2 = _term2_integral(common_args)
        T3 = _term3_integral(common_args)
        T4 = _term4_integral(common_args)
    else:
        max_workers = min(int(workers), 4)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut1 = ex.submit(_term1_integral, common_args)
            fut2 = ex.submit(_term2_integral, common_args)
            fut3 = ex.submit(_term3_integral, common_args)
            fut4 = ex.submit(_term4_integral, common_args)
            T1 = fut1.result()
            T2 = fut2.result()
            T3 = fut3.result()
            T4 = fut4.result()

    fprime = T1 + T2 + T3 + T4
    return fprime

def f_value_p(
    p: float,
    s: float,
    d: int,
    r: float,
    eps: float,
    workers: int = 1,
    dps: int = 60,
    tail_sigma: float = 12.0,
    chunks_per_piece: int = 2,
) -> float:
    """
    Wrapper for f(p) = δ^{right}(ε) with augmented leverage q = p + s.

    Parameters
    ----------
    p : float
        Leverage.
    s : float
        Residual share, so q = p + s.
    d, r, eps, workers, dps, tail_sigma, sign_w, chunks_per_piece :
        Passed through to delta_pq_ls_right.
    """
    q = p + s
    return float(
        delta_pq_ls_left(
            p,
            q,
            d,
            r,
            eps,
            workers=workers,
            dps=dps,
            chunks_per_piece=chunks_per_piece,
            tail_sigma=tail_sigma,
        )
    )


def _active_u_intervals(lam1: mp.mpf, lam3: mp.mpf, c: mp.mpf):
    """
    Return a list of (uL, uR) intervals in (0,1) such that
    a(x) < 0  iff  u ∈ ⋃(uL,uR), where u = Φ(x).

    a(x) = c + 0.5*lam1*x^2 - lam3*x
         = 0.5*lam1*x^2 - lam3*x + c.
    """
    zero = mp.mpf('0')
    one  = mp.mpf('1')

    A = lam1
    B = -2 * lam3
    C = 2 * c

    intervals = []

    # --- degenerate quadratic: A == 0 (linear or constant) ---
    if A == 0:
        # a(x) = c - lam3 x
        if lam3 == 0:
            # a(x) = c constant
            if c < 0:
                intervals.append((zero, one))  # negative everywhere
            return intervals

        x0 = c / lam3           # solve c - lam3 x = 0
        slope = -lam3           # derivative of a(x) wrt x

        if slope > 0:
            # a(x) increases; negative on (-∞, x0)
            uL, uR = zero, _Phi(x0)
            if uL < uR:
                intervals.append((uL, uR))
        else:
            # slope < 0: a(x) decreases; negative on (x0, ∞)
            uL, uR = _Phi(x0), one
            if uL < uR:
                intervals.append((uL, uR))

        return intervals

    # --- genuine quadratic: A != 0 ---
    disc = B*B - 4*A*C

    if A > 0:
        # Upward-opening: negative region is bounded (or empty)
        if disc <= 0:
            # a(x) >= 0 everywhere (disc<0 or tangency)
            return intervals

        sqrtD = mp.sqrt(disc)
        x1 = (-B - sqrtD) / (2*A)
        x2 = (-B + sqrtD) / (2*A)
        if x1 > x2:
            x1, x2 = x2, x1

        u1, u2 = _Phi(x1), _Phi(x2)
        if u1 < u2:
            intervals.append((u1, u2))
        return intervals

    else:
        # A < 0: downward-opening; negative on tails (or everywhere)
        if disc <= 0:
            # a(x) <= 0 for all x (at most one zero)
            intervals.append((zero, one))
            return intervals

        sqrtD = mp.sqrt(disc)
        x1 = (-B - sqrtD) / (2*A)
        x2 = (-B + sqrtD) / (2*A)
        if x1 > x2:
            x1, x2 = x2, x1

        u1, u2 = _Phi(x1), _Phi(x2)
        if zero < u1:
            intervals.append((zero, u1))
        if u2 < one:
            intervals.append((u2, one))
        return intervals


# ---------- active region in u-space ----------

def _active_u_intervals(lam1: mp.mpf, lam3: mp.mpf, c: mp.mpf):
    """
    Return a list of (uL, uR) intervals in (0,1) such that
    a(x) < 0  iff  u ∈ ⋃(uL, uR), where u = Φ(x).

    a(x) = c + 0.5*lam1*x^2 - lam3*x
         = 0.5*lam1*x^2 - lam3*x + c.
    """
    zero = mp.mpf('0')
    one  = mp.mpf('1')

    A = lam1
    B = -2 * lam3
    C = 2 * c

    intervals = []

    # --- degenerate quadratic: A == 0 (linear or constant) ---
    if A == 0:
        # a(x) = c - lam3 x
        if lam3 == 0:
            # constant
            if c < 0:
                intervals.append((zero, one))
            return intervals

        x0 = c / lam3
        slope = -lam3   # derivative of a(x)

        if slope > 0:
            # negative on (-∞, x0)
            uL, uR = zero, _Phi(x0)
            if uL < uR:
                intervals.append((uL, uR))
        else:
            # negative on (x0, ∞)
            uL, uR = _Phi(x0), one
            if uL < uR:
                intervals.append((uL, uR))

        return intervals

    # --- genuine quadratic: A != 0 ---
    disc = B*B - 4*A*C

    if A > 0:
        # Upward-opening: negative region bounded or empty
        if disc <= 0:
            # a(x) >= 0 everywhere
            return intervals

        sqrtD = mp.sqrt(disc)
        x1 = (-B - sqrtD) / (2*A)
        x2 = (-B + sqrtD) / (2*A)
        if x1 > x2:
            x1, x2 = x2, x1

        u1, u2 = _Phi(x1), _Phi(x2)
        if u1 < u2:
            intervals.append((u1, u2))
        return intervals

    else:
        # A < 0: downward-opening
        if disc <= 0:
            # a(x) <= 0 everywhere
            intervals.append((zero, one))
            return intervals

        sqrtD = mp.sqrt(disc)
        x1 = (-B - sqrtD) / (2*A)
        x2 = (-B + sqrtD) / (2*A)
        if x1 > x2:
            x1, x2 = x2, x1

        u1, u2 = _Phi(x1), _Phi(x2)
        if zero < u1:
            intervals.append((zero, u1))
        if u2 < one:
            intervals.append((u2, one))
        return intervals

# ---------- main change-of-variable implementation ----------

def fprime_p_cov(
    p: float,
    s: float,
    d: int,
    r: float,
    eps: float,
    dps: int = 60,
) -> mp.mpf:
    """
    Compute f'(p) via change-of-variable X ~ N(0,1) -> U = Φ(X) ~ Unif(0,1),
    integrating only over the active region where a(x) < 0.

    This mirrors your original four-term decomposition but works on u ∈ (0,1).
    """
    # ---- basic checks ----
    if r < 1:
        raise ValueError("Require r >= 1.")
    if d < 2:
        raise ValueError("Require d >= 2.")
    if not (0 < s < 1):
        raise ValueError("Require 0 < s < 1.")
    if not (0 < p < 1 - s):
        raise ValueError("Require 0 < p < 1 - s.")
    if eps < 0:
        raise ValueError("Require eps >= 0.")

    mp.mp.dps = int(dps)
    one  = mp.mpf('1')
    half = mp.mpf('0.5')

    p_mp   = mp.mpf(p)
    s_mp   = mp.mpf(s)
    r_mp   = mp.mpf(r)
    eps_mp = mp.mpf(eps)
    d_mp   = mp.mpf(d)

    # ---- parameters λ1, λ3, c and their derivatives ----
    lam1 = (p_mp**2 - p_mp + s_mp) / (one - p_mp)**2
    lam3 = -mp.sqrt(r_mp * p_mp * s_mp) * mp.sqrt(one - p_mp - s_mp) / (one - p_mp)**2
    c = (
        eps_mp
        + half * (d_mp * mp.log(one - p_mp - s_mp) - (d_mp + 1) * mp.log(one - p_mp))
        - half * r_mp * p_mp * s_mp / (one - p_mp)**2
    )

    lam1prime = (p_mp + 2*s_mp - one) / (one - p_mp)**3
    cprime = (
        -d_mp / (2 * (one - p_mp - s_mp))
        + (d_mp + 1) / (2 * (one - p_mp))
        - half * r_mp * s_mp * (one + p_mp) / (one - p_mp)**3
    )
    lam3prime = (
        mp.sqrt(r_mp * s_mp)
        * (2 * p_mp**2 + 3 * p_mp * s_mp - p_mp + s_mp - one)
        / (2 * mp.sqrt(p_mp) * (one - p_mp)**3 * mp.sqrt(one - p_mp - s_mp))
    )

    nu = (d_mp - 1) / 2
    t      = (one - p_mp) / (one - p_mp - s_mp)
    tprime = s_mp / (one - p_mp - s_mp)**2

    # ---- find active region in u ----
    u_intervals = _active_u_intervals(lam1, lam3, c)

    # No negative region => derivative is exactly 0
    if not u_intervals:
        return mp.mpf('0')

    # Slightly shrink intervals away from endpoints 0 and 1
    # to avoid hitting erfinv(±1) etc.
    eps_u = mp.mpf(10)**(- (dps // 2))
    zero  = mp.mpf('0')
    one   = mp.mpf('1')

    def a_of_x(x):
        return c + half * lam1 * (x*x) - lam3 * x

    def a_p_of_x(x):
        return cprime + half * lam1prime * (x*x) - lam3prime * x

    # ---------- term 1 ----------
    def integrand_u_T1(u):
        # Clamp u into (0,1) and catch small excursions
        if not (zero < u < one):
            return mp.mpf('0')
        x  = _x_from_u(u)
        ax = a_of_x(x)
        if ax >= 0:
            return mp.mpf('0')
        z1 = -ax * (one - p_mp) / s_mp
        w1 = _w_nu(nu, z1)
        ap = a_p_of_x(x)
        d_z1_dp = ax / s_mp - (one - p_mp) * ap / s_mp
        return w1 * d_z1_dp

    T1_parts = []
    for (uL, uR) in u_intervals:
        eff_L = max(uL, eps_u)
        eff_R = min(uR, one - eps_u)
        if eff_L < eff_R:
            T1_parts.append(mp.quad(integrand_u_T1, [eff_L, eff_R]))
    T1 = mp.fsum(T1_parts)

    # ---------- term 2 ----------
    coeff2 = -nu * t**(nu - 1) * tprime

    def integrand_u_T2(u):
        if not (zero < u < one):
            return mp.mpf('0')
        x  = _x_from_u(u)
        ax = a_of_x(x)
        if ax >= 0:
            return mp.mpf('0')
        z2 = -ax * (one - p_mp - s_mp) / s_mp
        Q2 = _Q_reg_upper(nu, z2)
        return mp.e**(ax) * (one - Q2)

    T2_parts = []
    for (uL, uR) in u_intervals:
        eff_L = max(uL, eps_u)
        eff_R = min(uR, one - eps_u)
        if eff_L < eff_R:
            T2_parts.append(mp.quad(integrand_u_T2, [eff_L, eff_R]))
    T2 = coeff2 * mp.fsum(T2_parts)

    # ---------- term 3 ----------
    coeff3 = -t**nu

    def integrand_u_T3(u):
        if not (zero < u < one):
            return mp.mpf('0')
        x  = _x_from_u(u)
        ax = a_of_x(x)
        if ax >= 0:
            return mp.mpf('0')
        ap = a_p_of_x(x)
        z2 = -ax * (one - p_mp - s_mp) / s_mp
        Q2 = _Q_reg_upper(nu, z2)
        return mp.e**(ax) * ap * (one - Q2)

    T3_parts = []
    for (uL, uR) in u_intervals:
        eff_L = max(uL, eps_u)
        eff_R = min(uR, one - eps_u)
        if eff_L < eff_R:
            T3_parts.append(mp.quad(integrand_u_T3, [eff_L, eff_R]))
    T3 = coeff3 * mp.fsum(T3_parts)

    # ---------- term 4 ----------
    coeff4 = -t**nu

    def integrand_u_T4(u):
        if not (zero < u < one):
            return mp.mpf('0')
        x  = _x_from_u(u)
        ax = a_of_x(x)
        if ax >= 0:
            return mp.mpf('0')
        z2 = -ax * (one - p_mp - s_mp) / s_mp
        w2 = _w_nu(nu, z2)
        ap = a_p_of_x(x)
        d_z2_dp = ax / s_mp - (one - p_mp - s_mp) * ap / s_mp
        return mp.e**(ax) * w2 * d_z2_dp

    T4_parts = []
    for (uL, uR) in u_intervals:
        eff_L = max(uL, eps_u)
        eff_R = min(uR, one - eps_u)
        if eff_L < eff_R:
            T4_parts.append(mp.quad(integrand_u_T4, [eff_L, eff_R]))
    T4 = coeff4 * mp.fsum(T4_parts)

    fprime = T1 + T2 + T3 + T4

    if mp.im(fprime) != 0:
        imag_part = abs(mp.im(fprime))
        if imag_part > mp.mpf('1e-10'):
            warnings.warn(
                f"fprime_p_cov: non-negligible imaginary part {imag_part:.6e}, using real part"
            )
    return mp.re(fprime)
