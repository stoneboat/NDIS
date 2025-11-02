import math
import multiprocessing
from functools import partial

import numpy as np
from numpy.random import MT19937, RandomState
import secrets
from scipy.linalg import sqrtm, eigh
from scipy.stats import norm
from numpy.linalg import inv as mat_inv
from math import sqrt, exp, pi, log2
from dataclasses import dataclass

from typing import Optional, Tuple
from scipy.stats import qmc
from scipy.special import erfi
from numpy.polynomial.hermite_e import hermegauss  
from numpy.random import Generator, MT19937  

def query(v, A, b, c):
    return 0.5 * v.T @ A @ v + b.T @ v + c


def _kd_func_expectation_estimation_workers(n_samples, dim, A, b, c):
    seed = secrets.randbits(128)
    rng = RandomState(MT19937(seed))

    mean = np.zeros(dim)
    cov = np.identity(dim)

    sum = 0
    v_list = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)
    for i in range(n_samples):
        sum += max(0, 1 - np.exp(query(v_list[i], A, b, c)))

    return sum / n_samples

def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n < 1:
        raise ValueError("n must be positive")
    return 1 if n == 1 else 1 << (n - 1).bit_length()


class BasicNDISEstimator:
    def __init__(
        self,
        mu1: np.ndarray,
        Sigma1: np.ndarray,
        mu2: np.ndarray,
        Sigma2: np.ndarray,
        workers: int = 12,
        num_samples: int = 10 ** 6,
    ) -> None:
        self.mu1 = mu1
        self.Sigma1 = Sigma1
        self.mu2 = mu2
        self.Sigma2 = Sigma2
        self.workers = workers
        self.num_samples = num_samples

        # Basic shape checks upfront
        assert isinstance(self.Sigma1, np.ndarray), "ERR: variable Sigma1 is required np.ndarray type"
        assert isinstance(self.Sigma2, np.ndarray), "ERR: variable Sigma2 is required np.ndarray type"
        assert self.Sigma1.ndim == 2, "ERR: Sigma1 must be 2D"
        assert self.Sigma2.ndim == 2, "ERR: Sigma2 must be 2D"
        assert self.Sigma1.shape[0] == self.Sigma1.shape[1] == self.Sigma2.shape[0] == self.Sigma2.shape[1]

        self.d = self.Sigma1.shape[0]
        
        sqrt_Sigma1 = sqrtm(self.Sigma1)
        inv_Sigma2 = mat_inv(self.Sigma2)
        D, U = eigh(np.identity(self.d) - sqrt_Sigma1 @ inv_Sigma2 @ sqrt_Sigma1)
        self.A = D
        self.b = -U.T @ sqrt_Sigma1 @ inv_Sigma2 @ (self.mu1 - self.mu2)

        sign1, logdet1 = np.linalg.slogdet(self.Sigma1)
        sign2, logdet2 = np.linalg.slogdet(self.Sigma2)
        assert sign1 > 0 and sign2 > 0, "Σ must be SPD"
        self.c = (0.5*(logdet1 - logdet2) - 0.5*(self.mu1 - self.mu2).T @ inv_Sigma2 @ (self.mu1 - self.mu2)).item()

    def _estimate_standard(self, epsilon: float) -> float:
        """Estimate delta using the standard multidimensional Monte Carlo method."""
        pool = multiprocessing.Pool(processes=self.workers)
        partial_worker = partial(
            _kd_func_expectation_estimation_workers, dim=self.d, A=np.diag(self.A), b=self.b, c=self.c+epsilon
        )

        chunk_size = np.ceil(self.num_samples / self.workers)
        input_list = []
        remains = self.num_samples
        for i in range(self.workers):
            input_list.append(int(min(chunk_size, remains)))
            remains -= chunk_size

        output_list = pool.map(partial_worker, input_list)
        ret = (np.dot(np.array(output_list, dtype=object).ravel(), np.array(input_list)) / self.num_samples)
        if isinstance(ret, np.ndarray):
            return ret.item()
        else:
            return ret
    
    def _estimate_pqmc(self, epsilon: float) -> Tuple[float, float]:
        estimation, se = pQMC_NDIS(self.A, self.b, self.c+epsilon).delta_rqmc(
            N=next_power_of_two(self.num_samples), 
            R=8, 
            workers=self.workers
        )
        return estimation, se

    def estimate(self, epsilon: float, method: str = "mc") -> float:
        """Main estimation function that routes to appropriate method based on dimensions and method type."""
        ret = {}
        if self.d == 1:
            ret["delta"] = delta_gauss_1d_different_covariance(np.sqrt(self.Sigma1[0, 0]), np.sqrt(self.Sigma2[0, 0]), self.mu2[0] - self.mu1[0], epsilon)
        else:
            if method == "mc":
                ret["delta"] = self._estimate_standard(epsilon)
            elif method == "pqmc":
                estimation, se = self._estimate_pqmc(epsilon)
                ret["delta"] = estimation
                ret["se"] = se
        return ret


# 1D NDIS estimation

@dataclass
class Inner1DKernel:
    """
    Reusable 1-D inner kernel for δ = E[(1 - exp(q(Z)))_+],
    with q(z)=0.5*a*z^2 + b*z + c'.
    """
    a: float
    b: float
    tol: float = 1e-14
    K: int = 64

    # ---------- active-set intervals ----------
    def _intervals(self, c_acc: float):
        a, b, tol = self.a, self.b, self.tol
        D = b*b - 2.0*a*c_acc
        if a > 0.0:
            if D < -tol:
                return []  # empty
            s = sqrt(max(D, 0.0))
            L = (-b - s)/a; R = (-b + s)/a
            return [(L, R)]
        elif a < 0.0:
            if D < -tol:
                return [(-np.inf, np.inf)]  # whole line
            s = sqrt(max(D, 0.0))
            l = (-b + s)/a; r = (-b - s)/a
            if l > r: l, r = r, l
            return [(-np.inf, l), (r, np.inf)]
        else:
            # a == 0: linear threshold
            if abs(b) < 1e-18:
                return [(-np.inf, np.inf)] if c_acc <= 0 else []
            thr = -c_acc/b
            return [(-np.inf, thr)] if b > 0 else [(thr, np.inf)]

    # ---------- ∫_S φ ----------
    def _J0(self, intervals):
        return sum(norm.cdf(R) - norm.cdf(L) for (L, R) in intervals)

    # ---------- ∫_S φ exp(0.5 a z^2 + b z) ----------
    def _J1_closed_or_numeric(self, intervals):
        """
        Compute J1 = ∫_S φ(z) * exp(0.5*a*z^2 + b*z) dz
        using closed forms in all regimes:
        - a < 1 : Gaussian CDF differences (complete the square)
        - a ≈ 1 or a ≈ 0 : linear-in-z case (same Φ-shift closed form)
        - a > 1 : exact erfi closed form on bounded interval(s)
        """
        a, b = self.a, self.b

        # ---- Linear case: a very close to 0 OR very close to 1
        # Note: for a == 0 and for a == 1 the integrand reduces to φ(z) * e^{b z}.
        if abs(a) < 1e-18 or abs(a - 1.0) < 1e-15:
            return sum(
                exp(0.5 * b * b) * (norm.cdf(R - b) - norm.cdf(L - b))
                for (L, R) in intervals
            )

        # ---- Comfortable regime: a < 1  (γ = (1-a)/2 > 0)
        if a < 1.0:
            gamma = (1.0 - a) / 2.0
            mu_t  = b / (1.0 - a)
            const = norm.pdf(0.0) * exp((b * b) / (2.0 * (1.0 - a)))  # 1/√(2π) * exp(...)
            scale = sqrt(pi / gamma)

            def Fu(z):
                if np.isneginf(z): return 0.0
                if np.isposinf(z): return 1.0
                return norm.cdf(sqrt(2.0 * gamma) * (z - mu_t))

            return sum(const * scale * (Fu(R) - Fu(L)) for (L, R) in intervals)

        # ---- a > 1 : exact erfi closed form on each bounded interval
        # Write: φ(z) * exp(0.5 a z^2 + b z) = (1/√(2π)) * exp(γ (z+β)^2 - γ β^2),
        # with γ = (a-1)/2 > 0, β = b/(a-1).
        gamma = 0.5 * (a - 1.0)           # > 0
        beta  = b / (a - 1.0)
        pref  = (1.0 / sqrt(2.0 * pi)) * exp(-gamma * beta * beta)
        root  = sqrt(gamma)

        def J1_interval(L, R):
            if not (np.isfinite(L) and np.isfinite(R)):
                raise ValueError("a>1 requires bounded active-set intervals; got unbounded.")
            uR = root * (R + beta)
            uL = root * (L + beta)
            # (√π / (2√γ)) * [erfi(uR) - erfi(uL)]
            return pref * (sqrt(pi) / (2.0 * root)) * (erfi(uR) - erfi(uL))

        return sum(J1_interval(L, R) for (L, R) in intervals)

    # ---------- public API ----------
    def eval(self, c_acc: float) -> float:
        intervals = self._intervals(c_acc)
        if not intervals:
            return 0.0
        J0 = self._J0(intervals)
        J1 = self._J1_closed_or_numeric(intervals)
        # δ = J0 - e^{c'} J1, clamped to [0,1]
        delta = J0 - exp(c_acc) * J1
        return max(0.0, min(1.0, delta))

# Thin wrapper for the specific 1-D Gaussian-vs-Gaussian parameters
def delta_gauss_1d_different_covariance(sigma1: float, sigma2: float, mu: float, eps: float,
                                        tol: float = 1e-14, K: int = 64) -> float:
    a = 1.0 - (sigma1**2)/(sigma2**2)
    b = (sigma1*mu)/(sigma2**2)
    c = eps + 0.5*np.log((sigma1**2)/(sigma2**2)) - 0.5*(mu**2)/(sigma2**2)
    kernel = Inner1DKernel(a=a, b=b, tol=tol, K=K)
    return kernel.eval(c)

def choose_innermost_index(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    # primary key: |a| ; secondary key: |b|
    keys = np.stack([np.abs(a), np.abs(b)], axis=1)
    # argmax lexicographically
    return int(np.lexsort(( -keys[:,1], -keys[:,0]))[0])  # largest |a| then largest |b|


def _process_chunk_helper(Z_chunk: np.ndarray, a: np.ndarray, b: np.ndarray, c: float, 
                          inner_a: float, inner_b: float) -> float:
    """
    Standalone helper function for parallel processing of Z chunks.
    Reconstructs the computation without requiring the pQMC_NDIS instance.
    """
    inner_kernel = Inner1DKernel(a=inner_a, b=inner_b)
    chunk_vals = []
    for z_out in Z_chunk:
        # Compute c' = c + Σ_j (0.5 a_j z_j^2 + b_j z_j)
        c_acc = float(c + 0.5 * np.dot(a, z_out * z_out) + np.dot(b, z_out))
        chunk_vals.append(inner_kernel.eval(c_acc))
    return float(np.mean(chunk_vals))

def _phi_inv_clip(u: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Map Sobol points u in (0,1)^m to N(0,1)^m with safe clipping."""
    u = np.clip(u, eps, 1.0 - eps)
    return norm.ppf(u)

@dataclass
class pQMC_NDIS:
    """
    High-d δ via nested 1D inner closed-form + pure RQMC outer.
    Inputs are already in the diagonal basis A = diag(a).
    """
    a: np.ndarray  # shape (d,)
    b: np.ndarray  # shape (d,)
    c: float
    i_star: Optional[int] = None  # innermost index; if None, auto-choose

    def __post_init__(self):
        self.a = np.asarray(self.a, dtype=float).copy()
        self.b = np.asarray(self.b, dtype=float).copy()
        assert self.a.ndim == 1 and self.b.ndim == 1 and self.a.shape == self.b.shape
        self.d = self.a.size

        # choose innermost
        if self.i_star is None:
            self.i_star = choose_innermost_index(self.a, self.b)

        # build inner kernel once
        self.inner = Inner1DKernel(a=float(self.a[self.i_star]), b=float(self.b[self.i_star]))

        # outer indices: sort by descending |a_j| (most influential first), excluding i_star
        mask = np.ones(self.d, dtype=bool); mask[self.i_star] = False
        J = np.where(mask)[0]
        order = np.argsort(-np.abs(self.a[J]))
        self.outer_idx = J[order]

        # MT19937 generator (fresh entropy each process start)
        self._rng = Generator(MT19937())  # no user seed -> OS entropy via SeedSequence

    def _accumulate_cprime(self, z_out: np.ndarray) -> float:
        """c' = c + Σ_j (0.5 a_j z_j^2 + b_j z_j) over outer dims (aligned with self.outer_idx)."""
        jj = self.outer_idx
        return float(self.c + 0.5*np.dot(self.a[jj], z_out*z_out) + np.dot(self.b[jj], z_out))

    def delta_rqmc(self, N: int = 1 << 14, R: int = 8, u_eps: float = 1e-15, workers: int = 12) -> Tuple[float, Optional[float]]:
        """
        Pure randomized QMC on m = d-1 outer dims.
        - N: points per scramble (must be power of 2 for Sobol).
        - R: number of independent Owen scrambles.
        - workers: number of parallel processes.
        Returns (estimate, standard_error or None if R<2).
        """
        if self.d <= 1:
            # no outer dims
            return self.inner.eval(self.c), None

        # require power of two for Sobol.random_base2
        if int(log2(N)) != np.log2(N):
            raise ValueError("N must be a power of 2 for Sobol.random_base2")

        m = self.d - 1
        estimates = []

        for _ in range(R):
            # independent scramble seed from MT19937
            scramble_seed = int(self._rng.integers(0, 2**31 - 1))
            sob = qmc.Sobol(d=m, scramble=True, seed=scramble_seed)
            U = sob.random_base2(int(log2(N)))  # (N, m)
            Z = _phi_inv_clip(U, u_eps)         # map to N(0,1)^m

            # Parallelize processing of Z array across workers
            if workers > 1 and N >= workers:
                pool = multiprocessing.Pool(processes=workers)
                chunk_size = int(np.ceil(N / workers))
                Z_chunks = [Z[i:i+chunk_size] for i in range(0, N, chunk_size)]
                
                # Create partial function with bound methods - need to pass data
                partial_worker = partial(
                    _process_chunk_helper,
                    a=self.a[self.outer_idx],
                    b=self.b[self.outer_idx],
                    c=self.c,
                    inner_a=float(self.a[self.i_star]),
                    inner_b=float(self.b[self.i_star])
                )
                
                chunk_results = pool.map(partial_worker, Z_chunks)
                pool.close()
                pool.join()
                
                # Combine results: weighted average by chunk sizes
                chunk_sizes = [len(chunk) for chunk in Z_chunks]
                estimate = float(np.average(chunk_results, weights=chunk_sizes))
            else:
                # Sequential fallback for small N or workers=1
                vals = np.empty(N, dtype=float)
                for n in range(N):
                    z_out = Z[n]
                    c_acc = self._accumulate_cprime(z_out)
                    vals[n] = self.inner.eval(c_acc)
                estimate = float(vals.mean())

            estimates.append(estimate)

        mean = float(np.mean(estimates))
        se = None if R < 2 else float(np.std(estimates, ddof=1) / np.sqrt(R))
        return mean, se

    