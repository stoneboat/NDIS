# The following only satisfies DP privacy guarantee
import numpy as np
import secrets
from numpy.random import MT19937, RandomState

from analysis.RP_privacy_analysis_advanced import compute_IS

def compute_largest_l2(D):
    row_norms = np.linalg.norm(D, axis=1)
    l = np.max(row_norms)
    return l

class OptimalRP_mech:
    def __init__(self, kwargs):
        self.D = kwargs["database"]

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.r = kwargs["r"]
        self.d = self.D.shape[1]
        self.n = self.D.shape[0]
        self.l = kwargs["l2"]

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
    
    def compute_leverage_upper_bound(self, epsilon, delta, low=1e-10, high=1-1e-8, tol=1e-8):
        # Ensure that the function value at the lower bound is less than delta
        if compute_IS(epsilon, low, self.r) > delta:
            raise ValueError("Please re-choose your down side")

        if compute_IS(epsilon, high, self.r) < delta:
            raise ValueError("Please re-choose your up side")

        # Binary search
        while high - low > tol:
            mid = (low + high) / 2

            if compute_IS(epsilon, mid, self.r) > delta:
                high = mid
            else:
                low = mid

        leverage_upper_bound = (low + high) / 2
        return leverage_upper_bound


    def find_minimal_sigma(self, epsilon, delta, low=1e-10, high=1-1e-8, tol=1e-8):
        s_bar = self.compute_leverage_upper_bound(epsilon, delta, low, high, tol)
        return self.l*np.sqrt(1/s_bar)

    def gen_samples(self, num_samples, epsilon, delta):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        return self._gen_samples(epsilon, delta, num_samples)

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        result_matrices = []

        sigma = self.find_minimal_sigma(epsilon, delta)
        X = np.concatenate((self.D, sigma * np.eye(self.d))).T

        for _ in range(self.r):
            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            result_matrices.append(X @ noise)

        # Concatenating the resulting matrices by row
        final_matrix = np.concatenate(result_matrices, axis=0).T

        return final_matrix


class one_short_PTR_RP_mech(OptimalRP_mech):
    def find_minimal_sigma(self, epsilon, delta, ratio=0.05, low=1e-10, high=1-1e-8, tol=1e-8):
        s_bar = self.compute_leverage_upper_bound(epsilon*ratio, delta*ratio, low, high, tol)
        eigenvalues = np.linalg.eigvalsh(self.D.T @ self.D)  # Use eigvalsh for symmetric matrices
        lam_min  = np.min(eigenvalues)

        l = compute_largest_l2(self.D)
        eps_T = (1-ratio) * epsilon
        delta_fail = (1-ratio) * delta

        alpha = (l**2 / eps_T) * np.log(1.0 / (2.0 * delta_fail))
        eta = float(self.rng.laplace(loc=0.0, scale=(l**2)/eps_T))
        lam_tilde = lam_min + eta

        lam_lb = max(lam_tilde - alpha, 0.0)
        sigma2 = max((l**2) / s_bar - lam_lb, 0.0)

        print(f"sigma: {np.sqrt(sigma2):.10e}")
        print(f"lam_min: {lam_min:.10e}")
        print(f"lam_lb: {lam_lb:.10e}")
        print(f"lam_tilde: {lam_tilde:.10e}")
        print(f"eta: {eta:.10e}")
        print(f"ratio: {ratio:.10e}")
        print(f"s_bar: {s_bar:.10e}")
        print(f"alpha: {alpha:.10e}")
        print(f"upper bound of sigma: {np.sqrt((l**2) / s_bar):.10e}")
        print(f"l: {l:.10e}")
        print(f"lam_min: {lam_min:.10e}")

        return np.sqrt(sigma2)
