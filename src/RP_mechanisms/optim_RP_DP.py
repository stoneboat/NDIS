# The following only satisfies DP privacy guarantee
import numpy as np
import secrets
from numpy.random import MT19937, RandomState

from RP_mechanisms.optim_RP import compute_IS

def compute_largest_l2(D):
    l = 0
    n = D.shape[0]
    for index in np.arange(n):
        v = D[index].copy().reshape(-1, 1)
        if np.linalg.norm(v) > l:
            l = np.linalg.norm(v)

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


    def find_minimal_sigma(self, epsilon, delta, low=1e-10, high=1-1e-8, tol=1e-8):
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

        s_bar = (low + high) / 2

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


class clipping_RP_mech(OptimalRP_mech):
    def compute_DP_LSV(self):
        # Compute the eigenvalues
        eigenvalues = np.linalg.eigvals(self.D.T@self.D)

        # Find the smallest eigenvalue
        lsv = np.sqrt(np.min(eigenvalues))

        return max(lsv, 0)

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        result_matrices = []

        sigma_prime = self.compute_DP_LSV()
        sigma = self.find_minimal_sigma(epsilon, delta) - sigma_prime
        X = np.concatenate((self.D, sigma * np.eye(self.d))).T

        for _ in range(self.r):
            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            result_matrices.append(X @ noise)

        # Concatenating the resulting matrices by row
        final_matrix = np.concatenate(result_matrices, axis=0).T

        return final_matrix

