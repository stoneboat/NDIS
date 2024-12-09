import numpy as np
import secrets
from numpy.random import MT19937, RandomState
from scipy import special

from analysis.commons import split_to_B_b, get_w, compute_xopt


def Gaussian_delta(epsilon, c):
    assert epsilon >= 0, "epsilon must be positive"
    term1 = 0.5 * (1 - np.exp(epsilon))
    term2 = special.erf(-epsilon / (np.sqrt(2) * c) + c / np.sqrt(8))
    term3 = np.exp(epsilon) * special.erf(epsilon / (np.sqrt(2) * c) + c / np.sqrt(8))
    ret = term1 + 0.5 * term2 + 0.5 * term3
    return ret


def find_minimal_c(delta, epsilon, low=1e-6, high=1e2, tol=1e-6):
    """
    Find the minimal value of c using binary search.

    Parameters:
    - delta: Target value
    - epsilon: Given epsilon
    - low: Lower bound for binary search
    - high: Upper bound for binary search
    - tol: Tolerance level for convergence

    Returns:
    - c: Minimal value of c such that Gaussian_delta(epsilon, c) < delta
    """

    # Ensure that the function value at the lower bound is less than delta
    if Gaussian_delta(epsilon, low) > delta:
        raise ValueError("Please re-choose your down side")

    if Gaussian_delta(epsilon, high) < delta:
        raise ValueError("Please re-choose your up side")

    # Binary search
    while high - low > tol:
        mid = (low + high) / 2
        if Gaussian_delta(epsilon, mid) < delta:
            low = mid
        else:
            high = mid

    return (low + high) / 2


class Gaussian_mech:
    def __init__(self, kwargs):
        self.D = kwargs["database"]
        self.B, self.b = split_to_B_b(self.D)
        self.b = self.b.reshape((-1, 1))

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.r = kwargs["r"]
        self.d = self.D.shape[1]-1
        self.n = self.D.shape[0]
        self.index, self.sensitivity = self.compute_sensitivity()

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def compute_sensitivity(self):
        largest_index = -1
        sensitivity = 0

        N = np.linalg.inv(self.B.T @ self.B)
        w = get_w(self.B, self.b).ravel()

        for index in np.arange(self.n):
            v = self.B[index].copy().reshape(-1, 1)
            leverage = (v.T @ N @ v).item()
            tmp = np.abs(w[index])*np.linalg.norm(N@v)/(1-leverage)
            if tmp > sensitivity:
                sensitivity = tmp
                largest_index = index

        return largest_index, sensitivity

    def find_minimal_sigma(self, epsilon, delta, low=1e-3, high=1e2, tol=1e-6):
        c = find_minimal_c(delta, epsilon, low, high, tol)
        sigma = self.sensitivity/c

        return sigma

    def gen_samples(self, num_samples, epsilon, delta):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        return self._gen_samples(epsilon, delta, num_samples)

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        xopt = compute_xopt(self.B, self.b)
        sigma = self.find_minimal_sigma(epsilon, delta)

        cov = sigma**2*np.identity(self.d)
        samples = self.rng.multivariate_normal(xopt, cov=cov, size=num_samples)

        return samples