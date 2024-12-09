# The following only satisfies relative-DP privacy guarantee
# Refer the definition of relative-DP from "eureka: a general framework for black-box differential privacy
# estimators" from SP2024 for details
import numpy as np
import secrets
from numpy.random import MT19937, RandomState

import multiprocessing
from functools import partial

from scipy.stats import chi2


def compute_IS(epsilon, leverage, r):
    # Compute the terms inside the probabilities
    term1 = (1 - leverage) * (2 * epsilon - r * np.log(1 - leverage)) / leverage

    term2 = (2 * epsilon - r * np.log(1 - leverage)) / leverage

    # Compute each probability term
    prob1 = 1 - chi2.cdf(term1, r)
    prob2 = 1 - chi2.cdf(term2, r)

    # Compute the exponential term
    exp_term = np.exp(epsilon - r / 2 * np.log(1 - leverage)) * (1 - leverage) ** (r / 2.0)

    # Combine the terms to get the result
    result = prob1 - exp_term * prob2

    return result


def find_leverage_bar(epsilon, delta, r, low=1e-8, high=1, tol=1e-7):
    if compute_IS(epsilon, low, r) > delta:
        raise ValueError("Please re-choose your down side")

    if compute_IS(epsilon, high, r) < delta:
        raise ValueError("Please re-choose your up side")

    # Binary search
    while high - low > tol:
        mid = (low + high) / 2

        if compute_IS(epsilon, mid, r) < delta:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def parallel_gen_samples(mech, epsilon, delta, num_samples, workers):
    pool = multiprocessing.Pool(processes=workers)
    sample_generating_func = partial(mech.gen_samples, epsilon=epsilon, delta=delta)
    input_list = int(np.ceil(num_samples / workers))*np.ones(workers)
    samples = np.vstack(pool.map(sample_generating_func, input_list))
    return samples


class OptimalRP_mech:
    def __init__(self, kwargs):
        self.D = kwargs["database"]

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.r = kwargs["r"]
        self.d = self.D.shape[1]
        self.n = self.D.shape[0]
        self.index, self.lev = self.compute_largest_leverage()
        self.l = self.compute_largest_l2()
        self.v = self.D[self.index].copy().reshape(-1, 1)

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def compute_largest_leverage(self):
        largest_index = -1
        largest_lev = 0

        M = np.linalg.inv(self.D.T @ self.D)

        for index in np.arange(self.n):
            v = self.D[index].copy().reshape(-1, 1)
            leverage = (v.T @ M @ v).item()
            if leverage > largest_lev:
                largest_index = index
                largest_lev = leverage

        return largest_index, largest_lev

    def compute_largest_l2(self):
        l = 0
        for index in np.arange(self.n):
            v = self.D[index].copy().reshape(-1, 1)
            if np.linalg.norm(v) > l:
                l = np.linalg.norm(v)

        return l

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
        if compute_IS(epsilon, self.lev, self.r) <= delta:
            X = self.D.T
            for _ in range(self.r):
                noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
            final_matrix = np.concatenate(result_matrices, axis=0).T
        else:
            sigma = self.find_minimal_sigma(epsilon, delta)
            X = np.concatenate((self.D, sigma * np.eye(self.d))).T

            for _ in range(self.r):
                noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
            final_matrix = np.concatenate(result_matrices, axis=0).T

        return final_matrix


class OptimalRP_mech_equ(OptimalRP_mech):
    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        X = self.D.T
        result_matrices = []
        if compute_IS(epsilon, self.lev, self.r) <= delta:
            for _ in range(self.r):
                noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
            final_matrix = np.concatenate(result_matrices, axis=0).T
        else:
            sigma = self.find_minimal_sigma(epsilon, delta)

            for _ in range(self.r):
                noise1 = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                noise2 = self.rng.normal(loc=0, scale=sigma, size=(X.shape[0], num_samples))
                result_matrices.append(X @ noise1 + noise2)

            # Concatenating the resulting matrices by row
            final_matrix = np.concatenate(result_matrices, axis=0).T

        return final_matrix