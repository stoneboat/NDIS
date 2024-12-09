# The following only satisfies relative-DP privacy guarantee
# Refer the definition of relative-DP from "eureka: a general framework for black-box differential privacy
# estimators" from SP2024 for details
import numpy as np
import secrets
from numpy.random import MT19937, RandomState

from analysis.commons import compute_xopt, split_to_B_b, get_w
from LS_mechanisms.optim_LS import lev_evaluate_ALS
from RP_mechanisms.optim_RP_DP import OptimalRP_mech


class ALS:
    def __init__(self, kwargs):
        self.D = kwargs["database"]

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.r = kwargs["r"]
        self.d = self.D.shape[1]
        self.n = self.D.shape[0]

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def gen_samples(self, num_samples, epsilon, delta):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        return self._gen_samples(num_samples)

    def _gen_samples(self, num_samples):
        num_samples = int(num_samples)
        X = self.D.T
        result_matrices = []

        for _ in range(self.r):
            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
        final_matrix = np.concatenate(result_matrices, axis=0).T

        samples = np.zeros((num_samples, self.d - 1))
        for i in range(num_samples):
            projected_database = final_matrix[i].reshape((self.r, self.d))
            B, b = split_to_B_b(projected_database)
            samples[i] = compute_xopt(B, b).reshape((1, -1))

        return samples


class OptimalLS_mech:
    def __init__(self, kwargs):
        self.D = kwargs["database"]
        self.l = kwargs["l2"]
        self.B, self.b = split_to_B_b(self.D)
        self.b = self.b.reshape((-1, 1))

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.r = kwargs["r"]
        self.d = self.D.shape[1]-1
        self.n = self.D.shape[0]

        # For privacy spectrum evaluation
        self.cyclimits = 20000000
        self.atol = 1e-8

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def find_minimal_sigma(self, epsilon, delta, low=1e-3, high=1e2, tol=1e-6):
        # Ensure that the function value at the lower bound is less than delta
        low_aug_lev= min(self.l**2*2/(low**2), 1-1e-8)
        low_lev = min(self.l**2/(low ** 2), 1-1e-8)

        if lev_evaluate_ALS(self.r, self.d, low_lev, low_aug_lev, epsilon,
                            cyclimits=self.cyclimits, atol=self.atol) < delta:
            raise ValueError("Please re-choose your down side")

        high_aug_lev = min(self.l ** 2*2/ (high ** 2), 1-1e-8)
        high_lev = min(self.l ** 2/ (high ** 2), 1-1e-8)
        if lev_evaluate_ALS(self.r, self.d, high_lev, high_aug_lev, epsilon,
                            cyclimits=self.cyclimits, atol=self.atol) > delta:
            raise ValueError("Please re-choose your up side")

        # Binary search
        while high - low > tol:
            mid = (low + high) / 2
            mid_aug_lev = min(self.l ** 2*2/(mid ** 2), 1-1e-8)
            mid_lev = min(self.l ** 2/(mid ** 2), 1-1e-8)

            if lev_evaluate_ALS(self.r, self.d, mid_lev, mid_aug_lev, epsilon,
                            cyclimits=self.cyclimits, atol=self.atol) < delta:
                high = mid
            else:
                low = mid

        return (low + high) / 2

    def gen_samples(self, num_samples, epsilon, delta):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        return self._gen_samples(epsilon, delta, num_samples)

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        sigma = self.find_minimal_sigma(epsilon, delta)

        X = np.vstack((self.D, sigma * np.eye(self.d+1)))
        B, b = split_to_B_b(X)

        xopt = compute_xopt(B, b)
        w = get_w(B, b)
        cov = (np.linalg.norm(w) ** 2 / self.r) * np.linalg.inv(B.T @ B)

        samples = self.rng.multivariate_normal(xopt, cov=cov, size=num_samples)

        return samples

class ALS_mech(OptimalLS_mech):
    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)

        sigma = self.find_minimal_sigma(epsilon, delta)
        X = np.vstack((self.D, sigma * np.eye(self.d + 1))).T

        result_matrices = []

        for _ in range(self.r):
            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
        final_matrix = np.concatenate(result_matrices, axis=0).T

        samples = np.zeros((num_samples, self.d))
        for i in range(num_samples):
            projected_database = final_matrix[i].reshape((self.r, self.d+1))
            B, b = split_to_B_b(projected_database)
            samples[i] = compute_xopt(B, b).reshape((1, -1))

        return samples

class LS_fromoptim_RP_mech(OptimalRP_mech):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        if "chunk_size" in kwargs:
            self.chunk_size = kwargs["chunk_size"]
        else:
            self.chunk_size = 1000

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        X = self.D
        samples = np.zeros((num_samples, self.d - 1))
        result_matrices = []

        sigma = self.find_minimal_sigma(epsilon, delta)

        for i in range(num_samples):
            r_piece = np.minimum(self.chunk_size, self.r)
            noise1 = self.rng.normal(loc=0, scale=1, size=(r_piece, X.shape[0]))
            pi_X = noise1 @ X
            pointer = r_piece

            while pointer < self.r:
                r_piece = np.minimum(self.chunk_size, self.r-pointer)
                noise1 = self.rng.normal(loc=0, scale=1, size=(r_piece, X.shape[0]))

                pi_X_piece = noise1 @ X
                pointer += r_piece
                pi_X = np.vstack((pi_X, pi_X_piece))

            noise2 = self.rng.normal(loc=0, scale=sigma, size=(self.r, self.d))
            projected_database = pi_X + noise2
            B, b = split_to_B_b(projected_database)
            samples[i] = compute_xopt(B, b).reshape((1, -1))

        return samples