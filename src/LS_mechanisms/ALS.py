# The following only satisfies relative-DP privacy guarantee
# Refer the definition of relative-DP from "eureka: a general framework for black-box differential privacy
# estimators" from SP2024 for details
import numpy as np
import secrets
from numpy.random import MT19937, RandomState

from analysis.commons import compute_xopt, split_to_B_b
from LS_mechanisms.optim_LS import OptimalLS_mech, lev_evaluate_ALS


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


class ALS_mech(OptimalLS_mech):
    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)

        if lev_evaluate_ALS(self.r, self.d, self.lev, self.lev + self.res, epsilon,
                            cyclimits=self.cyclimits, atol=self.atol) <= delta:
            print("No utility loss!")
            X = self.D.T
        else:
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
