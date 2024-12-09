# The following only satisfies relative-DP privacy guarantee
# Refer the definition of relative-DP from "eureka: a general framework for black-box differential privacy
# estimators" from SP2024 for details

import numpy as np
import secrets
from numpy.random import MT19937, RandomState

from analysis.commons import twoNorm


class ALT19RP_mech:
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

    def compute_constant(self, epsilon, delta):
        c1 = 0
        for index in np.arange(self.n):
            tmp = np.linalg.norm(self.D[index])
            if c1 < tmp:
                c1 = tmp

        c2 = np.sqrt(4*(c1**2)*(np.sqrt(2*self.r*np.log(4/delta)) + np.log(4/delta))/epsilon)
        return c2

    def gen_samples(self, num_samples, epsilon, delta):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        return self._gen_samples(epsilon, delta, num_samples)

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        c = self.compute_constant(epsilon, delta)
        X = np.concatenate((self.D, c*np.eye(self.d))).T
        result_matrices = []

        for _ in range(self.r):
            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
        final_matrix = np.concatenate(result_matrices, axis=0).T

        return final_matrix
