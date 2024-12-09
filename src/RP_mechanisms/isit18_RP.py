# This is an implementation of the paper, titled
# "Privacy-Utility Trade-off of Linear Regression under Random Projections and Additive Noise"
# Accessed from
# https://ieeexplore.ieee.org/abstract/document/8437722?casa_token=2DpQ1ZF2QDUAAAAA:PWr5Hhz7fzxiZECM6RfS-CjLh1G2t8aGlxwmFHn4r4YMpihxO_z0JLrpBewTLtRnLOib9_Fq4OA

# The following only satisfies relative-DP privacy guarantee
# Refer the definition of relative-DP from "eureka: a general framework for black-box differential privacy
# estimators" from SP2024 for details
import numpy as np
import secrets
from numpy.random import MT19937, RandomState

from analysis.commons import twoNorm


class ISIT18RP_mech:
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
        self.c = self.compute_constant()

    def compute_constant(self):
        tmp_array = np.ones(self.d)

        for i in np.arange(self.d):
            tmp_array[i] = np.sqrt(twoNorm(self.D.T[i]) ** 2 - np.max(self.D.T[i] ** 2))

        return np.min(tmp_array)

    def gen_samples(self, num_samples, epsilon, delta):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        return self._gen_samples(epsilon, delta, num_samples)

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)

        # since the paper claims eps-MI-DP, we first need to convert from standard DP to it
        # step 1 from (epsilon, delta)-DP to (epsilon_, delta_)-DP
        # based on Property 3, from paper
        # titled "Differential Privacy as a Mutual Information Constraint" CCS 16
        epsilon_ = 0
        delta_ = (np.exp(epsilon_) + 1) * (1 - delta) / (np.exp(epsilon) + 1)

        # step 2 from (0, delta_)-DP to eps-MI-DP
        # based on Corollary 1, from this ISIT18 paper
        eps = (delta_**2*np.log2(np.e))/2

        # compute the variance needed to add
        variance = max(self.r/(2**(2*eps)-1) - self.c**2, 0)

        # Start to generate sample
        X = self.D.T
        result_matrices = []

        for _ in range(self.r):
            noise1 = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            noise2 = self.rng.normal(loc=0, scale=np.sqrt(variance), size=(self.d, num_samples))
            result_matrices.append(X @ noise1 + noise2)

            # Concatenating the resulting matrices by row
        final_matrix = np.concatenate(result_matrices, axis=0).T

        return final_matrix
