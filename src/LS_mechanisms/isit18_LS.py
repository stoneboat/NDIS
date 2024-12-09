import numpy as np
import secrets
from numpy.random import MT19937, RandomState

from analysis.commons import split_to_B_b, compute_xopt


class ISIT18LS_mech:
    def __init__(self, kwargs):
        self.D = kwargs["database"]

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.r = kwargs["r"]
        if "chunk_size" in kwargs:
            self.chunk_size = kwargs["chunk_size"]
        else:
            self.chunk_size = 1000


        self.d = self.D.shape[1]
        self.n = self.D.shape[0]

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        self.c = self.compute_constant()

    def compute_constant(self):
        tmp_array = np.ones(self.d)

        for i in np.arange(self.d):
            tmp_array[i] = np.sqrt(np.linalg.norm(self.D.T[i]) ** 2 - np.max(self.D.T[i] ** 2))

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
        X = self.D
        result_matrices = []

        samples = np.zeros((num_samples, self.d - 1))
        for i in range(num_samples):
            r_piece = np.minimum(self.chunk_size, self.r)
            noise1 = self.rng.normal(loc=0, scale=1, size=(r_piece, X.shape[0]))
            pi_X = noise1@X
            pointer = r_piece

            while pointer < self.r:
                r_piece = np.minimum(self.chunk_size, self.r)
                noise1 = self.rng.normal(loc=0, scale=1, size=(r_piece, X.shape[0]))

                pi_X_piece = noise1@X
                pointer += r_piece
                pi_X = np.vstack((pi_X, pi_X_piece))

            noise2 = self.rng.normal(loc=0, scale=np.sqrt(variance), size=(self.r, self.d))
            projected_database = pi_X + noise2
            B, b = split_to_B_b(projected_database)
            samples[i] = compute_xopt(B, b).reshape((1, -1))

        # for _ in range(self.r):
        #     noise1 = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
        #     noise2 = self.rng.normal(loc=0, scale=np.sqrt(variance), size=(self.d, num_samples))
        #     result_matrices.append(X @ noise1 + noise2)
        #
        #     # Concatenating the resulting matrices by row
        # final_matrix = np.concatenate(result_matrices, axis=0).T
        #
        # samples = np.zeros((num_samples, self.d-1))
        # for i in range(num_samples):
        #     projected_database = final_matrix[i].reshape((self.r, self.d))
        #     B, b = split_to_B_b(projected_database)
        #     samples[i] = compute_xopt(B, b).reshape((1, -1))
        return samples