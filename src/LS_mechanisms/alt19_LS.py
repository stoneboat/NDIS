import numpy as np
import secrets
from numpy.random import MT19937, RandomState

from analysis.commons import twoNorm, compute_xopt, split_to_B_b


class ALT19LS_mech:
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

    def compute_constant(self, epsilon, delta):
        c1 = 0
        for index in np.arange(self.n):
            tmp = twoNorm(self.D[index])
            if c1 < tmp:
                c1 = tmp

        c2 = np.sqrt(4*(c1**2)*(np.sqrt(2*self.r*np.log(4/delta)) + np.log(4/delta))/epsilon)
        return c2

    def gen_samples(self, num_samples, epsilon, delta):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        return self._gen_samples(epsilon, delta, num_samples)

    # def _gen_samples(self, epsilon, delta, num_samples):
    #     num_samples = int(num_samples)
    #     c = self.compute_constant(epsilon, delta)
    #     X = np.concatenate((self.D, c*np.eye(self.d))).T
    #     result_matrices = []
    #
    #     for _ in range(self.r):
    #         noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
    #         result_matrices.append(X @ noise)
    #
    #         # Concatenating the resulting matrices by row
    #     final_matrix = np.concatenate(result_matrices, axis=0).T
    #
    #     samples = np.zeros((num_samples, self.d - 1))
    #     for i in range(num_samples):
    #         projected_database = final_matrix[i].reshape((self.r, self.d))
    #         B, b = split_to_B_b(projected_database)
    #         samples[i] = compute_xopt(B, b).reshape((1, -1))
    #
    #     return samples

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        c = self.compute_constant(epsilon, delta)
        X = np.concatenate((self.D, c*np.eye(self.d)))
        samples = np.zeros((num_samples, self.d - 1))

        for i in range(num_samples):
            r_piece = np.minimum(self.chunk_size, self.r)
            noise1 = self.rng.normal(loc=0, scale=1, size=(r_piece, X.shape[0]))
            pi_X = noise1 @ X
            pointer = r_piece

            while pointer < self.r:
                r_piece = np.minimum(self.chunk_size, self.r)
                noise1 = self.rng.normal(loc=0, scale=1, size=(r_piece, X.shape[0]))

                pi_X_piece = noise1 @ X
                pointer += r_piece
                pi_X = np.vstack((pi_X, pi_X_piece))

            projected_database = pi_X
            B, b = split_to_B_b(projected_database)
            samples[i] = compute_xopt(B, b).reshape((1, -1))

        return samples
