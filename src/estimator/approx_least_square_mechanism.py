import logging
import secrets
import os

import numpy as np
from numpy.linalg import inv as mat_inv
from numpy.random import MT19937, RandomState

from estimator.GRLC import _GeneralEstimator
from analysis.commons import split_to_B_b, get_w, compute_xopt, twoNorm, data_normalize_by_features, \
    concatenate_B_b, get_neighbor_B, generate_default_configuration
from Dataset.cancer_dataset_for_LS import load_cancer_dataset_matrices_for_LS, main_generate_cancer_matrices_for_LS

DUMMY_CONSTANT = 1000000000


class ApproximateLeastSquareGenerator:
    def __init__(self, kwargs):
        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]
        self.r = int(kwargs["dataset_settings"]["r"])
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]

        assert isinstance(self.X0, np.ndarray), "ERR: required np.ndarray type"
        assert isinstance(self.X1, np.ndarray), "ERR: required np.ndarray type"
        assert self.X0.shape == self.X1.shape, "ERR: neighboring database should be in the same shape"
        assert self.X0.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.B, self.b = split_to_B_b(self.X1)
        self.b = self.b.reshape((-1, 1))
        self.neighbor_B, self.neighbor_b = split_to_B_b(self.X0)
        self.neighbor_b = self.neighbor_b.reshape((-1, 1))
        assert self.B.shape[1] == np.linalg.matrix_rank(self.B), "ERR: expect database X1 has full column rank"
        assert self.neighbor_B.shape[1] == np.linalg.matrix_rank(self.neighbor_B), "ERR: expect database X0 has full " \
                                                                                   "column rank"
        # For the new distribution [M(D)]_{epsilon}
        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        # A row vector approx_xopt
        self.alternative_sample_noise = DUMMY_CONSTANT * np.ones((1, self.B.shape[1]))
        assert DUMMY_CONSTANT > 0, "ERR: negative dummy constant will potentially render " \
                                   "filter_dummy_sample function useless "
        self.dimensionality = self.B.shape[1]
        # same as dimensionality
        self.d = self.B.shape[1]
        # number of records that B has
        self.n = self.B.shape[0]

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def _gen_sample(self, generate_positive_sample, dummpy_sample, chunk_size=10000):
        """No distribution follows; return a row vector"""
        if dummpy_sample and generate_positive_sample:
            return self.alternative_sample_noise

        if generate_positive_sample:
            B = self.B
            b = self.b
        else:
            B = self.neighbor_B
            b = self.neighbor_b

        # Generate Π (pi) and compute sketch matrix chunk by chunk to save the memory
        r_piece = np.minimum(chunk_size, self.r)
        # generate a piece of Π (pi)
        pi_piece = self.rng.normal(size=(r_piece, self.n))
        # Compute a piece of sketch matrix
        pi_B = pi_piece @ B
        pi_b = pi_piece @ b
        # intialize pointer
        pointer = r_piece

        while pointer < self.r:
            r_piece = np.minimum(chunk_size, self.r - pointer)
            # generate a piece of Π (pi)
            pi_piece = self.rng.normal(size=(r_piece, self.n))
            # Compute a piece of sketch matrix
            pi_B_piece = pi_piece @ B
            pi_b_piece = pi_piece @ b

            # Update pointer
            pointer += r_piece
            # Assemble sketch matrix
            pi_B = np.vstack((pi_B, pi_B_piece))
            pi_b = np.vstack((pi_b, pi_b_piece))

        return (mat_inv(pi_B.T @ pi_B) @ (pi_B.T @ pi_b)).reshape((1, -1))

    def gen_samples(self, num_samples, generate_positive_sample, refresh_rate=1000):
        """
            Output form {'X': samples, 'y': labels};
            samples is a 2-dimensions array each row is a sample point; there are num_samples row
            labels is a 1-dimensions array each coordinate is a label for the corresponding sample
        """
        if generate_positive_sample:
            p = 1 - (self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample)
            X = self._gen_sample(generate_positive_sample=True, dummpy_sample=p[0])
            for i in range(1, num_samples):
                if i % refresh_rate == 0:
                    self.reset_randomness()
                    logging.info(f'Generate {i} samples')
                X = np.vstack((X, self._gen_sample(generate_positive_sample=True, dummpy_sample=p[i])))

            return {'X': X, 'y': np.ones(num_samples)}
        else:
            X = self._gen_sample(generate_positive_sample=False, dummpy_sample=False)
            for i in range(1, num_samples):
                if i % refresh_rate == 0:
                    self.reset_randomness()
                    logging.info(f'Generate {i} samples')
                X = np.vstack((X, self._gen_sample(generate_positive_sample=False, dummpy_sample=False)))

            return {'X': X, 'y': np.zeros(num_samples)}

    def parallel_gen_samples_class_label_first(self, generate_positive_sample, num_samples):
        """ If you want to use then gen_samples method in multiple processing,
            keep in mind that each process's copy should have fresh randomness, otherwise just accumulate error rather
            than accuracy
        """
        self.reset_randomness()
        return self.gen_samples(num_samples, generate_positive_sample)


class AsymptoticDistributionGenerator:
    def __init__(self, kwargs):
        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]
        self.r = int(kwargs["dataset_settings"]["r"])
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]

        assert isinstance(self.X0, np.ndarray), "ERR: required np.ndarray type"
        assert isinstance(self.X1, np.ndarray), "ERR: required np.ndarray type"
        assert self.X0.shape == self.X1.shape, "ERR: neighboring database should be in the same shape"
        assert self.X0.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.B, self.b = split_to_B_b(self.X1)
        self.b = self.b.reshape((-1, 1))
        self.mean = compute_xopt(self.B, self.b)
        self.cov = (twoNorm(get_w(self.B, self.b).ravel()) ** 2) * mat_inv(self.B.T @ self.B) / self.r

        self.neighbor_B, self.neighbor_b = split_to_B_b(self.X0)
        self.neighbor_b = self.neighbor_b.reshape((-1, 1))
        self.neighbor_mean = compute_xopt(self.neighbor_B, self.neighbor_b)
        self.neighbor_cov = (twoNorm(get_w(self.neighbor_B, self.neighbor_b).ravel()) ** 2) * mat_inv(
            self.neighbor_B.T @ self.neighbor_B) / self.r

        assert self.B.shape[1] == np.linalg.matrix_rank(self.B), "ERR: expect database X1 has full column rank"
        assert self.neighbor_B.shape[1] == np.linalg.matrix_rank(self.neighbor_B), "ERR: expect database X0 has full " \
                                                                                   "column rank"
        # For the new distribution [M(D)]_{epsilon}
        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        # A row vector approx_xopt
        self.alternative_sample_noise = DUMMY_CONSTANT * np.ones((1, self.B.shape[1]))
        assert DUMMY_CONSTANT > 0, "ERR: negative dummy constant will potentially render " \
                                   "filter_dummy_sample function useless "
        self.dimensionality = self.B.shape[1]
        # same as dimensionality
        self.d = self.B.shape[1]
        # number of records that B has
        self.n = self.B.shape[0]

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def parallel_gen_samples_class_label_first(self, generate_positive_sample, num_samples):
        """ If you want to use then gen_samples method in multiple processing,
            keep in mind that each process's copy should have fresh randomness, otherwise just accumulate error rather
            than accuracy
        """
        self.reset_randomness()
        return self.gen_samples(num_samples, generate_positive_sample)

    def gen_samples(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            samples = self.rng.multivariate_normal(mean=self.mean, cov=self.cov, size=num_samples)
            p = np.random.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            p = p.reshape((num_samples, 1)) * np.ones((num_samples, self.dimensionality))
            return {'X': p * samples + (1 - p) * self.alternative_sample_noise, 'y': np.ones(num_samples)}
        else:
            samples = self.rng.multivariate_normal(mean=self.neighbor_mean, cov=self.neighbor_cov, size=num_samples)
            return {'X': samples, 'y': np.zeros(num_samples)}


class ApproximateLeastSquareEstimator(_GeneralEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sample_generator = ApproximateLeastSquareGenerator(kwargs)


class AsymptoticDistributionEstimator(_GeneralEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sample_generator = AsymptoticDistributionGenerator(kwargs)


if __name__ == "__main__":
    try:
        X, y = load_cancer_dataset_matrices_for_LS(file_X_name=".././Dataset/cancer-LR-X.txt",
                                                   file_y_name=".././Dataset/cancer-LR-y.txt")
    except:
        main_generate_cancer_matrices_for_LS(file_X_name=".././Dataset/cancer-LR-X.txt",
                                             file_y_name=".././Dataset/cancer-LR-y.txt")
        X, y = load_cancer_dataset_matrices_for_LS(file_X_name=".././Dataset/cancer-LR-X.txt",
                                                   file_y_name=".././Dataset/cancer-LR-y.txt")

    B, b = data_normalize_by_features(X, y)

    # B = B[:, 10]
    # B = B.reshape((-1, 1))
    A = concatenate_B_b(B, b)

    claimed_epsilon = 0.1
    index = 186
    r = 1000

    workers = 10
    SAMPLE_SIZE = int(10 ** 2)
    neighbor_A = get_neighbor_B(index, A)
    kwargs = generate_default_configuration(claimed_ep=claimed_epsilon, sample_size=SAMPLE_SIZE, database_0=neighbor_A,
                                            database_1=A, r=r)

    generator = AsymptoticDistributionGenerator(kwargs)
    samples = generator.gen_samples(num_samples=SAMPLE_SIZE, generate_positive_sample=False)

    print(samples['X'].shape)
