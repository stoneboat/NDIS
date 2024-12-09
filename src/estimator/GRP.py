import secrets

import numpy as np
from numpy.random import MT19937, RandomState

from estimator.GRLC import _GeneralEstimator
from classifier.kNN import stack_samples
from utils.commons import convert_gb_to_bytes

DUMMY_CONSTANT = 1000000000


class GeneralRandomProjectionGenerator:
    def __init__(self, kwargs):
        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]
        self.r = kwargs["dataset_settings"]["r"]
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]

        assert isinstance(self.X0, np.ndarray), "ERR: required np.ndarray type"
        assert isinstance(self.X1, np.ndarray), "ERR: required np.ndarray type"
        assert self.X0.shape == self.X1.shape, "ERR: neighboring database should be in the same shape"
        assert self.X0.ndim == 2, f"ERR: data base input is in wrong shape, required 2 dimensions"

        # For the new distribution [M(D)]_{epsilon}
        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        self.alternative_sample_noise = DUMMY_CONSTANT * np.ones((self.X0.shape[0]*self.r, 1))
        assert DUMMY_CONSTANT > 0, "ERR: negative dummy constant will potentially render " \
                                   "filter_dummy_sample function useless "
        self.dimensionality = self.X1.shape[0]

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def reset_randomness(self):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def _gen_samples(self, num_samples, generate_positive_sample):
        """
            Output form {'X': samples, 'y': labels};
            samples is a 2-dimensions array each row is a sample point; there are num_samples row
            labels is a 1-dimensions array each coordinate is a label for the corresponding sample
        """
        if generate_positive_sample:
            X = self.X1
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample

            # List to store the resulting matrices
            result_matrices = []

            for _ in range(self.r):
                noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
            final_matrix = (p*np.concatenate(result_matrices, axis=0) + (1 - p) * self.alternative_sample_noise).T

            return {'X': final_matrix, 'y': np.ones(num_samples)}
        else:
            X = self.X0
            # List to store the resulting matrices
            result_matrices = []

            for _ in range(self.r):
                noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
            final_matrix = np.concatenate(result_matrices, axis=0).T

            return {'X': final_matrix, 'y': np.zeros(num_samples)}

    def gen_samples(self, num_samples, generate_positive_sample, allowed_memory_gb=1):
        X = self.X1
        chunk_size = int(convert_gb_to_bytes(allowed_memory_gb) / (X[0][0].nbytes * X.shape[1]))
        num_batch_samples = np.minimum(chunk_size, num_samples)
        batch_samples = self._gen_samples(num_batch_samples, generate_positive_sample)

        pointer = num_batch_samples
        while pointer < num_samples:
            num_batch_samples = np.minimum(chunk_size, num_samples - pointer)
            batch_samples = stack_samples(batch_samples, self._gen_samples(num_batch_samples, generate_positive_sample))

            # update pointer
            pointer += num_batch_samples

        return batch_samples

    def parallel_gen_samples_class_label_first(self, generate_positive_sample, num_samples, allowed_memory_gb=1):
        """ If you want to use then gen_samples method in multiple processing,
            keep in mind that each process's copy should have fresh randomness, otherwise just accumulate error rather
            than accuracy
        """
        self.reset_randomness()
        return self.gen_samples(num_samples, generate_positive_sample, allowed_memory_gb)


class NoiseRandomProjectionGenerator(GeneralRandomProjectionGenerator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sigma = kwargs["sigma"]

    def _gen_samples(self, num_samples, generate_positive_sample):
        """
            Output form {'X': samples, 'y': labels};
            samples is a 2-dimensions array each row is a sample point; there are num_samples row
            labels is a 1-dimensions array each coordinate is a label for the corresponding sample
        """
        if generate_positive_sample:
            X = self.X1.T
            d = X.shape[1]
            X = np.concatenate((X, self.sigma * np.eye(d))).T

            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample

            # List to store the resulting matrices
            result_matrices = []

            for _ in range(self.r):
                # noise1 = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                # noise2 = self.rng.normal(loc=0, scale=self.sigma ** 2, size=(X.shape[0], num_samples))
                # result_matrices.append(X @ noise1 + noise2)

                noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
            final_matrix = (p * np.concatenate(result_matrices, axis=0) + (1 - p) * self.alternative_sample_noise).T

            return {'X': final_matrix, 'y': np.ones(num_samples)}
        else:
            X = self.X0.T
            d = X.shape[1]
            X = np.concatenate((X, self.sigma * np.eye(d))).T
            # List to store the resulting matrices
            result_matrices = []

            for _ in range(self.r):
                # noise1 = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                # noise2 = self.rng.normal(loc=0, scale=self.sigma ** 2, size=(X.shape[0], num_samples))
                # result_matrices.append(X @ noise1 + noise2)

                noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
                result_matrices.append(X @ noise)

            # Concatenating the resulting matrices by row
            final_matrix = np.concatenate(result_matrices, axis=0).T

            return {'X': final_matrix, 'y': np.zeros(num_samples)}


class GeneralRandomProjectionEstimator(_GeneralEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sample_generator = GeneralRandomProjectionGenerator(kwargs)


class NoiseRandomProjectionEstimator(_GeneralEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sample_generator = NoiseRandomProjectionGenerator(kwargs)


def demo_generator():
    claimed_epsilon = 0.9

    dataset_settings = {
        'database_0': np.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        'database_1': np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        'claimed_epsilon': claimed_epsilon,
        'r':2
    }

    kwargs = {
        'dataset_settings': dataset_settings
    }

    generator = GeneralRandomProjectionGenerator(kwargs=kwargs)

    samples = generator.gen_samples(num_samples=5, generate_positive_sample=True)

    print(samples)
