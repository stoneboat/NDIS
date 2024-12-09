import gc
import multiprocessing
import os
import secrets
import logging
import time
from functools import partial

import numpy as np
from pympler.asizeof import asizeof
from numpy.random import MT19937, RandomState

from classifier.kNN import train_model, compute_testing_risk, stack_samples, _train_model, stack_parallel_samples, \
    train_model_with_stacked_samples
from utils.commons import compute_tight_bound, accuracy_to_delta, convert_bytes_to_gb, convert_gb_to_bytes

DUMMY_CONSTANT = 1000000000


class GeneralRandomLinearCombinationGenerator:
    def __init__(self, kwargs):
        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]

        assert isinstance(self.X0, np.ndarray), "ERR: required np.ndarray type"
        assert isinstance(self.X1, np.ndarray), "ERR: required np.ndarray type"
        assert self.X0.shape == self.X1.shape, "ERR: neighboring database should be in the same shape"
        assert self.X0.ndim == 2, f"ERR: data base input is in wrong shape, required 2 dimensions"

        # For the new distribution [M(D)]_{epsilon}
        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        self.alternative_sample_noise = DUMMY_CONSTANT * np.ones((self.X0.shape[0], 1))
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
            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            return {'X': (p * (X @ noise) + (1 - p) * self.alternative_sample_noise).T, 'y': np.ones(
                num_samples)}
        else:
            X = self.X0
            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            return {'X': (X @ noise).T, 'y': np.zeros(num_samples)}

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


class NoiseRandomLinearCombinationGenerator(GeneralRandomLinearCombinationGenerator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sigma = kwargs["sigma"]

    # def _gen_samples(self, num_samples, generate_positive_sample):
    #     """
    #         Output form {'X': samples, 'y': labels};
    #         samples is a 2-dimensions array each row is a sample point; there are num_samples row
    #         labels is a 1-dimensions array each coordinate is a label for the corresponding sample
    #     """
    #     if generate_positive_sample:
    #         X = self.X1
    #         noise1 = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
    #         noise2 = self.rng.normal(loc=0, scale=self.sigma**2, size=(X.shape[0], num_samples))
    #         p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
    #         return {'X': (p * (X @ noise1 + noise2) + (1 - p) * self.alternative_sample_noise).T, 'y': np.ones(
    #             num_samples)}
    #     else:
    #         X = self.X0
    #         noise1 = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
    #         noise2 = self.rng.normal(loc=0, scale=self.sigma ** 2, size=(X.shape[0], num_samples))
    #         return {'X': (X @ noise1 + noise2).T, 'y': np.zeros(num_samples)}
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

            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample
            return {'X': (p * (X @ noise) + (1 - p) * self.alternative_sample_noise).T, 'y': np.ones(
                num_samples)}
        else:
            X = self.X0.T
            d = X.shape[1]
            X = np.concatenate((X, self.sigma * np.eye(d))).T

            noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
            return {'X': (X @ noise).T, 'y': np.zeros(num_samples)}


class _GeneralEstimator:
    def __init__(self, kwargs):
        self.sample_generator = None
        self.training_set_size = kwargs["training_set_size"]
        self.validation_set_size = kwargs["validation_set_size"]
        self.gamma = kwargs["gamma"]
        self.dataset_settings = kwargs["dataset_settings"]

        if "batch_size" in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 10**8

        if "samples_file_pos" and "samples_file_neg" in kwargs:
            self.samples_file_pos = kwargs['samples_file_pos']
            self.samples_file_neg = kwargs['samples_file_neg']
        else:
            self.samples_file_pos = None
            self.samples_file_neg = None

        self.output_ = None
        self.model = None

    def read_samples(self, generate_positive_sample=True, file_prefix=None, num_samples=None, index=0):
        if file_prefix is None:
            if generate_positive_sample:
                assert self.samples_file_pos is not None, "Please initialize the estimator with file path"
                file_prefix = self.samples_file_pos
            else:
                assert self.samples_file_neg is not None, "Please initialize the estimator with file path"
                file_prefix = self.samples_file_neg

        file_index = int(np.floor(index/self.batch_size))
        file_extension = ".npy"
        sample_read = 0
        sample_index = index - file_index*self.batch_size
        samples = []

        while sample_read < num_samples:
            # generate the file name and load from it
            file_name = f"{file_prefix}-{file_index}{file_extension}"
            if not os.path.exists(file_name):
                logging.error(
                    f"Cannot load the file. Insufficient amount of data to load, required {index + num_samples} while "
                    f"have {sample_read}")
                logging.info(f"file name is {file_name}, other info {file_index}-{index}-{self.batch_size} ")
                break

            data = np.load(file_name)

            # Determine the number of samples to read in this file
            current_batch_size = min(data.shape[0] - sample_index, num_samples - sample_read)
            samples.append(data[sample_index:sample_index+current_batch_size])

            # Update the pointer
            sample_index = 0
            file_index += 1
            sample_read += current_batch_size

        end_index = sample_read + index

        return {'X': np.vstack(samples), 'y': np.full(end_index-index, generate_positive_sample, dtype=np.float64)}, \
               end_index

    def preprocess_sample(self, workers, generate_positive_sample=True, file_prefix=None, num_samples=None):
        if num_samples is None:
            num_samples = int((self.training_set_size + self.validation_set_size) / 2)

        assert isinstance(generate_positive_sample, bool), "the variable generate_positive_sample shall be a boolean"

        if file_prefix is None:
            if generate_positive_sample:
                assert self.samples_file_pos is not None, "Please initialize the estimator with file path"
                file_prefix = self.samples_file_pos
            else:
                assert self.samples_file_neg is not None, "Please initialize the estimator with file path"
                file_prefix = self.samples_file_neg

        input_list = np.full(workers, generate_positive_sample, dtype=bool)

        # Cause we only want to generate critical samples, we set the claimed_epsilon to zero, will restore back in
        # the end
        probability_of_alternative_sample = self.sample_generator.probability_of_alternative_sample
        self.sample_generator.probability_of_alternative_sample = 0

        file_index = 0
        file_extension = ".npy"
        samples_written = 0
        while samples_written < num_samples:
            # Determine the number of samples to generate in this batch
            current_batch_size = min(self.batch_size, num_samples - samples_written)

            # Generate the batch of random samples
            samples = self.parallel_gen_samples(current_batch_size, workers, input_list)

            # create the file and save into it
            file_name = f"{file_prefix}-{file_index}{file_extension}"
            if os.path.exists(file_name):
                os.remove(file_name)

            np.save(file_name, samples['X'][0:current_batch_size])

            # Update the total number of samples written
            file_index += 1
            samples_written += current_batch_size
            logging.info(f"write {current_batch_size} samples into file {file_name}, status {samples_written}/"
                         f"{num_samples}")

        # restore the value
        self.sample_generator.probability_of_alternative_sample = probability_of_alternative_sample

    def logging_model_size(self):
        if hasattr(self.model, "nbytes"):
            model_size = self.model.nbytes
        else:
            model_size = asizeof(self.model)
        return model_size

    def check_sample_generator(self):
        assert self.sample_generator is not None, "ERR: you need to use a super class or to define the sample " \
                                                  "generator first"

    def parallel_gen_samples(self, num_samples, workers, input_list=None):
        self.check_sample_generator()
        assert workers % 2 == 0, "ERR: expect even number of workers"
        pool = multiprocessing.Pool(processes=workers)

        sample_generating_func = partial(self.sample_generator.parallel_gen_samples_class_label_first,
                                         num_samples=int(np.ceil(num_samples / workers)))

        if input_list is None:
            input_list = np.concatenate((np.ones(int(workers / 2)), np.zeros(int(workers / 2)))).astype(dtype=bool)

        samples = stack_parallel_samples(pool.map(sample_generating_func, input_list))
        return samples

    def parallel_compute_testing_risk(self, samples, workers):
        # I want to run this code to reduce the memory usage before forking process
        gc.collect()
        time.sleep(2.4)

        pool = multiprocessing.Pool(processes=workers)
        partial_empirical_error_rate = partial(compute_testing_risk, model=self.model)

        testing_test_size = samples['y'].shape[0]
        batch_samples = int(np.ceil(testing_test_size / workers))

        samples_x = np.vsplit(samples['X'], indices_or_sections=range(batch_samples, testing_test_size, batch_samples))
        samples_y = np.vsplit(samples['y'].reshape((-1, 1)),
                              indices_or_sections=range(batch_samples, testing_test_size, batch_samples))
        input_list = [{'X': samples_x[i], 'y': samples_y[i].ravel()} for i in range(len(samples_x))]
        results_list = pool.map(partial_empirical_error_rate, input_list)
        return np.mean(np.array(results_list))

    @staticmethod
    def filter_dummy_sample(samples, threshold=DUMMY_CONSTANT / 100):
        """this function returns the samples set without dummy value and the number of dummy sample excluded"""
        idx = np.where((samples['X'].T[0] < threshold) == True)
        return {'X': samples['X'][idx], 'y': samples['y'][idx]}, samples['X'].shape[0] - idx[0].shape[0]

    def parallel_build_nn(self, workers=12, file_name="nn_files", classifier_args=None, sample_batch_size=10 ** 7,
                          preproceed_samples=False):
        """tailored version for using NN classifier"""
        # Generate training samples and train the model
        self.check_sample_generator()
        classifier = "NeuralNetwork"
        logging.info(f'Generate training samples')
        classifier_args['model'] = None
        pointer = 0

        if preproceed_samples:
            pos_file_index = 0
            neg_file_index = 0

        while pointer < self.training_set_size:
            num_sample_batch = np.minimum(sample_batch_size, self.training_set_size - pointer)

            # generate the sample
            tic = time.perf_counter()
            if preproceed_samples:
                n_pos_samples = self.sample_generator.rng.binomial(int(num_sample_batch / 2),
                                                                   self.sample_generator.probability_of_natural_sample)
                pos_samples, pos_file_index = self.read_samples(generate_positive_sample=True,
                                                                num_samples=n_pos_samples,
                                                                index=pos_file_index)
                neg_samples, neg_file_index = self.read_samples(generate_positive_sample=False,
                                                                num_samples=int(num_sample_batch / 2),
                                                                index=neg_file_index)
                samples = stack_samples(pos_samples, neg_samples)
            else:
                samples = self.parallel_gen_samples(num_samples=num_sample_batch, workers=workers)
            toc = time.perf_counter()

            logging.info(f"generate {num_sample_batch} training samples cost {toc - tic:0.4f} seconds, need "
                         f"{convert_bytes_to_gb(asizeof(samples))} GB memory")

            # update the model
            logging.info(f'Update {classifier} classifier')
            tic = time.perf_counter()
            self.model = train_model_with_stacked_samples(samples=samples,
                                                          n_features=self.sample_generator.dimensionality,
                                                          classifier=classifier, file_name=file_name, n_workers=workers,
                                                          classifier_args=classifier_args)
            classifier_args['model'] = self.model
            toc = time.perf_counter()

            # update the pointer and other info
            pointer += num_sample_batch
            model_size = self.logging_model_size()
            logging.critical(f"Update {classifier} classifier cost {toc - tic:0.4f} seconds, model need"
                             f" {convert_bytes_to_gb(model_size)} GB memory. {self.training_set_size - pointer} "
                             f"samples left")

        tic = time.perf_counter()
        if preproceed_samples:
            n_pos_samples = self.sample_generator.rng.binomial(int(self.validation_set_size / 2),
                                                               self.sample_generator.probability_of_natural_sample)
            pos_samples, pos_file_index = self.read_samples(generate_positive_sample=True,
                                                            num_samples=n_pos_samples,
                                                            index=pos_file_index)
            neg_samples, neg_file_index = self.read_samples(generate_positive_sample=False,
                                                            num_samples=int(self.validation_set_size / 2),
                                                            index=neg_file_index)
            samples = stack_samples(pos_samples, neg_samples)
        else:
            samples = self.parallel_gen_samples(num_samples=self.validation_set_size, workers=workers)
        toc = time.perf_counter()
        logging.info(f"generate {samples['X'].shape[0]} testing samples cost {toc - tic:0.4f} seconds, need "
                     f"{convert_bytes_to_gb(asizeof(samples))} GB memory")

        if preproceed_samples:
            filtered_samples = samples
            num_non_dummy_samples = samples['y'].shape[0]
            num_dummy_samples = int(self.validation_set_size / 2)-n_pos_samples
        else:
            filtered_samples, num_dummy_samples = self.filter_dummy_sample(samples, threshold=DUMMY_CONSTANT / 100)
            num_non_dummy_samples = samples['y'].shape[0] - num_dummy_samples

        # Test classifier
        logging.info(f'Test {classifier} classifier')
        tic = time.perf_counter()
        accuracy = (self.parallel_compute_testing_risk(samples=filtered_samples, workers=workers) *
                    num_non_dummy_samples + num_dummy_samples) / (num_non_dummy_samples + num_dummy_samples)

        toc = time.perf_counter()
        logging.critical(f"Compute the empirical error rate requires {toc - tic:0.4f} seconds")

        # Convert to delta
        logging.info('Compute estimated delta')
        mu = compute_tight_bound(
            gamma=self.gamma, n1=self.training_set_size,
            n2=self.validation_set_size, d=self.sample_generator.dimensionality,
            epsilon=self.dataset_settings['claimed_epsilon']
        )

        estimated_delta = accuracy_to_delta(accuracy, self.dataset_settings['claimed_epsilon'])
        estimated_range = (max(0, estimated_delta - mu), max(estimated_delta + mu, 0))
        self.output_ = {
            'estimated_delta': estimated_delta, 'accuracy': accuracy,
            'estimated_range': estimated_range, 'gamma': self.gamma,
            'training_set_size': self.training_set_size, 'validation_set_size': self.validation_set_size
        }

        return self.output_

    def parallel_build(self, classifier="kNN", workers=12, file_name="nn_files", classifier_args=None):
        # Generate training samples
        self.check_sample_generator()
        logging.info('Generate training samples')
        tic = time.perf_counter()
        samples = self.parallel_gen_samples(num_samples=self.training_set_size, workers=workers)
        toc = time.perf_counter()

        logging.info(f"generate {samples['X'].shape[0]} training samples cost {toc - tic:0.4f} seconds, need "
                     f"{convert_bytes_to_gb(asizeof(samples))} GB memory")

        # For high dimensional data, the NULL array (-10^9) in our case is computational inefficient to construct
        # KNN structure using L2 metric, so we can exclude it from the classification because it is very easy to
        # classify
        filtered_samples, num_dummy_samples = self.filter_dummy_sample(samples, threshold=DUMMY_CONSTANT / 100)
        # num_non_dummy_samples = samples['y'].shape[0] - num_dummy_samples

        # Train kNN classifier
        logging.info(f'Train {classifier} classifier')
        tic = time.perf_counter()

        self.model = train_model_with_stacked_samples(samples=filtered_samples,
                                                      n_features=self.sample_generator.dimensionality,
                                                      classifier=classifier, file_name=file_name, n_workers=workers,
                                                      classifier_args=classifier_args)
        toc = time.perf_counter()

        model_size = self.logging_model_size()
        logging.critical(
            f"Train {classifier} classifier cost {toc - tic:0.4f} seconds, model need {convert_bytes_to_gb(model_size)} GB"
            f"memory")

        tic = time.perf_counter()
        samples = self.parallel_gen_samples(num_samples=self.validation_set_size, workers=workers)
        toc = time.perf_counter()
        logging.info(f"generate {samples['X'].shape[0]} testing samples cost {toc - tic:0.4f} seconds, need "
                     f"{convert_bytes_to_gb(asizeof(samples))} GB memory")
        filtered_samples, num_dummy_samples = self.filter_dummy_sample(samples, threshold=DUMMY_CONSTANT / 100)
        num_non_dummy_samples = samples['y'].shape[0] - num_dummy_samples

        # Test kNN classifier
        logging.info(f'Test {classifier} classifier')
        tic = time.perf_counter()
        accuracy = (self.parallel_compute_testing_risk(samples=filtered_samples, workers=workers) *
                    num_non_dummy_samples + num_dummy_samples) / (num_non_dummy_samples + num_dummy_samples)


        toc = time.perf_counter()
        logging.critical(f"Compute the empirical error rate requires {toc - tic:0.4f} seconds")

        # Convert to delta
        logging.info('Compute estimated delta')
        mu = compute_tight_bound(
            gamma=self.gamma, n1=self.training_set_size,
            n2=self.validation_set_size, d=self.sample_generator.dimensionality,
            epsilon=self.dataset_settings['claimed_epsilon']
        )

        estimated_delta = accuracy_to_delta(accuracy, self.dataset_settings['claimed_epsilon'])
        estimated_range = (max(0, estimated_delta - mu), max(estimated_delta + mu, 0))
        self.output_ = {
            'estimated_delta': estimated_delta, 'accuracy': accuracy,
            'estimated_range': estimated_range, 'gamma': self.gamma,
            'training_set_size': self.training_set_size, 'validation_set_size': self.validation_set_size
        }

        return self.output_


class GeneralRandomLinearCombinationEstimator(_GeneralEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sample_generator = GeneralRandomLinearCombinationGenerator(kwargs)


class NoiseRandomLinearCombinationEstimator(_GeneralEstimator):
    def __init__(self, kwargs):
        super().__init__(kwargs=kwargs)
        self.sample_generator = NoiseRandomLinearCombinationGenerator(kwargs)


def demo_generator():
    claimed_epsilon = 0.9

    dataset_settings = {
        'database_0': np.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        'database_1': np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        'claimed_epsilon': claimed_epsilon,
    }

    kwargs = {
        'dataset_settings': dataset_settings
    }

    generator = GeneralRandomLinearCombinationGenerator(kwargs=kwargs)

    samples = generator.gen_samples(num_samples=5, generate_positive_sample=True)

    print(samples)
