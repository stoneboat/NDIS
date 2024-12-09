import logging
import multiprocessing
import os
import secrets
import time
import gc
from enum import Enum
from functools import partial
from math import ceil

import numpy as np
from numpy.random import RandomState, MT19937

from classifier.EDE import EmpiricalDistributionEstimator
from utils.commons import accuracy_to_delta


class SVTVariant(Enum):
    SVT1 = 1
    SVT3 = 3
    SVT4 = 4
    SVT5 = 5
    SVT6 = 6
    # really interested in get an Gaussian version SVT
    SVT7 = 7


class DataPattern(Enum):
    SIMPLE = 1
    RANDOM = 2


def gen_test_neighbors(query_length, sensitivity, svt_variant, data_pattern):
    if (svt_variant == SVTVariant.SVT1 or svt_variant == SVTVariant.SVT4 or svt_variant == SVTVariant.SVT6 or
        svt_variant == SVTVariant.SVT7) and \
            data_pattern == DataPattern.SIMPLE:
        c_array = np.full(query_length, sensitivity) - sensitivity/2
        c_neighbor_array = np.full(query_length, 0) - sensitivity/2
        return c_array, c_neighbor_array

    if (svt_variant == SVTVariant.SVT1 or svt_variant == SVTVariant.SVT4 or svt_variant == SVTVariant.SVT6 or
        svt_variant == SVTVariant.SVT7) and \
            data_pattern == DataPattern.RANDOM:
        seed = secrets.randbits(32)
        np.random.seed(seed)
        c_array = np.random.randint(2, size=query_length) * sensitivity - sensitivity/2
        c_neighbor_array = - c_array
        return c_array, c_neighbor_array

    if svt_variant == SVTVariant.SVT5:
        c_array = np.full(query_length, 0)  # decreasing order of gap
        c_array[0] = sensitivity
        c_neighbor_array = np.full(query_length, sensitivity)
        c_neighbor_array[0] = 0
        return c_array, c_neighbor_array


class SVTGenerator:
    """
    For the SVT1, SVT3, SVT4 we assume the number of query above the threshold should be 1
    """
    def __init__(self, svt_variant, kwargs):
        self.largevalue = 1000000000
        self.X0 = kwargs["dataset_settings"]["database_0"]
        self.X1 = kwargs["dataset_settings"]["database_1"]

        if not isinstance(self.X0, np.ndarray):
            self.X0 = np.array(self.X0)
        if not isinstance(self.X1, np.ndarray):
            self.X1 = np.array(self.X1)
        self.X0 = np.reshape(self.X0, (1, self.X0.size))
        self.X1 = np.reshape(self.X1, (1, self.X1.size))

        self.sensitivity = kwargs["dataset_settings"]["sensitivity"]
        self.one = np.ones((1,))
        self.zero = np.zeros((1,))
        self.epsilon = kwargs["dataset_settings"]["epsilon"]
        self.claimed_epsilon = kwargs["dataset_settings"]["claimed_epsilon"]

        try:
            self.delta = kwargs["dataset_settings"]["delta"]
        except:
            self.delta = None

        assert (self.X1.size == self.X0.size)
        assert isinstance(svt_variant, SVTVariant)
        self.svt_variant = svt_variant
        self.dim = self.dimensionality()

        self.probability_of_natural_sample = 1 / (np.exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        # we output an alternate sample that has negligible probability
        # (approx exp(-10000000)) of being generated by the laplace distribution
        self.alternative_sample_noise = -self.largevalue

        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def query_length(self):
        return self.X1.size

    def max_gap(self):
        return np.maximum(np.max(self.X1), np.max(self.X0))

    def dimensionality(self):
        assert isinstance(self.svt_variant, SVTVariant)
        if self.svt_variant == SVTVariant.SVT1:
            return 1
        if self.svt_variant == SVTVariant.SVT3:
            return 2
        if self.svt_variant == SVTVariant.SVT4:
            return 1
        if self.svt_variant == SVTVariant.SVT5:
            return 1
        if self.svt_variant == SVTVariant.SVT6:
            assert isinstance(self.X1, np.ndarray)
            return self.X1.size

    def less_than_func(self):
        def less_than_func_svt1(a, b):
            if a < b:
                return True
            else:
                return False

        if self.svt_variant == SVTVariant.SVT1:
            return less_than_func_svt1

    def gen_samples(self, num_samples, generate_positive_sample):
        if self.svt_variant == SVTVariant.SVT1:
            return self.gen_samples_SVT1(num_samples, generate_positive_sample)
        if self.svt_variant == SVTVariant.SVT4:
            return self.gen_samples_SVT4(num_samples, generate_positive_sample)
        if self.svt_variant == SVTVariant.SVT5:
            return self.gen_samples_SVT5(num_samples, generate_positive_sample)
        if self.svt_variant == SVTVariant.SVT7:
            return self.gen_samples_SVT7(num_samples, generate_positive_sample)

    def gen_samples_to_file(self, file_name,  num_samples, generate_positive_sample: bool, is_sorted: bool):
        samples = self.gen_samples(num_samples, generate_positive_sample)
        if is_sorted:
            samples['X'].sort()

        file_writer = np.memmap(file_name, dtype=samples['X'].dtype, mode='w+', shape=(num_samples, self.dim+1))
        buffer = np.hstack((samples['X'].reshape((num_samples, self.dim)), samples['y'].reshape((num_samples, 1))))
        file_writer[:] = buffer[:]

        file_writer.flush()
        return file_name

    def gen_samples_SVT1(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = self.one * np.ones(num_samples)

            rho = self.rng.laplace(scale=2 / self.epsilon, size=(num_samples, 1))
            nu = self.rng.laplace(scale=4 / self.epsilon, size=(num_samples, self.X1.size))
            b = np.reshape(np.ones(num_samples).astype(bool), (num_samples, 1))
            cmp = np.hstack((nu >= (rho + self.X1), b)).tolist()  # broadcasts rho horizontally
            X = np.array([cmp[i].index(True) for i in range(num_samples)])
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample

            return {'X': p * X + (1 - p) * self.alternative_sample_noise, 'y': y.astype("int")}
        else:
            y = self.zero * np.ones(num_samples)

            rho = self.rng.laplace(scale=2 / self.epsilon, size=(num_samples, 1))
            nu = self.rng.laplace(scale=4 / self.epsilon, size=(num_samples, self.X0.size))
            b = np.reshape(np.ones(num_samples).astype(bool), (num_samples, 1))
            cmp = np.hstack((nu >= (rho + self.X0), b)).tolist()
            X = np.array([cmp[i].index(True) for i in range(num_samples)])

            return {'X': X, 'y': y.astype("int")}

    def gen_samples_SVT7(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = self.one * np.ones(num_samples)
            gaussian_scale_1 = 2 * np.log(1.25 / self.delta) * np.power(self.sensitivity, 2) / np.power(
                self.epsilon/2, 2)

            gaussian_scale_2 = 2 * np.log(1.25 / self.delta) * np.power(self.sensitivity, 2) / np.power(
                self.epsilon/4, 2)

            rho = self.rng.normal(np.sqrt(gaussian_scale_1), size=(num_samples, 1))
            nu = self.rng.normal(np.sqrt(gaussian_scale_2), size=(num_samples, self.X1.size))
            b = np.reshape(np.ones(num_samples).astype(bool), (num_samples, 1))
            cmp = np.hstack((nu >= (rho + self.X1), b)).tolist()  # broadcasts rho horizontally
            X = np.array([cmp[i].index(True) for i in range(num_samples)])
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample

            return {'X': p * X + (1 - p) * self.alternative_sample_noise, 'y': y.astype("int")}
        else:
            y = self.zero * np.ones(num_samples)
            gaussian_scale_1 = 2 * np.log(1.25 / self.delta) * np.power(self.sensitivity, 2) / np.power(
                2 /self.epsilon, 2)

            gaussian_scale_2 = 2 * np.log(1.25 / self.delta) * np.power(self.sensitivity, 2) / np.power(
                4 / self.epsilon, 2)

            rho = self.rng.normal(np.sqrt(gaussian_scale_1), size=(num_samples, 1))
            nu = self.rng.normal(np.sqrt(gaussian_scale_2), size=(num_samples, self.X0.size))
            b = np.reshape(np.ones(num_samples).astype(bool), (num_samples, 1))
            cmp = np.hstack((nu >= (rho + self.X0), b)).tolist()
            X = np.array([cmp[i].index(True) for i in range(num_samples)])

            return {'X': X, 'y': y.astype("int")}

    def gen_samples_SVT4(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = self.one * np.ones(num_samples)

            rho = self.rng.laplace(scale=4 / self.epsilon, size=(num_samples, 1))
            nu = self.rng.laplace(scale=4 / (3*self.epsilon), size=(num_samples, self.X1.size))
            b = np.reshape(np.ones(num_samples).astype(bool), (num_samples, 1))
            cmp = np.hstack((nu >= (rho + self.X1), b)).tolist()  # broadcasts rho horizontally
            X = np.array([cmp[i].index(True) for i in range(num_samples)])
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample

            return {'X': p * X + (1 - p) * self.alternative_sample_noise, 'y': y.astype("int")}
        else:
            y = self.zero * np.ones(num_samples)

            rho = self.rng.laplace(scale=4 / self.epsilon, size=(num_samples, 1))
            nu = self.rng.laplace(scale=4 / (3*self.epsilon), size=(num_samples, self.X0.size))
            b = np.reshape(np.ones(num_samples).astype(bool), (num_samples, 1))
            cmp = np.hstack((nu >= (rho + self.X0), b)).tolist()
            X = np.array([cmp[i].index(True) for i in range(num_samples)])

            return {'X': X, 'y': y.astype("int")}

    def gen_samples_SVT5(self, num_samples, generate_positive_sample):
        if generate_positive_sample:
            y = self.one * np.ones(num_samples)

            rho = self.rng.laplace(scale=2 / self.epsilon, size=(num_samples, 1))
            cmp = (0 >= (rho + self.X1))
            X = np.packbits(cmp, axis=1, bitorder='little').ravel()
            p = self.rng.uniform(0, 1, num_samples) > self.probability_of_alternative_sample

            return {'X': p * X + (1 - p) * self.alternative_sample_noise, 'y': y.astype("int")}
        else:
            y = self.zero * np.ones(num_samples)

            rho = self.rng.laplace(scale=2 / self.epsilon, size=(num_samples, 1))
            cmp = (0 >= (rho + self.X0))
            X = np.packbits(cmp, axis=1, bitorder='little').ravel()

            return {'X': X, 'y': y.astype("int")}


def gen_empirical_estimator(generate_positive_sample, svt_variant, estimator_configure, number_samples):
    sample_generator = SVTGenerator(svt_variant=svt_variant, kwargs=estimator_configure)
    samples = sample_generator.gen_samples(number_samples, generate_positive_sample)
    estimator = EmpiricalDistributionEstimator()
    estimator.build(samples)
    return estimator


def gen_samples(generate_positive_sample, svt_variant, estimator_configure, number_samples):
    sample_generator = SVTGenerator(svt_variant=svt_variant, kwargs=estimator_configure)
    return sample_generator.gen_samples(number_samples, generate_positive_sample)


def parallel_gen_samples(number_samples, svt_variant, estimator_configure, batch_size, is_sorted, workers):
    if workers % 2 != 0:
        workers += 1
    number_round = ceil(number_samples/(batch_size*workers))
    pool = multiprocessing.Pool(processes=workers)
    partial_gen_samples = partial(gen_samples, svt_variant=svt_variant, estimator_configure=estimator_configure,
                                  number_samples=batch_size)

    tic = time.perf_counter()
    number_computed_samples = 0
    dim = None

    tmp_output_X = []
    tmp_output_y = []
    for round in range(number_round):
        input_list = np.hstack((np.zeros(int(workers/2)), np.ones(int(workers/2)))).astype(bool).tolist()
        output_list = pool.map(partial_gen_samples, input_list)
        dim = output_list[0]['X'].ndim
        if dim == 1:
            tmp_output_X.append(np.hstack([output_list[i]['X'] for i in range(workers)]))
            tmp_output_y.append(np.hstack([output_list[i]['y'] for i in range(workers)]))
        elif dim > 1:
            tmp_output_X.append(np.vstack([output_list[i]['X'] for i in range(workers)]))
            tmp_output_y.append(np.hstack([output_list[i]['y'] for i in range(workers)]))

        del output_list
        gc.collect()

        toc = time.perf_counter()
        number_computed_samples += batch_size*workers
        logging.info(f"{number_computed_samples} samples computed in {toc - tic:0.4f} seconds")

    if dim == 1:
        output_X = np.hstack(tmp_output_X)
        output_y = np.hstack(tmp_output_y)
    if dim > 1:
        output_X = np.vstack(tmp_output_X)
        output_y = np.hstack(tmp_output_y)

    del tmp_output_y
    del tmp_output_X
    gc.collect()

    if is_sorted:
        assert dim == 1
        tic = time.perf_counter()
        ids = output_X.argsort()
        samples = {'X': output_X[ids], 'y': output_y[ids]}
        del ids
        gc.collect()
        toc = time.perf_counter()
        logging.info(f"sort samples in {toc - tic:0.4f} seconds")
    else:
        samples = {'X': output_X, 'y': output_y}

    return samples


#   input_tuple[0] is a boolean value for generate_positive_sample; input_tuple[1] is the file name stored samples
def gen_samples_to_file(input_tuple, svt_variant, estimator_configure, number_samples, is_sorted):
    sample_generator = SVTGenerator(svt_variant=svt_variant, kwargs=estimator_configure)
    file_name = sample_generator.gen_samples_to_file(file_name=input_tuple[1], num_samples=number_samples,
                                                     generate_positive_sample=input_tuple[0], is_sorted=is_sorted)
    return file_name


def parallel_build_estimator(number_samples, svt_variant, estimator_configure, batch_size, workers):
    if workers % 2 != 0:
        workers += 1
    estimator = EmpiricalDistributionEstimator()
    number_round = ceil(number_samples/(batch_size*workers))
    pool = multiprocessing.Pool(processes=workers)
    partial_gen_samples = partial(gen_empirical_estimator, svt_variant=svt_variant, estimator_configure=estimator_configure,
                                  number_samples=batch_size)

    tic = time.perf_counter()
    number_computed_samples = 0
    for round in range(number_round):

        input_list = np.hstack((np.zeros(int(workers/2)), np.ones(int(workers/2)))).astype(bool).tolist()
        output_list = pool.map(partial_gen_samples, input_list)

        for sub_estimator in output_list:
            estimator.combine_distribution(sub_estimator)

        del output_list
        gc.collect()

        toc = time.perf_counter()
        number_computed_samples += batch_size*workers
        logging.info(f"{number_computed_samples} samples computed in {toc - tic:0.4f} seconds")

    return estimator


def parallel_gen_samples_to_files(number_samples, svt_variant, estimator_configure, batch_size,
                                  directory_name, is_sorted, workers):
    if workers % 2 != 0:
        workers += 1
    number_round = ceil(number_samples/(batch_size*workers))
    pool = multiprocessing.Pool(processes=workers)
    partial_gen_samples_to_file = partial(gen_samples_to_file, svt_variant=svt_variant,
                                          estimator_configure=estimator_configure, number_samples=batch_size,
                                          is_sorted=is_sorted)

    tic = time.perf_counter()
    number_computed_samples = 0
    file_name_lists = []

    for round in range(number_round):

        boolean_list = np.hstack((np.zeros(int(workers/2)), np.ones(int(workers/2)))).astype(bool).tolist()
        file_name_list = [os.path.join(directory_name, str(round)+"_"+str(i)) for i in range(workers)]
        input_list = list(zip(boolean_list, file_name_list))
        output_list_ = pool.map(partial_gen_samples_to_file, input_list)
        file_name_lists.extend(output_list_)
        toc = time.perf_counter()
        number_computed_samples += batch_size*workers
        logging.info(f"{number_computed_samples} samples computed and written to files in {toc - tic:0.4f} "
                     f"seconds")

    return file_name_lists


