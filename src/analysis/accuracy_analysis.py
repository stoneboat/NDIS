import secrets

import numpy as np
import time
import logging

import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

from numpy.linalg import inv as mat_inv

from analysis.commons import compute_xopt, sample_approx_xopt_error, twoNorm, \
    multiplication_sequence_generating_function, get_w
from Dataset.cancer_dataset_for_LS import load_cancer_dataset_matrices_for_LS, main_generate_cancer_matrices_for_LS


def compute_experimental_accuracy(B, b, r_list, n_repetitions=120, workers=12):
    """compute the error between approx_xopt - xopt experimentally by sampling"""
    assert isinstance(r_list, (np.ndarray)), "ERR: expect np.ndarray data type for r_list"
    assert r_list.ndim == 1, "ERR: expect one dimensional row vector"

    pool = multiprocessing.Pool(processes=workers)
    xopt = compute_xopt(B, b)
    partial_sample_approx_xopt_error = partial(sample_approx_xopt_error, B=B, b=b, xopt=xopt)

    experimental_error_list = []
    for r in r_list:
        tic = time.perf_counter()
        error = 0

        # Run several times and use the average
        input_list = [int(r)] * n_repetitions
        output_list = pool.map(partial_sample_approx_xopt_error, input_list)

        error = np.sum(np.array(output_list)) / n_repetitions

        # append the result to the error_list
        experimental_error_list.append(error)
        toc = time.perf_counter()
        logging.info(f"{r} is computed, error is {error}, time cost: {toc - tic:0.4f}s")

    return np.array(experimental_error_list)


def _compute_experimental_asymptotic_distribution_accuracy(B, b, r, num_samples=1000):
    xopt = compute_xopt(B, b)
    error_square = twoNorm(get_w(B, b))**2
    M = mat_inv(B.T @ B)

    seed = secrets.randbits(128)
    rng = np.random.RandomState(np.random.MT19937(seed))

    samples = rng.multivariate_normal(mean=xopt.ravel(), cov=error_square/r*M, size=num_samples) - xopt
    acc_error = 0
    for i in range(num_samples):
        acc_error += twoNorm(samples[i])**2

    return acc_error/num_samples


def _compute_experimental_asymptotic_distribution_accuracy_r_first(r, B, b, num_samples=10000):
    return _compute_experimental_asymptotic_distribution_accuracy(B, b, r, num_samples)


def compute_experimental_asymptotic_distribution_accuracy(B, b, r_list, num_samples=10000, workers=12):
    assert isinstance(r_list, (np.ndarray)), "ERR: expect np.ndarray data type for r_list"
    assert r_list.ndim == 1, "ERR: expect one dimensional row vector"

    pool = multiprocessing.Pool(processes=workers)
    partial_compute_experimental_asymptotic_distribution_accuracy_r_first = \
        partial(_compute_experimental_asymptotic_distribution_accuracy_r_first, B=B, b=b, num_samples=num_samples)

    return np.array(pool.map(partial_compute_experimental_asymptotic_distribution_accuracy_r_first, r_list))


def compute_theoretical_accuracy(B, b, r_list):
    """compute the theoretical error between approx_xopt - xopt"""
    assert isinstance(r_list, (np.ndarray)), "ERR: expect np.ndarray data type for r_list"
    assert r_list.ndim == 1, "ERR: expect one dimensional row vector"

    M = mat_inv(B.T @ B)
    P = B @ M @ B.T
    w = (np.identity(B.shape[0]) - P) @ b

    fixed_error_part = np.trace(M) * (twoNorm(w)**2)

    theoretical_error_list = []

    for r in r_list:
        # append the result to the error_list
        theoretical_error_list.append(fixed_error_part / r)

    return np.array(theoretical_error_list)


if __name__ == "__main__":
    n_instance = 10
    # gen = power_sequence_generating_function(step_size=1.25, first_element=100)
    gen = multiplication_sequence_generating_function(multiplier=2, first_element=10000)

    r_list = np.array([next(gen) for i in range(n_instance)])
    n_repetitions = 120
    workers = 12

    B = None
    b = None
    try:
        B, b = load_cancer_dataset_matrices_for_LS()
    except:
        main_generate_cancer_matrices_for_LS()
        B, b = load_cancer_dataset_matrices_for_LS()

    experimental_error_list = compute_experimental_accuracy(B, b, r_list, n_repetitions, workers)
    theoretical_error_list = compute_theoretical_accuracy(B, b, r_list)

    plt.title(r'LS comparison Xopt and Aprox_Xopt')
    plt.xlabel(r"$r$")
    plt.ylabel("standard error")
    plt.grid()
    plt.plot(r_list, experimental_error_list, color="red", label=r"experimental SE")
    plt.plot(r_list, theoretical_error_list, color="blue", label=r"theoretical SE")
    plt.legend(loc='upper right')
    # plt.savefig(os.getcwd() + "/cancer_accuracy_cmp.png", bbox_inches='tight', dpi=600);
    plt.show()