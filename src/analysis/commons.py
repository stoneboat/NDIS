import logging
import time
from functools import partial

import numpy as np
import mpmath as mp
import secrets
from sklearn import preprocessing
import multiprocessing

from numpy.linalg import inv as mat_inv
from numpy.random import RandomState, MT19937
from numpy.linalg import eigh
from scipy.stats import ncx2
from scipy.stats import chi2

from scipy.linalg import sqrtm

from utils.constants import WORKERS
from utils.empirical_bootstrap import EmpiricalBootstrap, SampleGenerator


def generate_default_configuration(claimed_ep, sample_size, database_0, database_1, r=0):
    claimed_epsilon = claimed_ep
    gamma = 0.01

    dataset_settings = {
        'database_0': database_0,
        'database_1': database_1,
        'claimed_epsilon': claimed_epsilon,
        'r': r,
    }

    kwargs = {
        'dataset_settings': dataset_settings, 'random_seed': int(time.time()),
        'gamma': gamma,
        'training_set_size': sample_size, 'validation_set_size': int(sample_size/10)
    }
    return kwargs


def twoNorm(x):
    """Take a one dimensional vector and output its Euclidean norm"""
    assert isinstance(x, (np.ndarray)), "ERR: expect np.ndarray data type"
    assert ((x.ndim == 1) or (x.ndim == 2 and x.shape[1] == 1)), "ERR: expect one dimensional vector or a column vector"
    sum = 0
    for i in range(len(x)):
        sum += x[i] ** 2

    if x.ndim == 1:
        return np.sqrt(sum)
    else:
        return np.sqrt(sum).item()



def concatenate_B_b(B, b):
    """return the concatenation [B, b]"""
    return np.hstack((B, b.reshape((-1, 1))))


def split_to_B_b(A):
    num_columns = A.shape[1] - 1
    b = A.T[num_columns].copy().ravel()
    B = np.delete(A, num_columns, 1)
    return B, b


def compute_xopt(B, b):
    """compute the optimal solution of LS: xopt = (B^T * B)^{-1}* B^T * b"""
    # return np.matmul(np.matmul(mat_inv(np.matmul(B.T, B)), B.T), b)
    # return an one dimensional vector, seems not reasonable
    return (mat_inv(B.T @ B) @ (B.T @ b)).ravel()


def compute_approx_xopt(r, B, b):
    """sample an approximate optimal solution of LS according to FOCS06 Sarlós."""

    # Generate randomness
    seed = secrets.randbits(128)
    rng = RandomState(MT19937(seed))

    # Number of column
    d = B.shape[1]
    # Number of row
    n = B.shape[0]
    # generate Π (pi)
    pi = rng.normal(size=(r, n))
    # Compute sketch matrix
    pi_B = pi @ B
    pi_b = pi @ b

    return compute_xopt(pi_B, pi_b)


def compute_approx_xopt_chunk(r, B, b, chunk_size=10000):
    """ sample an approximate optimal solution of LS according to FOCS06 Sarlós.
        This is an memory efficient version of compute_approx_xopt
    """
    # Generate randomness
    seed = secrets.randbits(128)
    rng = RandomState(MT19937(seed))

    # Number of column
    d = B.shape[1]
    # Number of row
    n = B.shape[0]
    b = b.reshape((-1, 1))

    # Generate Π (pi) and compute sketch matrix chunk by chunk to save the memory
    r_piece = np.minimum(chunk_size, r)
    # generate a piece of Π (pi)
    pi_piece = rng.normal(size=(r_piece, n))
    # Compute a piece of sketch matrix
    pi_B = pi_piece @ B
    pi_b = pi_piece @ b
    # intialize pointer
    pointer = r_piece

    while pointer < r:
        r_piece = np.minimum(chunk_size, r - pointer)
        # generate a piece of Π (pi)
        pi_piece = rng.normal(size=(r_piece, n))
        # Compute a piece of sketch matrix
        pi_B_piece = pi_piece @ B
        pi_b_piece = pi_piece @ b

        # Update pointer
        pointer += r_piece
        # Assemble sketch matrix
        pi_B = np.vstack((pi_B, pi_B_piece))
        pi_b = np.vstack((pi_b, pi_b_piece))

    pi_b = pi_b.ravel()
    return compute_xopt(pi_B, pi_b)


def sample_approx_xopt_error(r, B, b, xopt=None, memory_efficient=True):
    if xopt is None:
        xopt = compute_xopt(B, b)

    approx_xopt = None
    if memory_efficient:
        approx_xopt = compute_approx_xopt_chunk(r, B, b)
    else:
        approx_xopt = compute_approx_xopt(r, B, b)
    return twoNorm(xopt - approx_xopt)**2


def power_sequence_generating_function(step_size, first_element, length=100000):
    """this function generates power sequence generator"""
    assert step_size != 0, "step_size cannot be 0"
    assert first_element != 0, "first_element cannot be 0"
    output = first_element
    yield output
    for i in range(length - 1):
        output = output * step_size
        yield output


def multiplication_sequence_generating_function(multiplier, first_element, length=100000):
    """this function generates scalar multiplier sequence generator"""
    assert multiplier != 0, "step_size cannot be 0"
    assert first_element != 0, "first_element cannot be 0"
    for i in range(length):
        yield first_element * (i + 1) * multiplier


def calc_proj_matrix(A):
    """Take a m by n matrix and computes m by m projection matrix"""
    return A @ np.linalg.inv(A.T @ A) @ A.T


def clac_standard_vec(d, index):
    """
        Output the index-th d-dimensional standard vector
        note index is from 0 to d-1
    """
    arr = np.zeros(d)
    arr[index] = 1
    return arr.reshape(-1, 1)


def get_neighbor_b(index, b):
    """ Compute the neighbor vector b' of b such that b' is the one coordinate zero-out version of b """
    assert isinstance(b, (np.ndarray)), "ERR: expect np.ndarray data type for x"
    assert b.ndim == 1, "ERR: expect one dimensional matrix"
    assert index < len(b), "ERR: the index is out of bound"
    neighbor_b = np.copy(b)
    neighbor_b[index] = 0
    return neighbor_b


def get_neighbor_B(index, x):
    """ Compute the neighbor matrix B' of B such that B' is the one row zero-out version of B """
    assert isinstance(x, (np.ndarray)), "ERR: expect np.ndarray data type for B"
    assert x.ndim == 2, "ERR: expect two dimensional matrix"
    assert index < x.shape[0], "ERR: the index is out of bound"
    y = x.copy()
    v = y[index].copy().reshape(-1, 1)
    y[index][:] = 0 * y[index][:]
    assert x.shape[1] == np.linalg.matrix_rank(x), "ERR: expect x has full column rank"
    assert y.shape[1] == np.linalg.matrix_rank(y), "ERR: expect y has full column rank"
    return y


def get_neighbors_index_list(B, b):
    A = concatenate_B_b(B, b)
    return np.array([i for i in range(A.shape[0])])


def data_normalize_by_features(B, b):
    X = preprocessing.normalize(B, axis=0)
    y = preprocessing.normalize(b.reshape((-1, 1)), axis=0).ravel()
    return X, y


def data_normalize_by_sample(B, b):
    X = preprocessing.normalize(B, axis=1)
    y = preprocessing.normalize(b.reshape((-1, 1)), axis=1).ravel()
    return X, y


def get_w(B, b):
    """ B \in R^{n * d} ,b \in R^{n * 1} be a system of equation;
        P = B(B^TB)^{-1}B^T is the projection matrix of B, which projects any n-dimensional vector x into
        the column space of B;
        w = (I-P)b is the error (residual) vector, which is orthorgonal to the column space of B,
        and we know w + Pb = b;
        This function computes the error vector w corresponding to B,b
    """
    M = mat_inv(B.T @ B)

    # P = B @ M @ B.T
    # return (np.identity(B.shape[0]) - P) @ b

    return b - B @ (M @ (B.T @ b))


def get_neighbor_w(index, B, b):
    """ We consider the one row opt-out version of the B,b, we call that B', b';
        this function compute the error vector w' corresponding to B',b'
    """
    neighbor_B = get_neighbor_B(index, B)
    neighbor_b = get_neighbor_b(index, b)
    return get_w(neighbor_B, neighbor_b)


# def compute_projection_diagonal_elements(B):
#     """Let P be projection matrix of B, return the diagonal elements of the projection matrix for B"""
#     return calc_proj_matrix(B).diagonal()


def compute_projection_diagonal_elements(B):
    """Let P be projection matrix of B, return the diagonal elements of the projection matrix for B"""
    num = B.shape[0]

    # memory is enough
    if num < 10000:
        return calc_proj_matrix(B).diagonal()
    else:
        ret = np.zeros(num)
        M = np.linalg.inv(B.T @ B)
        for i in range(num):
            v = B[i].reshape((-1, 1))
            ret[i] = v.T@M@v
            
        return ret


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_equal(a, b, rtol=1e-05, atol=1e-08):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def gx2_params_norm_quad(mu, v, q):
    """A quadratic form of a normal variable is distributed as a generalized chi-squared.
        This function takes the normal parameters and the quadratic coeffs and x.TAx
        returns the parameters of the generalized chi-squared:
        w         the weights of the non-central chi-squares, a one-dimensional np.array
        k         the degrees of freedom of each non-central chi-squares, a one-dimensional np.array
        non_central_parameters the non-centrality paramaters (sum of squares of means) of the non-central chi-squares, , a one-dimensional np.array
        m         mean of normal term
        s         sandard deviation of normal term
    """

    assert isinstance(mu, (np.ndarray)), "ERR: expect np.ndarray data type for expectation vector mu"
    assert isinstance(v, (np.ndarray)), "ERR: expect np.ndarray data type for covariance matrix v"
    assert isinstance(q, (np.ndarray)), "ERR: expect np.ndarray data type for quadratic matrix q2"
    assert check_symmetric(q), "ERR: expect a symmetric quadratic form"
    mu = mu.reshape((-1, 1))
    n_variables = mu.shape[0]
    assert v.shape == (n_variables, n_variables), "ERR: covariance matrix v is not in good shape"
    assert q.shape == (n_variables, n_variables), "ERR: quadratic matrix q is not in good shape"

    # Standardize the multi-variate normal distribution with mean mu and covariance v;
    # This operation transforms the quadratic form to Z.Tq2Z + q1.TZ + q0
    S = sqrtm(v)
    q2 = S @ q @ S
    q1 = 2 * S @ q @ mu
    q0 = mu.T @ q @ mu

    # Diagonalized the q2 to make Z's coordinate independent of each other;
    # This operation transforms the quadratic form to X.TDX + b.TX + q0
    D, R = eigh(q2)
    b = R.T @ q1

    # Obtain all unique eigenvalues and their corresponding multiplicities
    w, k = np.unique(D, return_counts=True)

    # Compute noncetrality parameter for each independent non-central chi-square distribution
    non_central_parameters = np.zeros(w.shape)
    for i in range(w.shape[0]):
        numerator = b[D == w[i]]
        non_central_parameters[i] = np.sum(numerator * numerator) / (4 * w[i] ** 2)

    # Compute the normal parameter
    m = (q0 - w @ non_central_parameters).ravel()[0]
    s = twoNorm(b[D == 0].ravel())

    return w, k, non_central_parameters, m, s


def gx2cdf_ruben(x, w, k, non_central_parameters, m, side="lower", N=100):
    """ This functions given a generalized chi-square distribution (a weighted sum of non-central chi-squares
        with all weights the same sign), using Ruben's [1962] method.
        output the cdf of it evaluating on the point x;
        w         the weights of the non-central chi-squares, a one-dimensional np.array
        k         the degrees of freedom of each non-central chi-squares, a one-dimensional np.array
        non_central_parameters the non-centrality paramaters (sum of squares of means) of the non-central chi-squares, , a one-dimensional np.array
        m         mean of normal term
        side      "lower" for cdf and "upper" for sf; when cdf is too small it is recommended computing sf instead
        N         no. of terms in the approximation. Default = 100.

        Outputs
        p         computed cdf
        errbnd    upper error bound of the computed cdf
    """

    assert np.isreal(x), "ERR: x must be a real number"
    assert isinstance(w, (np.ndarray)), "ERR: w must be a one dimensional np array"
    assert isinstance(k, (np.ndarray)), "ERR: k must be a one dimensional np array"
    assert isinstance(non_central_parameters,
                      (np.ndarray)), "ERR: non_central_parameters must be a one dimensional np array"
    assert k.shape == w.shape == non_central_parameters.shape
    assert k.ndim == w.ndim == non_central_parameters.ndim == 1
    assert np.isreal(m), "ERR: m must be a real number"
    if side != "lower" and side != "upper":
        raise TypeError("side variable must be assigned lower or upper")
    assert isinstance(N, int), "ERR: m must be a positive integer"
    assert N > 0, "ERR: m must be a positive integer"
    if not (np.all(w > 0) or np.all(w < 0)):
        raise ValueError("all weights must be the same sign")

    non_central_parameters_pos = True
    if np.all(w < 0):
        w = -w
        x = -x
        m = -m
        non_central_parameters_pos = False

    beta = 0.90625 * np.amin(w)
    M = np.sum(k)
    n = np.array([i for i in range(1, N)]).reshape(-1, 1)

    # compute the g
    g = np.sum(k * ((1 - beta / w) ** n), axis=1) + (
            (beta * n * ((1 - beta / w) ** (n - 1))) @ ((non_central_parameters / w).reshape((-1, 1)))).ravel()

    # compute the expansion coefficients
    a = np.zeros(N)
    a[0] = np.sqrt(np.exp(-np.sum(non_central_parameters)) * (beta ** M) * np.prod(w ** (-k)))

    if a[0] < np.finfo(a.dtype).tiny:
        # raise ValueError("Underflow error: some series coefficients are smaller than machine precision.")
        a[0] = 0
    for i in range(1, N):
        a[i] = np.dot(np.flip(g[:i]), a[:i]) / (2 * i)

    # compute the central chi-squared integrals
    xg, mg = np.meshgrid((x - m) / beta, np.array([i for i in range(M, M + 2 * N, 2)]))
    xg = xg.ravel()
    mg = mg.ravel()

    F = np.zeros(N)
    for i in range(N):
        F[i] = chi2.cdf(x=xg[i], df=mg[i])

    # compute the integral
    p = np.dot(a, F)

    # flip if necessary
    if (non_central_parameters_pos and side == "upper") or (not non_central_parameters_pos and side == "lower"):
        p = 1 - p

    # compute the truncation error
    errbnd = (1 - np.sum(a)) * chi2.cdf((x - m) / beta, M + 2 * N)

    return p, errbnd


def gx2cdf_davies(x, w, k, non_central_parameters, m, s, side="lower", rtol=1e-05, atol=1e-08, dps=400):
    """ This functions given a generalized chi-square distribution (a weighted sum of non-central chi-squares
        and a normal), using Davies' [1973] method.
        output the cdf of it evaluating on the point x;
        w         the weights of the non-central chi-squares, a one-dimensional np.array
        k         the degrees of freedom of each non-central chi-squares, a one-dimensional np.array
        non_central_parameters the non-centrality paramaters (sum of squares of means) of the non-central chi-squares, , a one-dimensional np.array
        m         mean of normal term
        s         sandard deviation of normal term
        side      "lower" for cdf and "upper" for sf; when cdf is too small it is recommended computing sf instead
        rtol      relative tolerant error
        atol      absolute tolerant error

        Outputs
        p         computed cdf
        flag      =true if output was too close to 0 or 1 to compute exactly with
                  default settings. Try stricter tolerances
        errbnd    upper error bound of the computed cdf
    """

    # Comment it out the old code for integral
    # def davies_integrand(u, x, w, k, non_central_parameters, s):
    #     """ this integrand function evaluating over u
    #         x,s must be positive real number
    #         w, k, non_central_parameters must be column vector
    #     """
    #     theta = np.sum(
    #         k * np.arctan(u * w) + (non_central_parameters * (u * w)) / (1 + (u ** 2) * (w ** 2))) / 2 - u * x / 2
    #     rho = np.prod(((1 + (u ** 2) * (w ** 2)) ** (k / 4)) * np.exp(
    #         ((u ** 2) * (w ** 2) * non_central_parameters) / (2 * ((1 + (u ** 2) * (w ** 2)))))) * np.exp(
    #         (u ** 2) * (s ** 2) / 8)
    #     return np.sin(theta) / (u * rho)
    #
    # davies_integral, errbnd = integrate.quad(davies_integrand, 0, np.inf,
    #                                          args=(point_copy, w_copy, k_copy, non_central_parameters_copy, s),
    #                                          limit=300)

    def davies_integrand_mp(u, x, w, k, non_central_parameters, s):
        """ this integrand function evaluating over u
            x,s must be positive real number
            w, k, non_central_parameters must be column vector
        """
        mp_exp_array = np.frompyfunc(mp.exp, 1, 1)
        mp_atan_array = np.frompyfunc(mp.atan, 1, 1)
        theta = np.sum(
            k * mp_atan_array(u * w) + (non_central_parameters * (u * w)) / (1 + (u ** 2) * (w ** 2))) / 2 - u * x / 2

        part1 = ((u ** 2) * (w ** 2) * non_central_parameters) / (2 * ((1 + (u ** 2) * (w ** 2))))

        rho = np.prod(((1 + (u ** 2) * (w ** 2)) ** (k / 4)) * mp_exp_array(part1)) * mp_exp_array(
            (u ** 2) * (s ** 2) / 8)

        return mp.sin(theta) / (u * rho)

    mp.mp.dps = dps  # Set the decimal places for higher precision
    assert np.isreal(x), "ERR: x must be a real number"
    assert isinstance(w, (np.ndarray)), "ERR: w must be a one dimensional np array"
    assert isinstance(k, (np.ndarray)), "ERR: k must be a one dimensional np array"
    assert isinstance(non_central_parameters,
                      (np.ndarray)), "ERR: non_central_parameters must be a one dimensional np array"
    assert k.shape == w.shape == non_central_parameters.shape
    assert k.ndim == w.ndim == non_central_parameters.ndim == 1
    assert np.isreal(m), "ERR: m must be a real number"
    assert np.isreal(s), "ERR: s must be a real number"
    assert rtol > 0, "ERR: relative error must be larger than 0"
    assert atol > 0, "ERR: absolute error must be larger than 0"
    if side != "lower" and side != "upper":
        raise TypeError("side variable must be assigned lower or upper")


    # to do the integral we maintain an column vector copy
    w_copy = w.reshape((-1, 1))
    k_copy = k.reshape((-1, 1))
    non_central_parameters_copy = non_central_parameters.reshape((-1, 1))
    point_copy = x - m

    # Compute the integral
    partial_davies_integrand_mp = partial(davies_integrand_mp, x=point_copy, w=w_copy, k=k_copy,
                                          non_central_parameters=non_central_parameters_copy, s=s)

    # mp.inf introduces more error ...
    # davies_integral = float(mp.quad(partial_davies_integrand_mp, [0, mp.inf]))
    davies_integral = float(mp.quad(partial_davies_integrand_mp, [0, 1000]))

    if side == "lower":
        p = 0.5 - davies_integral / np.pi
    elif side == "upper":
        p = 0.5 + davies_integral / np.pi

    # Handle failure case
    flag = (p < 0) or (p > 1)
    p = max(p, 0)
    p = min(p, 1)

    return p, flag


def gx2cdf(x, w, k, non_central_parameters, m, s, side="lower", rtol=1e-05, atol=1e-08):
    """ This functions given a generalized chi-square distribution
        output the cdf of it evaluating on the point x;
        w         the weights of the non-central chi-squares, a one-dimensional np.array
        k         the degrees of freedom of each non-central chi-squares, a one-dimensional np.array
        non_central_parameters the non-centrality paramaters (sum of squares of means) of the non-central chi-squares, , a one-dimensional np.array
        m         mean of normal term
        s         sandard deviation of normal term
        side      "lower" for cdf and "upper" for sf; when cdf is too small it is recommended computing sf instead
        rtol      relative tolerant error
        atol      absolute tolerant error
    """
    assert np.isreal(x), "ERR: x must be a real number"
    assert isinstance(w, (np.ndarray)), "ERR: w must be a one dimensional np array"
    assert isinstance(k, (np.ndarray)), "ERR: k must be a one dimensional np array"
    assert isinstance(non_central_parameters,
                      (np.ndarray)), "ERR: non_central_parameters must be a one dimensional np array"
    assert k.shape == w.shape == non_central_parameters.shape
    assert k.ndim == w.ndim == non_central_parameters.ndim == 1
    assert np.isreal(m), "ERR: m must be a real number"
    assert np.isreal(s), "ERR: s must be a real number"
    assert rtol > 0, "ERR: relative error must be larger than 0"
    assert atol > 0, "ERR: absolute error must be larger than 0"
    if side != "lower" and side != "upper":
        raise TypeError("side variable must be assigned lower or upper")

    if np.isclose(s, 0, rtol, atol, equal_nan=False) and len(np.unique(w)) == 1:
        # native ncx2 fallback
        if (np.sign(np.unique(w)[0]) == 1 and side == "lower") or (
                (np.sign(np.unique(w)[0]) == -1 and side == "upper")):
            p = ncx2.cdf(x=(x - m) / np.unique(w)[0], df=np.sum(k), nc=np.sum(non_central_parameters))
        else:
            p = ncx2.sf(x=(x - m) / np.unique(w)[0], df=np.sum(k), nc=np.sum(non_central_parameters))

    elif np.isclose(s, 0, rtol, atol, equal_nan=False) and (np.all(w > 0) or np.all(w < 0)):
        try:
            p, errbnd = gx2cdf_ruben(x, w, k, non_central_parameters, m, side)
            logging.debug("use the ruben chi-square cdf method")
        except ValueError:
            p, flag = gx2cdf_davies(x, w, k, non_central_parameters, m, s, side, rtol, atol)
            logging.debug("use the davies chi-square cdf method")

    else:
        p, flag = gx2cdf_davies(x, w, k, non_central_parameters, m, s, side, rtol, atol)

        logging.debug("use the davies chi-square cdf method")
        if flag:
            logging.debug("the output is too close to 0 or 1, within 10e-08")

    return p


def compute_cardinality(a_list, b_list):
    """compute the number of distinct intersection elements"""
    assert isinstance(a_list, (np.ndarray)), "ERR: expect np.ndarray data type for a_list"
    assert a_list.ndim == 1, "ERR: expect one dimensional vector for a_list"
    assert isinstance(b_list, (np.ndarray)), "ERR: expect np.ndarray data type for b_list"
    assert b_list.ndim == 1, "ERR: expect one dimensional vector for b_list"

    return len(np.intersect1d(a_list, b_list))


def store_array_str(data_array):
    converted_str = np.array_str(data_array)
    converted_str = converted_str.replace(' ', ',')
    return converted_str

def estimator_process_wrapper(kwargs):
    estimator_cls = kwargs['estimator_cls']
    config = kwargs['config']
    estimator = estimator_cls(config)
    return estimator.build()


def batch_estimator_total_report(kwargs_lists, workers):
    pool = multiprocessing.Pool(processes=workers)
    results_list = pool.map(estimator_process_wrapper, kwargs_lists)
    return results_list


def prune_pos_samples(samples, threshold=1000, dim=1):
    if dim == 1:
        index = np.argwhere(np.abs(samples) < threshold)
        index = index.transpose()[0]
        return samples[index]
    else:
        index = np.argwhere(np.linalg.norm(samples, ord=np.inf, axis=1) < threshold)
        index = index.transpose()[0]
        return samples[index]


def batch_estimator_estimated_delta(kwargs_lists, workers):
    pool = multiprocessing.Pool(processes=workers)
    results_list = pool.map(estimator_process_wrapper, kwargs_lists)

    estimated_delta = np.zeros(len(kwargs_lists))
    for i in range(len(kwargs_lists)):
        estimated_delta[i] = results_list[i]['estimated_delta']

    return estimated_delta


def compute_bootstrap_range(estimator_cls, config, n_samples, workers=WORKERS, confidence_interval_prob=0.9,
                            bootstrap_samples=10):
    kwargs = {'estimator_cls': estimator_cls, 'config': config}
    input_list = []
    for i in range(n_samples):
        input_list.append(kwargs)

    pool = multiprocessing.Pool(processes=workers)

    results_list = pool.map(estimator_process_wrapper, input_list)

    estimated_delta = np.zeros(n_samples)
    for i in range(n_samples):
        estimated_delta[i] = results_list[i]['estimated_delta']

    bootstrap = EmpiricalBootstrap(sample_generator=SampleGenerator(data=estimated_delta))

    boot_res = bootstrap.bootstrap_confidence_bounds(
        confidence_interval_prob=confidence_interval_prob,
        n_samples=bootstrap_samples
    )
    logging.critical(boot_res)
    return boot_res