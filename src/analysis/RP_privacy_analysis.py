import numpy as np

from scipy import special
import multiprocessing
from functools import partial
from scipy.stats import norm

from analysis.commons import clac_standard_vec, calc_proj_matrix, concatenate_B_b, \
    get_neighbors_index_list, twoNorm, check_equal


def compute_analytical_delta_gaussian_general(mu1, sigma1, mu2, sigma2, epsilon):
    delta1 = _compute_analytical_delta_gaussian_general(mu1, sigma1, mu2, sigma2, epsilon)
    delta2 = _compute_analytical_delta_gaussian_general(mu2, sigma2, mu1, sigma1, epsilon)
    return max(delta1, delta2)


def _compute_analytical_delta_gaussian_general(mu1, sigma1, mu2, sigma2, epsilon):
    assert not check_equal(sigma1, sigma2), "ERR: the scale of the gaussian should not be the same"
    a = 1 / (sigma2 ** 2) - 1 / (sigma1 ** 2)
    b = 2 * mu1 / (sigma1 ** 2) - 2 * mu2 / (sigma2 ** 2)
    c = (mu2 ** 2) / (sigma2 ** 2) - (mu1 ** 2) / (sigma1 ** 2)

    v = 2 / a * (epsilon - np.log(sigma2) + np.log(sigma1)) - c / a + (b ** 2) / (4 * (a ** 2))

    if v < 0:
        return 0

    y1_mu = mu1 + b / (2 * a)
    y2_mu = mu2 + b / (2 * a)

    if (a > 0):
        p1 = 1 - (norm.cdf(np.sqrt(v), loc=y1_mu, scale=sigma1) - norm.cdf(-np.sqrt(v), loc=y1_mu, scale=sigma1))
        p2 = 1 - (norm.cdf(np.sqrt(v), loc=y2_mu, scale=sigma2) - norm.cdf(-np.sqrt(v), loc=y2_mu, scale=sigma2))
    else:
        p1 = (norm.cdf(np.sqrt(v), loc=y1_mu, scale=sigma1) - norm.cdf(-np.sqrt(v), loc=y1_mu, scale=sigma1))
        p2 = (norm.cdf(np.sqrt(v), loc=y2_mu, scale=sigma2) - norm.cdf(-np.sqrt(v), loc=y2_mu, scale=sigma2))

    return min(max(p1 - np.exp(epsilon) * p2, 0), 1)


def compute_analytical_delta_1dRLC(x, y, epsilon):
    """
        take two vectors x,y, output the (epsilon, delta) distance between distributions <x,g> and <y,g>,
        where epsilon is given and g is a standard multivariate normal distribution, operator <> is dot product
    """
    delta1 = _compute_analytical_delta_1dRLC(x, y, epsilon)
    delta2 = _compute_analytical_delta_1dRLC(y, x, epsilon)
    return max(delta1, delta2)


def _compute_analytical_delta_1dRLC(x, y, epsilon):
    """
        take two vectors x,y (ordered), output the (epsilon, delta) distance between distributions <x,g> and <y,g>,
        where epsilon is given and g is a standard multivariate normal distribution, operator <> is dot product
    """
    c = twoNorm(x) / twoNorm(y)
    assert len(x) == len(y), "ERR: two vectors have different dimensions"
    assert epsilon >= 0, "ERR: expect non-negative epsilon"

    # It is possible that when epsilon too big, the inner value of sqrt smaller than 0
    b1 = np.sqrt(max((epsilon + np.log(c)) / (c ** 2 - 1), 0))
    b2 = b1 * c

    delta = 1 - np.exp(epsilon) + np.exp(epsilon) * special.erf(b2) - special.erf(b1)

    return min(max(delta, 0), 1)


def compute_analytical_delta_kdRLC(index, x, epsilon):
    """
        take a n by d matrices x, an row index and a privacy parameter epsilon
        create a neighbor y of x, y is the one-row (the indexed row, say, v) opt-out version of x
        that is, x, y has the relationship x = y + e_index*v.T;
        the program outputs the (epsilon, delta) distance between distributions x^Tg and y^Tg,
        where epsilon is given and g is a standard multivariate normal distribution
    """

    assert isinstance(x, (np.ndarray)), "ERR: expect np.ndarray data type for x"
    assert x.ndim == 2, "ERR: expect two dimensional matrix"
    assert epsilon >= 0, "ERR: expect positive epsilon"
    y = x.copy()
    v = y[index].copy().reshape(-1, 1)
    y[index][:] = 0 * y[index][:]
    assert x.shape[1] == np.linalg.matrix_rank(x), "ERR: expect x has full column rank"
    assert y.shape[1] == np.linalg.matrix_rank(y), "ERR: expect y has full column rank"
    e = clac_standard_vec(x.shape[0], index)
    assert np.sum(x - y - e @ v.T) == 0, "ERR: matrix doesn't in the shape of x = y + e_index*v.T"

    p = calc_proj_matrix(x)[index][index]

    a = 0.5*(np.log(1-p))
    # compute delta_{A, A'} and delta_{A', A}
    delta2 = special.erf(np.sqrt((-epsilon-a)/p)) - np.exp(epsilon)*special.erf(np.sqrt((epsilon+a)*(p-1)/p))
    delta1 = 1 - np.exp(epsilon) + np.exp(epsilon)*special.erf(np.sqrt((epsilon-a)/p)) - special.erf(np.sqrt((epsilon-a)*(1-p)/p))

    return min(max(max(delta1, 0), delta2), 1)


def compute_analytical_delta_RP(index, x, r, epsilon):
    """
        take a n by d matrices x, an row index and a privacy parameter epsilon
        create a neighbor y of x, y is the one-row (the indexed row, say, v) opt-out version of x
        that is, x, y has the relationship x = y + e_index*v.T;
        the program outputs the (epsilon, delta) distance between distributions x^TG and y^TG,
        where epsilon is given and G is an n by r standard Gaussian matrix
    """

    assert isinstance(x, (np.ndarray)), "ERR: expect np.ndarray data type for x"
    assert x.ndim == 2, "ERR: expect two dimensional matrix"
    assert epsilon >= 0, "ERR: expect positive epsilon"
    y = x.copy()
    v = y[index].copy().reshape(-1, 1)
    y[index][:] = 0 * y[index][:]
    assert x.shape[1] == np.linalg.matrix_rank(x), "ERR: expect x has full column rank"
    assert y.shape[1] == np.linalg.matrix_rank(y), "ERR: expect y has full column rank"
    e = clac_standard_vec(x.shape[0], index)
    assert np.sum(x - y - e @ v.T) == 0, "ERR: matrix doesn't in the shape of x = y + e_index*v.T"

    p = calc_proj_matrix(x)[index][index]

    a = 0.5*(np.log(1-p))*r
    # compute delta_{A, A'} and delta_{A', A}
    delta2 = special.erf(np.sqrt((-epsilon-a)/(r*p))) - np.exp(epsilon)*special.erf(np.sqrt((epsilon+a)*(p-1)/(r*p)))
    delta1 = 1 - np.exp(epsilon) + np.exp(epsilon)*special.erf(np.sqrt((epsilon-a)/(r*p))) - special.erf(np.sqrt((epsilon-a)*(1-p)/(r*p)))

    return min(max(max(delta1, 0), delta2), 1)


def compute_analytical_delta_kdRLC_for_all_neighbors(B, b, epsilon, workers=12):
    """Given epsilon, compute the corresponding delta for all neighboring pair [B b].Tg and [B b]'.Tg"""
    pool = multiprocessing.Pool(processes=workers)
    A = concatenate_B_b(B, b)
    partial_compute_analytical_delta_kdRLC = partial(compute_analytical_delta_kdRLC, x=A, epsilon=epsilon)

    neighbor_list = get_neighbors_index_list(B, b)
    output_list = pool.map(partial_compute_analytical_delta_kdRLC, neighbor_list)

    return np.array(output_list)


def compute_analytical_delta_kdRLC_epsilon_first(epsilon, index, x):
    return compute_analytical_delta_kdRLC(index, x, epsilon)


def compute_analytical_delta_kdRLC_for_epsilon_lists(B, b, epsilon_list, index, workers=12):
    """Given epsilon list, compute the corresponding delta for neighboring pair #index [B b].Tg and [B b]'.Tg"""
    assert isinstance(epsilon_list, (np.ndarray)), "ERR: expect np.ndarray data type for epsilon_list"
    assert epsilon_list.ndim == 1, "ERR: expect one dimensional row vector"

    pool = multiprocessing.Pool(processes=workers)
    A = concatenate_B_b(B, b)
    partial_compute_analytical_delta_kdRLC_epsilon_first = \
        partial(compute_analytical_delta_kdRLC_epsilon_first, index=index, x=A)

    output_list = pool.map(partial_compute_analytical_delta_kdRLC_epsilon_first, epsilon_list)

    return np.array(output_list)


def compute_sequential_composition_delta_RP(index, x, r, epsilon):
    """
        take two n by d matrices x,y and a d-dimensional column vector v such that x = y + e_index*v.T
        output the (epsilon, delta) distance between distributions x^TG and y^TG,
        where epsilon is given and G is a standard n by r gaussian matrix, that is, each of its entry is an independent
        standard normal random variable
    """
    piece_delta = compute_analytical_delta_kdRLC(index, x, epsilon/r)
    return min(r*piece_delta, 1)


def compute_sequential_composition_delta_RP_for_all_neighbors(B, b, r, epsilon, workers=12):
    """
        Given epsilon, compute the corresponding delta for all neighboring pair [B b].TG and [B b]'.TG
        where r is the number of G's column
    """
    pool = multiprocessing.Pool(processes=workers)
    A = concatenate_B_b(B, b)
    partial_compute_sequential_composition_delta_RP = \
        partial(compute_sequential_composition_delta_RP, x=A, r=r, epsilon=epsilon)

    neighbor_list = get_neighbors_index_list(B, b)
    output_list = pool.map(partial_compute_sequential_composition_delta_RP, neighbor_list)

    return np.array(output_list)


def compute_sequential_composition_delta_RP_epsilon_first(epsilon, index, x, r):
    return compute_sequential_composition_delta_RP(index, x, r, epsilon)


def compute_sequential_composition_delta_RP_for_epsilon_lists(B, b, r, epsilon_list, index, workers=12):
    """
        Given epsilon lists, compute the corresponding delta for all neighboring #index [B b].TG and [B b]'.TG
        where r is the number of G's column
    """
    assert isinstance(epsilon_list, (np.ndarray)), "ERR: expect np.ndarray data type for epsilon_list"
    assert epsilon_list.ndim == 1, "ERR: expect one dimensional row vector"

    pool = multiprocessing.Pool(processes=workers)
    A = concatenate_B_b(B, b)
    partial_compute_sequential_composition_delta_RP_epsilon_first = \
        partial(compute_sequential_composition_delta_RP_epsilon_first, index=index, x=A, r=r)

    output_list = pool.map(partial_compute_sequential_composition_delta_RP_epsilon_first, epsilon_list)

    return np.array(output_list)
