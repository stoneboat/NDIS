import numpy as np
from numpy.linalg import inv as mat_inv

import multiprocessing
from functools import partial

from analysis.commons import get_w, twoNorm, calc_proj_matrix, get_neighbor_w, gx2_params_norm_quad, \
    gx2cdf, get_neighbors_index_list, get_neighbor_B, get_neighbor_b, compute_projection_diagonal_elements


def get_matrix_Q(B, b, r):
    w = get_w(B, b)
    M = mat_inv(B.T @ B)

    return (twoNorm(w)**2)/r*M


def get_neighbor_matrix_Q(index, B, b, r):
    neighbor_B = get_neighbor_B(index, B)
    neighbor_b = get_neighbor_b(index, b)

    return get_matrix_Q(neighbor_B, neighbor_b, r)


def compute_c1_c2(index, B, b):
    """return c1 = 1/||w'||^2;  c2 = (||w'||^2-||w||^2)/||w'||^2||w||^2 """
    error_square = twoNorm(get_w(B, b))**2
    neighbor_error_square = twoNorm(get_neighbor_w(index, B, b))**2
    c1 = 1/neighbor_error_square
    c2 = (neighbor_error_square-error_square)/(error_square*neighbor_error_square)
    return c1, c2


def compute_c_ratio(index, B, b):
    c1, c2 = compute_c1_c2(index, B, b)
    return c1/c2


def compute_quadratic_matrx(index, B, b):
    """return c1 vv^T + c2 B^TB, B = B_neighbor + e_index*v.T"""
    v = B[index].copy().reshape(-1 , 1)
    c1, c2 = compute_c1_c2(index, B, b)
    return c1*v@v.T + c2*B.T@B


def get_constant_list_a(r, B, b):
    w = get_w(B, b)
    error_square = twoNorm(w) ** 2
    Pi = calc_proj_matrix(B).diagonal()
    wi_square = w * w
    d = B.shape[1]

    a = 0.5 * (d*np.log(1 - wi_square/((1-Pi)*error_square)) - np.log(1 - Pi) +
               (r * wi_square) / (error_square * (1 - Pi) - wi_square) -
               (r * (wi_square ** 2) * (1 - Pi)) / (
                wi_square * error_square * (1 - Pi ** 2) - wi_square ** 2 - (error_square ** 2) * Pi * ((1 - Pi)**2)))
    return a


def get_constant_a(r, index, B, b):
    """ For each index, we know a is a linear function of r;
        a(r) = c1 + c2r, where c1 = 0.5*(np.log(1 - P_diagonal - w_square/error_norm_square))
        and c2 = 0.5*(w_square/(error_norm_square*(1-P_diagonal)) + (w_square*w_square)/(error_norm_square*w_square+error_norm_square*error_norm_square*P_diagonal*(1-P_diagonal))) here
    """
    w = get_w(B, b)
    error_square = twoNorm(get_w(B, b))**2

    M = np.linalg.inv(B.T @ B)
    v = B[index].reshape((-1, 1))
    Pi = v.T@M@v

    wi = w[index]
    wi_square = wi*wi
    d = B.shape[1]

    a = 0.5*(d*np.log(1 - (wi**2)/((1-Pi)*(twoNorm(w)**2))) - np.log(1 - Pi) +
             (r*wi_square)/(error_square*(1-Pi) - wi_square) -
             (r*(wi_square**2)*(1-Pi))/(wi_square*error_square*(1-Pi**2) - wi_square**2 - (error_square**2)*Pi*((
                    1-Pi)**2)))

    return a.item()


def get_constant_a_ratio(index, B, b):
    """ For each index, we know a is a linear function of r;
        a(r) = c1 + c2r, where c1 = 0.5*(np.log(1 - P_diagonal - w_square/error_norm_square))
        and c2 = 0.5*(w_square/(error_norm_square*(1-P_diagonal)) + (w_square*w_square)/(error_norm_square*w_square+error_norm_square*error_norm_square*P_diagonal*(1-P_diagonal))) here
        this function return c2
    """
    raise ValueError("the function is deprecated due to the constant a's expression has bug")
    w = get_w(B, b)
    error_norm_square = twoNorm(get_w(B, b))**2
    P_diagonal = calc_proj_matrix(B).diagonal()[index]
    w_square = w[index]*w[index]
    return 0.5*(w_square/(error_norm_square*(1-P_diagonal)) + (w_square*w_square)/(error_norm_square*w_square+error_norm_square*error_norm_square*P_diagonal*(1-P_diagonal)))


def get_constant_a_constant(index, B, b):
    """ For each index, we know a is a linear function of r;
        a(r) = c1 + c2r, where c1 = 0.5*(np.log(1 - P_diagonal - w_square/error_norm_square))
        and c2 = 0.5*(w_square/(error_norm_square*(1-P_diagonal)) + (w_square*w_square)/(error_norm_square*w_square+error_norm_square*error_norm_square*P_diagonal*(1-P_diagonal))) here
        this function return c1
    """
    raise ValueError("the function is deprecated due to the constant a's expression has bug")
    w = get_w(B, b)
    error_norm_square = twoNorm(get_w(B, b))**2
    P_diagonal = calc_proj_matrix(B).diagonal()[index]
    w_square = w[index]*w[index]
    return 0.5*(np.log(1 - P_diagonal - w_square/error_norm_square))


def compute_multivariate_Y_pair_character(r, index, B, b):
    w = get_w(B, b)
    wi = w[index]

    P_diagonal = compute_projection_diagonal_elements(B)
    c_ratio = compute_c_ratio(index, B, b)
    M = mat_inv(B.T@B)

    neighbor_B = get_neighbor_B(index, B)
    neighbor_M = mat_inv(neighbor_B.T@neighbor_B)
    v = B[index].copy().reshape(-1 , 1)

    Y_mean = (-np.sqrt(r)*wi*c_ratio/(1+c_ratio*P_diagonal[index]))*(M@v)
    Y_cov = (twoNorm(w)**2)*M

    neighbor_Y_mean = (-np.sqrt(r)*wi*(1 + c_ratio)/((1-P_diagonal[index])*(1+c_ratio*P_diagonal[index])))*(M@v)
    neighbor_w = get_neighbor_w(index, B, b)
    neighbor_Y_cov = (twoNorm(neighbor_w)**2)*neighbor_M
    return Y_mean, Y_cov, neighbor_Y_mean, neighbor_Y_cov


def compute_analytical_asymptotic_dist_delta(index, B, b, epsilon, r):
    point = 2 * (get_constant_a(r, index, B, b) - epsilon)
    quadratic_mat = compute_quadratic_matrx(index, B, b)
    Y_mean, Y_cov, neighbor_Y_mean, neighbor_Y_cov = compute_multivariate_Y_pair_character(r, index, B, b)

    w, k, non_central_parameters, m, s = gx2_params_norm_quad(Y_mean, Y_cov, quadratic_mat)
    p1 = gx2cdf(point, w, k, non_central_parameters, m, s, "lower")

    w, k, non_central_parameters, m, s = gx2_params_norm_quad(neighbor_Y_mean, neighbor_Y_cov, quadratic_mat)
    p2 = gx2cdf(point, w, k, non_central_parameters, m, s, "lower")

    #     print(f"point: {point} p1: {p1} p2:{p2}")

    return min(max(max(p1 - np.exp(epsilon) * p2, 0), p2 - np.exp(epsilon) * p1), 1)


def compute_analytical_asymptotic_dist_delta_epsilon_first(epsilon, index, B, b, r):
    return compute_analytical_asymptotic_dist_delta(index, B, b, epsilon, r)


def compute_analytical_asymptotic_dist_delta_r_first(r, index, B, b, epsilon):
    return compute_analytical_asymptotic_dist_delta(index, B, b, epsilon, r)


def compute_analytical_asymptotic_dist_delta_index_first(index, B, b, epsilon, r):
    return compute_analytical_asymptotic_dist_delta(index, B, b, epsilon, r)


def compute_analytical_asymptotic_delta_for_all_neighbors(B, b, r, epsilon, workers=12):
    pool = multiprocessing.Pool(processes=workers)
    partial_compute_analytical_asymptotic_dist_delta_index_first = \
        partial(compute_analytical_asymptotic_dist_delta_index_first, B=B, b=b, r=r, epsilon=epsilon)

    neighbor_list = get_neighbors_index_list(B, b)
    output_list = pool.map(partial_compute_analytical_asymptotic_dist_delta_index_first, neighbor_list)

    return np.array(output_list)


def compute_analytical_asymptotic_delta_for_epsilon_lists(B, b, r, epsilon_list, index, workers=12):
    assert isinstance(epsilon_list, (np.ndarray)), "ERR: expect np.ndarray data type for epsilon_list"
    assert epsilon_list.ndim == 1, "ERR: expect one dimensional row vector"

    pool = multiprocessing.Pool(processes=workers)
    partial_compute_analytical_asymptotic_dist_delta_epsilon_first = \
        partial(compute_analytical_asymptotic_dist_delta_epsilon_first, index=index, B=B, b=b, r=r)

    output_list = pool.map(partial_compute_analytical_asymptotic_dist_delta_epsilon_first, epsilon_list)

    return np.array(output_list)


def compute_analytical_asymptotic_delta_for_r_lists(B, b, r_list, epsilon, index, workers=12):
    assert isinstance(r_list, (np.ndarray)), "ERR: expect np.ndarray data type for epsilon_list"
    assert r_list.ndim == 1, "ERR: expect one dimensional row vector"

    pool = multiprocessing.Pool(processes=workers)
    partial_compute_analytical_asymptotic_dist_delta_r_first = \
        partial(compute_analytical_asymptotic_dist_delta_r_first, index=index, B=B, b=b, epsilon=epsilon)

    output_list = pool.map(partial_compute_analytical_asymptotic_dist_delta_r_first, r_list)

    return np.array(output_list)