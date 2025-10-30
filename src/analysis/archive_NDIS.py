import math
import multiprocessing
from functools import partial

from numpy.random import MT19937, RandomState
import secrets

import numpy as np
from scipy.stats import norm
from scipy.linalg import sqrtm, eigh
from numpy.linalg import inv as mat_inv

from analysis.commons import check_equal, twoNorm, compute_xopt, get_w, calc_proj_matrix, get_neighbor_w, \
    get_neighbor_B, get_neighbor_b
from analysis.ALS_privacy_analysis import get_constant_a, compute_quadratic_matrx, \
    compute_multivariate_Y_pair_character, compute_c1_c2, compute_c_ratio, get_matrix_Q, get_neighbor_matrix_Q


def check_kd_asymptotic_dist_delta(index, B, b, epsilon, r):
    neighbor_B = get_neighbor_B(index, B)
    neighbor_b = get_neighbor_b(index, b)

    #  Compute the two multi-dimensional Gaussians' parameters
    mu1 = compute_xopt(B, b).reshape((-1, 1))
    Sigma1 = get_matrix_Q(B, b, r)
    mu2 = compute_xopt(neighbor_B, neighbor_b).reshape((-1, 1))
    Sigma2 = get_matrix_Q(neighbor_B, neighbor_b, r)

    # Compute delta
    delta1 = _check_kd_asymptotic_dist_delta(mu1=mu1, Sigma1=Sigma1, mu2=mu2, Sigma2=Sigma2, epsilon=epsilon)
    delta2 = _check_kd_asymptotic_dist_delta(mu1=mu2, Sigma1=Sigma2, mu2=mu1, Sigma2=Sigma1, epsilon=epsilon)

    return max(delta1, delta2)


def check_expression_correct(mu1, Sigma1, mu2, Sigma2, epsilon):
    d = Sigma1.shape[0]

    #  Compute the two multi-dimensional Gaussians' parameters
    sqrt_Sigma1 = sqrtm(Sigma1)
    inv_Sigma2 = mat_inv(Sigma2)
    D, U = eigh(np.identity(d) - sqrt_Sigma1 @ inv_Sigma2 @ sqrt_Sigma1)
    A = np.diag(D)
    b = -U.T @ sqrt_Sigma1 @ inv_Sigma2 @ (mu1 - mu2)
    c = (epsilon + np.log(np.sqrt(np.linalg.det(Sigma1)/np.linalg.det(Sigma2))) - 0.5*(mu1-mu2).T@inv_Sigma2@(
            mu1-mu2)).item()

    # Check the integral expression's correctness
    v = np.random.uniform(size=(d, 1))
    z = U @ v
    x = sqrtm(Sigma1) @ z + mu1

    expression1 = np.exp(-0.5 * v.T @ v) / ((2 * np.pi) ** (d / 2)) * (1 - np.exp(0.5 * v.T @ A @ v + b.T @ v + c))

    expression2 = np.exp(-0.5*z.T@z)/((2 * np.pi) ** (d / 2))\
                  *(1 - np.exp(0.5*z.T@(np.identity(d) - sqrtm(Sigma1)@mat_inv(Sigma2)@sqrtm(Sigma1))@z -
                    (mu1-mu2).T@mat_inv(Sigma2)@sqrtm(Sigma1)@z + epsilon +
                               np.log(np.sqrt(np.linalg.det(Sigma1)/np.linalg.det(Sigma2))) -
                               0.5*(mu1-mu2).T@mat_inv(Sigma2)@(mu1-mu2)))

    assert check_equal(expression1, expression2)

    expression3 = np.exp(-0.5*z.T@z)/((2 * np.pi) ** (d / 2)) - np.exp(epsilon)*np.sqrt(np.linalg.det(
        Sigma1)/np.linalg.det(Sigma2))/((2 * np.pi) ** (d / 2))*np.exp(-0.5*(sqrtm(Sigma1)@z + mu1 - mu2).T@mat_inv(
        Sigma2)@(sqrtm(Sigma1)@z + mu1 - mu2))

    assert check_equal(expression1, expression3)

    expression4 = (np.exp(-0.5*(x-mu1).T@mat_inv(Sigma1)@(x-mu1))/((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(
        Sigma1))) - np.exp(epsilon - 0.5*(x-mu2).T@mat_inv(Sigma2)@(x-mu2))/((2 * np.pi) ** (d / 2)*np.sqrt(np.linalg.det(
        Sigma2))))*np.sqrt(np.linalg.det(Sigma1))

    assert check_equal(expression1, expression4)

    expression5 = (np.exp(-0.5 * v.T @ v) - np.exp(0.5 * v.T @ (A-np.identity(d)) @ v + b.T @ v + c)) / ((2 * np.pi) ** (d / 2))

    return check_equal(expression1, expression5)


def kd_func(v, A, b, c):
    return max(0, 1 - np.exp(0.5 * v.T @ A @ v + b.T @ v + c))


def kd_func_expectation_estimation_workers(n_samples, dim, A, b, c):
    seed = secrets.randbits(128)
    rng = RandomState(MT19937(seed))

    mean = np.zeros(dim)
    cov = np.identity(dim)

    sum = 0
    v_list = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)
    for i in range(n_samples):
        sum += kd_func(v_list[i], A, b, c)

    return sum/n_samples


def _check_kd_asymptotic_dist_delta(mu1, Sigma1, mu2, Sigma2, epsilon, num_samples=10**6, workers=12):
    # Check covariance matrices are in good shape
    assert isinstance(Sigma1, np.ndarray), "ERR: variable Sigma1 is required np.ndarray type"
    assert isinstance(Sigma2, np.ndarray), "ERR: variable Sigma1 is required np.ndarray type"
    assert Sigma1.ndim == 2, f"ERR: data base input is in wrong shape, matrix Sigma1 required 2 dimensions"
    assert Sigma2.ndim == 2, f"ERR: data base input is in wrong shape, matrix Sigma1 required 2 dimensions"

    assert Sigma1.shape[0] == Sigma1.shape[1] == Sigma2.shape[0] == Sigma2.shape[1]
    d = Sigma1.shape[0]

    #  Compute the two multi-dimensional Gaussians' parameters
    sqrt_Sigma1 = sqrtm(Sigma1)
    inv_Sigma2 = mat_inv(Sigma2)
    D, U = eigh(np.identity(d) - sqrt_Sigma1 @ inv_Sigma2 @ sqrt_Sigma1)
    A = np.diag(D)
    b = -U.T @ sqrt_Sigma1 @ inv_Sigma2 @ (mu1 - mu2)

    scale = 1
    # try to do something when Sigma1 and Sigma2's determinant is too small, render the ratio no computable
    while math.isnan(np.linalg.det(Sigma1*scale)/np.linalg.det(Sigma2*scale)) and scale < 2**20:
        scale = scale*2

    c = (epsilon + np.log(np.sqrt(np.linalg.det(Sigma1*scale)/np.linalg.det(Sigma2*scale))) - 0.5*(mu1-mu2).T@inv_Sigma2@(
            mu1-mu2)).item()

    pool = multiprocessing.Pool(processes=workers)
    partial_kd_func_expectation_estimation_workers = partial(kd_func_expectation_estimation_workers, dim=d, A=A, b=b, c=c)

    chunk_size = np.ceil(num_samples / workers)
    input_list = []
    remains = num_samples
    for i in range(workers):
        input_list.append(int(min(chunk_size, remains)))
        remains -= chunk_size

    output_list = pool.map(partial_kd_func_expectation_estimation_workers, input_list)
    ret = (np.dot(np.array(output_list, dtype=object).ravel(), np.array(input_list)) / num_samples)
    if isinstance(ret, np.ndarray):
        return ret.item()
    else:
        return ret


def check_1d_asymptotic_dist_delta(index, B, b, epsilon, r):
    """This function aims for testing the cdf numerical function
       Assume the statement before proposition 8 is correct, then this step check theorem 6 should be correct
    """
    assert isinstance(B, np.ndarray), "ERR: variable B is required np.ndarray type"
    assert B.ndim == 2, f"ERR: data base input is in wrong shape, matrix B required 2 dimensions"
    assert B.shape[1] == 1, f"ERR: data base input is in wrong shape, matrix B required has only one row"

    constant = get_constant_a(r, index, B, b)
    quadratic_mat = compute_quadratic_matrx(index, B, b)
    Y_mean, Y_cov, neighbor_Y_mean, neighbor_Y_cov = compute_multivariate_Y_pair_character(r, index, B, b)

    point = np.sqrt(2 * (constant - epsilon) / quadratic_mat.ravel().item())
    mu = Y_mean.ravel().item()
    sigma = np.sqrt(Y_cov.ravel().item())
    neighbor_mu = neighbor_Y_mean.ravel().item()
    neighbor_sigma = np.sqrt(neighbor_Y_cov.ravel().item())

    p1 = norm.cdf(point, loc=mu, scale=sigma) - norm.cdf(-point, loc=mu, scale=sigma)
    p2 = norm.cdf(point, loc=neighbor_mu, scale=neighbor_sigma) - norm.cdf(-point, loc=neighbor_mu, scale=neighbor_sigma)

    # return min(max(max(p1 - np.exp(epsilon) * p2, 0), p2 - np.exp(epsilon) * p1), 1)
    return min(max(p1 - np.exp(epsilon) * p2, 0), 1)


def check_1d_asymdptotic_dist_delta_epsilon_first(epsilon, index, B, b, r):
    return check_1d_asymptotic_dist_delta(index, B, b, epsilon, r)


def check_1d_asymptotic_dist_delta_for_epsilon_lists(B, b, r, epsilon_list, index, workers=12):
    assert isinstance(epsilon_list, (np.ndarray)), "ERR: expect np.ndarray data type for epsilon_list"
    assert epsilon_list.ndim == 1, "ERR: expect one dimensional row vector"

    pool = multiprocessing.Pool(processes=workers)
    partial_check_1d_asymdptotic_dist_delta_epsilon_first = \
        partial(check_1d_asymdptotic_dist_delta_epsilon_first, index=index, B=B, b=b, r=r)

    output_list = pool.map(partial_check_1d_asymdptotic_dist_delta_epsilon_first, epsilon_list)

    return np.array(output_list)


def check_quadratic_form_transformation_1(d=1, repeat_times=100):
    """We want to check the following equation
       Let x be an arbitrary d by 1 column vector, A be a d by d invertible matrix, b be an arbitrary d by 1 column
       vector
       Compute h = -0.5* inv(A)@b  and k = -0.25* b.T@inv(A)@b
       Then the Equality x.T@A@x + b.T@x = (x-h).T@A@(x-h) + k holds

       According to the test, we will sample x and b according to uniform distribution
       we sample B from uniform distribution and compute A = B.T@B
    """
    for _ in range(repeat_times):
        x = np.random.uniform(size=(d, 1))
        b = np.random.uniform(size=(d, 1))
        B = np.random.uniform(size=(d * 200, d))
        A = B.T @ B
        assert np.linalg.matrix_rank(A) == d, "ERR: the input matrix does not have full column rank"

        h = -0.5 * mat_inv(A) @ b
        k = -0.25 * b.T @ mat_inv(A) @ b
        assert check_equal(x.T @ A @ x + b.T @ x, (x - h).T @ A @ (x - h) + k), "ERR: fail to pass the test"


def check_quadratic_form_transformation_2(B_matrix, b_vector, index):
    """Check part of proposition 7: If claim 1 is correct then claim 2 should be correct"""
    assert isinstance(b_vector, np.ndarray), "ERR: variable b is required np.ndarray type"
    assert b_vector.ndim == 2, f"ERR: data base input is in wrong shape, matrix b required 2 dimensions"
    assert b_vector.shape[1] == 1, f"ERR: data base input is in wrong shape, matrix b required has only one row"

    v = B_matrix[index].copy().reshape(-1, 1)
    c1, c2 = compute_c1_c2(index, B_matrix, b_vector.ravel())
    c_ratio = compute_c_ratio(index, B_matrix, b_vector.ravel())
    M = mat_inv(B_matrix.T @ B_matrix)
    w = get_w(B_matrix, b_vector)
    wi = w[index]
    wi_square = w[index] ** 2
    error_square = twoNorm(w) ** 2
    P_diagonal = calc_proj_matrix(B_matrix).diagonal()
    Pi = P_diagonal[index]

    # Let x be a d by 1 column vector and B_matrix be a n by d matrices, b_vector be a n by 1 column vector
    # the following is a generalized quadratic form x.T@A@x + b.T@x + c_before, that is the claim 1
    A = compute_quadratic_matrx(index, B_matrix, b_vector.ravel())
    b = -(2 * b_vector[index].item() * c1 * v.T + 2 * c2 * b_vector.T @ B_matrix).T
    c_before = (b_vector[index].item() ** 2) * c1 + (twoNorm(b_vector.ravel()) ** 2) * c2

    # We first compute the test version after transformation
    h_test = -0.5 * mat_inv(A) @ b
    k_test = -0.25 * b.T @ mat_inv(A) @ b + c_before
    print(h_test)
    print(k_test)

    # We then compute the computation version after transformation, that is the claim 2
    # The expression of claim 2 is a standard quadratic form (x-h).T@A@(x-h) + k
    h_compute = compute_xopt(B_matrix, b_vector).reshape((-1, 1)) + (c_ratio * wi) / (1 + c_ratio * Pi) * M @ v
    k_compute = -(wi ** 2) / ((twoNorm(w) ** 2) * (1 - Pi) - wi ** 2) + (1 - Pi) * (wi ** 4) / (
                (twoNorm(w) ** 2) * (wi ** 2) * (1 - Pi ** 2) - (wi ** 4) - (twoNorm(w) ** 4) * Pi * ((1 - Pi)**2))

    print(h_compute)
    print(k_compute)


def check_quadratic_form_transformation_3(B_matrix, b_vector, index, r):
    """Check part of proposition 7: If the condition is correct then claim 1 should be correct """
    assert isinstance(b_vector, np.ndarray), "ERR: variable b is required np.ndarray type"
    assert b_vector.ndim == 2, f"ERR: data base input is in wrong shape, matrix b required 2 dimensions"
    assert b_vector.shape[1] == 1, f"ERR: data base input is in wrong shape, matrix b required has only one row"

    d = B_matrix.shape[1]
    # Uniformly sample a random vector
    x = np.random.uniform(size=(d, 1))
    c1, c2 = compute_c1_c2(index, B_matrix, b_vector.ravel())
    v = B_matrix[index].copy().reshape(-1, 1)

    # condition
    Q = get_matrix_Q(B_matrix, b_vector.ravel(), r)
    xopt = compute_xopt(B_matrix, b_vector.ravel()).reshape((-1, 1))
    neighbor_B = get_neighbor_B(index, B_matrix)
    neighbor_b = get_neighbor_b(index, b_vector.ravel())
    neighbor_Q = get_matrix_Q(neighbor_B, neighbor_b, r)
    neighbor_xopt = compute_xopt(neighbor_B, neighbor_b).reshape((-1, 1))
    expression1 = (x - xopt).T @ mat_inv(Q) @ (x - xopt) - (x - neighbor_xopt).T @ mat_inv(neighbor_Q) @ (x -
                                                                                                          neighbor_xopt)

    # claim 1
    A = compute_quadratic_matrx(index, B_matrix, b_vector.ravel())
    coeff1 = -(2 * b_vector[index].item() * c1 * v.T + 2 * c2 * b_vector.T @ B_matrix).T
    c_before = (b_vector[index].item() ** 2) * c1 + (twoNorm(b_vector.ravel()) ** 2) * c2
    expression2 = r * (x.T @ A @ x + coeff1.T @ x + c_before)

    print(f"{expression1} v.s. {expression2}")

    return check_equal(expression1, expression2)


def check_quadratic_form_transformation_4(B_matrix, b_vector, index, r):
    """Check correctness of Proposition 7"""
    assert isinstance(b_vector, np.ndarray), "ERR: variable b is required np.ndarray type"
    assert b_vector.ndim == 2, f"ERR: data base input is in wrong shape, matrix b required 2 dimensions"
    assert b_vector.shape[1] == 1, f"ERR: data base input is in wrong shape, matrix b required has only one row"

    d = B_matrix.shape[1]
    # Uniformly sample a random vector
    x = np.random.uniform(size=(d, 1))
    v = B_matrix[index].copy().reshape(-1, 1)
    c1, c2 = compute_c1_c2(index, B_matrix, b_vector.ravel())
    c_ratio = compute_c_ratio(index, B_matrix, b_vector.ravel())
    M = mat_inv(B_matrix.T @ B_matrix)
    w = get_w(B_matrix, b_vector)
    wi = w[index]
    P_diagonal = calc_proj_matrix(B_matrix).diagonal()
    Pi = P_diagonal[index]

    # condition
    Q = get_matrix_Q(B_matrix, b_vector.ravel(), r)
    xopt = compute_xopt(B_matrix, b_vector.ravel()).reshape((-1, 1))
    neighbor_B = get_neighbor_B(index, B_matrix)
    neighbor_b = get_neighbor_b(index, b_vector.ravel())
    neighbor_Q = get_matrix_Q(neighbor_B, neighbor_b, r)
    neighbor_xopt = compute_xopt(neighbor_B, neighbor_b).reshape((-1, 1))
    expression1 = (x - xopt).T @ mat_inv(Q) @ (x - xopt) - (x - neighbor_xopt).T @ mat_inv(neighbor_Q) @ (x -
                                                                                                          neighbor_xopt)

    # Claim 2
    A = compute_quadratic_matrx(index, B_matrix, b_vector.ravel())
    h_compute = xopt + (c_ratio * wi) / (1 + c_ratio * Pi) * M @ v
    k_compute = -(wi ** 2) / ((twoNorm(w) ** 2) * (1 - Pi) - wi ** 2) + (1 - Pi) * (wi ** 4) / (
                (twoNorm(w) ** 2) * (wi ** 2) * (1 - Pi ** 2) - (wi ** 4) - (twoNorm(w) ** 4) * Pi * ((1 - Pi)**2))

    expression2 = r*((x-h_compute).T@A@(x-h_compute) + k_compute)

    print(f"{expression1} v.s. {expression2}")
    return check_equal(expression1, expression2)


def check_expression_w_neighbors(index, B, b):
    """Check proposition 5's correctness"""
    neighbor_w = get_neighbor_w(index, B, b)
    w = get_w(B, b)

    wi = w[index]
    P_diagonal = calc_proj_matrix(B).diagonal()
    Pi = P_diagonal[index]

    expression1 = twoNorm(w) ** 2 - twoNorm(neighbor_w) ** 2
    expression2 = (wi ** 2) / (1 - Pi)

    return check_equal(expression1, expression2)


def check_expression_Q_ratio(index, B, b):
    """Check proposition 6's correctness"""

    r = 1000  # r doesn't matter since it will be canceled during ratio
    det_Q = np.linalg.det(get_matrix_Q(B, b, r))
    det_neighbor_Q = np.linalg.det(get_neighbor_matrix_Q(index, B, b, r))
    expression1 = det_neighbor_Q/det_Q

    d = B.shape[1]
    w = get_w(B, b)
    wi = w[index]
    P_diagonal = calc_proj_matrix(B).diagonal()
    Pi = P_diagonal[index]
    expression2 = 1/(1-Pi)*((1-(wi**2)/((1-Pi)*(twoNorm(w)**2)))**d)
    print(f"{expression1} v.s. {expression2}")

    return check_equal(expression1, expression2)


def check_expression_Q_log_ratio(index, B, b):
    """Check proposition 8 in parts"""

    r = 1000  # r doesn't matter since it will be canceled during ratio
    det_Q = np.linalg.det(get_matrix_Q(B, b, r))
    det_neighbor_Q = np.linalg.det(get_neighbor_matrix_Q(index, B, b, r))
    expression1 = np.log(np.sqrt(det_neighbor_Q)/np.sqrt(det_Q))

    d = B.shape[1]
    w = get_w(B, b)
    wi = w[index]
    P_diagonal = calc_proj_matrix(B).diagonal()
    Pi = P_diagonal[index]
    expression2 = 0.5*(d*np.log(1 - (wi**2)/((1-Pi)*(twoNorm(w)**2))) - np.log(1 - Pi))
    print(f"{expression1} v.s. {expression2}")

    return check_equal(expression1, expression2)


def check_expression_b_transpose_B(index, B_matrix, b_vector):
    """Check proposition 3's correctness"""
    assert isinstance(b_vector, np.ndarray), "ERR: variable b is required np.ndarray type"
    assert b_vector.ndim == 2, f"ERR: data base input is in wrong shape, matrix b required 2 dimensions"
    assert b_vector.shape[1] == 1, f"ERR: data base input is in wrong shape, matrix b required has only one row"

    neighbor_B = get_neighbor_B(index, B_matrix)
    neighbor_b = get_neighbor_b(index, b_vector.ravel()).reshape((-1, 1))
    v = B_matrix[index].copy().reshape(-1, 1)

    expression1 = neighbor_b.T@neighbor_B
    expression2 = b_vector.T@B_matrix - b_vector[index].item()*v.T

    print(f"{expression1} v.s. {expression2}")

    return check_equal(expression1, expression2)


def check_expression_xopt_neighbor(index, B_matrix, b_vector):
    """Check proposition 4's correctness"""
    assert isinstance(b_vector, np.ndarray), "ERR: variable b is required np.ndarray type"
    assert b_vector.ndim == 2, f"ERR: data base input is in wrong shape, matrix b required 2 dimensions"
    assert b_vector.shape[1] == 1, f"ERR: data base input is in wrong shape, matrix b required has only one row"

    neighbor_B = get_neighbor_B(index, B_matrix)
    neighbor_b = get_neighbor_b(index, b_vector.ravel()).reshape((-1, 1))
    v = B_matrix[index].copy().reshape(-1, 1)
    bi = b_vector[index].item()

    xopt = compute_xopt(B_matrix, b_vector.ravel()).reshape((-1, 1))
    neighbor_xopt = compute_xopt(neighbor_B, neighbor_b.ravel()).reshape((-1, 1))
    expression1 = neighbor_xopt - xopt

    P_diagonal = calc_proj_matrix(B_matrix).diagonal()
    Pi = P_diagonal[index]

    M = mat_inv(B_matrix.T @ B_matrix)
    expression2 = (v.T@xopt - bi)/(1-Pi)*M@v

    print(f"{expression1} v.s. {expression2}")
    return check_equal(expression1, expression2)

