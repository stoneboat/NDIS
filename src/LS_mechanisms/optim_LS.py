import logging

import numpy as np
import secrets
from numpy.random import MT19937, RandomState

import multiprocessing
from functools import partial

from chi2comb import chi2comb_cdf, ChiSquared

from analysis.commons import split_to_B_b, compute_xopt, get_w
from RP_mechanisms.optim_RP import OptimalRP_mech, compute_IS


def lev_evaluate_ALS_right(r, d, leverage, aug_leverage, epsilon, cyclimits=200000, atol=1e-8):
    neg_lev = 1 - leverage
    neg_aug_lev = 1 - aug_leverage

    point = -r * leverage * (neg_lev - neg_aug_lev) / (neg_lev ** 2 - neg_aug_lev) + d * np.log(neg_aug_lev) - (
                d + 1) * np.log(neg_lev) - 2 * epsilon

    w11 = 1 - (neg_lev ** 2) / neg_aug_lev
    w12 = 1 - neg_lev / neg_aug_lev

    lam1 = r * leverage * (neg_lev ** 2) * (neg_lev - neg_aug_lev) / (neg_lev ** 2 - neg_aug_lev) ** 2

    w21 = (neg_aug_lev - neg_lev ** 2) / (neg_lev ** 2)
    w22 = (neg_aug_lev - neg_lev) / neg_lev

    lam2 = r * leverage * neg_aug_lev * (neg_lev - neg_aug_lev) / (neg_lev ** 2 - neg_aug_lev) ** 2

    chi2s = [ChiSquared(w11, lam1, 1), ChiSquared(w12, 0, d - 1)]
    p1, errno1, info = chi2comb_cdf(point, chi2s, gcoef=0, lim=cyclimits, atol=atol)
    if errno1 != 0:
        logging.error(f"ERROR! p1 is {p1}, error code is {errno1} and other algorithm info {info}")

    chi2s = [ChiSquared(w21, lam2, 1), ChiSquared(w22, 0, d - 1)]
    p2, errno2, info = chi2comb_cdf(point, chi2s, gcoef=0, lim=cyclimits, atol=atol)
    if errno2 != 0:
        logging.error(f"ERROR! p2 is {p1}, error code is {errno2} and other algorithm info {info}")

    ret = p1 - np.exp(epsilon) * p2
    return ret, max(errno1, errno2)


def lev_evaluate_ALS_left(r, d, leverage, aug_leverage, epsilon, cyclimits=200000, atol=1e-8):
    neg_lev = 1 - leverage
    neg_aug_lev = 1 - aug_leverage

    point = r * leverage * (neg_lev - neg_aug_lev) / (neg_lev ** 2 - neg_aug_lev) - d * np.log(neg_aug_lev) + (
                d + 1) * np.log(neg_lev) - 2 * epsilon

    w11 = 1 - neg_aug_lev / (neg_lev ** 2)
    w12 = 1 - neg_aug_lev / neg_lev
    lam1 = r * leverage * neg_aug_lev * (neg_lev - neg_aug_lev) / (neg_lev ** 2 - neg_aug_lev) ** 2

    w21 = (neg_lev ** 2 - neg_aug_lev) / (neg_aug_lev)
    w22 = (neg_lev - neg_aug_lev) / neg_aug_lev
    lam2 = r * leverage * (neg_lev ** 2) * (neg_lev - neg_aug_lev) / (neg_lev ** 2 - neg_aug_lev) ** 2

    chi2s = [ChiSquared(w11, lam1, 1), ChiSquared(w12, 0, d - 1)]
    p1, errno1, info = chi2comb_cdf(point, chi2s, gcoef=0, lim=cyclimits, atol=atol)
    if errno1 != 0:
        logging.error(f"ERROR! p1 is {p1}, error code is {errno1} and other algorithm info {info}")

    chi2s = [ChiSquared(w21, lam2, 1), ChiSquared(w22, 0, d - 1)]
    p2, errno2, info = chi2comb_cdf(point, chi2s, gcoef=0, lim=cyclimits, atol=atol)
    if errno2 != 0:
        logging.error(f"ERROR! p2 is {p1}, error code is {errno2} and other algorithm info {info}")

    ret = p1 - np.exp(epsilon) * p2

    return ret, max(errno1, errno2)


def lev_evaluate_ALS(r, d, leverage, aug_leverage, epsilon, cyclimits=200000, atol=1e-8):
    a1, errno1 = lev_evaluate_ALS_right(r, d, leverage, aug_leverage, epsilon, cyclimits, atol)
    a2, errno2 = lev_evaluate_ALS_left(r, d, leverage, aug_leverage, epsilon, cyclimits, atol)

    if max(errno1, errno2) != 0:
        raise ValueError(f"Fail to evaluate generalized chi-squared distribution, erro code {max(errno1, errno2)}")

    return max(a1, a2)


class OptimalLS_mech:
    def __init__(self, kwargs):
        self.D = kwargs["database"]
        self.B, self.b = split_to_B_b(self.D)
        self.b = self.b.reshape((-1, 1))

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.r = kwargs["r"]
        self.d = self.D.shape[1]-1
        self.n = self.D.shape[0]
        self.index, self.lev, self.res = self.compute_largest_leverage()
        self.l = self.compute_largest_l2()

        # For privacy spectrum evaluation
        self.cyclimits = 20000000
        self.atol = 1e-8

        # Prepare the randomness
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))

    def compute_largest_leverage(self):
        largest_index = -1
        q = 0
        largest_lev = 0
        largest_res = 0

        M = np.linalg.inv(self.D.T @ self.D)
        N = np.linalg.inv(self.B.T @ self.B)

        for index in np.arange(self.n):
            v = self.D[index].copy().reshape(-1, 1)
            leverage = (v.T @ M @ v).item()
            if leverage > q:
                largest_index = index
                u = self.B[index].copy().reshape(-1, 1)
                q = leverage
                largest_lev = (u.T @ N @ u).item()
                largest_res = q-largest_lev

        return largest_index, largest_lev, largest_res

    def compute_largest_l2(self):
        l = 0
        for index in np.arange(self.n):
            v = self.D[index].copy().reshape(-1, 1)
            if np.linalg.norm(v) > l:
                l = np.linalg.norm(v)

        return l

    def find_minimal_sigma(self, epsilon, delta, low=1e-3, high=1e2, tol=1e-6):
        # Ensure that the function value at the lower bound is less than delta
        low_aug_lev= min(self.l**2*2/(low**2), 1-1e-8)
        low_lev = min(self.l**2/(low ** 2), 1-1e-8)

        if lev_evaluate_ALS(self.r, self.d, low_lev, low_aug_lev, epsilon,
                            cyclimits=self.cyclimits, atol=self.atol) < delta:
            raise ValueError("Please re-choose your down side")

        high_aug_lev = min(self.l ** 2*2/ (high ** 2), 1-1e-8)
        high_lev = min(self.l ** 2/ (high ** 2), 1-1e-8)
        if lev_evaluate_ALS(self.r, self.d, high_lev, high_aug_lev, epsilon,
                            cyclimits=self.cyclimits, atol=self.atol) > delta:
            raise ValueError("Please re-choose your up side")

        # Binary search
        while high - low > tol:
            mid = (low + high) / 2
            mid_aug_lev = min(self.l ** 2*2/(mid ** 2), 1-1e-8)
            mid_lev = min(self.l ** 2/(mid ** 2), 1-1e-8)

            if lev_evaluate_ALS(self.r, self.d, mid_lev, mid_aug_lev, epsilon,
                            cyclimits=self.cyclimits, atol=self.atol) < delta:
                high = mid
            else:
                low = mid

        return (low + high) / 2

    def gen_samples(self, num_samples, epsilon, delta):
        seed = secrets.randbits(128)
        self.rng = RandomState(MT19937(seed))
        return self._gen_samples(epsilon, delta, num_samples)

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)

        if lev_evaluate_ALS(self.r, self.d, self.lev, self.lev+self.res, epsilon,
                                              cyclimits=self.cyclimits, atol=self.atol) <= delta:

            print("hit here!")
            xopt = compute_xopt(self.B, self.b)
            w = get_w(self.B, self.b)
            cov = (np.linalg.norm(w)**2/self.r)*np.linalg.inv(self.B.T@self.B)

            samples = self.rng.multivariate_normal(xopt, cov=cov, size=num_samples)
        else:
            sigma = self.find_minimal_sigma(epsilon, delta)

            X = np.vstack((self.D, sigma * np.eye(self.d+1)))
            B, b = split_to_B_b(X)

            xopt = compute_xopt(B, b)
            w = get_w(B, b)
            cov = (np.linalg.norm(w) ** 2 / self.r) * np.linalg.inv(B.T @ B)

            samples = self.rng.multivariate_normal(xopt, cov=cov, size=num_samples)

        return samples


class LS_fromoptim_RP_mech(OptimalRP_mech):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        if "chunk_size" in kwargs:
            self.chunk_size = kwargs["chunk_size"]
        else:
            self.chunk_size = 1000

    def _gen_samples(self, epsilon, delta, num_samples):
        num_samples = int(num_samples)
        X = self.D
        samples = np.zeros((num_samples, self.d - 1))
        result_matrices = []
        if compute_IS(epsilon, self.lev, self.r) <= delta:
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
        else:
            sigma = self.find_minimal_sigma(epsilon, delta)

            for i in range(num_samples):
                r_piece = np.minimum(self.chunk_size, self.r)
                noise1 = self.rng.normal(loc=0, scale=1, size=(r_piece, X.shape[0]))
                pi_X = noise1 @ X
                pointer = r_piece

                while pointer < self.r:
                    r_piece = np.minimum(self.chunk_size, self.r-pointer)
                    noise1 = self.rng.normal(loc=0, scale=1, size=(r_piece, X.shape[0]))

                    pi_X_piece = noise1 @ X
                    pointer += r_piece
                    pi_X = np.vstack((pi_X, pi_X_piece))

                noise2 = self.rng.normal(loc=0, scale=sigma, size=(self.r, self.d))
                projected_database = pi_X + noise2
                B, b = split_to_B_b(projected_database)
                samples[i] = compute_xopt(B, b).reshape((1, -1))

        return samples


# class LS_fromoptim_RP_mech(OptimalRP_mech):
#     def _gen_samples(self, epsilon, delta, num_samples):
#         num_samples = int(num_samples)
#         X = self.D.T
#         result_matrices = []
#         if compute_IS(epsilon, self.lev, self.r) <= delta:
#             for _ in range(self.r):
#                 noise = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
#                 result_matrices.append(X @ noise)
#
#             # Concatenating the resulting matrices by row
#             final_matrix = np.concatenate(result_matrices, axis=0).T
#         else:
#             sigma = self.find_minimal_sigma(epsilon, delta)
#
#             for _ in range(self.r):
#                 noise1 = self.rng.normal(loc=0, scale=1, size=(X.shape[1], num_samples))
#                 noise2 = self.rng.normal(loc=0, scale=sigma, size=(X.shape[0], num_samples))
#                 result_matrices.append(X @ noise1 + noise2)
#
#             # Concatenating the resulting matrices by row
#             final_matrix = np.concatenate(result_matrices, axis=0).T
#
#         samples = np.zeros((num_samples, self.d-1))
#         for i in range(num_samples):
#             projected_database = final_matrix[i].reshape((self.r, self.d))/np.sqrt(self.r)
#             B, b = split_to_B_b(projected_database)
#             samples[i] = compute_xopt(B, b).reshape((1, -1))
#
#         return samples

