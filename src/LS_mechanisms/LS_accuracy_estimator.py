import numpy as np

from analysis.commons import compute_xopt, split_to_B_b
from utils.empirical_bootstrap import EmpiricalBootstrap, SampleGenerator


class LS_accuracy_estimator:
    def __init__(self, kwargs):
        self.CI = kwargs["CI"]
        assert 0 < self.CI < 1, "confidence interval should have value between 0 and 1"
        self.bootstrap_samples = kwargs["bootstrap_samples"]

        self.D = kwargs["database"]
        self.B, self.b = split_to_B_b(self.D)

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.d = self.D.shape[1]
        self.n = self.D.shape[0]

    def estimate_square_error(self, samples):
        assert samples.shape[0] >= self.bootstrap_samples, "insufficient samples provided"
        assert samples.shape[1] == self.d-1
        B = self.B
        b = self.b
        xopt = compute_xopt(B, b).ravel()

        constant = np.linalg.norm(xopt)

        estimates_error_ratio = np.zeros(self.bootstrap_samples)
        estimates_length_ratio = np.zeros(self.bootstrap_samples)

        for i in np.arange(self.bootstrap_samples):
            estimates_error_ratio[i] = np.linalg.norm(samples[i].ravel() - xopt)/constant
            # estimates[i] = np.linalg.norm(samples[i].ravel()-xopt)/np.linalg.norm(samples[i].ravel())
            estimates_length_ratio[i] = constant / np.linalg.norm(samples[i].ravel())

        bootstrap_error_ratio = EmpiricalBootstrap(sample_generator=SampleGenerator(data=estimates_error_ratio))
        bootstrap_length_ratio = EmpiricalBootstrap(sample_generator=SampleGenerator(data=estimates_length_ratio))

        boot_res_error_ratio = bootstrap_error_ratio.bootstrap_confidence_bounds(
            confidence_interval_prob=self.CI,
            n_samples=self.bootstrap_samples
        )

        boot_res_length_ratio = bootstrap_length_ratio.bootstrap_confidence_bounds(
            confidence_interval_prob=self.CI,
            n_samples=self.bootstrap_samples
        )

        return boot_res_error_ratio, boot_res_length_ratio