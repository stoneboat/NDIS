import numpy as np

from utils.empirical_bootstrap import EmpiricalBootstrap, SampleGenerator


class RP_accuracy_estimator:
    def __init__(self, kwargs):
        self.CI = kwargs["CI"]
        assert 0 < self.CI < 1, "confidence interval should have value between 0 and 1"
        self.bootstrap_samples = kwargs["bootstrap_samples"]

        self.D = kwargs["database"]

        assert isinstance(self.D, np.ndarray), "ERR: required np.ndarray type"
        assert self.D.ndim == 2, f"ERR: database input is in wrong shape, required 2 dimensions"

        self.r = kwargs["r"]
        self.d = self.D.shape[1]
        self.n = self.D.shape[0]

    def estimate_pairwise_distance_acc(self, samples):
        assert samples.shape[0] >= self.bootstrap_samples, "insufficient samples provided"
        assert samples.shape[1] == self.r*self.d
        database = self.D.T

        estimates = np.zeros(self.bootstrap_samples)

        for i in np.arange(self.bootstrap_samples):
            sum = 0
            cnt = 0
            projected_database = samples[i].reshape((self.r, self.d)).T
            # Since the true random projection matrix is not with variance 1 but variance 1/r
            projected_database = projected_database/np.sqrt(self.r)
            for j in np.arange(1, self.d):
                for k in np.arange(j):
                    cnt += 1
                    sum += np.linalg.norm(projected_database[j]-projected_database[k])/np.linalg.norm(database[j]-database[k])

            estimates[i] = sum/cnt

        bootstrap = EmpiricalBootstrap(sample_generator=SampleGenerator(data=estimates))

        boot_res = bootstrap.bootstrap_confidence_bounds(
            confidence_interval_prob=self.CI,
            n_samples=self.bootstrap_samples
        )

        return boot_res

    def estimate_dot_product_acc(self, samples):
        assert samples.shape[0] >= self.bootstrap_samples, "insufficient samples provided"
        assert samples.shape[1] == self.r*self.d
        database = self.D.T

        estimates = np.zeros(self.bootstrap_samples)

        for i in np.arange(self.bootstrap_samples):
            projected_database = samples[i].reshape((self.r, self.d)).T
            # Since the true random projection matrix is not with variance 1 but variance 1/np.sqrt(r)
            projected_database = projected_database/np.sqrt(self.r)

            Dot_database = (database@database.T).ravel()
            Dot_projected_database = (projected_database@projected_database.T).ravel()

            cov = np.cov(Dot_database, Dot_projected_database)[0,1]
            std_database = np.std(Dot_database)
            std_projected_database = np.std(Dot_projected_database)

            estimates[i] = cov/(std_database*std_projected_database)

        bootstrap = EmpiricalBootstrap(sample_generator=SampleGenerator(data=estimates))

        boot_res = bootstrap.bootstrap_confidence_bounds(
            confidence_interval_prob=self.CI,
            n_samples=self.bootstrap_samples
        )

        return boot_res