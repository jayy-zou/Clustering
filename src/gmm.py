import numpy as np
from src import KMeans
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, n_clusters, covariance_type):
        self.n_clusters = n_clusters
        allowed_covariance_types = ['spherical', 'diagonal']
        if covariance_type not in allowed_covariance_types:
            raise ValueError(f'covariance_type must be in {allowed_covariance_types}')
        self.covariance_type = covariance_type

        self.means = None
        self.covariances = None
        self.mixing_weights = None
        self.max_iterations = 200

    def fit(self, features):
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(features)
        self.means = kmeans.means

        self.covariances = self._init_covariance(features.shape[-1])

        self.mixing_weights = np.random.rand(self.n_clusters)
        self.mixing_weights /= np.sum(self.mixing_weights)

        prev_log_likelihood = -float('inf')
        log_likelihood = self._overall_log_likelihood(features)

        n_iter = 0
        while log_likelihood - prev_log_likelihood > 1e-4 and n_iter < self.max_iterations:
            prev_log_likelihood = log_likelihood

            assignments = self._e_step(features)
            self.means, self.covariances, self.mixing_weights = (
                self._m_step(features, assignments)
            )

            log_likelihood = self._overall_log_likelihood(features)
            n_iter += 1

    def predict(self, features):
        posteriors = self._e_step(features)
        return np.argmax(posteriors, axis=-1)

    def _e_step(self, features):
        expectation = np.ones((np.shape(features)[0], self.n_clusters))

        for index in range(self.n_clusters):
            expectation[:, index] = self._posterior(features, index)

        return expectation

    def _m_step(self, features, assignments):
        R = np.sum(assignments, axis=0)

        means = np.dot(np.transpose(assignments), features)/R[:,np.newaxis]

        cov = self.covariances.copy()
        if self.covariance_type == 'spherical':
            for i in range(self.n_clusters):
                cov[i] = np.mean(np.diag(np.dot(assignments[:,i]*np.transpose(features-means[i]),(features-means[i]))/R[i]))
            return means, cov, R / np.sum(R)
        else:
            for i in range(self.n_clusters):
                cov[i] = np.diag(np.dot(assignments[:,i]*np.transpose(features-means[i]),(features-means[i]))/R[i])
            return means, cov, R / np.sum(R)



    def _init_covariance(self, n_features):
        if self.covariance_type == 'spherical':
            return np.random.rand(self.n_clusters)
        elif self.covariance_type == 'diagonal':
            return np.random.rand(self.n_clusters, n_features)

    def _log_likelihood(self, features, k_idx):
        log=np.log(self.mixing_weights[k_idx])
        pdf=multivariate_normal.logpdf(features, self.means[k_idx], self.covariances[k_idx])

        return log+pdf

    def _overall_log_likelihood(self, features):
        denom = [
            self._log_likelihood(features, j) for j in range(self.n_clusters)
        ]
        return np.sum(denom)

    def _posterior(self, features, k):
        num = self._log_likelihood(features, k)
        denom = np.array([
            self._log_likelihood(features, j)
            for j in range(self.n_clusters)
        ])
        
        max_value = denom.max(axis=0, keepdims=True)
        denom_sum = max_value + np.log(np.sum(np.exp(denom - max_value), axis=0))
        posteriors = np.exp(num - denom_sum)
        return posteriors
