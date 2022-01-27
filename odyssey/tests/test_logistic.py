import unittest

import numpy as np

from odyssey.logistic import LogisticRegression, sigmoid


class TestLogisticRegression(unittest.TestCase):
    def test_logistic_regression(self):
        # Generate logistic regression data.
        num_obs, num_dims = 100, 5
        x = np.random.normal(size=(num_obs, num_dims))
        b = np.ones((x.shape[-1], ))
        p = sigmoid(x@b)
        y = np.random.binomial(1, p)
        alpha = 0.5

        distr = LogisticRegression(x, y, alpha)
        log_posterior = lambda q: distr.riemannian_quantities(q)[0]
        grad_log_posterior = lambda q: distr.riemannian_quantities(q)[1]
        metric = lambda q: distr.riemannian_quantities(q)[2]
        jac_metric = lambda q: distr.riemannian_quantities(q)[3]

        # Check the gradients of the posterior using finite differences.
        delta = 1e-6
        u = np.random.normal(size=b.shape)
        fd = (log_posterior(b + 0.5*delta*u) - log_posterior(b - 0.5*delta*u)) / delta
        dd = grad_log_posterior(b)@u
        self.assertTrue(np.allclose(fd, dd))

        # Check the gradient of the metric using finite differences.
        fd = (metric(b + 0.5*delta*u) - metric(b - 0.5*delta*u)) / delta
        dG = jac_metric(b)
        self.assertTrue(np.allclose(fd, dG@u))
