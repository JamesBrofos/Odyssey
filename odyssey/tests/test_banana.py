import unittest

import numpy as np

from odyssey.banana import generate_data, Banana


class TestBanana(unittest.TestCase):
    def test_banana(self):
        # Generate data.
        t = 0.5
        sigma_theta = 2.
        sigma_y = 2.
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)
        distr = Banana(y, sigma_y, sigma_theta)
        log_posterior = lambda q: distr.riemannian_quantities(q)[0]
        grad_log_posterior = lambda q: distr.riemannian_quantities(q)[1]
        metric = lambda q: distr.riemannian_quantities(q)[2]
        jac_metric = lambda q: distr.riemannian_quantities(q)[3]

        # Check the gradients of the posterior using finite differences.
        delta = 1e-6
        u = np.random.normal(size=(2, ))
        fd = (log_posterior(theta + 0.5*delta*u) - log_posterior(theta - 0.5*delta*u)) / delta
        dd = grad_log_posterior(theta)@u
        self.assertTrue(np.allclose(fd, dd))

        # Check the gradient of the metric using finite differences.
        fd = (metric(theta + 0.5*delta*u) - metric(theta - 0.5*delta*u)) / delta
        dG = jac_metric(theta)
        self.assertTrue(np.allclose(fd, dG@u))
