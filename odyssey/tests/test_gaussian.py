import unittest

import numpy as np

from odyssey.gaussian import Gaussian


class TestGaussian(unittest.TestCase):
    def test_posterior(self):
        n = int(np.ceil(100*np.random.uniform()))

        L = np.random.normal(size=(n, n))
        Sigma = L@L.T
        mu = np.random.normal(size=n)

        distr = Gaussian(mu, Sigma)
        log_posterior = lambda q: distr.euclidean_quantities(q)[0]
        grad_log_posterior = lambda q: distr.euclidean_quantities(q)[1]
        x = distr.sample()

        # Check the gradients of the posterior using finite differences.
        delta = 1e-6
        u = np.random.normal(size=(n, ))
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))
        fd = (grad_log_posterior(x + 0.5*delta*u) - grad_log_posterior(x - 0.5*delta*u)) / delta
        dd = -distr.iSigma@u
        self.assertTrue(np.allclose(fd, dd))
