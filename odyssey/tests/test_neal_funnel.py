import unittest

import numpy as np

from odyssey.neal_funnel import NealFunnel



class TestNealFunnel(unittest.TestCase):
    def test_neal_funnel(self):
        num_dims = int(np.ceil(10*np.random.uniform()))
        distr = NealFunnel(num_dims)
        log_density = lambda q: distr.softabs_quantities(q)[0]
        grad_log_density = lambda q: distr.softabs_quantities(q)[1]
        hess_log_density = lambda q: distr.softabs_quantities(q)[2]

        x, v = distr.sample()
        q = np.hstack((x, v))
        delta = 1e-5
        u = np.random.normal(size=q.shape)
        fd = (log_density(q + 0.5*delta*u) - log_density(q - 0.5*delta*u)) / delta
        _, glp, H, dH = distr.softabs_quantities(q)
        self.assertTrue(np.allclose(fd, glp@u))
        fd = (grad_log_density(q + 0.5*delta*u) - grad_log_density(q - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, H@u))
        fd = (hess_log_density(q + 0.5*delta*u) - hess_log_density(q - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, dH@u))
