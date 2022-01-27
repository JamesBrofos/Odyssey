import unittest

import numpy as np

from odyssey.fitzhugh_nagumo import generate_data, FitzhughNagumo


class TestFitzhughNagumo(unittest.TestCase):
    def test_fitzhugh_nagumo(self):
        # Integrator parameters.
        rtol = None
        atol = None
        hmax = 0.0
        hmin = 0.0
        mxstep = 0
        # Generate observations from the Fitzhugh-Nagumo model.
        a = 0.2
        b = 0.2
        c = 3.0
        sigma = 0.5
        correct = True

        state = np.array([-1.0, 1.0])
        t = np.linspace(0.0, 10.0, 200)
        y = generate_data(state, t, sigma, a, b, c, rtol, atol, hmax, hmin, mxstep=mxstep)
        distr = FitzhughNagumo(state, y, t, sigma, rtol, atol, hmax, hmin, mxstep, correct)
        log_posterior = lambda q: distr.riemannian_quantities(q)[0]
        grad_log_posterior = lambda q: distr.riemannian_quantities(q)[1]
        metric = lambda q: distr.riemannian_quantities(q)[2]
        jac_metric = lambda q: distr.riemannian_quantities(q)[3]

        # Check the gradients of the posterior using finite differences.
        q = np.random.uniform(size=(3, ))
        delta = 1e-6
        u = np.random.normal(size=(3, ))
        fd = (log_posterior(q + 0.5*delta*u) - log_posterior(q - 0.5*delta*u)) / delta
        dd = grad_log_posterior(q)@u
        self.assertTrue(np.allclose(fd, dd))

        # Check the gradient of the metric using finite differences.
        fd = (metric(q + 0.5*delta*u) - metric(q - 0.5*delta*u)) / delta
        dG = jac_metric(q)
        self.assertTrue(np.allclose(fd, dG@u))
