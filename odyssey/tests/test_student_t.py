import unittest

import numpy as np

from odyssey.t import T

class TestStudentT(unittest.TestCase):
    def test_posterior(self):
        n = int(np.ceil(20*np.random.uniform()))
        dof = int(np.ceil(50*np.random.uniform()))
        L = np.random.normal(size=(n, n))
        Sigma = L@L.T
        distr = T(Sigma, dof)
        log_posterior = lambda q: distr.riemannian_quantities(q)[0]
        grad_log_posterior = lambda q: distr.riemannian_quantities(q)[1]
        metric = lambda q: distr.riemannian_quantities(q)[2]
        jac_metric = lambda q: distr.riemannian_quantities(q)[3]
        hessian = lambda q: distr.softabs_quantities(q)[2]
        jac_hessian = lambda q: distr.softabs_quantities(q)[3]

        delta = 1e-5
        x = distr.sample()
        u = np.random.normal(size=x.shape)
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (grad_log_posterior(x + 0.5*delta*u) - grad_log_posterior(x - 0.5*delta*u)) / delta
        dd = hessian(x)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (hessian(x + 0.5*delta*u) - hessian(x - 0.5*delta*u)) / delta
        dd = jac_hessian(x)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (metric(x + 0.5*delta*u) - metric(x - 0.5*delta*u)) / delta
        dd = jac_metric(x)@u
        self.assertTrue(np.allclose(fd, dd))

        G = metric(x)
        self.assertTrue(np.allclose(G, G.T))
        w, v = np.linalg.eigh(G)
        self.assertTrue(np.all(w > 0.0))
