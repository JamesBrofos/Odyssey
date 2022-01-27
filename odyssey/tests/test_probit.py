import unittest

import numpy as np

from odyssey.probit import generate_data, Probit
from odyssey.probit.probit import afn, bfn, cfn, dfn, xi, dxi


class TestProbit(unittest.TestCase):
    def test_probit(self):
        x, y = generate_data()
        delta = 1e-5
        qt = np.random.normal(size=(2, ))
        beta, log_sigmasq = qt
        sigmasq = np.exp(log_sigmasq)

        distr = Probit(x, y)
        log_posterior = lambda q: distr.riemannian_quantities(q)[0]
        grad_log_posterior = lambda q: distr.riemannian_quantities(q)[1]
        metric = lambda q: distr.riemannian_quantities(q)[2]
        jac_metric = lambda q: distr.riemannian_quantities(q)[3]

        pert = np.random.normal(size=(2, ))
        fd = (log_posterior(qt + 0.5*delta*pert) - log_posterior(qt - 0.5*delta*pert)) / delta
        dd = grad_log_posterior(qt)@pert
        self.assertTrue(np.allclose(fd, dd))

        G = metric(qt)
        e, _ = np.linalg.eigh(G)
        self.assertTrue(np.all(e > 0))

        dG = jac_metric(qt)
        pert = np.random.normal(size=(2, ))
        fd = (metric(qt + 0.5*delta*pert) - metric(qt - 0.5*delta*pert)) / delta
        dd = dG@pert
        self.assertTrue(np.allclose(fd, dd))

        z = np.random.uniform()
        fd = (xi(z+0.5*delta) - xi(z-0.5*delta)) / delta
        dd = dxi(z)
        self.assertTrue(np.allclose(fd, dd))

        fd = (afn(beta + 0.5*delta, log_sigmasq, x, y)[0] - afn(beta - 0.5*delta, log_sigmasq, x, y)[0]) / delta
        dd = afn(beta, log_sigmasq, x, y)[1]
        self.assertTrue(np.allclose(fd, dd))
        fd = (afn(beta, log_sigmasq + 0.5*delta, x, y)[0] - afn(beta, log_sigmasq - 0.5*delta, x, y)[0]) / delta
        dd = afn(beta, log_sigmasq, x, y)[2]
        self.assertTrue(np.allclose(fd, dd))

        fd = (bfn(beta + 0.5*delta, log_sigmasq, x, y)[0] - bfn(beta - 0.5*delta, log_sigmasq, x, y)[0]) / delta
        dd = bfn(beta, log_sigmasq, x, y)[1]
        self.assertTrue(np.allclose(fd, dd))
        fd = (bfn(beta, log_sigmasq + 0.5*delta, x, y)[0] - bfn(beta, log_sigmasq - 0.5*delta, x, y)[0]) / delta
        dd = bfn(beta, log_sigmasq, x, y)[2]
        self.assertTrue(np.allclose(fd, dd))

        fd = (cfn(beta + 0.5*delta, log_sigmasq, x, y)[0] - cfn(beta - 0.5*delta, log_sigmasq, x, y)[0]) / delta
        dd = cfn(beta, log_sigmasq, x, y)[1]
        self.assertTrue(np.allclose(fd, dd))
        fd = (cfn(beta, log_sigmasq + 0.5*delta, x, y)[0] - cfn(beta, log_sigmasq - 0.5*delta, x, y)[0]) / delta
        dd = cfn(beta, log_sigmasq, x, y)[2]
        self.assertTrue(np.allclose(fd, dd))

        fd = (dfn(beta + 0.5*delta, log_sigmasq, x, y)[0] - dfn(beta - 0.5*delta, log_sigmasq, x, y)[0]) / delta
        dd = dfn(beta, log_sigmasq, x, y)[1]
        self.assertTrue(np.allclose(fd, dd))
        fd = (dfn(beta, log_sigmasq + 0.5*delta, x, y)[0] - dfn(beta, log_sigmasq - 0.5*delta, x, y)[0]) / delta
        dd = dfn(beta, log_sigmasq, x, y)[2]
        self.assertTrue(np.allclose(fd, dd))
