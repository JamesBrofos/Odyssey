import unittest

import numpy as np

from odyssey.cox_poisson import generate_data, prior, transforms
from odyssey.cox_poisson import LatentIntensity, Hyperparameters


class TestCoxPoisson(unittest.TestCase):
    def test_prior(self):
        def transformed_log_prior(qt):
            return prior.log_prior(*transforms.inverse_transform(qt)[0])

        transformed_grad_log_prior = lambda qt: prior.grad_log_prior(*qt)
        transformed_hess_log_prior = lambda qt: prior.hess_log_prior(*qt)
        transformed_grad_hess_log_prior = lambda qt: prior.jac_hess_log_prior(*qt)

        q = np.random.uniform(size=(2, ))
        qt, _ = transforms.forward_transform(q)

        delta = 1e-5

        u = np.random.normal(size=qt.shape)
        fd = (transformed_log_prior(qt + 0.5*delta*u) - transformed_log_prior(qt - 0.5*delta*u)) / delta
        dd = transformed_grad_log_prior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (transformed_grad_log_prior(qt + 0.5*delta*u) - transformed_grad_log_prior(qt - 0.5*delta*u)) / delta
        dd = transformed_hess_log_prior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

        fd = (transformed_hess_log_prior(qt + 0.5*delta*u) - transformed_hess_log_prior(qt - 0.5*delta*u)) / delta
        dd = transformed_grad_hess_log_prior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

    def test_latent_intensity(self):
        sigmasq, beta = np.random.uniform(size=(2, ))
        mu = np.log(126.0) - sigmasq / 2.0
        dist, x, y = generate_data(10, mu, beta, sigmasq)

        distr = LatentIntensity(dist, mu, sigmasq, beta, y)
        log_posterior = lambda q: distr.euclidean_quantities(q)[0]
        grad_log_posterior = lambda q: distr.euclidean_quantities(q)[1]
        delta = 1e-6

        u = np.random.normal(size=x.shape)
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))

    def test_hyperparameters(self):
        sigmasq, beta = np.random.uniform(size=(2, ))
        mu = np.log(126.0) - sigmasq / 2.0
        dist, x, y = generate_data(10, mu, beta, sigmasq)

        distr = Hyperparameters(dist, mu, x, y)
        log_posterior = lambda q: distr.riemannian_quantities(q)[0]
        grad_log_posterior = lambda q: distr.riemannian_quantities(q)[1]
        metric = lambda q: distr.riemannian_quantities(q)[2]
        jac_metric = lambda q: distr.riemannian_quantities(q)[3]

        # Check the gradients of the posterior using finite differences.
        delta = 1e-6
        q = np.array([sigmasq, beta])
        qt, _ = transforms.forward_transform(q)
        u = np.random.normal(size=qt.shape)
        fd = (log_posterior(qt + 0.5*delta*u) - log_posterior(qt - 0.5*delta*u)) / delta
        dd = grad_log_posterior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

        # Check the gradient of the metric using finite differences.
        fd = (metric(qt + 0.5*delta*u) - metric(qt - 0.5*delta*u)) / delta
        dG = jac_metric(qt)
        self.assertTrue(np.allclose(fd, dG@u))
