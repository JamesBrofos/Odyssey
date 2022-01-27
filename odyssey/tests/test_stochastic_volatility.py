import unittest

import numpy as np

from odyssey.stochastic_volatility import generate_data, prior, transforms
from odyssey.stochastic_volatility import LatentVolatilities, Hyperparameters


class TestStochasticVolatility(unittest.TestCase):
    def test_prior(self):
        def transformed_log_prior(qt):
            return prior.log_prior(*transforms.inverse_transform(qt)[0])

        transformed_grad_log_prior = lambda qt: prior.grad_log_prior(*qt)
        transformed_hess_log_prior = lambda qt: prior.hess_log_prior(*qt)
        transformed_grad_hess_log_prior = lambda qt: prior.jac_hess_log_prior(*qt)

        q = np.random.uniform(size=(3, ))
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

    def test_latent_volatilities(self):
        sigma, phi, beta = np.random.uniform(size=(3, ))
        T = int(np.ceil(1000*np.random.uniform()))
        x, y = generate_data(T, sigma, phi, beta)
        distr = LatentVolatilities(sigma, phi, beta, y)
        log_posterior = lambda q: distr.euclidean_quantities(q)[0]
        grad_log_posterior = lambda q: distr.euclidean_quantities(q)[1]
        delta = 1e-6

        u = np.random.normal(size=x.shape)
        fd = (log_posterior(x + 0.5*delta*u) - log_posterior(x - 0.5*delta*u)) / delta
        dd = grad_log_posterior(x)@u
        self.assertTrue(np.allclose(fd, dd))

    def test_hyperparameters(self):
        sigma, phi, beta = np.random.uniform(size=(3, ))
        T = int(np.ceil(1000*np.random.uniform()))
        x, y = generate_data(T, sigma, phi, beta)
        q = np.array([sigma, phi, beta])
        qt = transforms.forward_transform(q)[0]
        distr = Hyperparameters(x, y)
        log_posterior = lambda q: distr.riemannian_quantities(q)[0]
        grad_log_posterior = lambda q: distr.riemannian_quantities(q)[1]
        metric = lambda q: distr.riemannian_quantities(q)[2]
        jac_metric = lambda q: distr.riemannian_quantities(q)[3]

        # Check the gradients of the posterior using finite differences.
        delta = 1e-6
        u = np.random.normal(size=qt.shape)
        fd = (log_posterior(qt + 0.5*delta*u) - log_posterior(qt - 0.5*delta*u)) / delta
        dd = grad_log_posterior(qt)@u
        self.assertTrue(np.allclose(fd, dd))

        # Check the gradient of the metric using finite differences.
        fd = (metric(qt + 0.5*delta*u) - metric(qt - 0.5*delta*u)) / delta
        dG = jac_metric(qt)
        self.assertTrue(np.allclose(fd, dG@u))
