from typing import Callable, Tuple

import numpy as np

from odyssey.distribution import Distribution


def sigmoid(x: np.ndarray) -> np.ndarray:
    """This implementation of the sigmoid function is from [1].

    [1] https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python

    Args:
        x: Input to the sigmoid function.

    Returns:
        z: Output of the sigmoid function.

    """
    idx = x >= 0.0
    z = np.zeros(len(x))
    z[idx] = 1.0 / (1.0 + np.exp(-x[idx]))
    p = np.exp(x[~idx])
    z[~idx] = p / (1.0 + p)
    return z

def sigmoid_p(z):
    """First derivative of the sigmoid function."""
    p = sigmoid(z)
    return p*(1.0 - p)

def sigmoid_pp(z):
    """Second derivative of the sigmoid function."""
    p = sigmoid(z)
    pp = sigmoid_p(z)
    return pp - 2*p*pp

def sample_posterior_precision(beta: np.ndarray, k: float, theta: float) -> float:
    """Samples the posterior distribution of the precision parameter, which can be
    shown to be a Gamma distribution with a prescribed shape and scale.

    Args:
        beta: The current linear coefficients.
        k: The shape of the precision Gamma prior.
        theta: The scale of the precision Gamma prior.

    Returns:
        alpha: The precision parameter.

    """
    d = len(beta)
    shape = k + 0.5*d
    scale = np.reciprocal(0.5*np.sum(np.square(beta)) + np.reciprocal(theta))
    alpha = np.random.gamma(shape, scale)
    return alpha

def log_posterior(lin: np.ndarray, beta: np.ndarray, y: np.ndarray, alpha: float):
    ll = -np.sum(np.maximum(lin, 0.0) - lin*y + np.log(1.0 + np.exp(-np.abs(lin))))
    lbeta = -0.5 * alpha * np.sum(np.square(beta))
    lp = ll + lbeta
    return lp

def grad_log_posterior(lin: np.ndarray, beta: np.ndarray, x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    yp = sigmoid(lin)
    glp = (y - yp)@x - alpha * beta
    return glp

def metric(lin: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    L = sigmoid_p(lin)
    G = (x.T*L)@x + alpha * np.eye(x.shape[-1])
    return G

def jac_metric(lin: np.ndarray, x: np.ndarray) -> np.ndarray:
    o = sigmoid_pp(lin)
    Q = o[..., np.newaxis] * x
    dG = x.T@(Q[..., np.newaxis] * x[:, np.newaxis]).swapaxes(0, 1)
    return dG

class LogisticRegression(Distribution):
    """Factory function that yields further functions to compute the log-posterior
    of a Bayesian logistic regression model, the gradient of the log-posterior,
    the Fisher information metric, and the gradient of the Fisher information
    metric. It is defined by the following generative model:

        alpha ~ Gamma(k, theta)
        beta | alpha ~ Normal(0, (1 / alpha) * Id)
        y[i] | x[i], beta ~ Bernoulli(sigma(x[i] @ beta))

    However, this distribution is conditional on a value of the precision
    variable `alpha`; the precision can be sampled exactly in a Gibbs step.

    Parameters:
        x: Covariates of the logistic regression.
        y: Binary targets of the logistic regression.
        alpha: The precision of the normal prior over the linear coefficients.

    """
    def __init__(self, x: np.ndarray, y: np.ndarray, alpha: float):
        super().__init__()
        self.x = x
        self.y = y
        self.alpha = alpha

    def log_density(self, qt):
        lin = self.x@qt
        lp = log_posterior(lin, qt, self.y, self.alpha)
        return lp

    def sample(self):
        raise NotImplementedError()

    def forward_transform(self, q):
        return q, 0.0

    def inverse_transform(self, qt):
        return qt, 0.0

    def hessian(self, qt):
        raise NotImplementedError()

    def euclidean_metric(self):
        Id = np.eye(self.x.shape[-1])
        return Id, Id, Id

    def riemannian_metric(self, qt):
        lin = self.x@qt
        G = metric(lin, self.x, self.alpha)
        return G

    def riemannian_metric_and_jacobian(self, qt):
        lin = self.x@qt
        G = metric(lin, self.x, self.alpha)
        dG = jac_metric(lin, self.x)
        return G, dG

    def euclidean_quantities(self, qt):
        lin = self.x@qt
        lp = log_posterior(lin, qt, self.y, self.alpha)
        glp = grad_log_posterior(lin, qt, self.x, self.y, self.alpha)
        return lp, glp

    def riemannian_quantities(self, qt):
        lin = self.x@qt
        lp = log_posterior(lin, qt, self.y, self.alpha)
        glp = grad_log_posterior(lin, qt, self.x, self.y, self.alpha)
        G = metric(lin, self.x, self.alpha)
        dG = jac_metric(lin, self.x)
        return lp, glp, G, dG

    def lagrangian_quantities(self, qt):
        lin = self.x@qt
        lp = log_posterior(lin, qt, self.y, self.alpha)
        G = metric(lin, self.x, self.alpha)
        return lp, G

    def softabs_quantities(self, qt):
        raise NotImplementedError()
