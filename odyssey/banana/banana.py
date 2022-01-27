from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution


def generate_data(t: float, sigma_y: float, sigma_theta: float, num_obs: int) -> Tuple[np.ndarray]:
    """Generate data from the banana-shaped posterior distribution.

    Args:
        t: Free-parameter determining the thetas.
        sigma_y: Noise standard deviation.
        sigma_theta: Prior standard deviation over the thetas.
        num_obs: Number of observations to generate.

    Returns:
        theta: Linear coefficients of the banana-shaped distribution.
        y: Observations from the unidentifiable model.

    """
    theta = np.array([t, np.sqrt(1.0 - t)])
    y = theta[0] + np.square(theta[1]) + sigma_y * np.random.normal(size=(num_obs, ))
    return theta, y

def log_posterior(theta: np.ndarray, y: np.ndarray, sigma_sq_y: float, sigma_sq_theta: float) -> float:
    """The banana-shaped distribution log-posterior.

    Args:
        theta: Linear coefficients.
        y: Observations of the banana model.
        sigma_y: Standard deviation of the observations.
        sigma_theta: Standard deviation of prior over linear coefficients.

    Returns:
        lp: The log-posterior of the banana-shaped distribution.

    """
    p = theta[0] + np.square(theta[1])
    ll = -0.5 / sigma_sq_y * np.square(y - p).sum()
    lpr = -0.5 / sigma_sq_theta * np.square(theta).sum()
    lp = ll + lpr
    return lp

def grad_log_posterior(theta: np.ndarray, y: np.ndarray, sigma_sq_y: float, sigma_sq_theta: float) -> np.ndarray:
    """Gradient of the banana-shaped distribution log-posterior with respect to the
    linear coefficients.

    Args:
        theta: Linear coefficients.
        y: Observations of the banana model.
        sigma_y: Standard deviation of the observations.
        sigma_theta: Standard deviation of prior over linear coefficients.

    Returns:
        glp: The gradient of the log-posterior of the banana-shaped distribution
            with respect to the linear coefficients.

    """
    p = theta[0] + np.square(theta[1])
    d = np.sum(y - p)
    ga = d / sigma_sq_y - theta[0] / sigma_sq_theta
    gb = 2.0*d / sigma_sq_y * theta[1] - theta[1] / sigma_sq_theta
    glp = np.hstack((ga, gb))
    return glp

def metric(theta: np.ndarray, y: np.ndarray, sigma_sq_y: float, sigma_sq_theta: float) -> np.ndarray:
    """The Riemannian metric is the sum of the Fisher information and the negative
    Hessian of the log-prior.

    Args:
        theta: Linear coefficients.
        y: Observations of the banana model.
        sigma_y: Standard deviation of the observations.
        sigma_theta: Standard deviation of prior over linear coefficients.

    Returns:
        G: The Riemannian metric of the banana-shaped distribution.

    """
    n = y.size
    s = 2.0*n*theta[1] / sigma_sq_y
    G = np.array([[n / sigma_sq_y + 1.0 / sigma_sq_theta, s],
                  [s, 4.0*n*np.square(theta[1]) / sigma_sq_y + 1.0 / sigma_sq_theta]])
    return G

def jac_metric(theta: np.ndarray, y: np.ndarray, sigma_sq_y: float, sigma_sq_theta: float) -> np.ndarray:
    """The gradient of the Riemannian metric with respect to the linear
    coefficients.

    Args:
        theta: Linear coefficients.
        y: Observations of the banana model.
        sigma_y: Standard deviation of the observations.
        sigma_theta: Standard deviation of prior over linear coefficients.

    Returns:
        dG: The Jacobian of the Riemannian metric of the banana-shaped
            distribution.

    """
    n = y.size
    dG = np.array([
        [[0.0, 0.0], [0.0, 2.0*n / sigma_sq_y]],
        [[0.0, 2.0*n / sigma_sq_y], [0.0, 8.0*n*theta[1] / sigma_sq_y]]
    ])
    return dG

class Banana(Distribution):
    """The banana distribution is a distribution that exhibits a characteristic
    banana-shaped ridge that resembles the posterior that can emerge from
    models that are not identifiable. The distribution is the posterior of the
    following generative model.

        y ~ Normal(theta[0] + theta[1]**2, sigma_sq_y)
        theta[i] ~ Normal(0, sigma_sq_theta)

    Parameters:
        y: Observations of the banana model.
        sigma_y: Standard deviation of the observations.
        sigma_theta: Standard deviation of prior over linear coefficients.

    """
    def __init__(self, y: np.ndarray, sigma_y: float, sigma_theta: float):
        super().__init__()
        self.y = y
        self.sigma_sq_y = np.square(sigma_y)
        self.sigma_sq_theta = np.square(sigma_theta)

    def log_density(self, qt):
        ld = log_posterior(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        return ld

    def sample(self):
        raise NotImplementedError()

    def forward_transform(self, q):
        return q, 0.0

    def inverse_transform(self, qt):
        return qt, 0.0

    def hessian(self, qt):
        raise NotImplementedError()

    def euclidean_metric(self):
        Id = np.eye(2)
        return Id, Id, Id

    def riemannian_metric(self, qt):
        G = metric(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        return G

    def riemannian_metric_and_jacobian(self, qt):
        G = metric(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        dG = jac_metric(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        return G, dG

    def euclidean_quantities(self, qt):
        lp = log_posterior(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        glp = grad_log_posterior(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        return lp, glp

    def riemannian_quantities(self, qt):
        lp = log_posterior(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        glp = grad_log_posterior(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        G = metric(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        dG = jac_metric(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        return lp, glp, G, dG

    def lagrangian_quantities(self, qt):
        lp = log_posterior(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        G = metric(qt, self.y, self.sigma_sq_y, self.sigma_sq_theta)
        return lp, G

    def softabs_quantities(self, qt):
        raise NotImplementedError()
