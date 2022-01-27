from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from iliad.linalg import solve_psd

from odyssey.distribution import Distribution
from odyssey.cox_poisson import prior, transforms


def generate_data(num_grid: int, mu: float, beta: float=1.0 / 33, sigmasq: float=1.91) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data from the log-Gaussian Cox-Poisson process.

    Args:
        num_grid: The number of grid elements for the spatial data.
        mu: Mean value of the Gaussian process.
        sigmasq: Amplitude of the Gaussian process kernel.
        beta: Length scale of the Gaussian process kernel.

    Returns:
        dist: Matrix of pairwise distances.
        x: The Gaussian process.
        y: Count observations from the Poisson process.

    """
    num_grid_sq = np.square(num_grid)
    lin = np.linspace(0.0, 1.0, num_grid)
    [I, J] = np.meshgrid(lin, lin)
    grid = np.stack((I.ravel(), J.ravel())).T
    dist = cdist(grid, grid) / num_grid
    K = sigmasq * np.exp(-dist / beta)
    L = np.linalg.cholesky(K)
    x = L@np.random.normal(size=(num_grid_sq, )) + mu
    m = 1.0 / num_grid_sq
    e = m * np.exp(x)
    y = np.random.poisson(e)
    return dist, x, y

def update_intensity(dist, mu, sigmasq, beta, y, m):
    K = sigmasq * np.exp(-dist / beta)
    iK, _ = solve_psd(K)
    # I think there should be a factor of one-half multiplying the diagonal
    # because of the expectation of the exponential of a normal random
    # variable. This differs from the RMHMC paper.
    Lambda = m * np.exp(mu + 0.5 * np.diag(K))
    G = iK.copy()
    G[np.diag_indices(len(G))] += Lambda
    return K, iK, G

class LatentIntensity(Distribution):
    """Factory to produce functions for computing the log-posterior, the gradient
    of the log-posterior and the Fisher information metric of the log-Gaussian
    Cox-Poisson process given values of `sigma` and `beta`.

    Parameters:
        dist: Matrix of pairwise distances.
        mu: Mean value of the Gaussian process.
        sigmasq: Amplitude of the Gaussian process kernel.
        beta: Length scale of the Gaussian process kernel.
        y: Count observations from the Poisson process.

    """
    def __init__(self, dist: np.ndarray, mu: float, sigmasq: float, beta: float, y: np.ndarray):
        super().__init__()
        self.dist = dist
        self.mu = mu
        self.y = y
        self.num_grid_sq = dist.shape[0]
        self.m = 1.0 / self.num_grid_sq

        self.sigmasq = sigmasq
        self.beta = beta

        self.K, self.iK, self.G = update_intensity(dist, mu, sigmasq, beta, y, self.m)

    def log_density(self, qt):
        x = qt
        o = x - self.mu
        e = self.m * np.exp(x)
        iKo = self.iK@o
        lp = np.sum(self.y*x - e) - 0.5*o.dot(iKo)
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
        Ginv, Gchol = solve_psd(self.G)
        return self.G, Gchol, Ginv

    def riemannian_metric(self, qt):
        raise NotImplementedError()

    def riemannian_metric_and_jacobian(self, qt):
        raise NotImplementedError()

    def euclidean_quantities(self, qt):
        x = qt
        o = x - self.mu
        e = self.m * np.exp(x)
        iKo = self.iK@o
        lp = np.sum(self.y*x - e) - 0.5*o.dot(iKo)
        glp = self.y - e - iKo
        return lp, glp

    def riemannian_quantities(self, qt):
        raise NotImplementedError()

    def lagrangian_quantities(self, qt):
        raise NotImplementedError()

    def softabs_quantities(self, qt):
        raise NotImplementedError()


def kernel(sigmasq: float, beta: float, dist_div_beta: np.ndarray) -> np.ndarray:
    """The smooth kernel defining spatial correlation.

    Args:
        phis: Reparameterized aplitude of the Gaussian process kernel.
        phib: Reparameterized length scale of the Gaussian process kernel.

    Returns:
        K: The kernel.

    """
    K = sigmasq * np.exp(-dist_div_beta)
    return K

def grad_kernel(K: np.ndarray, dist_div_beta: np.ndarray) -> Tuple[np.ndarray]:
    """The Hessian of the smooth kernel defining spatial correlation with respect
    to the reparameterized model parameters.

    Args:
        K: The Gaussian process covariance.
        dist_div_beta: The pairwise distances divided by the length scale

    Returns:
        dKphis: The derivative of the covariance with respect to the
            reparameterized amplitude.
        dKphib: The derivative of the covariance with respect to the
            reparameterized length scale.

    """
    dKphis = K
    dKphib = K * dist_div_beta
    return dKphis, dKphib

def hess_kernel(K: np.ndarray, dKphib: np.ndarray, dist_div_beta: np.ndarray) -> Tuple[np.ndarray]:
    """The Hessian of the smooth kernel defining spatial correlation with respect
    to the reparameterized model parameters.

    Args:
        K: The Gaussian process covariance.
        dKphib: The derivative of the covariance with respect to the
            reparameterized length scale.
        dist_div_beta: The pairwise distances divided by the length scale

    Returns:
        ddKphis: The hessian of the kernel with respect to the reparameterized
            amplitude.
        ddKphib: The hessian of the kernel with respect to the reparameterized
            length scale.
        ddKdsdb: The hessian of the kernel with respect to the reparameterized
            length scale and the reparameterized amplitude.

    """
    ddKphis = K
    ddKdsdb = dKphib
    ddKphib = K * np.square(dist_div_beta) - ddKdsdb
    return ddKphis, ddKphib, ddKdsdb

def hyperparameter_log_posterior(sigmasq: float, beta: float, oiKo: np.ndarray, L: np.ndarray) -> float:
    logdet = 2*np.sum(np.log(np.diag(L)))
    ll = -0.5*logdet - 0.5*oiKo
    pr = prior.log_prior(sigmasq, beta)
    lp = ll + pr
    return lp

def hyperparameter_grad_log_posterior(
        phis: float,
        phib: float,
        sigmasq: float,
        beta: float,
        K: np.ndarray,
        iK: np.ndarray,
        iKo: np.ndarray,
        oiKo: np.ndarray,
        dKphis: np.ndarray,
        dKphib: np.ndarray,
        iKdKphib: np.ndarray
) -> np.ndarray:
    dphis, dphib = prior.grad_log_prior(phis, phib)
    dphis += -0.5*K.shape[0] + 0.5*oiKo
    dphib += -0.5*np.trace(iKdKphib) + 0.5*iKo@dKphib@iKo
    return np.array([dphis, dphib])

def hyperparameter_metric(
        phis: float,
        phib: float,
        K: np.ndarray,
        iKdKphib: np.ndarray
) -> np.ndarray:
    a = K.shape[0]
    d = np.trace(iKdKphib)
    c = np.trace(iKdKphib@iKdKphib)
    G = 0.5 * np.array([[a, d], [d, c]])
    H = prior.hess_log_prior(phis, phib)
    F = G - H
    return F

def hyperparameter_jac_metric(
        phis: float,
        phib: float,
        K: np.ndarray,
        iK: np.ndarray,
        dKphis: np.ndarray,
        dKphib: np.ndarray,
        iKdKphib: np.ndarray,
        dist_div_beta: np.ndarray,
):
    dH = prior.jac_hess_log_prior(phis, phib)
    ddKphis, ddKphib, ddKdsdb = hess_kernel(K, dKphib, dist_div_beta)
    bb, sb = np.hsplit(iK@np.hstack((ddKphib, ddKdsdb)), 2)
    b = iKdKphib
    b_b = b@b
    s_b_b = b_b
    # I think that the partial derivative of the metric with respect to the
    # amplitude vanishes because the amplitude cancels in the metric.
    dGs = np.array([[0.0, 0.0], [0.0, 0.0]])
    od = 0.5*np.trace(-s_b_b + bb)
    b_bb = b@bb
    dGb = np.array([[0.0, od], [od, np.trace(-b@b_b + b_bb)]])
    dG = np.array([dGs, dGb]).swapaxes(0, -1)
    return dG - dH

def base_quantities(qt, dist, o):
    phis, phib = qt
    sigmasq = np.exp(phis)
    beta = np.exp(phib)
    dist_div_beta = dist / beta
    K = kernel(sigmasq, beta, dist_div_beta)
    iK, L = solve_psd(K)
    iKo = iK@o
    oiKo = o@iKo
    return phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo

class Hyperparameters(Distribution):
    """Factory to produce the log-posterior, the gradient of the log-posterior, the
    Fisher information metric and the gradient of the Fisher information metric
    for the log-Gaussian Cox-Poisson process given the underlying Gaussian
    process.

    Parameters:
        dist: Matrix of pairwise distances.
        mu: Mean value of the Gaussian process.
        x: The Gaussian process.
        y: Count observations from the Poisson process.

    """
    def __init__(self, dist: np.ndarray, mu: float, x: np.ndarray, y: np.ndarray):
        super().__init__()
        self.dist = dist
        self.mu = mu
        self.x = x
        self.y = y

    @property
    def o(self):
        return self.x - self.mu

    def log_density(self, qt):
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, self.dist, self.o)
        lp = hyperparameter_log_posterior(sigmasq, beta, oiKo, L)
        return lp

    def sample(self):
        raise NotImplementedError()

    def forward_transform(self, q):
        qt, ildj = transforms.forward_transform(q)
        return qt, ildj

    def inverse_transform(self, qt):
        q, fldj = transforms.inverse_transform(qt)
        return q, fldj

    def hessian(self, qt):
        raise NotImplementedError()

    def euclidean_metric(self):
        Id = np.eye(2)
        return Id, Id, Id

    def riemannian_metric(self, qt):
        phis, phib = qt
        sigmasq = np.exp(phis)
        beta = np.exp(phib)
        dist_div_beta = self.dist / beta
        K = kernel(sigmasq, beta, dist_div_beta)
        iK, L = solve_psd(K)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        iKdKphib = iK@dKphib
        G = hyperparameter_metric(phis, phib, K, iKdKphib)
        return G

    def riemannian_metric_and_jacobian(self, qt):
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, self.dist, self.o)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        iKdKphib = iK@dKphib
        G = hyperparameter_metric(phis, phib, K, iKdKphib)
        dG = hyperparameter_jac_metric(phis, phib, K, iK, dKphis, dKphib, iKdKphib, dist_div_beta)
        return G, dG

    def euclidean_quantities(self, qt):
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, self.dist, self.o)
        lp = hyperparameter_log_posterior(sigmasq, beta, oiKo, L)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        iKdKphib = iK@dKphib
        glp = hyperparameter_grad_log_posterior(phis, phib, sigmasq, beta, K, iK, iKo, oiKo, dKphis, dKphib, iKdKphib)
        return lp, glp

    def riemannian_quantities(self, qt):
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, self.dist, self.o)
        lp = hyperparameter_log_posterior(sigmasq, beta, oiKo, L)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        iKdKphib = iK@dKphib
        glp = hyperparameter_grad_log_posterior(phis, phib, sigmasq, beta, K, iK, iKo, oiKo, dKphis, dKphib, iKdKphib)
        G = hyperparameter_metric(phis, phib, K, iKdKphib)
        dG = hyperparameter_jac_metric(phis, phib, K, iK, dKphis, dKphib, iKdKphib, dist_div_beta)
        return lp, glp, G, dG

    def lagrangian_quantities(self, qt):
        phis, phib, sigmasq, beta, dist_div_beta, K, iK, L, iKo, oiKo = base_quantities(qt, self.dist, self.o)
        lp = hyperparameter_log_posterior(sigmasq, beta, oiKo, L)
        dKphis, dKphib = grad_kernel(K, dist_div_beta)
        iKdKphib = iK@dKphib
        G = hyperparameter_metric(phis, phib, K, iKdKphib)
        return lp, G

    def softabs_quantities(self, qt):
        raise NotImplementedError()
