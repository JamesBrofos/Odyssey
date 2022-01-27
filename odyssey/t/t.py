from typing import Tuple

import numpy as np

from iliad.linalg import solve_psd
from iliad.statistics import rvs

from odyssey.distribution import Distribution


def sample(sqrtm: np.ndarray, dof: float) -> np.ndarray:
    """Draws samples from the multivariate Student-t distribution given square root
    of the covariance matrix and the degrees of freedom.

    Args:
        sqrtm: The square root of the covariance matrix.
        dof: The degrees of freedom.

    Returns:
        x: A sample from the multivariate Student-t distribution.

    """
    zero = np.zeros(len(sqrtm))
    y = rvs(sqrtm, zero)
    u = np.random.chisquare(dof)
    x = y * np.sqrt(dof / u)
    return x

def base_quantities(x: np.ndarray, iSigma: np.ndarray, dof: float) -> Tuple[np.ndarray]:
    """Computes the preconditioning effect of the inverse covariance matrix on the
    input and the inner product regularized by the degrees of freedom.

    Args:
        x: A point in Euclidean space.
        iSigma: The inverse of the covariance matrix.
        dof: The degrees of freedom.

    Returns:
        iSigma_x: The preconditioning of `x` by the inverse covariance matrix.
        ip: The Mahalanobis inner product regularized by the degrees of freedom.

    """
    iSigma_x = iSigma@x
    ip = 1 + x@iSigma_x / dof
    return iSigma_x, ip

def log_posterior(ip: np.ndarray, dof: float, n: int) -> float:
    """Computes the log-density of the Student-t distribution given the inner
    product, the degrees of freedom, and the dimensionality of the space.

    Args:
        ip: The Mahalanobis inner product regularized by the degrees of freedom.
        dof: The degrees of freedom.
        n: The dimensionality of the Student-t distribution.

    Returns:
        lp: The log-density.

    """
    lp = -0.5*(dof + n)*np.log(ip)
    return lp

def grad_log_posterior(iSigma_x: np.ndarray, ip: np.ndarray, dof: float, n: int) -> np.ndarray:
    """Computes the gradient of the log-density of the Student-t distribution with
    respect to a location in Euclidean space.

    Args:
        iSigma_x: The preconditioning of `x` by the inverse covariance matrix.
        ip: The Mahalanobis inner product regularized by the degrees of freedom.
        dof: The degrees of freedom.
        n: The dimensionality of the Student-t distribution.

    Returns:
        glp: The gradient of the log-posterior.

    """
    glp = -(dof + n) / ip * iSigma_x / dof
    return glp

def hessian(iSigma: np.ndarray, iSigma_x: np.ndarray, ip: np.ndarray, dof: float, n: int) -> np.ndarray:
    """Computes the Hessian of the log-density of the Student-t distribution with
    respect to a location in Euclidean space.

    Args:
        iSigma_x: The preconditioning of `x` by the inverse covariance matrix.
        ip: The Mahalanobis inner product regularized by the degrees of freedom.
        dof: The degrees of freedom.
        n: The dimensionality of the Student-t distribution.

    Returns:
        H: The Hessian of the log-posterior.

    """
    k = dof + n
    H = (
        -k / ip * iSigma / dof
        + 2*k / dof**2 / ip**2 * np.outer(iSigma_x, iSigma_x)
    )
    return H

def jac_hessian(x: np.ndarray, iSigma: np.ndarray, iSigma_x: np.ndarray, ip: np.ndarray, dof: float, n: int) -> np.ndarray:
    """Computes the Jacobian of the Hessian of the log-density of the Student-t
    distribution with respect to a location in Euclidean space.

    Args:
        x: The location at which to compute the Jacobian of the Hessian.
        iSigma_x: The preconditioning of `x` by the inverse covariance matrix.
        ip: The Mahalanobis inner product regularized by the degrees of freedom.
        dof: The degrees of freedom.
        n: The dimensionality of the Student-t distribution.

    Returns:
        dH: The Jacobian of the Hessian of the log-posterior.

    """
    k = dof + n
    Id = np.eye(n)
    o = np.einsum('ki,j->ijk', Id, x) + np.einsum('i,kj->ijk', x, Id)
    o = np.swapaxes(o, 0, -1)
    o = np.swapaxes(iSigma@o@iSigma, 0, -1)
    dH = (
        2*k / ip**2 * np.einsum('ij,k->ijk', iSigma, iSigma_x) / dof**2
        + 2*k / dof**2 / ip**2 * o
        - 8*k / dof**3 / ip**3 * np.einsum('i,j,k->ijk', iSigma_x, iSigma_x, iSigma_x)
    )
    return dH

def metric(iSigma: np.ndarray, iSigma_x: np.ndarray, ip: np.ndarray, dof: float, n: int) -> np.ndarray:
    """Computes a Riemannian metric for the multivariate Student-t distribution.
    This is done by taking the positive definite part of the negative Hessian
    of the log-density.

    Args:
        iSigma_x: The preconditioning of `x` by the inverse covariance matrix.
        ip: The Mahalanobis inner product regularized by the degrees of freedom.
        dof: The degrees of freedom.
        n: The dimensionality of the Student-t distribution.

    Returns:
        G: The Riemannian metric.

    """
    k = dof + n
    G = k / ip * iSigma / dof
    return G

def jac_metric(x: np.ndarray, iSigma: np.ndarray, iSigma_x: np.ndarray, ip: np.ndarray, dof: float, n: int) -> np.ndarray:
    """Computes the Jacobian of the Riemannian metric for the multivariate
    Student-t distribution.

    Args:
        x: The location at which to compute the Jacobian of the Riemannian metric.
        iSigma_x: The preconditioning of `x` by the inverse covariance matrix.
        ip: The Mahalanobis inner product regularized by the degrees of freedom.
        dof: The degrees of freedom.
        n: The dimensionality of the Student-t distribution.

    Returns:
        dG: The Jacobian of the Riemannian metric.

    """
    k = dof + n
    dG = -2*k / ip**2 * np.einsum('ij,k->ijk', iSigma, iSigma_x) / dof**2
    return dG

class T(Distribution):
    """Constructs the posterior distribution corresponding to a multivariate
    Student-t distribution given the covariance matrix and the degrees of
    freedom.

    Parameters:
        Sigma: The covariance matrix of the Student-t distribution.
        dof: The degrees of freedom of the Student-t distribution.

    """
    def __init__(self, Sigma: np.ndarray, dof: float):
        self.Sigma = Sigma
        self.dof = dof
        self.iSigma, self.L = solve_psd(Sigma)
        self.n = len(Sigma)

    def log_density(self, qt):
        iSigma_x, ip = base_quantities(qt, self.iSigma, self.dof)
        lp = log_posterior(ip, self.dof, self.n)
        return lp

    def sample(self):
        x = sample(self.L, self.dof)
        return x

    def forward_transform(self, q):
        return q, 0.0

    def inverse_transform(self, qt):
        return qt, 0.0

    def hessian(self, qt):
        iSigma_x, ip = base_quantities(qt, self.iSigma, self.dof)
        H = hessian(self.iSigma, iSigma_x, ip, self.dof, self.n)
        return H

    def euclidean_metric(self):
        Id = np.eye(self.n)
        return Id, Id, Id

    def riemannian_metric(self, qt):
        iSigma_x, ip = base_quantities(qt, self.iSigma, self.dof)
        G = metric(self.iSigma, iSigma_x, ip, self.dof, self.n)
        return G

    def riemannian_metric_and_jacobian(self, qt):
        iSigma_x, ip = base_quantities(qt, self.iSigma, self.dof)
        G = metric(self.iSigma, iSigma_x, ip, self.dof, self.n)
        dG = jac_metric(qt, self.iSigma, iSigma_x, ip, self.dof, self.n)
        return G, dG

    def euclidean_quantities(self, qt):
        iSigma_x, ip = base_quantities(qt, self.iSigma, self.dof)
        lp = log_posterior(ip, self.dof, self.n)
        glp = grad_log_posterior(iSigma_x, ip, self.dof, self.n)
        return lp, glp

    def riemannian_quantities(self, qt):
        iSigma_x, ip = base_quantities(qt, self.iSigma, self.dof)
        lp = log_posterior(ip, self.dof, self.n)
        glp = grad_log_posterior(iSigma_x, ip, self.dof, self.n)
        G = metric(self.iSigma, iSigma_x, ip, self.dof, self.n)
        dG = jac_metric(qt, self.iSigma, iSigma_x, ip, self.dof, self.n)
        return lp, glp, G, dG

    def lagrangian_quantities(self, qt):
        iSigma_x, ip = base_quantities(qt, self.iSigma, self.dof)
        lp = log_posterior(ip, self.dof, self.n)
        G = metric(self.iSigma, iSigma_x, ip, self.dof, self.n)
        return lp, G

    def softabs_quantities(self, qt):
        iSigma_x, ip = base_quantities(qt, self.iSigma, self.dof)
        lp = log_posterior(ip, self.dof, self.n)
        glp = grad_log_posterior(iSigma_x, ip, self.dof, self.n)
        H = hessian(self.iSigma, iSigma_x, ip, self.dof, self.n)
        dH = jac_hessian(qt, self.iSigma, iSigma_x, ip, self.dof, self.n)
        return lp, glp, H, dH
