import numpy as np
import scipy.linalg as spla

from iliad.statistics import rvs
from odyssey.distribution import Distribution


class Gaussian(Distribution):
    """Implements sampling from a multivariate normal distribution. Constructs
    functions for the log-density of the normal distribution and for the
    gradient of the log-density.

    Parameters:
        mu: The mean of the multivariate normal distribution.
        Sigma: The covariance matrix of the multivariate normal distribution.

    """
    def __init__(self, mu: np.ndarray, Sigma: np.ndarray):
        super().__init__()
        self.mu = np.atleast_1d(mu)
        self.Sigma = np.atleast_2d(Sigma)
        self.n = len(self.mu)
        self.L = spla.cholesky(self.Sigma)
        self.iL = spla.solve_triangular(self.L, np.eye(self.n))
        self.logdet = 2.0 * np.sum(np.log(np.diag(self.L)))
        self.iSigma = self.iL@self.iL.T

    def log_density(self, qt):
        o = qt - self.mu
        glp = -o@self.iSigma
        maha = np.sum(glp*o, axis=-1)
        lp = -0.5*self.n*np.log(2.0*np.pi) - 0.5*self.logdet + 0.5*maha
        return lp

    def sample(self):
        x = rvs(self.L, self.mu)
        return x

    def forward_transform(self, q):
        return q, 0.0

    def euclidean_metric(self):
        Id = np.eye(self.n)
        return Id, Id, Id

    def riemannian_metric(self, qt):
        raise NotImplementedError()

    def riemannian_metric_and_jacobian(self, qt):
        raise NotImplementedError()

    def inverse_transform(self, qt):
        return qt, 0.0

    def hessian(self, qt):
        raise NotImplementedError()

    def euclidean_quantities(self, qt):
        o = qt - self.mu
        glp = -o@self.iSigma
        maha = np.sum(glp*o, axis=-1)
        lp = -0.5*self.n*np.log(2.0*np.pi) - 0.5*self.logdet + 0.5*maha
        return lp, glp

    def riemannian_quantities(self, qt):
        raise NotImplementedError()

    def lagrangian_quantities(self, qt):
        raise NotImplementedError()

    def softabs_quantities(self, qt):
        raise NotImplementedError()
