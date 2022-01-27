from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution


def log_density(
        x: np.ndarray,
        v: float,
        num_dims: int,
        s: float,
        ssq: float,
        x_div_ssq: np.ndarray,
        ssx_div_ssq: np.ndarray
) -> float:
    """Log-density of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        num_dims: The dimensionality of the non-hierarchical parameters.
        s: The standard deviation of the non-hierarchical parameters.
        ssq: The variance of the non-hierarchical parameters.
        x_div_ssq: The non-hierarchical parameters divided by their variance.
        ssx_div_ssq: The sum-of-squares of the non-hierarchical parameters
            divided by their variance.

    Returns:
        ld: The log-density of Neal's funnel.

    """
    ldx = -0.5*ssx_div_ssq - 0.5*num_dims*np.log(2*np.pi) - num_dims*np.log(s)
    ldv = -0.5*np.square(v) / 9.0 - 0.5*np.log(2*np.pi) - np.log(3.0)
    ld = ldv + ldx
    return ld

def grad_log_density(
        x: np.ndarray,
        v: float,
        num_dims: int,
        s: float,
        ssq: float,
        x_div_ssq: np.ndarray,
        ssx_div_ssq: np.ndarray
) -> np.ndarray:
    """Gradient of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        num_dims: The dimensionality of the non-hierarchical parameters.
        s: The standard deviation of the non-hierarchical parameters.
        ssq: The variance of the non-hierarchical parameters.
        x_div_ssq: The non-hierarchical parameters divided by their variance.
        ssx_div_ssq: The sum-of-squares of the non-hierarchical parameters
            divided by their variance.

    Returns:
        out: The gradient of the log-density of Neal's funnel.

    """
    glp = np.hstack([
        -x_div_ssq,
        -v/9.0 - 0.5 * ssx_div_ssq + 0.5 * num_dims])
    return glp

def hess_log_density(
        x: np.ndarray,
        v: float,
        num_dims: int,
        s: float,
        ssq: float,
        x_div_ssq: np.ndarray,
        ssx_div_ssq: np.ndarray
) -> np.ndarray:
    """Hessian of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        num_dims: The dimensionality of the non-hierarchical parameters.
        s: The standard deviation of the non-hierarchical parameters.
        ssq: The variance of the non-hierarchical parameters.
        x_div_ssq: The non-hierarchical parameters divided by their variance.
        ssx_div_ssq: The sum-of-squares of the non-hierarchical parameters
            divided by their variance.

    Returns:
        H: The Hessian of the log-density of Neal's funnel.

    """
    dvdv = -1.0/9.0 - 0.5 * ssx_div_ssq
    dvdx = -x_div_ssq
    dxdx = -np.eye(num_dims) / ssq
    H = np.vstack((
        np.hstack((dxdx, dvdx[..., np.newaxis])),
        np.hstack((dvdx, dvdv))
    ))
    return H

def jac_hess_log_density(
        x: np.ndarray,
        v: float,
        num_dims: int,
        s: float,
        ssq: float,
        x_div_ssq: np.ndarray,
        ssx_div_ssq: np.ndarray
) -> np.ndarray:
    """Jacobian of the Hessian of Neal's funnel distribution.

    Args:
        x: Conditionally-independent samples from Neal's funnel.
        v: Variance of Neal's funnel.
        num_dims: The dimensionality of the non-hierarchical parameters.
        s: The standard deviation of the non-hierarchical parameters.
        ssq: The variance of the non-hierarchical parameters.
        x_div_ssq: The non-hierarchical parameters divided by their variance.
        ssx_div_ssq: The sum-of-squares of the non-hierarchical parameters
            divided by their variance.

    Returns:
        dH: The higher-order derivatives of the log-density of Neal's funnel.

    """
    dvdvdv = -0.5 * ssx_div_ssq
    dxdxdv = -np.eye(num_dims) / ssq
    dvdvdx = -x_div_ssq

    Z = np.zeros((num_dims, num_dims, num_dims))
    da = np.concatenate((Z, dxdxdv[..., np.newaxis]), axis=-1)
    mm = np.hstack((dxdxdv, dvdvdx[..., np.newaxis]))
    rr = np.vstack((mm, np.hstack((dvdvdx, dvdvdv))))
    db = np.concatenate((da, mm[:, np.newaxis]), axis=1)
    dH = np.concatenate((db, rr[np.newaxis]))
    return dH

separator = lambda q: (q[:-1], q[-1])

def base_quantities(q):
    x, v = separator(q)
    s = np.exp(-0.5*v)
    ssq = np.square(s)
    x_div_ssq = x / ssq
    ssx_div_ssq = np.square(x).sum() / ssq
    return x, v, s, ssq, x_div_ssq, ssx_div_ssq

class NealFunnel(Distribution):
    """Posterior factory function for Neal's funnel distribution. This is a density
    that exhibits extreme variation in the dimensions and may therefore present
    a challenge for leapfrog integrators. Therefore, the posterior is also
    equipped with the softabs metric which adapts the generalized leapfrog
    integrator to the local geometry. The softabs metric is a transformation of
    the Hessian to make it positive definite.

    It is a curious attribute of this posterior that for a larger size of the
    posterior, larger step-sizes are better behaved that for a smaller size of
    the posterior.

    """
    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims

    def log_density(self, qt):
        x, v, s, ssq, x_div_ssq, ssx_div_ssq = base_quantities(qt)
        lp = log_density(x, v, self.num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        return lp

    def sample(self):
        v = np.random.normal(0.0, 3.0)
        s = np.exp(-0.5*v)
        x = np.random.normal(0.0, s, size=(self.num_dims, ))
        return x, v

    def forward_transform(self, q):
        return q, 0.0

    def inverse_transform(self, qt):
        return qt, 0.0

    def hessian(self, qt):
        x, v, s, ssq, x_div_ssq, ssx_div_ssq = base_quantities(qt)
        H = hess_log_density(x, v, self.num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        return H

    def euclidean_metric(self):
        Id = np.eye(self.num_dims + 1)
        return Id, Id, Id

    def riemannian_metric(self, qt):
        raise NotImplementedError()

    def riemannian_metric_and_jacobian(self, qt):
        raise NotImplementedError()

    def euclidean_quantities(self, qt):
        x, v, s, ssq, x_div_ssq, ssx_div_ssq = base_quantities(qt)
        lp = log_density(x, v, self.num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        glp = grad_log_density(x, v, self.num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        return lp, glp

    def riemannian_quantities(self, qt):
        raise NotImplementedError()

    def lagrangian_quantities(self, qt):
        raise NotImplementedError()

    def softabs_quantities(self, qt):
        x, v, s, ssq, x_div_ssq, ssx_div_ssq = base_quantities(qt)
        lp = log_density(x, v, self.num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        glp = grad_log_density(x, v, self.num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        H = hess_log_density(x, v, self.num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        dH = jac_hess_log_density(x, v, self.num_dims, s, ssq, x_div_ssq, ssx_div_ssq)
        return lp, glp, H, dH
