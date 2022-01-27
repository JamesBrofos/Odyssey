import abc
from typing import Tuple

import numpy as np


class Distribution(abc.ABC):
    """Abstract base class implementing the methods required for sampling from
    probability distributions. At the very least, we should be able to compute
    the log-density of the probability distribution, known up to an additive
    constant. In order to employ Hamiltonian Monte Carlo, we require the
    gradient of the log-density. For methods inspired by geometric concepts,
    such as Riemannian Manifold Hamiltonian Monte Carlo or Lagrangian Monte
    Carlo, we additionally require a metric and the Jacobian of the metric.

    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def log_density(self, qt: np.ndarray):
        """Computes the log-density of the distribution at the given input.

        Args:
            qt: The transformed variables.

        Returns:
            ld: The log-density of the distribution.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self):
        """Returns an independent random sample from the distribution.

        Returns:
            q: An untransformed sample from the target distribution.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_transform(self, q: np.ndarray) -> Tuple[np.ndarray, float]:
        """Computes the forward transformation of the variables to be samples and the
        Jacobian determinant of the transformation. This can be useful for
        transforming variables to unconstrained representations.

        Args:
            q: The original variables.

        Returns:
            qt: The transformed variables.
            ildj: The Jacobian log-determinant of the inverse transformation.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform(self, qt: np.ndarray) -> Tuple[np.ndarray, float]:
        """Computes the inverse transformation, which maps unconstrained
        representations to their constrained counterparts.

        Args:
            qt: The transformed variables.

        Returns:
            q: The original variables.
            fldj: The Jacobian log-determinant of the forward transformation.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def hessian(self, qt: np.ndarray) -> np.ndarray:
        """The Hessian of the log-density of the distribution.

        Args:
            qt: The transformed variables.

        Returns:
            H: The Hessian of the log-density.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def euclidean_metric(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """When employing Euclidean HMC, the metric is constant. This method computes a
        constant metric to employ; in addition to the metric we also compute
        the inverse of the metric and its matrix square root. The square root
        is the Cholesky factor.

        Returns:
            G: The constant metric.
            Gchol: The Cholesky decomposition of the constant metric.
            Ginv: The inverse of the constant metric.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def riemannian_metric(self, qt: np.ndarray) -> np.ndarray:
        """Computes the Riemannian metric, which may depend on the position variable.

        Args:
            qt: The transformed variables.

        Returns:
            G: The Riemannian metric.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def riemannian_metric_and_jacobian(self, qt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the Riemannian metric and the Jacobian of the Riemannian metric.
        These quantities can be used in resolving the implicit update to
        position using Newton's method in the generalized leapfrog algorithm.

        Args:
            qt: The transformed variables.

        Returns:
            G: The Riemannian metric.
            dG: The Jacobian of the Riemannian metric.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def euclidean_quantities(self, qt: np.ndarray) -> Tuple[float, np.ndarray]:
        """Function to compute the variables that are required by HMC on flat manifolds
        (i.e. those with constant Euclidean metrics). The required quantities
        are the log-density of the distribution and the gradient of the
        log-density.

        Args:
            qt: The transformed variables.

        Returns:
            ld: The log-density of the distribution.
            grad_ld: The gradient of the log-density of the distribution.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def riemannian_quantities(self, qt: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Function to compute the variables that are required by the Riemannian
        variants of HMC which incorporate local geometry. This function
        computes the log-density of the distribution, the gradient of the
        log-density, the Riemannian metric to employ, and the Jacobian of the
        Riemannian metric.

        Args:
            qt: The transformed variables.

        Returns:
            ld: The log-density of the distribution.
            grad_ld: The gradient of the log-density of the distribution.
            G: The Riemannian metric.
            dG: The Jacobian of the Riemannian metric.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def lagrangian_quantities(self, qt: np.ndarray) -> Tuple[float, np.ndarray]:
        """Function to compute quantities that may be used in the Lagranian Monte Carlo
        algorithm. This function computes the log-density of the distribution
        and the Riemannian metric. This function may only be called by
        Lagrangian Monte Carlo implementations that invert the order of
        integration.

        Args:
            qt: The transformed variables.

        Returns:
            ld: The log-density of the distribution.
            G: The Riemannian metric.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def softabs_quantities(self, qt: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Computes the quantities required by the implementation of RMHMC that
        involves the SoftAbs Riemannian metric. The SoftAbs metric is a smooth
        transformation of the Hessian of the log-density. Therefore, in
        addition to the log-density and the gradient of the log-density, we
        also compute the Hessian and the Jacobian of the Hessian.

        Args:
            qt: The transformed variables.

        Returns:
            ld: The log-density of the distribution.
            grad_ld: The gradient of the log-density of the distribution.
            H: The Hessian of the log-density.
            dH: The Jacobian of the Hessian of the log-density.

        """
        raise NotImplementedError()
