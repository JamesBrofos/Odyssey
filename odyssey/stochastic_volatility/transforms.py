from typing import Tuple

import numpy as np


def forward_transform(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Transform parameters frm their constrained representation to their
    unconstrained representation.

    Args:
        q: The constrained parameter representation.

    Returns:
        qt: The unconstrained parameter representation.
        ildj: The logarithm of the Jacobian determinant of the inverse
            transformation.

    """
    sigma, phi, beta = q
    gamma, alpha = np.log(sigma), np.arctanh(phi)
    qt = np.array([gamma, alpha, beta])
    ildj = gamma + np.log(1.0 - np.square(phi))
    return qt, ildj

def inverse_transform(qt: np.ndarray) -> Tuple[np.ndarray, float]:
    """Transform parameters from their unconstrained representation to their
    constrained representation.

    Args:
        qt: The unconstrained parameter representation.

    Returns:
        q: The constrained parameter representation.
        fldj: The logarithm of the Jacobian determinant of the forward
            transformation.

    """
    gamma, alpha, beta = qt
    sigma, phi = np.exp(gamma), np.tanh(alpha)
    q = np.array([sigma, phi, beta])
    fldj = -(gamma + np.log(1.0 - np.square(phi)))
    return q, fldj
