from typing import Tuple

import numpy as np


def forward_transform(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Transform parameters from their constrained representation to their
    unconstrained representation.

    Args:
        q: The constrained parameter representation.

    Returns:
        qt: The unconstrained parameter representation.
        ildj: The logarithm of the Jacobian determinant of the inverse
            transformation.

    """
    sigmasq, beta = q
    phis, phib = np.log(sigmasq), np.log(beta)
    qt = np.array([phis, phib])
    ildj = phis + phib
    return qt, ildj

def inverse_transform(qt: np.ndarray) -> Tuple[np.ndarray, float]:
    """Transform parameters from their unconstrained representation to their
    constrained representation.

    Args:
        qt: The unconstrained parameter representation.

    Returns:
        q: The constrained parameter representation.

    """
    phis, phib = qt
    sigmasq, beta = np.exp(phis), np.exp(phib)
    q = np.array([sigmasq, beta])
    fldj = -(phis + phib)
    return q, fldj
