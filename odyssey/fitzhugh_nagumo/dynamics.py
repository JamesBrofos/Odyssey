from typing import Tuple

import numba as nb
import numpy as np


@nb.njit
def fn_dynamics(state: np.ndarray, t: float, *args: Tuple) -> np.ndarray:
    """Definition of the dynamics for the Fitzhugh-Nagumo differential equation
    model.

    Args:
        state: The current state of the system.
        t: The current time of the system.
        args: Parameters of the Fitzhugh-Nagumo model.

    Returns:
        ds: The time derivative of the state.

    """
    a, b, c = args
    v, r = state[0], state[1]
    ds = np.array([c * (v - np.power(v, 3.0) / 3.0 + r), -(v - a + b * r) / c])
    return ds

@nb.njit
def fn_sensitivity(state: np.ndarray, t: float, *args: Tuple) -> np.ndarray:
    """The sensitivity of the states of the Fitzhugh-Nagumo model with respect to
    the model parameters at each time.

    Args:
        state: The expanded state of the system, including states representing
            the sensitivity of the state with respect to the parameters.
        t: The current time of the system.
        args: Parameters of the Fitzhugh-Nagumo model.

    Returns:
        ret: The time derivative of the expanded state.

    """
    # Parameters of the Fitzhugh-Nagumo model.
    a, b, c = args
    # Compute the dynamics of the Fitzhugh-Nagumo differential equation as we
    # progress through time.
    s = state[:2]
    ds = fn_dynamics(s, t, a, b, c)
    v, r = s[0], s[1]
    # Compute the state sensitivities.
    csq = np.square(c)
    fx = np.array([[c - c*np.square(v), c], [-1.0 / c,  -b / c]])
    fo = np.array([[0.0, 0.0, v - np.power(v, 3.0) / 3.0 + r],
                   [1.0 / c, -r / c, (v - a + b*r) / csq]])

    # Here is the layout of the sensitivities with the index of these
    # quantities in the output of the ODE solver shown in parentheses.
    #
    # Index      Interpretation
    # --------------------------------------------------
    # 0 (2)      Sensitivity of `v` with respect to `a`.
    # 1 (3)      Sensitivity of `r` with respect to `a`.
    # 2 (4)      Sensitivity of `v` with respect to `b`.
    # 3 (5)      Sensitivity of `r` with respect to `b`.
    # 4 (6)      Sensitivity of `v` with respect to `c`.
    # 5 (7)      Sensitivity of `r` with respect to `c`.
    # --------------------------------------------------
    va, ra, vb, rb, vc, rc = state[2:8]
    vsq = np.square(v)
    vsq_m_one = vsq - 1.0
    neg_b_div_c = -b/c
    dS = np.array([
        (-c*vsq_m_one)*va + c*ra,
        -va/c + neg_b_div_c*ra + 1/c,
        (-c*vsq_m_one)*vb + c*rb,
        -vb/c + neg_b_div_c*rb- r/c,
        (-c*vsq_m_one)*vc + (c)*rc + v - v**3/3 + r,
        -vc/c + neg_b_div_c*rc + (v - a + r*b)/csq
    ])
    # Stack the state dynamics and the sensitivity dynamics.
    ret = np.hstack((ds, dS))
    return ret

@nb.njit
def fn_higher_sensitivity(state: np.ndarray, t: float, *args: Tuple) -> np.ndarray:
    """The second order sensitivities of the states of the Fitzhugh-Nagumo model
    with respect to the model parameters at each time.

    Args:
        state: The expanded state of the system, including states representing
            the sensitivity of the state with respect to the parameters.
        t: The current time of the system.
        args: Parameters of the Fitzhugh-Nagumo model.

    Returns:
        ret: The time derivative of the expanded state.

    """
    # Unpack variables and computed repeated quantities.
    a, b, c, correct = args
    v, r = state[:2]
    va, ra, vb, rb, vc, rc = state[2:8]
    vaa, raa, vab, rab, vac, rac, vbb, rbb, vbc, rbc, vcc, rcc = state[8:20]
    two_c_v = 2*c*v
    vsq = np.square(v)
    vsq_m_one = vsq - 1.0
    neg_b_div_c = -b/c
    csq = np.square(c)

    # Decide whether or not to break HMC by invalidating the symmetry of
    # partial derivatives.
    if correct:
        d1 = -vsq_m_one*va - two_c_v*va*vc - c*vsq_m_one*vac + ra + c*rac
        d2 = -vsq_m_one*vb - two_c_v*vc*vb - c*vsq_m_one*vbc + rb + c*rbc
        d3 = -vsq_m_one*vc - two_c_v*vc*vc - c*vsq_m_one*vcc + 2*rc + c*rcc + vc - vsq*vc
    else:
        d1 = -two_c_v*vc*va + 1 - vsq*va - c*vsq_m_one*vac + ra + c*rac
        d2 = -two_c_v*vc*vb + 1 - vsq*vb - c*vsq_m_one*vbc + rb + c*rbc
        d3 = -two_c_v*vc*vc + 1 - vsq*vc - c*vsq_m_one*vcc + 1 - vsq*vc + 2*rc + c*rcc

    # Index        Interpretation
    # ------------------------------------------------------------------------
    # 0 (8)        Sensitivity of `v` with respect to `a` with respect to `a`.
    # 1 (9)        Sensitivity of `r` with respect to `a` with respect to `a`.
    # 2 (10)       Sensitivity of `v` with respect to `a` with respect to `b`.
    # 3 (11)       Sensitivity of `r` with respect to `a` with respect to `b`.
    # 4 (12)       Sensitivity of `v` with respect to `a` with respect to `c`.
    # 5 (13)       Sensitivity of `r` with respect to `a` with respect to `c`.
    # 6 (14)       Sensitivity of `v` with respect to `b` with respect to `b`.
    # 7 (15)       Sensitivity of `r` with respect to `b` with respect to `b`.
    # 8 (16)       Sensitivity of `v` with respect to `b` with respect to `c`.
    # 9 (17)       Sensitivity of `r` with respect to `b` with respect to `c`.
    # 10 (18)      Sensitivity of `v` with respect to `c` with respect to `c`.
    # 11 (19)      Sensitivity of `r` with respect to `c` with respect to `c`.
    # ------------------------------------------------------------------------
    dsens = np.array([
        -two_c_v*va*va - c*vsq_m_one*vaa + c*raa,
        -vaa/c - b/c*raa,
        -two_c_v*vb*va - c*vsq_m_one*vab + c*rab,
        -vab/c - ra/c - b/c*rab,
        d1,
        va/csq - vac/c + b/csq*ra - b/c*rac - 1/csq,
        -two_c_v*vb*vb - c*vsq_m_one*vbb + c*rbb,
        -vbb/c - rb/c - b/c*rbb - rb/c,
        d2,
        vb/csq - vbc/c + b/csq*rb - b/c*rbc - rc/c + r/csq,
        d3,
        vc/csq - vcc/c + vc/csq + b/csq*rc - b/c*rcc + b/csq*rc - (2*(v - a + r*b))/c**3
    ])
    # Stack the time derivative of the sensitivity dynamics.
    sens = fn_sensitivity(state, t, a, b, c)
    ret = np.hstack((sens, dsens))
    return ret
