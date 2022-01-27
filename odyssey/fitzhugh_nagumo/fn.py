from typing import Callable, Optional, Tuple

import numpy as np
import scipy.integrate

from odyssey.distribution import Distribution

from .dynamics import fn_dynamics, fn_sensitivity, fn_higher_sensitivity


def generate_data(
        state: np.ndarray,
        t: np.ndarray,
        sigma: float,
        a: float=0.2,
        b: float=0.2,
        c: float=3.0,
        rtol: float=None,
        atol: float=None,
        hmax: float=0.0,
        hmin: float=0.0,
        mxstep: int=0
) -> np.ndarray:
    """Generate random observations from the Fitzhugh-Nagumo model.

    Args:
        state: The current state of the system.
        t: The current time of the system.
        sigma: The noise level of the system.
        a: Parameter of the Fitzhugh-Nagumo model.
        b: Parameter of the Fitzhugh-Nagumo model.
        c: Parameter of the Fitzhugh-Nagumo model.
        rtol: Relative error tolerance of the numerical integrator.
        atol: Absolute error tolerance of the numerical integrator.
        hmax: Maximum integration step-size.
        mxstep: Maximum number of internal integration steps.

    Returns:
        y: Noise-corrupted observations of the Fitzhugh-Nagumo model.

    """
    y = scipy.integrate.odeint(fn_dynamics, state, t, (a, b, c),
                               rtol=rtol,
                               atol=atol,
                               hmax=hmax,
                               hmin=hmin,
                               mxstep=mxstep)
    y += np.random.normal(0.0, sigma, size=y.shape)
    return y

def six_to_nine(a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """Helper function to stack matrices with redundancy."""
    return np.array([
        [a, b, c],
        [b, d, e],
        [c, e, f]
    ])

def log_posterior(y: np.ndarray, sigma: float, sens: np.ndarray, a: float, b: float, c: float) -> float:
    yh = sens[:, :2]
    ll = -0.5*np.sum(np.square((y - yh) / sigma)) - np.log(sigma) - 0.5*np.log(2*np.pi)
    lpa = -0.5*a**2
    lpb = -0.5*b**2
    lpc = -0.5*c**2
    lp = ll + lpa + lpb + lpc
    return lp

def grad_log_posterior(y: np.ndarray, sigmasq: float, sens: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    yh = sens[:, :2]
    dsa, dsb, dsc = sens[:, 2:4], sens[:, 4:6], sens[:, 6:8]
    r = (y - yh) / sigmasq
    da = np.sum(r*dsa) - a
    db = np.sum(r*dsb) - b
    dc = np.sum(r*dsc) - c
    glp = np.array([da, db, dc])
    return glp

def metric(sigmasq: float, Id: np.ndarray, sens: np.ndarray) -> np.ndarray:
    dsa, dsb, dsc = sens[:, 2:4], sens[:, 4:6], sens[:, 6:8]
    Ga = dsa[:, 0]@dsa[:, 0] + dsa[:, 1]@dsa[:, 1]
    Gb = dsa[:, 0]@dsb[:, 0] + dsa[:, 1]@dsb[:, 1]
    Gc = dsa[:, 0]@dsc[:, 0] + dsa[:, 1]@dsc[:, 1]
    Gd = dsb[:, 0]@dsb[:, 0] + dsb[:, 1]@dsb[:, 1]
    Ge = dsb[:, 0]@dsc[:, 0] + dsb[:, 1]@dsc[:, 1]
    Gf = dsc[:, 0]@dsc[:, 0] + dsc[:, 1]@dsc[:, 1]
    G = six_to_nine(Ga, Gb, Gc, Gd, Ge, Gf)
    G /= sigmasq
    # Add in the negative Hessian of the log-prior.
    G += Id
    return G

def grad_metric(sigmasq: float, sens: np.ndarray) -> np.ndarray:
    S, dS = sens[:, 2:8], sens[:, 8:]
    dG = np.zeros((3, 3, 3))

    # Derivatives with respect to `a`.
    dGaa = 2*dS[:, 0]@S[:, 0] + 2*dS[:, 1]@S[:, 1]
    dGab = dS[:, 0]@S[:, 2] + S[:, 0]@dS[:, 2] + dS[:, 1]@S[:, 3] + S[:, 1]@dS[:, 3]
    dGac = dS[:, 0]@S[:, 4] + S[:, 0]@dS[:, 4] + dS[:, 1]@S[:, 5] + S[:, 1]@dS[:, 5]
    dGad = 2*dS[:, 2]@S[:, 2] + 2*dS[:, 3]@S[:, 3]
    dGae = dS[:, 2]@S[:, 4] + S[:, 2]@dS[:, 4] + dS[:, 3]@S[:, 5] + S[:, 3]@dS[:, 5]
    dGaf = 2*dS[:, 4]@S[:, 4] + 2*dS[:, 5]@S[:, 5]
    dGa = six_to_nine(dGaa, dGab, dGac, dGad, dGae, dGaf)

    # Derivatives with respect to `b`.
    dGba = 2*dS[:, 2]@S[:, 0] + 2*dS[:, 3]@S[:, 1]
    dGbb = dS[:, 2]@S[:, 2] + S[:, 0]@dS[:, 6] + dS[:, 3]@S[:, 3] + S[:, 1]@dS[:, 7]
    dGbc = dS[:, 2]@S[:, 4] + S[:, 0]@dS[:, 8] + dS[:, 3]@S[:, 5] + S[:, 1]@dS[:, 9]
    dGbd = 2*dS[:, 6]@S[:, 2] + 2*dS[:, 7]@S[:, 3]
    dGbe = dS[:, 6]@S[:, 4] + S[:, 2]@dS[:, 8] + dS[:, 7]@S[:, 5] + S[:, 3]@dS[:, 9]
    dGbf = 2*dS[:, 8]@S[:, 4] + 2*dS[:, 9]@S[:, 5]
    dGb = six_to_nine(dGba, dGbb, dGbc, dGbd, dGbe, dGbf)

    # Derivatives with respect to `c`.
    dGca = 2*dS[:, 4]@S[:, 0] + 2*dS[:, 5]@S[:, 1]
    dGcb = dS[:, 4]@S[:, 2] + S[:, 0]@dS[:, 8] + dS[:, 5]@S[:, 3] + S[:, 1]@dS[:, 9]
    dGcc = dS[:, 4]@S[:, 4] + S[:, 0]@dS[:, 10] + dS[:, 5]@S[:, 5] + S[:, 1]@dS[:, 11]
    dGcd = 2*dS[:, 8]@S[:, 2] + 2*dS[:, 9]@S[:, 3]
    dGce = dS[:, 8]@S[:, 4] + S[:, 2]@dS[:, 10] + dS[:, 9]@S[:, 5] + S[:, 3]@dS[:, 11]
    dGcf = 2*dS[:, 10]@S[:, 4] + 2*dS[:, 11]@S[:, 5]
    dGc = six_to_nine(dGca, dGcb, dGcc, dGcd, dGce, dGcf)

    # Stack the component matrices.
    dG = np.array([dGa, dGb, dGc]).swapaxes(0, -1)
    dG /= sigmasq
    return dG

class FitzhughNagumo(Distribution):
    """Factory function that yields functions to compute the log-posterior of the
    Fitzhugh-Nagumo model, the gradient of the log-posterior, the Fisher
    information metric and the gradient of the Fisher information metric.

    Parameters:
        state: Initial state of the Fitzhugh-Nagumo dynamics.
        y: Observations from the Fitzhugh-Nagumo model.
        t: Time points at which observations from the Fitzhugh-Nagumo model
            were collected.
        sigma: Standard deviation of noise.
        rtol: Relative error tolerance of the numerical integrator.
        atol: Absolute error tolerance of the numerical integrator.
        hmax: Maximum integration step-size.
        hmin: Minimum integration step-size.
        mxstep: Maximum number of internal integration steps.
        correct: Whether or not to invalidate the detailed balance of RMHMC by
            breaking the symmetry of partial derivatives.

    """
    def __init__(
            self,
            state: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            sigma: float,
            rtol: float=None,
            atol: float=None,
            hmax: float=0.0,
            hmin: float=0.0,
            mxstep: int=0,
            correct: bool=True
    ):
        super().__init__()
        self.state = state
        self.y = y
        self.t = t
        self.sigma = sigma
        self.rtol = rtol
        self.atol = atol
        self.hmax = hmax
        self.hmin = hmin
        self.mxstep = mxstep
        self.correct = correct

        # Precompute assumed noise variance.
        self.sigmasq = np.square(sigma)
        # Precompute identity matrix.
        self.Id = np.eye(3)
        # Precompute augmented state with and without higher order sensitivities.
        self.aug = np.hstack((state, np.zeros(6)))
        self.augh = np.hstack((state, np.zeros(18)))

    def odeint(self, dynamics, state, t, params):
        return scipy.integrate.odeint(
            dynamics,
            state,
            t,
            params,
            rtol=self.rtol,
            atol=self.atol,
            hmax=self.hmax,
            hmin=self.hmin,
            mxstep=self.mxstep
        )

    def log_density(self, qt):
        a, b, c = qt
        sens = self.odeint(fn_sensitivity, self.aug, self.t, (a, b, c))
        lp = log_posterior(self.y, self.sigma, sens, a, b, c)
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
        Id = np.eye(3)
        return Id, Id, Id

    def riemannian_metric(self, qt):
        a, b, c = qt
        sens = self.odeint(fn_sensitivity, self.aug, self.t, (a, b, c))
        G = metric(self.sigmasq, self.Id, sens)
        return G

    def riemannian_metric_and_jacobian(self, qt):
        a, b, c = qt
        sens = self.odeint(fn_higher_sensitivity, self.augh, self.t, (a, b, c, self.correct))
        G = metric(self.sigmasq, self.Id, sens)
        dG = grad_metric(self.sigmasq, sens)
        return G, dG

    def euclidean_quantities(self, qt):
        a, b, c = qt
        sens = self.odeint(fn_sensitivity, self.aug, self.t, (a, b, c))
        lp = log_posterior(self.y, self.sigma, sens, a, b, c)
        glp = grad_log_posterior(self.y, self.sigmasq, sens, a, b, c)
        return lp, glp

    def riemannian_quantities(self, qt):
        a, b, c = qt
        sens = self.odeint(fn_higher_sensitivity, self.augh, self.t, (a, b, c, self.correct))
        lp = log_posterior(self.y, self.sigma, sens, a, b, c)
        glp = grad_log_posterior(self.y, self.sigmasq, sens, a, b, c)
        G = metric(self.sigmasq, self.Id, sens)
        dG = grad_metric(self.sigmasq, sens)
        return lp, glp, G, dG

    def lagrangian_quantities(self, qt):
        a, b, c = qt
        sens = self.odeint(fn_sensitivity, self.aug, self.t, (a, b, c))
        lp = log_posterior(self.y, self.sigma, sens, a, b, c)
        G = metric(self.sigmasq, self.Id, sens)
        return lp, G

    def softabs_quantities(self, qt):
        raise NotImplementedError()
