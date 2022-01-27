from typing import Tuple

import numpy as np
import scipy.special as spsp

from odyssey.distribution import Distribution


def generate_data(n: int=2000, beta: float=-2.0, sigma: float=2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generates data from the probit model. Given a linear combination of
    features, we perturb them by gaussian random noise with a prescribed noise
    level. These values are then thresholded to produce the observed binary
    outcome.

    Args:
        n: The number of observations.
        beta: The linear coefficient.
        sigma: The noise scale.

    Returns:
        x: The covariate.
        y: The binary outcome.

    """
    x = np.random.uniform(size=(n, )) - 0.5
    w = x*beta + sigma * np.random.normal(size=(n, ))
    y = w < 0.0
    return x, y

def normpdf(z: np.ndarray) -> np.ndarray:
    """The density of the normal distribution.

    Args:
        z: The locations at which to compute the density of the normal
            distribution.

    Returns:
        d: The density of the normal distribution at the given points.

    """
    d = np.exp(-0.5*np.square(z)) / np.sqrt(2*np.pi)
    return d

def normcdf(z: np.ndarray) -> np.ndarray:
    """The cumulative distribution function of the normal distribution.

    Args:
        z: The locations at which to compute the cumulative distribution function
            of the normal distribution.

    Returns:
        d: The cumulative distribution function of the normal distribution at the
            given points.

    """
    p = 0.5*(1 + spsp.erf(z / np.sqrt(2.0)))
    return p

def xi(z: np.ndarray) -> np.ndarray:
    """Function to compute the ratio of the density and the cumulative distribution
    function of the normal distribution.

    Args:
        z: The locations at which to evaluate the ratio.

    Returns:
        v: The ratio of the density and the cumulative distribution function.

    """
    v = normpdf(z) / normcdf(z)
    return v

def dxi(z: np.ndarray) -> np.ndarray:
    """Function to compute the derivative of the ratio of the density and the
    cumulative distribution function of the normal distribution.

    Args:
        z: The locations at which to evaluate the derivative of the ratio.

    Returns:
        d: The derivative of the ratio of the density and the cumulative
            distribution function.

    """
    zsq = np.square(z)
    e = spsp.erfc(-z / np.sqrt(2))
    d = -np.exp(-zsq)*(2 + np.exp(0.5*zsq)*np.sqrt(2*np.pi)*z*e) / (np.pi*e**2)
    return d

def log_posterior(qt, x, y):
    beta, log_sigmasq = qt
    sigmasq = np.exp(log_sigmasq)
    sigma = np.sqrt(sigmasq)
    l = x*beta / sigma
    ll = np.sum(y*np.log(normcdf(l)) + (1-y)*np.log(normcdf(-l)))
    pbeta = -0.5*np.square(beta) / 100.0
    psigma = -(1.5+1)*log_sigmasq - 1.0 / (6*sigmasq)
    lp = ll + pbeta + psigma
    return lp

def grad_log_posterior(qt, x, y, vals):
    a, b, c, d = vals
    beta, log_sigmasq = qt
    sigmasq = np.exp(log_sigmasq)
    grad_beta = np.sum(a*y + b) - beta / 100.0
    grad_log_sigma = np.sum(c*y + d) - 2.5 + 1.0 / (6*sigmasq)
    grad = np.hstack((grad_beta, grad_log_sigma))
    return grad

def metric(qt, x, y, vals):
    a, b, c, d = vals
    beta, log_sigmasq = qt
    sigmasq = np.exp(log_sigmasq)
    sigma = np.sqrt(sigmasq)
    p = normcdf(beta*x/sigma)
    diag_beta = np.sum(b**2 + p*(a**2 + 2*a*b))
    diag_log_sigmasq = np.sum(d**2 + p*(c**2 + 2*c*d))
    off_diag = np.sum(b*d + p*(a*c + a*d + b*c))
    H = np.array([[-1.0 / 100, 0.0], [0.0, -1.0 / (6*sigmasq)]])
    G = np.array([
        [diag_beta, off_diag],
        [off_diag, diag_log_sigmasq]
    ]) - H
    return G

def grad_metric(qt, x, y, vals, dbeta, dlog_sigmasq):
    a, b, c, d = vals
    dadbeta, dbdbeta, dcdbeta, dddbeta = dbeta
    dadlog_sigmasq, dbdlog_sigmasq, dcdlog_sigmasq, dddlog_sigmasq = dlog_sigmasq
    beta, log_sigmasq = qt
    sigmasq = np.exp(log_sigmasq)
    sigma = np.sqrt(sigmasq)
    z = beta*x/sigma
    p = normcdf(z)
    dp = normpdf(z)
    alpha = np.sum(2*b*dbdbeta + dp*x/sigma*(a**2 + 2*a*b) + p*(2*a*dadbeta + 2*dadbeta*b + 2*a*dbdbeta))
    gamma = np.sum(dbdbeta*d + b*dddbeta + dp*x/sigma*(a*c + a*d + b*c) + p*(dadbeta*c + a*dcdbeta + dadbeta*d + a*dddbeta + dbdbeta*c + b*dcdbeta))
    delta = np.sum(2*d*dddbeta + dp*x/sigma*(c**2 + 2*c*d) + p*(2*c*dcdbeta + 2*dcdbeta*d + 2*c*dddbeta))
    dGdbeta = np.array([
        [alpha, gamma],
        [gamma, delta]
    ])
    theta = np.sum(2*b*dbdlog_sigmasq - dp*beta*x/sigma*0.5*(a**2 + 2*a*b) + p*(2*a*dadlog_sigmasq + 2*dadlog_sigmasq*b + 2*a*dbdlog_sigmasq))
    kappa = np.sum(dbdlog_sigmasq*d + b*dddlog_sigmasq - dp*beta*x/sigma*0.5*(a*c + a*d + b*c) + p*(dadlog_sigmasq*c + a*dcdlog_sigmasq + dadlog_sigmasq*d + a*dddlog_sigmasq + dbdlog_sigmasq*c + b*dcdlog_sigmasq))
    omega = np.sum(2*d*dddlog_sigmasq - dp*beta*x/sigma*0.5*(c**2 + 2*c*d) + p*(2*c*dcdlog_sigmasq + 2*dcdlog_sigmasq*d + 2*c*dddlog_sigmasq)) - 1.0 / (6*sigmasq)
    dGdlog_sigmasq = np.array([
        [theta, kappa],
        [kappa, omega]
    ])
    dG = np.array([dGdbeta, dGdlog_sigmasq]).swapaxes(0, -1)
    return dG

def afn(beta, log_sigmasq, x, y):
    sigmasq = np.exp(log_sigmasq)
    sigma = np.sqrt(sigmasq)
    r = x / sigma
    p = beta*r
    m = xi(p)
    nm = xi(-p)
    a = r * (m + nm)
    dbeta = r*(dxi(p)*r - dxi(-p)*r)
    dsigma = -x / sigmasq * (m + nm) + r * (dxi(p)*-x*beta/sigmasq + dxi(-p)*x*beta/sigmasq)
    dlog_sigmasq = dsigma * 0.5*sigma
    return a, dbeta, dlog_sigmasq

def bfn(beta, log_sigmasq, x, y):
    sigmasq = np.exp(log_sigmasq)
    sigma = np.sqrt(sigmasq)
    r = x / sigma
    p = beta*r
    nm = xi(-p)
    b = -r*nm
    dbeta = r*dxi(-p)*r
    dsigma = x / sigmasq*nm - r*dxi(-p)*x*beta/sigmasq
    dlog_sigmasq = dsigma * 0.5*sigma
    return b, dbeta, dlog_sigmasq

def cfn(beta, log_sigmasq, x, y):
    sigmasq = np.exp(log_sigmasq)
    sigma = np.sqrt(sigmasq)
    r = x / sigma
    p = beta*r
    m = xi(p)
    nm = xi(-p)
    c = -0.5*beta*r*(m + nm)
    dbeta = -0.5*r*(m + nm) - 0.5*beta*r*(dxi(p)*r - dxi(-p)*r)
    dsigma = 0.5*beta*x/sigmasq*(m + nm) - 0.5*beta*r*(dxi(p)*-x*beta/sigmasq + dxi(-p)*x*beta/sigmasq)
    dlog_sigmasq = dsigma * 0.5*sigma
    return c, dbeta, dlog_sigmasq

def dfn(beta, log_sigmasq, x, y):
    sigmasq = np.exp(log_sigmasq)
    sigma = np.sqrt(sigmasq)
    r = x / sigma
    p = beta*r
    nm = xi(-p)
    d = 0.5*beta*r*nm
    dbeta = 0.5*r*nm - 0.5*beta*r*dxi(-p)*r
    dsigma = -0.5*beta*x/sigmasq*nm + 0.5*beta*r*dxi(-p)*x*beta/sigmasq
    dlog_sigmasq = dsigma * 0.5*sigma
    return d, dbeta, dlog_sigmasq

class Probit(Distribution):
    """We consider the probit regression model proposed by Stathopoulos and
    Filippone. In this example, we consider a generative model of the form:

        b ~ Normal(0, 100)
        e ~ Normal(0, sigmasq)
        w = x b + e
        y = { 0 if w < 0 otherwise 1 }

    Given the observations `y`, we seek to infer the posterior of `b` and
    `sigmasq`. However, the model is such that only the ratio is identifiable.

    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__()
        self.x = x
        self.y = y

    def log_density(self, qt):
        lp = log_posterior(qt, self.x, self.y)
        return lp

    def sample(self):
        raise NotImplementedError()

    def forward_transform(self, q):
        beta, sigmasq = q
        log_sigmasq = np.log(sigmasq)
        qt = np.array([beta, log_sigmasq])
        ildj = log_sigmasq
        return qt, ildj

    def inverse_transform(self, qt):
        beta, log_sigmasq = qt
        sigmasq = np.exp(log_sigmasq)
        q = np.array([beta, sigmasq])
        fldj = -log_sigmasq
        return q, fldj

    def hessian(self, qt):
        raise NotImplementedError()

    def euclidean_metric(self):
        Id = np.eye(2)
        return Id, Id, Id

    def riemannian_metric(self, qt):
        beta, log_sigmasq = qt
        a, dadbeta, dadlog_sigmasq = afn(beta, log_sigmasq, self.x, self.y)
        b, dbdbeta, dbdlog_sigmasq = bfn(beta, log_sigmasq, self.x, self.y)
        c, dcdbeta, dcdlog_sigmasq = cfn(beta, log_sigmasq, self.x, self.y)
        d, dddbeta, dddlog_sigmasq = dfn(beta, log_sigmasq, self.x, self.y)
        vals = (a, b, c, d)
        G = metric(qt, self.x, self.y, vals)
        return G

    def riemannian_metric_and_jacobian(self, qt):
        beta, log_sigmasq = qt
        a, dadbeta, dadlog_sigmasq = afn(beta, log_sigmasq, self.x, self.y)
        b, dbdbeta, dbdlog_sigmasq = bfn(beta, log_sigmasq, self.x, self.y)
        c, dcdbeta, dcdlog_sigmasq = cfn(beta, log_sigmasq, self.x, self.y)
        d, dddbeta, dddlog_sigmasq = dfn(beta, log_sigmasq, self.x, self.y)
        vals = (a, b, c, d)
        dbeta = (dadbeta, dbdbeta, dcdbeta, dddbeta)
        dlog_sigmasq = (dadlog_sigmasq, dbdlog_sigmasq, dcdlog_sigmasq, dddlog_sigmasq)
        G = metric(qt, self.x, self.y, vals)
        dG = grad_metric(qt, self.x, self.y, vals, dbeta, dlog_sigmasq)
        return G, dG

    def euclidean_quantities(self, qt):
        beta, log_sigmasq = qt
        a, dadbeta, dadlog_sigmasq = afn(beta, log_sigmasq, self.x, self.y)
        b, dbdbeta, dbdlog_sigmasq = bfn(beta, log_sigmasq, self.x, self.y)
        c, dcdbeta, dcdlog_sigmasq = cfn(beta, log_sigmasq, self.x, self.y)
        d, dddbeta, dddlog_sigmasq = dfn(beta, log_sigmasq, self.x, self.y)
        vals = (a, b, c, d)
        lp = log_posterior(qt, self.x, self.y)
        glp = grad_log_posterior(qt, self.x, self.y, vals)
        return lp, glp

    def riemannian_quantities(self, qt):
        beta, log_sigmasq = qt
        a, dadbeta, dadlog_sigmasq = afn(beta, log_sigmasq, self.x, self.y)
        b, dbdbeta, dbdlog_sigmasq = bfn(beta, log_sigmasq, self.x, self.y)
        c, dcdbeta, dcdlog_sigmasq = cfn(beta, log_sigmasq, self.x, self.y)
        d, dddbeta, dddlog_sigmasq = dfn(beta, log_sigmasq, self.x, self.y)
        vals = (a, b, c, d)
        dbeta = (dadbeta, dbdbeta, dcdbeta, dddbeta)
        dlog_sigmasq = (dadlog_sigmasq, dbdlog_sigmasq, dcdlog_sigmasq, dddlog_sigmasq)
        lp = log_posterior(qt, self.x, self.y)
        glp = grad_log_posterior(qt, self.x, self.y, vals)
        G = metric(qt, self.x, self.y, vals)
        dG = grad_metric(qt, self.x, self.y, vals, dbeta, dlog_sigmasq)
        return lp, glp, G, dG

    def lagrangian_quantities(self, qt):
        beta, log_sigmasq = qt
        a, dadbeta, dadlog_sigmasq = afn(beta, log_sigmasq, self.x, self.y)
        b, dbdbeta, dbdlog_sigmasq = bfn(beta, log_sigmasq, self.x, self.y)
        c, dcdbeta, dcdlog_sigmasq = cfn(beta, log_sigmasq, self.x, self.y)
        d, dddbeta, dddlog_sigmasq = dfn(beta, log_sigmasq, self.x, self.y)
        vals = (a, b, c, d)
        dbeta = (dadbeta, dbdbeta, dcdbeta, dddbeta)
        dlog_sigmasq = (dadlog_sigmasq, dbdlog_sigmasq, dcdlog_sigmasq, dddlog_sigmasq)
        lp = log_posterior(qt, self.x, self.y)
        dG = grad_metric(qt, self.x, self.y, vals, dbeta, dlog_sigmasq)
        return lp, G

    def softabs_quantities(self, qt):
        raise NotImplementedError()
