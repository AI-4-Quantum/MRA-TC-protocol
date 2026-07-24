"""Jastrow erfc Gaussian-sum fitting (pure NumPy/SciPy). No vampyr import.

Only responsibility: turn the radial erfc(mu r)/r and erfc(mu r)^2 kernels into
finite all-positive Gaussian sums  kernel(r) = sum_m c_m exp(-z_m r^2)  via
Gauss-Legendre quadrature of their exact integral representations, so the Part 2
notebook can apply them with MRA CartesianConvolution/ConvolutionOperator.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import erfc as _erfc

def erfc_over_r_gl(mu, rmin, n):
    """erfc(mu r)/r = sum c_m exp(-z_m r^2); s = mu e^t, t in [0, log((4/rmin)/mu)]."""
    big_t = math.log((4.0 / rmin) / mu)
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * big_t * (x + 1.0)
    wt = 0.5 * big_t * w
    s = mu * np.exp(t)
    return (2.0 / math.sqrt(math.pi)) * s * wt, s * s

def erfc_sq_gl(mu, rmin, n):
    """erfc(mu r)^2 = r^2 sum c_m exp(-z_m r^2); q = sqrt2 mu e^tau (positive terms)."""
    big_t = math.log((4.0 / rmin) / (math.sqrt(2.0) * mu))
    x, w = np.polynomial.legendre.leggauss(n)
    tau = 0.5 * big_t * (x + 1.0)
    wt = 0.5 * big_t * w
    q = math.sqrt(2.0) * mu * np.exp(tau)
    coef = (4.0 / math.pi) * q * q * (math.pi / 2.0 - 2.0 * np.arcsin(np.minimum(mu / q, 1.0))) * wt
    keep = coef > 0.0
    return coef[keep], (q * q)[keep]

def q1_terms(mu, rmin, n):
    c, z = erfc_over_r_gl(mu, rmin, n)
    return 0.5 * c, z                      # q1 = erfc/(2r)

def q2_terms(mu, rmin, n):
    c, z = erfc_over_r_gl(mu, rmin, n)     # q2 = erfc/r - (mu/sqrt pi) e^{-mu^2 r^2}
    return np.append(c, -(mu / math.sqrt(math.pi))), np.append(z, mu * mu)  # signed

def q3_terms(mu, rmin, n):
    c, z = erfc_sq_gl(mu, rmin, n)
    return 0.25 * c, z                     # q3 = erfc^2/(4 r^2), applied via r^2 pow-2

def analytic_kernels(mu, r):
    r = np.asarray(r, float); ec = _erfc(mu * r); g = np.exp(-(mu * r) ** 2)
    return {
        "u": 0.5 * r * ec - g / (2 * math.sqrt(math.pi) * mu),
        "up": 0.5 * ec,
        "q1": ec / (2 * r),
        "q2": ec / r - (mu / math.sqrt(math.pi)) * g,
        "q3": ec * ec / (4 * r * r),
    }

def kernel_errors(mu, x_min, n, x_max=25.0):
    rmin = x_min / mu
    w1, z1 = erfc_over_r_gl(mu, rmin, n)
    w3, z3 = erfc_sq_gl(mu, rmin, n)
    x = np.geomspace(x_min, x_max, 4000); r = x / mu
    a1 = (w1 @ np.exp(-np.outer(z1, r * r))) / mu
    a3 = 0.25 * r * r * (w3 @ np.exp(-np.outer(z3, r * r)))
    return {
        "erfc_over_x_max_rel_err": float(np.max(np.abs(a1 - _erfc(x) / x)) / np.max(np.abs(_erfc(x) / x))),
        "erfc_sq_over_4_max_err": float(np.max(np.abs(a3 - 0.25 * _erfc(x) ** 2))),
        "all_positive": bool(np.all(w1 > 0.0) and np.all(w3 > 0.0)),
    }
