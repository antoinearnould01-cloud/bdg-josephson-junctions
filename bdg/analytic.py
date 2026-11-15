"""Analytical and semi-analytical results used as references.

Uniform s-wave superconductor (Chapter 4)
    Mean-field gap equation of the attractive Hubbard model, with the same
    convention as the BdG solver, Delta = -(U/2) F:
        1 = -(U / 4pi) * integral_0^{2pi} dk  tanh(E_k / 2T) / E_k     (1D)
        1 = -(U a^2 / 8pi^2) * integral_BZ d^2k  tanh(E_k / 2T) / E_k   (2D)
    with E_k = sqrt(xi_k^2 + Delta^2), xi_k = -mu - 2t cos(k a) (1D).

SNS junctions (Chapters 6 and 7), currents in units of 2et/hbar
    short ballistic junction  I = (Delta/2) sin(phi/2) tanh(Delta cos(phi/2) / 2T)
    Beenakker, one channel    I = (Delta/4) T sin(phi) / sqrt(1 - T sin^2(phi/2))
                                    * tanh(Delta sqrt(1 - T sin^2(phi/2)) / 2T)
    Ishii, long junction, T=0 I = phi / (pi L) * (v_F / 2t),  |phi| < pi
"""

import numpy as np
from scipy.integrate import dblquad, quad
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# Uniform superconductor
# --------------------------------------------------------------------------

def critical_temperature_1d(mu: float, u: float = -2.0, t: float = 1.0, a: float = 1.0) -> float:
    """T_c from the linearised 1D gap equation (0 if there is no solution in [1e-4, 5])."""
    def gap_equation(temp):
        def integrand(k):
            xi = abs(mu + 2 * t * np.cos(k * a))
            if xi < 1e-12:
                return 1.0 / (2.0 * temp)            # tanh(x)/x -> 1
            return np.tanh(xi / (2.0 * temp)) / xi
        integral, _ = quad(integrand, 0, 2 * np.pi, limit=200)
        return -(u / (4 * np.pi)) * integral - 1.0
    try:
        return brentq(gap_equation, 1e-4, 5.0)
    except ValueError:
        return 0.0


def critical_temperature_2d(mu: float, u: float = -6.0, t: float = 1.0, a: float = 1.0) -> float:
    """T_c from the linearised gap equation on the square lattice."""
    def gap_equation(temp):
        def integrand(ky, kx):
            xi = abs(-mu - 2 * t * (np.cos(kx * a) + np.cos(ky * a)))
            if xi < 1e-12:
                return 1.0 / (2.0 * temp)
            return np.tanh(xi / (2.0 * temp)) / xi
        integral, _ = dblquad(integrand, 0, 2 * np.pi, lambda x: 0, lambda x: 2 * np.pi,
                              epsabs=1e-8, epsrel=1e-8)
        return -(u * a**2) / (8 * np.pi**2) * integral - 1.0
    try:
        return brentq(gap_equation, 1e-4, 5.0)
    except ValueError:
        return 0.0


def gap_1d(temp: float, mu: float = 0.0, u: float = -2.0, t: float = 1.0, a: float = 1.0) -> float:
    """Bulk gap Delta(T) of the 1D chain (0 above T_c)."""
    temp = max(temp, 1e-12)
    def gap_equation(delta):
        def integrand(k):
            e = np.sqrt(delta**2 + (mu + 2 * t * np.cos(k * a))**2)
            return np.tanh(e / (2 * temp)) / e
        return -(u / (4 * np.pi)) * quad(integrand, 0, 2 * np.pi)[0] - 1.0
    try:
        return brentq(gap_equation, 1e-9, 2.0)
    except ValueError:
        return 0.0


def gap_2d(temp: float, mu: float = 0.0, u: float = -2.0, t: float = 1.0, a: float = 1.0) -> float:
    """Bulk gap Delta(T) on the square lattice (0 above T_c)."""
    temp = max(temp, 1e-12)

    def gap_equation(delta):
        def integrand(ky, kx):
            xi = -mu - 2 * t * (np.cos(kx * a) + np.cos(ky * a))
            e = np.sqrt(xi**2 + delta**2)
            if e < 1e-12:
                return 0.5 / delta * np.tanh(delta / (2 * temp))
            return 0.5 / e * np.tanh(e / (2 * temp))
        integral, _ = dblquad(integrand, 0, 2 * np.pi, lambda x: 0, lambda x: 2 * np.pi,
                              epsabs=1e-8, epsrel=1e-8)
        return 1.0 + u * a**2 / (8 * np.pi**2) * integral
    try:
        if gap_equation(1e-9) > 0:
            return 0.0
        return brentq(gap_equation, 1e-9, 5.0)
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------
# Josephson junctions
# --------------------------------------------------------------------------

def fermi_wavevector(mu: float, t: float = 1.0, a: float = 1.0) -> float:
    """k_F a = arccos(-mu / 2t) of the tight-binding band."""
    return np.arccos(-mu / (2 * t)) / a


def transmission(mu_n: float, mu_s: float, length: float, t: float = 1.0, a: float = 1.0) -> np.ndarray:
    """Normal-state transmission of the N region with a Fermi-wavevector mismatch:
    T = [1 + (kN^2 - kS^2)^2 / (4 kN^2 kS^2) sin^2(kN L)]^{-1}."""
    kn = fermi_wavevector(mu_n, t, a)
    ks = fermi_wavevector(mu_s, t, a)
    return 1.0 / (1.0 + (kn**2 - ks**2)**2 / (4 * kn**2 * ks**2) * np.sin(kn * length)**2)


def cpr_short_ballistic(phase, delta: float, temp: float):
    """Short transparent junction (bound-state current only), units of 2et/hbar."""
    return 0.5 * delta * np.sin(phase / 2) * np.tanh(delta * np.cos(phase / 2) / (2 * temp))


def cpr_beenakker(phase, delta: float, temp: float, trans: float):
    """Short single-channel junction with transmission `trans`, units of 2et/hbar."""
    root = np.sqrt(1 - trans * np.sin(phase / 2)**2)
    return 0.25 * delta * trans * np.sin(phase) / root * np.tanh(delta * root / (2 * temp))


def ishii_slope(length: float, mu: float = 0.0, t: float = 1.0) -> float:
    """dI/dphi of the T = 0 sawtooth, I = e v_F phi / (pi L), in units of 2et/hbar."""
    v_f = 2 * t * np.sin(fermi_wavevector(mu, t))
    return v_f / (2 * t) / (np.pi * length)


def critical_point_of(cpr, phases: np.ndarray) -> tuple[float, float]:
    """(phi_max, I_max) of a CPR sampled on `phases`."""
    values = cpr(phases)
    i = int(np.argmax(values))
    return phases[i], values[i]
