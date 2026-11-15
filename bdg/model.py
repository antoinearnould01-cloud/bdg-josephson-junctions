"""Tight-binding model of a one-dimensional SNS junction.

The chain is ordered as  left lead | normal region | right lead:

    sites 0 ... n_s-1             left superconductor  (U = u0, mu = mu_s)
    sites n_s ... n_s+n_n-1       normal metal         (U = 0,  mu = mu_n)
    sites n_s+n_n ... n_s+n_n+n_s-1   right superconductor

All energies are in units of the hopping t, lengths in units of the lattice
spacing a, and currents in units of 2et/hbar.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Junction:
    """Parameters of a 1D SNS junction.

    Attributes
    ----------
    n_s : number of sites in each superconducting lead.
    n_n : number of sites in the normal region.
    t : nearest-neighbour hopping.
    u0 : on-site attraction in the leads (negative).
    mu_s, mu_n : chemical potentials in the leads and in the normal region.
    delta0 : amplitude of the initial (or fixed) pair potential.
    """

    n_s: int
    n_n: int
    t: float = 1.0
    u0: float = -2.0
    mu_s: float = 0.0
    mu_n: float = 0.0
    delta0: float = 0.3

    @property
    def n_sites(self) -> int:
        return 2 * self.n_s + self.n_n

    @property
    def normal(self) -> slice:
        """Sites of the normal region."""
        return slice(self.n_s, self.n_s + self.n_n)

    def with_(self, **changes) -> "Junction":
        """Return a copy with some parameters changed."""
        return replace(self, **changes)

    def interaction(self) -> np.ndarray:
        """On-site attraction U_i (non-zero only in the leads)."""
        u = np.zeros(self.n_sites)
        u[0:self.n_s] = self.u0
        u[self.n_s + self.n_n:self.n_sites] = self.u0
        return u

    def chemical_potential(self) -> np.ndarray:
        mu = np.zeros(self.n_sites)
        mu[0:self.n_s] = self.mu_s
        mu[self.n_s:self.n_s + self.n_n] = self.mu_n
        mu[self.n_s + self.n_n:self.n_sites] = self.mu_s
        return mu


def hamiltonian(junction: Junction, delta: np.ndarray) -> np.ndarray:
    """BdG matrix  [[H0, Delta], [Delta*, -H0*]]  of size 2N x 2N."""
    n = junction.n_sites
    hop = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    h0 = -junction.t * hop - np.diag(junction.chemical_potential())
    return np.block([[h0, np.diag(delta)], [np.diag(np.conj(delta)), -h0.conj()]])


# --------------------------------------------------------------------------
# Initial / fixed pair potentials
# --------------------------------------------------------------------------

def gap_linear_in_normal(junction: Junction, phase: float) -> np.ndarray:
    """Constant amplitude delta0 everywhere, phase -phase/2 in the left lead,
    +phase/2 in the right lead, and a linear phase ramp across the normal region."""
    n, s, nn = junction.n_sites, junction.n_s, junction.n_n
    delta = np.zeros(n, dtype=complex)
    delta[0:s] = junction.delta0 * np.exp(-1j * phase / 2)
    delta[s + nn:n] = junction.delta0 * np.exp(1j * phase / 2)
    for i in range(s, s + nn):
        delta[i] = junction.delta0 * np.exp(1j * (-phase / 2 + phase * (i - s) / nn))
    return delta


def gap_linear_between_pins(junction: Junction, phase: float, pin_width: int) -> np.ndarray:
    """Constant amplitude delta0, fixed phases -phase/2 and +phase/2 on the
    pin_width outermost sites, and a linear phase ramp over all sites in between."""
    n = junction.n_sites
    delta = np.zeros(n, dtype=complex)
    delta[0:pin_width] = junction.delta0 * np.exp(-1j * phase / 2)
    delta[n - pin_width:n] = junction.delta0 * np.exp(1j * phase / 2)
    for i in range(pin_width, n - pin_width):
        frac = (i - pin_width) / (n - 2 * pin_width)
        delta[i] = junction.delta0 * np.exp(1j * (-phase / 2 + phase * frac))
    return delta


def gap_step(junction: Junction, phase: float, amplitude: float | None = None) -> np.ndarray:
    """Step-function pair potential: amplitude in the leads, zero in N.

    This is the frozen square-well profile used for the one-shot
    diagonalisations of Chapter 6.
    """
    amp = junction.delta0 if amplitude is None else amplitude
    n, s = junction.n_sites, junction.n_s
    delta = np.zeros(n, dtype=complex)
    delta[0:s] = amp * np.exp(-1j * phase / 2)
    delta[n - s:n] = amp * np.exp(1j * phase / 2)
    return delta


def gap_pins_only(junction: Junction, phase: float, pin_width: int) -> np.ndarray:
    """delta0 on the pin_width outermost sites only, zero elsewhere."""
    n = junction.n_sites
    delta = np.zeros(n, dtype=complex)
    delta[0:pin_width] = junction.delta0 * np.exp(-1j * phase / 2)
    delta[n - pin_width:n] = junction.delta0 * np.exp(1j * phase / 2)
    return delta
