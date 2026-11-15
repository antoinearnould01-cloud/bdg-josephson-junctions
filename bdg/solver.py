"""Diagonalisation, self-consistency loop and bond currents."""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from .model import Junction, hamiltonian


class ConvergenceError(RuntimeError):
    """Raised when the self-consistency loop does not reach the tolerance."""


def occupation(energies: np.ndarray, beta: float) -> np.ndarray:
    """Fermi-Dirac occupation f(E) = 1 / (1 + exp(beta E)).

    beta = np.inf gives the zero-temperature step function (f(0) = 1/2).
    """
    if np.isinf(beta):
        return np.where(energies < 0, 1.0, np.where(energies > 0, 0.0, 0.5))
    return expit(-beta * energies)


@dataclass
class Eigensystem:
    """Eigenvalues and eigenvectors of the BdG matrix, split into u and v."""

    energies: np.ndarray
    vectors: np.ndarray

    @property
    def n(self) -> int:
        return self.vectors.shape[0] // 2

    @property
    def u(self) -> np.ndarray:
        return self.vectors[:self.n, :]

    @property
    def v(self) -> np.ndarray:
        return self.vectors[self.n:, :]


def diagonalize(junction: Junction, delta: np.ndarray) -> Eigensystem:
    energies, vectors = np.linalg.eigh(hamiltonian(junction, delta))
    return Eigensystem(energies, vectors)


def pair_amplitude(eig: Eigensystem, beta: float) -> np.ndarray:
    """Anomalous correlator F_i = <c_{i,down} c_{i,up}> = sum_n u_i v_i* (1 - 2 f(E_n))."""
    f = occupation(eig.energies, beta)
    return np.sum(eig.u * np.conj(eig.v) * (1 - 2 * f), axis=1)


def bond_currents(eig: Eigensystem, beta: float, states: np.ndarray | None = None) -> np.ndarray:
    """Supercurrent on every bond, in units of 2et/hbar.

    Element i (i >= 1) is the current on the bond (i-1, i); element 0 is 0.
    `states` optionally selects a subset of eigenstates (boolean mask), which
    is used to separate bound-state and continuum contributions.
    """
    u, v, energies = eig.u, eig.v, eig.energies
    if states is not None:
        u, v, energies = u[:, states], v[:, states], energies[states]
    f = occupation(energies, beta)
    term = np.sum(np.conj(u[:-1, :]) * u[1:, :] * f
                  + v[:-1, :] * np.conj(v[1:, :]) * (1 - f), axis=1)
    current = np.zeros(eig.n)
    current[1:] = np.imag(term)
    return current


@dataclass
class Solution:
    """Converged state of the self-consistent BdG loop."""

    junction: Junction
    beta: float
    delta: np.ndarray          # pair potential after the last update
    pair: np.ndarray           # anomalous correlator F_i
    eig: Eigensystem           # eigensystem of the last Hamiltonian
    iterations: int
    converged: bool = True
    error_history: list = field(default_factory=list)

    def currents(self) -> np.ndarray:
        return bond_currents(self.eig, self.beta)


def solve_self_consistent(
    junction: Junction,
    phase: float,
    beta: float,
    pin_width: int | None = None,
    initial: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 100_000,
    eps: float = 1e-12,
    error_norm: str = "sum_of_ratios",
    raise_on_failure: bool = True,
    record_history: bool = False,
) -> Solution:
    """Iterate the gap equation Delta_i = -(U_i / 2) F_i to self-consistency.

    Phase boundary conditions
    -------------------------
    After every update the phase of Delta is reset to -phase/2 on the first
    `pin_width` sites and to +phase/2 on the last `pin_width` sites; only the
    amplitude there is free.
      * pin_width = n_s           -> fixed phase in the leads (FPL)
      * pin_width = a few sites   -> fixed phase at the boundaries (FPB)
    Default: n_s (FPL).

    Convergence
    -----------
    error_norm = "sum_of_ratios":  sum_i |dDelta_i|^2 / (|Delta_i|^2 + eps)
    error_norm = "ratio_of_sums":  sum_i |dDelta_i|^2 / (sum_i |Delta_i|^2 + eps)
    The loop stops when the error is below `tol`.
    """
    from .model import gap_linear_in_normal  # default initial guess

    n = junction.n_sites
    w = junction.n_s if pin_width is None else pin_width
    u = junction.interaction()
    left_phase = np.exp(-1j * phase / 2)
    right_phase = np.exp(1j * phase / 2)

    delta = gap_linear_in_normal(junction, phase) if initial is None else initial
    history = []
    eig, new_delta, pair = None, delta, None

    for k in range(max_iter):
        eig = diagonalize(junction, delta)
        pair = pair_amplitude(eig, beta)
        new_delta = -(u / 2) * pair

        new_delta[0:w] = np.abs(new_delta[0:w]) * left_phase
        new_delta[n - w:n] = np.abs(new_delta[n - w:n]) * right_phase

        diff = np.abs(new_delta - delta) ** 2
        if error_norm == "sum_of_ratios":
            error = np.sum(diff / (np.abs(delta) ** 2 + eps))
        elif error_norm == "ratio_of_sums":
            error = np.sum(diff) / (np.sum(np.abs(delta) ** 2) + eps)
        else:
            raise ValueError(f"unknown error_norm {error_norm!r}")
        if record_history:
            history.append(error)

        if error < tol:
            return Solution(junction, beta, new_delta, pair, eig, k, True, history)
        delta = new_delta

    if raise_on_failure:
        raise ConvergenceError(
            f"no convergence after {max_iter} iterations "
            f"(n_n={junction.n_n}, mu_n={junction.mu_n}, phase={phase:.4f}, beta={beta})")
    return Solution(junction, beta, delta, pair, eig, max_iter, False, history)


def solve_fixed_gap(junction: Junction, delta: np.ndarray, beta: float) -> Solution:
    """One-shot diagonalisation with a frozen pair potential (no self-consistency)."""
    eig = diagonalize(junction, delta)
    return Solution(junction, beta, delta, pair_amplitude(eig, beta), eig, 0)
