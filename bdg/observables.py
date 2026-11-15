"""Quantities extracted from a converged solution."""

import numpy as np

from .solver import Solution, bond_currents


def normal_current(sol: Solution, drop_last_bond: bool = False, include_left_bond: bool = False) -> float:
    """Average bond current inside the normal region.

    With the bond convention of `bond_currents` (element i = bond (i-1, i)),
    the default averages the n_n bonds that end on a normal site.
      drop_last_bond     : average over n_n - 1 bonds (used for the FPB runs)
      include_left_bond  : also include the S-N bond on the left
    """
    j = sol.junction
    current = sol.currents()
    start = j.n_s - 1 if include_left_bond else j.n_s
    stop = j.n_s + j.n_n - (1 if drop_last_bond else 0)
    return float(np.mean(current[start:stop]))


def effective_phase(sol: Solution, left_site: int | None = None) -> float:
    """Phase difference of F across the normal region, in (-pi, pi].

    Measured between the last site of the left lead (or `left_site`) and the
    first site of the right lead.
    """
    j = sol.junction
    left = j.n_s - 1 if left_site is None else left_site
    return float(np.angle(sol.pair[j.n_s + j.n_n] / sol.pair[left]))


def lead_gap(sol: Solution, margin: int = 0) -> float:
    """Mean |Delta| deep inside the two leads (excluding `margin` sites at each interface)."""
    j = sol.junction
    left = np.abs(sol.delta[:j.n_s - margin])
    right = np.abs(sol.delta[j.n_s + j.n_n + margin:])
    return 0.5 * (np.mean(left) + np.mean(right))


def split_current(sol: Solution, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Bond currents carried by sub-gap states (|E| < gap) and by the continuum."""
    sub = np.abs(sol.eig.energies) < gap
    return bond_currents(sol.eig, sol.beta, sub), bond_currents(sol.eig, sol.beta, ~sub)
