"""Self-consistent tight-binding Bogoliubov-de Gennes solver for 1D SNS junctions."""

from .model import (Junction, hamiltonian, gap_linear_in_normal, gap_linear_between_pins,
                    gap_step, gap_pins_only)
from .solver import (ConvergenceError, Eigensystem, Solution, occupation, diagonalize,
                     pair_amplitude, bond_currents, solve_self_consistent, solve_fixed_gap)
from .observables import normal_current, effective_phase, lead_gap, split_current
from .critical import CriticalPoint, find_critical_point

__all__ = [
    "Junction", "hamiltonian", "gap_linear_in_normal", "gap_linear_between_pins", "gap_step",
    "gap_pins_only", "ConvergenceError", "Eigensystem", "Solution", "occupation", "diagonalize",
    "pair_amplitude", "bond_currents", "solve_self_consistent", "solve_fixed_gap",
    "normal_current", "effective_phase", "lead_gap", "split_current", "CriticalPoint",
    "find_critical_point",
]
