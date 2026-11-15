"""Regression and consistency tests for the bdg package.

Reference values are given for fixed parameters.
Run with:  python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bdg  # noqa: E402
from bdg.analytic import cpr_beenakker, cpr_short_ballistic, transmission  # noqa: E402


def test_fpl_current_reference():
    j = bdg.Junction(n_s=30, n_n=10, mu_n=0.7)
    sol = bdg.solve_self_consistent(j, 2.2, 100.0, tol=1e-10)
    assert bdg.normal_current(sol) == pytest.approx(0.035647716140625604, rel=1e-12)


def test_fpb_current_and_phase_reference():
    j = bdg.Junction(n_s=30, n_n=20)
    sol = bdg.solve_self_consistent(j, 1.3, 50.0, pin_width=3, tol=1e-10)
    assert bdg.normal_current(sol, drop_last_bond=True) == pytest.approx(0.005003593791489742, rel=1e-10)
    assert bdg.effective_phase(sol) == pytest.approx(0.36869712790724224, rel=1e-10)


def test_subgap_spectrum_reference():
    j = bdg.Junction(n_s=60, n_n=20)
    sol = bdg.solve_self_consistent(j, 0.0, 20.0, pin_width=3, tol=1e-8, error_norm="ratio_of_sums",
                                    initial=bdg.gap_linear_between_pins(j, 0.0, 3))
    lowest = np.sort(np.abs(sol.eig.energies))[:4]
    assert lowest == pytest.approx([0.11249267623580321] * 4, rel=1e-10)


def test_particle_hole_symmetry_of_spectrum():
    j = bdg.Junction(n_s=20, n_n=6, mu_n=0.4)
    energies = bdg.diagonalize(j, bdg.gap_step(j, 1.0)).energies
    assert np.sort(energies) == pytest.approx(np.sort(-energies), abs=1e-12)


def test_current_is_conserved_in_the_normal_region():
    j = bdg.Junction(n_s=30, n_n=10)
    current = bdg.solve_self_consistent(j, 1.0, 50.0, tol=1e-12).currents()
    inside = current[j.n_s + 1:j.n_s + j.n_n]
    assert np.ptp(inside) < 1e-10 * np.max(np.abs(inside))


def test_no_current_without_phase_difference():
    j = bdg.Junction(n_s=30, n_n=10)
    sol = bdg.solve_self_consistent(j, 0.0, 50.0, tol=1e-12)
    assert np.max(np.abs(sol.currents())) < 1e-12


def test_critical_point_search_on_a_known_function():
    cp = bdg.find_critical_point(lambda p: np.sin(p) * np.exp(-0.1 * p), min_step=1e-8)
    assert cp.reliable
    assert cp.phase == pytest.approx(np.arctan(10.0), abs=1e-7)


def test_beenakker_reduces_to_ballistic_formula_for_full_transmission():
    phases = np.linspace(0.01, np.pi - 0.01, 50)
    assert cpr_beenakker(phases, 0.01, 0.02, 1.0) == pytest.approx(cpr_short_ballistic(phases, 0.01, 0.02))


def test_transmission_is_one_without_mismatch():
    assert transmission(0.3, 0.3, length=7) == pytest.approx(1.0)
