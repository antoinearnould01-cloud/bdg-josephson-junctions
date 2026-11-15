# Bogoliubov-de Gennes study of SNS Josephson junctions

Self-consistent tight-binding Bogoliubov-de Gennes (BdG) solver for one-dimensional
superconductor / normal metal / superconductor (SNS) junctions, and every script needed to
reproduce the figures of the report [`report/report.pdf`](report/report.pdf).

The solver computes the current-phase relation (CPR) of junctions of varying length,
temperature and normal-region transparency, and compares it with the analytical limits:
the short-junction ballistic CPR, the single-channel Beenakker formula with Fabry-Pérot
resonances, and the Ishii sawtooth of long junctions.

## Repository layout

```
bdg/                  the solver (importable package)
  model.py            Junction parameters, BdG Hamiltonian, initial / frozen gap profiles
  solver.py           diagonalisation, gap equation, self-consistency loop, bond currents
  observables.py      current in N, effective phase, lead gap, bound-state / continuum split
  critical.py         search for the critical current I_max and critical phase phi_max
  analytic.py         uniform BCS gap equations and analytical junction CPRs
scripts/              one script per report figure (or group of panels), named chX_*.py
  _common.py          paths, data cache, plotting style, critical-current scans
data/                 simulation results (CSV) used to draw the figures
figures/              output folder of the scripts (not tracked)
report/               LaTeX source of the report, its bibliography and figures, compiled PDF
tests/                regression and consistency tests of the package
run_all.py            regenerate all figures
```

## Installation and use

```bash
pip install -r requirements.txt

python scripts/ch5_cpr_temperature.py              # one figure, from data/ if present
python scripts/ch5_cpr_temperature.py --recompute  # rerun the simulation
python run_all.py                                  # every figure
python run_all.py ch6 ch7                          # only Chapters 6 and 7
python -m pytest tests                             # tests (a few seconds)
```

Every script first looks for its results in `data/`; without them (or with `--recompute`)
it runs the simulation and saves the CSV before plotting. Scripts are independent, and the
docstring at the top of each one states the physical parameters and the expected runtime.

## How the code works

**Model** (`bdg/model.py`). The junction is a chain of `n_s + n_n + n_s` sites with
nearest-neighbour hopping `t`, chemical potential `mu_s` in the leads and `mu_n` in the
normal region, and an on-site attraction `u0 < 0` in the leads only. The BdG matrix is

    H = [[H0, diag(Delta)], [diag(Delta*), -H0*]],     H0 = -t (hopping) - diag(mu)

Energies are in units of `t`, lengths in lattice sites and currents in units of `2et/hbar`.

**Self-consistency** (`bdg/solver.py`, report Appendix "Numerical implementation").
Starting from an initial gap profile, `solve_self_consistent` repeats

1. diagonalise `H(Delta)`;
2. compute the pair amplitude `F_i = sum_n u_i v_i* (1 - 2 f(E_n))`;
3. update `Delta_i = -(U_i / 2) F_i`;
4. reset the phase of `Delta` on the pinned sites (amplitude stays free);

until `sum_i |dDelta_i|^2 / (|Delta_i|^2 + eps)` is below the tolerance. The phase boundary
condition is set by `pin_width`:

* **FPL** (fixed phase in the leads), `pin_width = n_s`: phases `-phi/2` and `+phi/2` are
  imposed on the whole leads, so the full phase drop occurs across N;
* **FPB** (fixed phase at the boundaries), e.g. `pin_width = 3`: only the outermost sites
  are pinned, the leads can carry a phase gradient and current is conserved. The phase
  difference actually seen by N is then `effective_phase(sol)`.

`solve_fixed_gap` instead diagonalises once with a frozen step-function gap; it is used in
Chapter 6 to test the analytical formulas under their own assumptions.

**Current** (`bond_currents`, report eq. `J_final`). The current on each bond is
`Im sum_n [u_i* u_{i+1} f(E_n) + v_i v_{i+1}* (1 - f(E_n))]`; `normal_current` averages it
over the bonds of the normal region. Restricting the sum to `|E_n| < Delta` or
`|E_n| > Delta` (`split_current`) separates bound states from the continuum (Chapter 7).

**Critical current** (`bdg/critical.py`). Each point of `I(phi)` costs a full
self-consistent solve, so `find_critical_point` climbs the CPR without derivatives: it
moves `phi` towards the larger neighbour, then divides the step until it is below
`min_step`. A maximum that is not the largest current met during the search is flagged as
unreliable and left out of the figures. Scans over T, mu_N or N_N warm-start each search
from the previous optimum (`scripts/_common.py: scan_critical`).

**Analytical references** (`bdg/analytic.py`). Mean-field gap equation of the uniform
superconductor in 1D and 2D (T_c, Delta(T)); short ballistic CPR; Beenakker CPR with the
mismatch transmission `T(mu_N, mu_S, L)`; Ishii slope `1/(pi L)` at mu = 0.

## Figures of the report

Each figure of the report, the script that produces it and the data it is drawn from.
Scripts write their output to `figures/`; the figures as they appear in the report are in
`report/figures/`.

| Fig. | Content | Report figure | Script | Data |
|---|---|---|---|---|
| 4.1a | Phase diagram T_c(μ), 1D chain | [`ch4_tc_vs_mu_1d.png`](report/figures/ch4_tc_vs_mu_1d.png) | [`ch4_tc_vs_mu.py`](scripts/ch4_tc_vs_mu.py) | `data/ch4_tc_vs_mu_1d.csv` |
| 4.1b | Phase diagram T_c(μ), square lattice | [`ch4_tc_vs_mu_2d.png`](report/figures/ch4_tc_vs_mu_2d.png) | [`ch4_tc_vs_mu.py`](scripts/ch4_tc_vs_mu.py) | `data/ch4_tc_vs_mu_2d.csv` |
| 4.2a | Δ(T) and √(T_c−T) law, 1D | [`ch4_gap_vs_temperature_1d.png`](report/figures/ch4_gap_vs_temperature_1d.png) | [`ch4_gap_vs_temperature.py`](scripts/ch4_gap_vs_temperature.py) | `data/ch4_gap_vs_temperature_1d.csv` |
| 4.2b | Δ(T) and √(T_c−T) law, 2D | [`ch4_gap_vs_temperature_2d.png`](report/figures/ch4_gap_vs_temperature_2d.png) | [`ch4_gap_vs_temperature.py`](scripts/ch4_gap_vs_temperature.py) | `data/ch4_gap_vs_temperature_2d.csv` |
| 4.3 | Convergence of Δ with system size | [`ch4_finite_size_convergence.png`](report/figures/ch4_finite_size_convergence.png) | [`ch4_finite_size_convergence.py`](scripts/ch4_finite_size_convergence.py) | `data/ch4_finite_size_convergence.csv` |
| 5.4a | Gap, phase and current profiles, FPL | [`ch5_phase_profile_fpl.png`](report/figures/ch5_phase_profile_fpl.png) | [`ch5_phase_profiles.py`](scripts/ch5_phase_profiles.py) | `data/ch5_phase_profile_fpl.csv` |
| 5.4b | Gap, phase and current profiles, FPB | [`ch5_phase_profile_fpb.png`](report/figures/ch5_phase_profile_fpb.png) | [`ch5_phase_profiles.py`](scripts/ch5_phase_profiles.py) | `data/ch5_phase_profile_fpb.csv` |
| 5.5a | CPR, FPB vs FPL, long junction | [`ch5_cpr_fpb_vs_fpl_long.png`](report/figures/ch5_cpr_fpb_vs_fpl_long.png) | [`ch5_cpr_fpb_vs_fpl.py`](scripts/ch5_cpr_fpb_vs_fpl.py) | `data/ch5_cpr_fpb_vs_fpl_long.csv` |
| 5.5b | CPR, FPB vs FPL, short junction | [`ch5_cpr_fpb_vs_fpl_short.png`](report/figures/ch5_cpr_fpb_vs_fpl_short.png) | [`ch5_cpr_fpb_vs_fpl.py`](scripts/ch5_cpr_fpb_vs_fpl.py) | `data/ch5_cpr_fpb_vs_fpl_short.csv` |
| 5.6a | CPR at several T, short junction | [`ch5_cpr_temperature_short.png`](report/figures/ch5_cpr_temperature_short.png) | [`ch5_cpr_temperature.py`](scripts/ch5_cpr_temperature.py) | `data/ch5_cpr_temperature_short.csv` |
| 5.6b | CPR at several T, long junction | [`ch5_cpr_temperature_long.png`](report/figures/ch5_cpr_temperature_long.png) | [`ch5_cpr_temperature.py`](scripts/ch5_cpr_temperature.py) | `data/ch5_cpr_temperature_long.csv` |
| 5.7a | φ_max and I_max vs T, short | [`ch5_critical_vs_temperature_short.png`](report/figures/ch5_critical_vs_temperature_short.png) | [`ch5_critical_vs_temperature.py`](scripts/ch5_critical_vs_temperature.py) | `data/ch5_critical_vs_temperature_short.csv` |
| 5.7b | φ_max and I_max vs T, long, FPL and FPB | [`ch5_critical_vs_temperature_long.png`](report/figures/ch5_critical_vs_temperature_long.png) | [`ch5_critical_vs_temperature.py`](scripts/ch5_critical_vs_temperature.py) | `data/ch5_critical_vs_temperature_long.csv` |
| 5.8a | φ_max and I_max vs μ_N, short | [`ch5_critical_vs_muN_short.png`](report/figures/ch5_critical_vs_muN_short.png) | [`ch5_critical_vs_muN.py`](scripts/ch5_critical_vs_muN.py) | `data/ch5_critical_vs_muN_short.csv` |
| 5.8b | φ_max and I_max vs μ_N, long, FPL | [`ch5_critical_vs_muN_long_fpl.png`](report/figures/ch5_critical_vs_muN_long_fpl.png) | [`ch5_critical_vs_muN.py`](scripts/ch5_critical_vs_muN.py) | `data/ch5_critical_vs_muN_long_fpl.csv` |
| 5.8c | φ_max and I_max vs μ_N, long, FPB | [`ch5_critical_vs_muN_long_fpb.png`](report/figures/ch5_critical_vs_muN_long_fpb.png) | [`ch5_critical_vs_muN.py`](scripts/ch5_critical_vs_muN.py) | `data/ch5_critical_vs_muN_long_fpb.csv` |
| 5.9a | (μ_N, μ_S) maps, T = 10⁻⁴ | [`ch5_critical_map_mu_low_temperature.png`](report/figures/ch5_critical_map_mu_low_temperature.png) | [`ch5_critical_map_mu.py`](scripts/ch5_critical_map_mu.py) | `data/ch5_critical_map_mu_low_temperature.csv` |
| 5.9b | (μ_N, μ_S) maps, T = 0.024 | [`ch5_critical_map_mu_high_temperature.png`](report/figures/ch5_critical_map_mu_high_temperature.png) | [`ch5_critical_map_mu.py`](scripts/ch5_critical_map_mu.py) | `data/ch5_critical_map_mu_high_temperature.csv` |
| 5.10 | I_max vs length of N | [`ch5_critical_vs_length.png`](report/figures/ch5_critical_vs_length.png) | [`ch5_critical_vs_length.py`](scripts/ch5_critical_vs_length.py) | `data/ch5_critical_vs_length.csv` |
| 6.1 | Self-consistent vs frozen step gap | [`ch6_square_well_comparison.png`](report/figures/ch6_square_well_comparison.png) | [`ch6_square_well_comparison.py`](scripts/ch6_square_well_comparison.py) | `data/ch6_square_well_comparison.csv` |
| 6.2 | Number of sub-gap states vs length | [`ch6_number_of_bound_states.png`](report/figures/ch6_number_of_bound_states.png) | [`ch6_number_of_bound_states.py`](scripts/ch6_number_of_bound_states.py) | `data/ch6_number_of_bound_states.csv` |
| 6.3a | φ_c, I_c vs T: short ballistic vs theory | [`ch6_short_junction_theory_short_ballistic.png`](report/figures/ch6_short_junction_theory_short_ballistic.png) | [`ch6_short_junction_theory.py`](scripts/ch6_short_junction_theory.py) | `data/ch6_short_junction_theory_short_ballistic.csv` |
| 6.3b | φ_c, I_c vs T: long junction vs theory | [`ch6_short_junction_theory_long.png`](report/figures/ch6_short_junction_theory_long.png) | [`ch6_short_junction_theory.py`](scripts/ch6_short_junction_theory.py) | `data/ch6_short_junction_theory_long.csv` |
| 6.3c | φ_c, I_c vs T: mismatch vs theory | [`ch6_short_junction_theory_short_mismatch.png`](report/figures/ch6_short_junction_theory_short_mismatch.png) | [`ch6_short_junction_theory.py`](scripts/ch6_short_junction_theory.py) | `data/ch6_short_junction_theory_short_mismatch.csv` |
| 6.5 | Beenakker formula with mismatch transmission | [`ch6_beenakker_mismatch.png`](report/figures/ch6_beenakker_mismatch.png) | [`ch6_short_junction_theory.py`](scripts/ch6_short_junction_theory.py) | `data/ch6_short_junction_theory_short_mismatch.csv` |
| 6.6 | Analytical transmission vs critical maps | [`ch6_transmission_vs_critical_map.png`](report/figures/ch6_transmission_vs_critical_map.png) | [`ch5_critical_map_mu.py`](scripts/ch5_critical_map_mu.py) | `data/ch5_critical_map_mu_high_temperature.csv` |
| 7.1a | T = 0 CPR, FPB (Ishii) | [`ch7_sawtooth_fpb.png`](report/figures/ch7_sawtooth_fpb.png) | [`ch7_long_junction.py`](scripts/ch7_long_junction.py) | `data/ch7_sawtooth_fpb.csv` |
| 7.1b | T = 0 CPR over [−2π, 2π], FPL | [`ch7_sawtooth_fpl.png`](report/figures/ch7_sawtooth_fpl.png) | [`ch7_long_junction.py`](scripts/ch7_long_junction.py) | `data/ch7_sawtooth_fpl.csv` |
| 7.2 | Bound-state and continuum currents | [`ch7_continuum_compensation.png`](report/figures/ch7_continuum_compensation.png) | [`ch7_long_junction.py`](scripts/ch7_long_junction.py) | `data/ch7_continuum_compensation.csv` |
| 7.3 | CPR slope vs length (1/L law) | [`ch7_slope_vs_length.png`](report/figures/ch7_slope_vs_length.png) | [`ch7_long_junction.py`](scripts/ch7_long_junction.py) | `data/ch7_slope_vs_length.csv` |

## Data

`data/` contains the simulation results (one CSV per figure) from which the figures are
drawn. A script recomputes its data when the CSV is missing or with `--recompute`.

## Tests

`tests/test_bdg.py` checks the solver against reference values, the particle-hole symmetry
of the spectrum, current conservation in the normal region, the absence of current at zero
phase, the critical-point search and the analytical limits (`python -m pytest tests`).

## Conventions

* Energies in units of `t`, lengths in lattice sites, currents in units of `2et/hbar`.
* Temperatures in Chapter 5 are given as `T / T_c` with `T_c = 0.24` for the leads.
* Chapter 7 is at `T = 0` (`beta = numpy.inf`).
