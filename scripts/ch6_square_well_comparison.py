"""Chapter 6 - self-consistent CPR versus one-shot diagonalisation with a step gap.

Report figure: "square well comparison".
N_S = 20, N_N = 5, mu_N = mu_S = 0, beta = 10, FPL.
  self-consistent : full BdG loop, CPR plotted against the effective phase
  one-shot        : frozen step gap (eq. Delta_square) of amplitude |Delta|
                    on the last lead site of the self-consistent solution
Runtime: ~30 s.
"""

import numpy as np
import pandas as pd
from _common import cached, parse_args, plt, save

import bdg

JUNCTION = bdg.Junction(n_s=20, n_n=5, u0=-2.0, mu_n=0.0, mu_s=0.0, delta0=0.3)
BETA, TOL = 10.0, 1e-10
PHASES = np.linspace(0, np.pi, 20)

# Normalisation of the one-shot current used in the report.
ONE_SHOT_FACTOR = 2


def compute():
    rows = []
    j = JUNCTION
    for phi in PHASES:
        sol = bdg.solve_self_consistent(j, phi, BETA, initial=bdg.gap_step(j, phi), tol=TOL)
        lead_edge_gap = abs(sol.delta[j.n_s - 1])
        fixed = bdg.solve_fixed_gap(j, bdg.gap_step(j, phi, amplitude=lead_edge_gap), BETA)
        rows.append(dict(phase=phi, phi_eff=bdg.effective_phase(sol),
                         I_self_consistent=bdg.normal_current(sol, include_left_bond=True),
                         I_one_shot=ONE_SHOT_FACTOR * bdg.normal_current(fixed, include_left_bond=True)))
    df = pd.DataFrame(rows)
    df["phi_eff"] = np.unwrap(df.phi_eff)
    return df


def main():
    args = parse_args(__doc__)
    df = cached("ch6_square_well_comparison", compute, args.recompute)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df.phi_eff, df.I_self_consistent, "o-", ms=8, color="C0", label="Self-consistent")
    ax.plot(df.phase, df.I_one_shot, "s--", ms=8, color="C1", label="Non-self-consistent")
    ax.set_xlabel(r"Phase difference $\varphi$")
    ax.set_ylabel(r"Current $I$")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right", fontsize=12)
    j = JUNCTION
    ax.text(0.98, 0.80, "\n".join([f"$L_S = {j.n_s}$", f"$L_N = {j.n_n}$", rf"$\Delta_0 = {j.delta0}$",
                                   f"$t = {j.t}$", f"$U_0 = {j.u0}$", rf"$\mu_N = {j.mu_n}$",
                                   rf"$\mu_S = {j.mu_s}$", rf"$\beta = {BETA}$"]),
            transform=ax.transAxes, fontsize=12, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    save(fig, "ch6_square_well_comparison")


if __name__ == "__main__":
    main()
