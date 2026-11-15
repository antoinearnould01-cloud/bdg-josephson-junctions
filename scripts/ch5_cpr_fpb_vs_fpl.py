"""Chapter 5 - CPR of a fully transparent junction with the FPL and FPB schemes.

Report figures:
  "FPB vs FPL long junction"   N_N = 30, BC = 3
  "simple CPR short junction"  N_N = 10, BC = 3
N_S = 30, mu_N = mu_S = 0, T = 1/beta = 0.025.
The imposed phase is scanned from 0 to pi; the CPR is plotted against the
effective phase across N (for FPL it equals the imposed phase). With FPB on a
short junction most of the bias drops in the leads, so only phi_eff < ~0.5 is
reached (discussed in the report).
Runtime: ~15 min (FPB solves dominate).
"""

import numpy as np
import pandas as pd
from _common import cached, param_box, parse_args, plt, save

import bdg

BETA, TOL, BC_WIDTH = 40.0, 1e-10, 3
PHASES = np.linspace(0, np.pi, 50)
CASES = {"long": 30, "short": 10}


def compute(n_n):
    j = bdg.Junction(n_s=30, n_n=n_n)
    rows = []
    for scheme, width in (("FPL", j.n_s), ("FPB", BC_WIDTH)):
        for phi in PHASES:
            init = bdg.gap_linear_between_pins(j, phi, width)
            sol = bdg.solve_self_consistent(j, phi, BETA, pin_width=width, initial=init, tol=TOL)
            rows.append(dict(scheme=scheme, phase=phi, phi_eff=bdg.effective_phase(sol),
                             current=bdg.normal_current(sol, drop_last_bond=True)))
    return pd.DataFrame(rows)


def main():
    args = parse_args(__doc__)
    for case, n_n in CASES.items():
        df = cached(f"ch5_cpr_fpb_vs_fpl_{case}", lambda: compute(n_n), args.recompute)
        fig, ax = plt.subplots(figsize=(8, 6))
        for scheme, style in (("FPB", "-"), ("FPL", "--")):
            d = df[df.scheme == scheme]
            ax.plot(d.phi_eff, d.current, style, lw=2, label=scheme)
        ax.set_xlabel(r"Phase difference $\phi$")
        ax.set_ylabel(r"Current $I$ $(2et/\hbar)$")
        ax.grid(True)
        ax.legend(loc="upper right")
        param_box(ax, [rf"$N_N$ = {n_n}, $N_S$ = 30", rf"$BC$ = {BC_WIDTH}", "$t$ = 1.0, $U_0$ = -2.0",
                       r"$\mu_N$ = 0.0, $\mu_S$ = 0.0", rf"$T=1/\beta$ = {1 / BETA:.3f}"], fontsize=11)
        save(fig, f"ch5_cpr_fpb_vs_fpl_{case}")


if __name__ == "__main__":
    main()
