"""Chapter 5 - current-phase relation at several temperatures.

Report figures:
  "CPR for diff temp short junction"  N_N = 10, FPL, T/T_c = 0.001 ... 0.5 (T_c = 0.24)
  "CPR for diff temp long junction"   N_N = 30, FPB with BC = 3, T/T_c = 0.09 ... 0.5 (T_c = 0.23)
N_S = 30, mu_N = mu_S = 0. The imposed phase is scanned over [0, pi] and the
CPR is drawn against the effective phase across N. FPB points whose solve
does not converge are skipped.
Runtime: ~3 min (short), ~25 min (long).
"""

import numpy as np
import pandas as pd
from _common import cached, param_box, parse_args, plt, save

import bdg

CASES = {
    "short": dict(n_n=10, width=None, tc=0.24, tol=1e-12, n_phases=200,
                  temps=[0.00024, 0.012, 0.024, 0.2 * 0.24, 0.12]),
    "long": dict(n_n=30, width=3, tc=0.23, tol=1e-6, n_phases=50,
                 temps=[0.09 * 0.23, 0.1 * 0.23, 0.15 * 0.23, 0.2 * 0.23, 0.5 * 0.23]),
}


def compute(case):
    j = bdg.Junction(n_s=30, n_n=case["n_n"])
    width = case["width"] or j.n_s
    rows = []
    for temp in case["temps"]:
        for phi in np.linspace(0, np.pi, case["n_phases"]):
            try:
                init = bdg.gap_linear_between_pins(j, phi, width)
                sol = bdg.solve_self_consistent(j, phi, 1 / temp, pin_width=width, initial=init,
                                                tol=case["tol"])
            except bdg.ConvergenceError as err:
                print(f"skipped: {err}")
                continue
            rows.append(dict(T=temp, phase=phi, phi_eff=bdg.effective_phase(sol),
                             current=bdg.normal_current(sol, drop_last_bond=True)))
    return pd.DataFrame(rows)


def main():
    args = parse_args(__doc__)
    for name, case in CASES.items():
        df = cached(f"ch5_cpr_temperature_{name}", lambda: compute(case), args.recompute)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(case["temps"])))
        for color, temp in zip(colors, case["temps"]):
            d = df[np.isclose(df["T"], temp)]
            d = d[(d.phi_eff >= -0.1) & (d.phi_eff <= np.pi + 0.1)]
            digits = 4 if name == "short" else 3
            ax.plot(d.phi_eff, d.current, "-", lw=2, color=color,
                    label=f"$T/T_c$ = {temp / case['tc']:.{digits}f}")
        ax.set_xlabel(r"Phase difference $\phi$")
        ax.set_ylabel(r"Current $I$ $(2et/\hbar)$")
        ax.grid(True)
        lines = [f"$N_N$ = {case['n_n']}, $N_S$ = 30"]
        if case["width"]:
            lines.append(f"$BC$ = {case['width']}")
        lines += ["$t$ = 1.0, $U_0$ = -2.0", r"$\mu_N$ = 0.0, $\mu_S$ = 0.0"]
        param_box(ax, lines, fontsize=11)
        ax.legend(title="Temperature", loc="upper right" if name == "long" else "center left")
        save(fig, f"ch5_cpr_temperature_{name}")


if __name__ == "__main__":
    main()
