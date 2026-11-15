"""Chapter 5 - critical phase and current versus the chemical potential mu_N of N.

Report figures:
  "crit curr and phase as func of muN short junction"  FPL, N_N = 10
  "FPL_as_func_of_muN_long_junction"                   FPL, N_N = 30
  "FPB_as_func_of_muN_long_junction"                   FPB (BC = 3), N_N = 30
N_S = 30, mu_S = 0, T_c = 0.24. The Fabry-Perot oscillations come from the
normal reflection caused by the Fermi-wavevector mismatch (Chapter 6).
The two long-junction data sets take several hours each to compute.
Runtime: ~1 h (short); several hours per long-junction data set.
"""

import numpy as np
import pandas as pd
from _common import TC_LEADS, cached, param_box, parse_args, plt, save, scan_critical

import bdg

CASES = {
    "short": dict(n_n=10, width=None, temps=[0.001 * 0.24, 0.01 * 0.24, 0.05 * 0.24, 0.1 * 0.24, 0.2 * 0.24],
                  mu=np.linspace(0, 2, 200), tol=1e-8, min_step=1e-6, shrink=2),
    "long_fpl": dict(n_n=30, width=None, temps=[0.00024, 0.0024, 0.012, 0.024, 0.048],
                     mu=np.linspace(0, 2, 200), tol=1e-10, min_step=1e-5, shrink=2),
    "long_fpb": dict(n_n=30, width=3, temps=[0.02, 0.021, 0.024, 0.036, 0.048, 0.072],
                     mu=np.linspace(0, 2, 100), tol=1e-12, min_step=1e-5, shrink=10),
}


def compute(case):
    rows = []
    for temp in case["temps"]:
        def current_at(mu_n):
            j = bdg.Junction(n_s=30, n_n=case["n_n"], mu_n=mu_n)

            def current(phi):
                try:
                    sol = bdg.solve_self_consistent(j, phi, 1 / temp, pin_width=case["width"], tol=case["tol"])
                except bdg.ConvergenceError:
                    return np.nan
                return bdg.normal_current(sol)
            return current

        found = scan_critical(case["mu"], current_at, min_step=case["min_step"], shrink=case["shrink"],
                              nan_safe=True, label=lambda m, T=temp: f"T={T:.5f} mu_N={m:.3f}")
        for mu_n, cp in zip(case["mu"], found):
            phi = np.nan
            if cp.reliable:
                phi = cp.phase
                if case["width"]:          # FPB: report the effective phase across N
                    j = bdg.Junction(n_s=30, n_n=case["n_n"], mu_n=mu_n)
                    sol = bdg.solve_self_consistent(j, cp.phase, 1 / temp, pin_width=case["width"],
                                                    tol=case["tol"])
                    phi = bdg.effective_phase(sol)
            rows.append(dict(T=temp, mu_N=mu_n, I_max=cp.current if cp.reliable else np.nan, phi_max=phi))
    return pd.DataFrame(rows)


def main():
    args = parse_args(__doc__)
    for name, case in CASES.items():
        df = cached(f"ch5_critical_vs_muN_{name}", lambda: compute(case), args.recompute)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
        temps = sorted(df["T"].unique())
        for color, temp in zip(plt.cm.viridis(np.linspace(0, 1, len(temps))), temps):
            d = df[df["T"] == temp].sort_values("mu_N")
            ax1.plot(d.mu_N, d.phi_max, "-", color=color)
            ax2.plot(d.mu_N, d.I_max, "-", color=color, label=f"$T/T_c$={temp / TC_LEADS:.3f}")
        ax1.set_ylabel(r"Critical phase $\phi_{\max}$")
        ax2.set_xlabel(r"Chemical potential $\mu_N$")
        ax2.set_ylabel(r"Critical current $I_{\max}$ $(2et/\hbar)$")
        ax2.set_yscale("log")
        for ax in (ax1, ax2):
            ax.grid(True)
        ax2.legend(title="Temperature", ncol=2, loc="lower center")
        lines = ["$N_S$ = 30", f"$N_N$ = {case['n_n']}", "$t$ = 1.0", "$U_0$ = -2.0", r"$\mu_S$ = 0.0"]
        if case["width"]:
            lines.append(f"BC width = {case['width']}")
        param_box(ax2, lines, loc="lower left")
        save(fig, f"ch5_critical_vs_muN_{name}")


if __name__ == "__main__":
    main()
