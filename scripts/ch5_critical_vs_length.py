"""Chapter 5 - critical current versus the length N_N of the normal region.

Report figure: "crit curr L".
FPL, N_S = 30, mu_N = mu_S = 0, N_N = 10 ... 50, T/T_c = 0.05, 0.1, 0.2, 0.5 (T_c = 0.24).
Runtime: ~10 min.
"""

import numpy as np
import pandas as pd
from _common import TC_LEADS, cached, param_box, parse_args, plt, save, scan_critical

import bdg

LENGTHS = np.arange(10, 51, 5)
T_OVER_TC = [0.5, 0.2, 0.1, 0.05]


def compute():
    rows = []
    for ratio in T_OVER_TC:
        temp = ratio * TC_LEADS

        def current_at(n_n):
            j = bdg.Junction(n_s=30, n_n=int(n_n))
            return lambda phi: bdg.normal_current(bdg.solve_self_consistent(j, phi, 1 / temp, tol=1e-12))

        found = scan_critical(LENGTHS, current_at, warm_start=False, min_step=1e-5, shrink=2,
                              coarse_phases=np.linspace(0.5, 2.5, 10),
                              label=lambda n, r=ratio: f"T/Tc={r} N_N={n}")
        rows += [dict(T=temp, N_N=n, phi_max=cp.phase if cp.reliable else np.nan,
                      I_max=cp.current if cp.reliable else np.nan) for n, cp in zip(LENGTHS, found)]
    return pd.DataFrame(rows)


def main():
    args = parse_args(__doc__)
    df = cached("ch5_critical_vs_length", compute, args.recompute)
    fig, ax = plt.subplots(figsize=(8, 6))
    for color, ratio in zip(plt.cm.viridis(np.linspace(0, 1, len(T_OVER_TC))), T_OVER_TC):
        d = df[np.isclose(df["T"], ratio * TC_LEADS)]
        ax.plot(d.N_N, d.I_max, "s-", color=color, label=f"$T/T_c$={ratio:.3f}")
    ax.set_xlabel(r"$N$ (normal region length)")
    ax.set_ylabel(r"Critical current $I_{\max}$ $(2et/\hbar)$")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(title="Temperature", loc="lower center")
    param_box(ax, ["$N_S$ = 30", "$t$ = 1.0", "$U_0$ = -2.0", r"$\mu_S$ = 0.0", r"$\mu_N$ = 0.0"],
              loc="lower left", fontsize=11)
    save(fig, "ch5_critical_vs_length")


if __name__ == "__main__":
    main()
