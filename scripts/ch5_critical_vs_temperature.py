"""Chapter 5 - critical phase phi_max and critical current I_max versus temperature.

Report figures:
  "crit curr and phase as func of T short junction mu zero"
        FPL, N_N = 8, 10, 12, T from 0.001 to 0.75 T_c
  "FPB and FPL crit curr and phase as func of T long junction"
        N_N = 20, 30, 40, T from 0.001 to 0.5 T_c, FPL (phase pinned in the
        whole leads) and FPB (BC = 3; phi_max is the effective phase across N)
N_S = 30, mu_N = mu_S = 0, T_c = 0.24. Each point is a hill-climb on the CPR
(bdg.critical), warm-started from the previous temperature.
Runtime: ~10 min (short); FPB long junction several hours.
"""

import numpy as np
import pandas as pd
from _common import TC_LEADS, cached, param_box, parse_args, plt, save, scan_critical

import bdg

SHORT = dict(lengths=[8, 10, 12], temps=np.linspace(0.001, 0.24 * 0.75, 20))
LONG = dict(lengths=[20, 30, 40], temps=np.linspace(0.001, 0.24 * 0.5, 30), widths=[30, 3])


def compute_short():
    rows = []
    for n_n in SHORT["lengths"]:
        j = bdg.Junction(n_s=30, n_n=n_n)

        def current_at(temp):
            return lambda phi: bdg.normal_current(bdg.solve_self_consistent(j, phi, 1 / temp, tol=1e-14))

        found = scan_critical(SHORT["temps"], current_at, min_step=1e-6, shrink=2,
                              coarse_phases=np.linspace(0.5, 2.5, 10),
                              label=lambda T, n=n_n: f"N_N={n} T={T:.4f}")
        rows += [dict(N_N=n_n, T=T, phi_max=cp.phase if cp.reliable else np.nan,
                      I_max=cp.current if cp.reliable else np.nan)
                 for T, cp in zip(SHORT["temps"], found)]
    return pd.DataFrame(rows)


def compute_long():
    rows = []
    for width in LONG["widths"]:
        for n_n in LONG["lengths"]:
            j = bdg.Junction(n_s=30, n_n=n_n)

            def current_at(temp):
                def current(phi):
                    try:
                        sol = bdg.solve_self_consistent(j, phi, 1 / temp, pin_width=width, tol=1e-14)
                    except bdg.ConvergenceError:
                        return np.nan
                    return bdg.normal_current(sol, drop_last_bond=True)
                return current

            found = scan_critical(LONG["temps"], current_at, min_step=1e-5, shrink=10,
                                  label=lambda T, n=n_n, w=width: f"BC={w} N_N={n} T={T:.4f}")
            for T, cp in zip(LONG["temps"], found):
                keep = cp.reliable and not (width == 3 and cp.phase > np.pi)
                phi_eff = np.nan
                if keep:
                    sol = bdg.solve_self_consistent(j, cp.phase, 1 / T, pin_width=width, tol=1e-14)
                    phi_eff = bdg.effective_phase(sol)
                rows.append(dict(BC=width, N_N=n_n, T=T, phi_max=phi_eff,
                                 I_max=cp.current if keep else np.nan))
    return pd.DataFrame(rows)


def two_panels():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    ax1.set_ylabel(r"Critical phase $\phi_{\max}$")
    ax2.set_xlabel(r"Temperature $T/T_c$")
    ax2.set_ylabel(r"Critical current $I_{\max}$ $(2et/\hbar)$")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.grid(True)
    return fig, ax1, ax2


def plot_short(recompute=False):
    df = cached("ch5_critical_vs_temperature_short", compute_short, recompute)
    fig, ax1, ax2 = two_panels()
    colors = plt.cm.viridis(np.linspace(0, 1, len(SHORT["lengths"])))
    for color, n_n in zip(colors, SHORT["lengths"]):
        d = df[df.N_N == n_n]
        ax1.plot(d["T"] / TC_LEADS, d.phi_max, "o-", color=color, label=f"N={n_n}")
        ax2.plot(d["T"] / TC_LEADS, d.I_max, "s-", color=color, label=f"N={n_n}")
    ax2.legend(title="Normal region length", loc="lower center")
    param_box(ax2, ["$N_S$ = 30", "$t$ = 1.0", "$U_0$ = -2.0", r"$\mu_S$ = 0.0", r"$\mu_N$ = 0.0"],
              loc="lower left")
    save(fig, "ch5_critical_vs_temperature_short")


def plot_long(recompute=False):
    df = cached("ch5_critical_vs_temperature_long", compute_long, recompute)
    fig, ax1, ax2 = two_panels()
    styles = {30: ("FPL", ["red", "orange", "darkred"]), 3: ("FPB", ["darkblue", "deepskyblue", "navy"])}
    for width, (name, colors) in styles.items():
        for i, n_n in enumerate(LONG["lengths"]):
            d = df[(df.BC == width) & (df.N_N == n_n)].dropna()
            kw = dict(color=colors[i], marker="os^"[i], ls=["-", "--", "-."][i], ms=5,
                      label=f"{name}, $N_N$={n_n}")
            ax1.plot(d["T"] / TC_LEADS, d.phi_max, **kw)
            ax2.plot(d["T"] / TC_LEADS, d.I_max, **kw)
    ax1.legend(title="Boundary condition, N", fontsize=8, loc="upper right")
    param_box(ax2, ["$N_S$ = 30", "$t$ = 1.0", "$U_0$ = -2.0", r"$\mu_S$ = 0.0", r"$\mu_N$ = 0.0",
                    "BC width = 3"], loc="lower left")
    save(fig, "ch5_critical_vs_temperature_long")


def main():
    args = parse_args(__doc__)
    plot_short(args.recompute)
    plot_long(args.recompute)


if __name__ == "__main__":
    main()
