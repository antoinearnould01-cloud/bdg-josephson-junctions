"""Chapter 5 and 6 - maps of phi_max and I_max in the (mu_N, mu_S) plane.

Report figures:
  "color grid muN muS low temp"   T = 0.0001
  "color grid muN muS high temp"  T = 0.024
  "Transmittance_1" (Chapter 6)   the T = 0.024 maps next to the analytical
                                  transmission T(mu_N, mu_S) of the N region
FPL, N_S = 30, N_N = 10, 40 x 40 grid. Each pixel is a critical-current search.
Runtime: ~2 h per temperature (40 x 40 grid); use GRID = 20 for a quick look.
"""

import numpy as np
import pandas as pd
from _common import cached, parse_args, plt, save, scan_critical
from matplotlib.colors import LogNorm

import bdg
from bdg.analytic import transmission

N_N, GRID = 10, 40
MU = np.linspace(0, 2, GRID)
TEMPS = {"high_temperature": 0.024, "low_temperature": 0.0001}


def compute(temp):
    rows = []
    for mu_s in MU:
        def current_at(mu_n, mu_s=mu_s):
            j = bdg.Junction(n_s=30, n_n=N_N, mu_n=mu_n, mu_s=mu_s)
            return lambda phi: bdg.normal_current(bdg.solve_self_consistent(j, phi, 1 / temp, tol=1e-10))

        found = scan_critical(MU, current_at, min_step=1e-5, shrink=2,
                              label=lambda m, s=mu_s: f"T={temp} mu_S={s:.3f} mu_N={m:.3f}")
        rows += [dict(mu_S=mu_s, mu_N=mu_n, phi_max=cp.phase if cp.reliable else np.nan,
                      I_max=cp.current if cp.reliable else np.nan) for mu_n, cp in zip(MU, found)]
    return pd.DataFrame(rows)


def grid(df, column):
    return df.pivot(index="mu_S", columns="mu_N", values=column)


def draw_maps(fig, axes, df):
    phase, current = grid(df, "phi_max"), grid(df, "I_max")
    im = axes[0].pcolormesh(phase.columns, phase.index, phase.values, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="rad")
    axes[0].set_title(r"Critical phase $\phi_{\max}$")
    im = axes[1].pcolormesh(current.columns, current.index, current.values, shading="auto",
                            cmap="plasma", norm=LogNorm())
    fig.colorbar(im, ax=axes[1], label=r"$\log_{10}(I_{\max})$")
    axes[1].set_title(r"Critical current $I_{\max}$ $(2et/\hbar)$")
    for ax in axes:
        ax.set_xlabel(r"$\mu_N$")
        ax.set_ylabel(r"$\mu_S$")


def box(fig, temp):
    fig.text(0.01, 0.99, f"$N_S$ = 30\n$N_N$ = {N_N}\n$t$ = 1.0\n$U_0$ = -2.0\n$T$ = {temp}",
             fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))


def plot_transmission(df):
    """Chapter 6: high-temperature maps next to the analytical transmission."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    draw_maps(fig, axes[:2], df)
    mu_n, mu_s = np.meshgrid(np.linspace(0, 2, 400), np.linspace(0, 2, 400))
    with np.errstate(divide="ignore", invalid="ignore"):
        trans = transmission(mu_n, mu_s, length=N_N)
    im = axes[2].pcolormesh(mu_n, mu_s, trans, shading="auto", cmap="coolwarm")
    fig.colorbar(im, ax=axes[2], label=r"$\mathcal{T}$")
    axes[2].set_title(r"Theoretical transparency $\mathcal{T}$")
    axes[2].set_xlabel(r"$\mu_N$")
    axes[2].set_ylabel(r"$\mu_S$")
    box(fig, TEMPS["high_temperature"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "ch6_transmission_vs_critical_map")


def main():
    args = parse_args(__doc__)
    for name, temp in TEMPS.items():
        df = cached(f"ch5_critical_map_mu_{name}", lambda: compute(temp), args.recompute)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        draw_maps(fig, axes, df)
        box(fig, temp)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save(fig, f"ch5_critical_map_mu_{name}")
        if name == "high_temperature":
            plot_transmission(df)


if __name__ == "__main__":
    main()
