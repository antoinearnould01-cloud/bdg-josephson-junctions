"""Chapter 5 - profiles of |Delta|, |F|, arg F and the bond current, FPL vs FPB.

Report figures: "FPL" and "FPB" (Section "Phase profile: FPL and FPB").
One self-consistent solution each, N_S = 40, N_N = 30, T = 0.02, phi = pi/2.
  FPL: phase pinned on the whole leads.
  FPB: phase pinned on the 5 outermost sites only.
Runtime: ~30 s.
"""

import numpy as np
import pandas as pd
from _common import cached, parse_args, plt, save
from matplotlib.ticker import NullFormatter

import bdg

JUNCTION = bdg.Junction(n_s=40, n_n=30, u0=-2.0, mu_s=0.0, mu_n=0.0, delta0=0.3)
BETA, PHASE, TOL, BC_WIDTH = 50.0, np.pi / 2, 1e-14, 5


def compute(scheme):
    if scheme == "fpl":
        sol = bdg.solve_self_consistent(JUNCTION, PHASE, BETA, tol=TOL)
    else:
        init = bdg.gap_linear_between_pins(JUNCTION, PHASE, BC_WIDTH)
        sol = bdg.solve_self_consistent(JUNCTION, PHASE, BETA, pin_width=BC_WIDTH, initial=init, tol=TOL)
    return pd.DataFrame({"site": np.arange(JUNCTION.n_sites), "abs_delta": np.abs(sol.delta),
                         "abs_F": np.abs(sol.pair), "arg_F": np.angle(sol.pair),
                         "current": sol.currents()})


def plot(df, scheme):
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax1.plot(df.site, df.abs_delta, color="C0", lw=1.5, label=r"Gap $|\Delta_i|$")
    ax1.plot(df.site, df.abs_F, "--", color="C0", alpha=0.7, label=r"Correlation $|F_i|$")
    ax1.set_yscale("log")
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_ylabel(r"$|\Delta_i|,\ |F_i|$")
    ax1.legend(loc="lower left", fontsize=10)
    ax2.plot(df.site, df.arg_F, color="C1", lw=1.5)
    ax2.set_ylabel(r"$\arg(F_i)$")
    ax3.plot(df.site, df.current, color="C2", lw=1.5)
    ax3.set_ylabel(r"$J_{i\to i+1}\ (2te/\hbar)$")
    ax3.set_xlabel("Site index")

    j = JUNCTION
    lines = [rf"$N_L=N_R={j.n_s}$, $N_N={j.n_n}$", rf"$\mu_S={j.mu_s:g}$, $\mu_N={j.mu_n:g}$",
             rf"$U={j.u0:.2f}$, $T=1/\beta={1 / BETA:.2f}$",
             rf"$\phi={PHASE / np.pi:.2f}\,\pi$" + (f", $BC={BC_WIDTH}$" if scheme == "fpb" else "")]
    ax1.text(0.98, 0.96, "\n".join(lines), transform=ax1.transAxes, fontsize=11, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, j.n_sites - 1)
        ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout(pad=0.5)
    save(fig, f"ch5_phase_profile_{scheme}")


def main():
    args = parse_args(__doc__)
    for scheme in ("fpl", "fpb"):
        plot(cached(f"ch5_phase_profile_{scheme}", lambda: compute(scheme), args.recompute), scheme)


if __name__ == "__main__":
    main()
