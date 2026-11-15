"""Chapter 6 - critical phase and current versus T: one-shot BdG against analytical CPRs.

Report figures (one-shot diagonalisation with the step gap, Delta_0 = 0.01, N_S = 30):
  "comparison_short_ballistic"  N_N = 1,  mu_N = 0     vs short ballistic formula
  "comparison_long"             N_N = 20, mu_N = 0     vs short ballistic formula
  "comparison_short_mismatch"   N_N = 1,  mu_N = 1.5   vs short ballistic formula
  "method with mismatch"        N_N = 1,  mu_N = 1.5   vs Beenakker with T(mu_N, mu_S, L)
                                                       and the ballistic formula (T = 1)
Analytical currents (bdg.analytic) are in units of 2et/hbar.
For each temperature the maximum of the CPR is located with bdg.critical.
Runtime: ~1 min.
"""

import numpy as np
import pandas as pd
from _common import cached, param_box, parse_args, plt, save
from scipy.optimize import minimize_scalar

import bdg
from bdg.analytic import cpr_beenakker, cpr_short_ballistic, transmission

DELTA0 = 0.01
TEMPS = np.linspace(0.01, 0.15, 50)
CASES = {
    "short_ballistic": dict(n_n=1, mu_n=0.0),
    "long": dict(n_n=20, mu_n=0.0),
    "short_mismatch": dict(n_n=1, mu_n=1.5),
}
# Normalisation of the one-shot current used in the report.
ONE_SHOT_FACTOR = 2


def numerical(n_n, mu_n):
    j = bdg.Junction(n_s=30, n_n=n_n, mu_n=mu_n, delta0=DELTA0)
    rows = []
    for temp in TEMPS:
        def current(phi):
            sol = bdg.solve_fixed_gap(j, bdg.gap_step(j, phi), 1 / temp)
            return ONE_SHOT_FACTOR * bdg.normal_current(sol, include_left_bond=True)
        cp = bdg.find_critical_point(current, phase_guess=np.pi / 2, step=0.5, min_step=1e-6)
        rows.append(dict(T=temp, phi_c=cp.phase, I_c=cp.current))
    return pd.DataFrame(rows)


def analytical(cpr):
    """(phi_c, I_c) of an analytical CPR for every temperature."""
    out = []
    for temp in TEMPS:
        res = minimize_scalar(lambda p: -cpr(p, temp), bounds=(0, np.pi), method="bounded",
                              options={"xatol": 1e-10})
        out.append((res.x, -res.fun))
    return np.array(out)


def axes_pair():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    ax1.set_ylabel(r"Critical phase $\phi_c$")
    ax2.set_ylabel(r"Critical current $I_c$")
    ax2.set_xlabel("Temperature $T$")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    return fig, ax1, ax2


def box(ax, n_n, mu_n, loc="upper right"):
    param_box(ax, ["Parameters:", f"$N_S = 30, N_N = {n_n}$", rf"$\Delta_0 = {DELTA0}, t = 1.0$",
                   rf"$U_0 = -2.0, \mu_N = {mu_n:g}$", r"$\mu_S = 0$"], loc=loc)


def main():
    args = parse_args(__doc__)
    ballistic = analytical(lambda p, T: cpr_short_ballistic(p, DELTA0, T))

    for name, case in CASES.items():
        df = cached(f"ch6_short_junction_theory_{name}", lambda: numerical(**case), args.recompute)
        fig, ax1, ax2 = axes_pair()
        ax1.plot(TEMPS, ballistic[:, 0], "-", label="Analytical")
        ax1.plot(df["T"], df.phi_c, "o-", ms=3, label="BdG numerical")
        ax2.plot(TEMPS, ballistic[:, 1], "-")
        ax2.plot(df["T"], df.I_c, "o-", ms=3)
        ax1.legend(loc="center right")
        box(ax1, **case)
        fig.tight_layout()
        save(fig, f"ch6_short_junction_theory_{name}")

    # Beenakker formula with the mismatch transmission
    case = CASES["short_mismatch"]
    trans = transmission(case["mu_n"], 0.0, length=case["n_n"])
    beenakker = analytical(lambda p, T: cpr_beenakker(p, DELTA0, T, trans))
    df = cached("ch6_short_junction_theory_short_mismatch", lambda: numerical(**case))
    fig, ax1, ax2 = axes_pair()
    fig.set_size_inches(8, 7)
    for ax, col in ((ax1, 0), (ax2, 1)):
        ax.plot(TEMPS, beenakker[:, col], "-", color="C0", label="Analytical solution including mismatch")
        ax.plot(TEMPS, ballistic[:, col], "-", color="C1", label="Analytical solution without mismatch")
    ax1.plot(df["T"], df.phi_c, "s-", ms=3, color="C2", label="BdG numerical")
    ax2.plot(df["T"], df.I_c, "s-", ms=3, color="C2")
    ax1.set_ylim(1.570, 1.583)
    ax1.legend(loc="upper right")
    box(ax2, **case)
    fig.tight_layout()
    save(fig, "ch6_beenakker_mismatch")


if __name__ == "__main__":
    main()
