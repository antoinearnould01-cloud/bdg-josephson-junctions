"""Chapter 7 - long ballistic junctions at T = 0 and the Ishii sawtooth.

Report figures:
  "sawtooth cpr ishii FPB"               N_N = 30, N_S = 30, BC = 5, phi_eff in [0, ~1.1]
  "sawtooth cpr ishii FPL"               N_N = 30, N_S = 30, imposed phase in [-2pi, 2pi]
  "sawtooth cpr ishii comparison continu" N_N = 400, N_S = 100, BC = 5: total current split
                                         into sub-gap (|E| < Delta) and continuum states
  "slope CPR vs length long junc"        slope dI/dphi at small phase, N_N = 30 ... 200
mu_N = mu_S = 0, U_0 = -2, T = 0 (beta = inf).
For the N_N = 30 CPRs consecutive phases are warm-started from the previous
converged gap; the N_N = 400 decomposition solves every phase from scratch.
Runtime: ~5 min (N_N = 30 figures), ~15 min (slope), ~10 h (N_N = 400).
"""

import numpy as np
import pandas as pd
from _common import cached, param_box, parse_args, plt, save

import bdg

BETA, TOL, BC_WIDTH = np.inf, 1e-10, 5


def rephase(delta, old, new):
    """Shift a converged gap from imposed phase `old` to `new` with a linear phase ramp."""
    x = np.linspace(-0.5, 0.5, len(delta))
    return delta * np.exp(1j * (new - old) * x)


def cpr(junction, phases, width, split=False, warm_start=True):
    rows, prev = [], None
    for phi in phases:
        init = (bdg.gap_linear_between_pins(junction, phi, width) if prev is None
                else rephase(prev[0], prev[1], phi))
        sol = bdg.solve_self_consistent(junction, phi, BETA, pin_width=width, initial=init, tol=TOL)
        if warm_start:
            prev = (sol.delta, phi)
        row = dict(phase=phi, phi_eff=bdg.effective_phase(sol),
                   current=bdg.normal_current(sol, drop_last_bond=True))
        if split:
            gap = junction.delta0
            sub, cont = bdg.split_current(sol, gap)
            normal = slice(junction.n_s, junction.n_s + junction.n_n - 1)
            row.update(gap=gap, current_subgap=float(np.mean(sub[normal])),
                       current_continuum=float(np.mean(cont[normal])))
        print(f"N_N={junction.n_n} phi={phi:+.3f} phi_eff={row['phi_eff']:+.4f} I={row['current']:.5e}"
              f" ({sol.iterations} it)", flush=True)
        rows.append(row)
    return pd.DataFrame(rows)


def box(ax, j, width=None, loc="upper left"):
    lines = [f"$N$ = {j.n_sites}", f"$N_N$ = {j.n_n}", f"$N_S$ = {j.n_s}"]
    if width:
        lines.append(f"$BC$ = {width}")
    lines += ["$t$ = 1.0", "$U_0$ = -2.0", r"$\mu_N$ = 0.0, $\mu_S$ = 0.0", r"$T = 1/\beta$ = 0"]
    param_box(ax, lines, loc=loc, fontsize=11)


def cpr_axes():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r"Phase difference $\phi$")
    ax.set_ylabel(r"Current $I$ $(2et/\hbar)$")
    ax.grid(True)
    return fig, ax


def main():
    args = parse_args(__doc__)
    short = bdg.Junction(n_s=30, n_n=30)

    df = cached("ch7_sawtooth_fpb", lambda: cpr(short, np.linspace(0, np.pi, 40), BC_WIDTH), args.recompute)
    fig, ax = cpr_axes()
    ax.plot(df.phi_eff, df.current, lw=2)
    box(ax, short, BC_WIDTH)
    save(fig, "ch7_sawtooth_fpb")

    # FPL: start at 0 and go outwards in both directions so that warm starts stay close
    def fpl():
        up = cpr(short, np.linspace(0, 2 * np.pi, 201), short.n_s)
        down = cpr(short, np.linspace(0, -2 * np.pi, 201)[1:], short.n_s)
        return pd.concat([down, up]).sort_values("phase")
    df = cached("ch7_sawtooth_fpl", fpl, args.recompute)
    fig, ax = cpr_axes()
    ax.plot(df.phase, df.current, lw=2)
    box(ax, short, loc="lower left")
    save(fig, "ch7_sawtooth_fpl")

    lengths = np.arange(30, 201, 10)

    def slopes():
        rows = []
        for n_n in lengths:
            j = bdg.Junction(n_s=30, n_n=int(n_n))
            d = cpr(j, [0.2], BC_WIDTH)
            rows.append(dict(N_N=n_n, slope=d.current[0] / d.phi_eff[0]))
        return pd.DataFrame(rows)
    df = cached("ch7_slope_vs_length", slopes, args.recompute)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.loglog(df.N_N, df.slope, "o-")
    ax.set_xlabel("Normal region length")
    ax.set_ylabel(r"CPR slope $dI/d\phi$")
    ax.grid(True, which="both", alpha=0.4)
    param_box(ax, ["$U_0$ = -2.00", r"$\mu_N$ = 0", r"$\mu_S$ = 0", r"$T = 1/\beta$ = 0",
                   r"$\Delta_0$ = 0.3", f"BC width = {BC_WIDTH}", "$N_S$ = 30", "$t$ = 1.0"],
              loc="lower left", fontsize=11)
    save(fig, "ch7_slope_vs_length")

    long = bdg.Junction(n_s=100, n_n=400)
    df = cached("ch7_continuum_compensation",
                lambda: cpr(long, np.linspace(0, np.pi, 25), BC_WIDTH, split=True, warm_start=False), args.recompute)
    fig, ax = cpr_axes()
    ax.plot(df.phi_eff, df.current, "-", lw=2, label="Total current")
    ax.plot(df.phi_eff, df.current_subgap, "--", label=r"Subgap current ($|E| < \Delta_0$)")
    ax.plot(df.phi_eff, df.current_continuum, ":", lw=2, label=r"Above-gap current ($|E| > \Delta_0$)")
    ax.legend(loc="center right")
    box(ax, long, BC_WIDTH)
    save(fig, "ch7_continuum_compensation")


if __name__ == "__main__":
    main()
