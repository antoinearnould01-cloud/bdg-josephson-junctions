"""Chapter 6 - number of sub-gap (Andreev) states versus the length of N.

Report figure: "nb_ABS_vs_L".
N_S = 300, N_N = 20 ... 200, phi = 0, T = 1/beta = 0.05, FPB with BC = 3.
A state is counted as sub-gap if |E| < gap, where the gap is the mean |Delta|
in the leads at least 50 sites away from the interfaces.
Runtime: ~20 min.
"""

import numpy as np
import pandas as pd
from _common import cached, param_box, parse_args, plt, save

import bdg

LENGTHS = np.arange(20, 201, 18)
BETA, BC_WIDTH = 20.0, 3


def compute():
    rows = []
    for n_n in LENGTHS:
        j = bdg.Junction(n_s=300, n_n=int(n_n))
        sol = bdg.solve_self_consistent(j, 0.0, BETA, pin_width=BC_WIDTH, tol=1e-8,
                                        initial=bdg.gap_linear_between_pins(j, 0.0, BC_WIDTH),
                                        error_norm="ratio_of_sums", raise_on_failure=False)
        gap = bdg.lead_gap(sol, margin=50)
        count = int(np.sum(np.abs(sol.eig.energies) < gap))
        print(f"N_N={n_n}: gap={gap:.4f}, sub-gap states={count}", flush=True)
        rows.append(dict(N_N=n_n, gap=gap, n_subgap=count))
    return pd.DataFrame(rows)


def main():
    args = parse_args(__doc__)
    df = cached("ch6_number_of_bound_states", compute, args.recompute)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df.N_N, df.n_subgap, "o-", color="teal", lw=2, ms=8, markeredgecolor="black")
    ax.set_xlabel("Normal region length", fontsize=14)
    ax.set_ylabel("Number of sub-gap states", fontsize=14)
    ax.grid(True, alpha=0.3)
    param_box(ax, ["$N_S = 300$", r"$t = 1.0,\ U_0 = -2.0$", r"$\mu_N = 0.0,\ \mu_S = 0.0$",
                   rf"$T = 1/\beta = {1 / BETA:.2f},\ \phi = 0$"], fontsize=12)
    save(fig, "ch6_number_of_bound_states")


if __name__ == "__main__":
    main()
