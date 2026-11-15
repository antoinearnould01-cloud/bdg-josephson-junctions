"""Chapter 4 - bulk order parameter Delta(T) and the mean-field law A sqrt(T_c - T).

Report figures: "Delta vs T in 1D", "Delta vs T in 2D".
The amplitude A is fitted on the points with 0.9 T_c < T < T_c.
Runtime: ~5 s (1D), ~5 min (2D).
"""

import numpy as np
import pandas as pd
from _common import cached, param_box, parse_args, plt, save

from bdg.analytic import gap_1d, gap_2d

CASES = {
    "1d": dict(temps=np.linspace(0, 0.23, 200), gap=gap_1d, box="lower left"),
    "2d": dict(temps=np.linspace(0, 0.04, 41), gap=gap_2d, box="lower left"),
}
U, MU = -2.0, 0.0


def main():
    args = parse_args(__doc__)
    for dim, case in CASES.items():
        df = cached(f"ch4_gap_vs_temperature_{dim}",
                    lambda: pd.DataFrame({"T": case["temps"],
                                          "Delta": [case["gap"](T, mu=MU, u=U) for T in case["temps"]]}),
                    args.recompute)
        temps, gaps = df["T"].to_numpy(), df["Delta"].to_numpy()

        tc = temps[gaps > 0].max()
        near = (temps > 0.9 * tc) & (gaps > 0)
        amp = np.sum(gaps[near] * np.sqrt(tc - temps[near])) / np.sum(tc - temps[near])
        t_fit = np.linspace(0.5 * tc, tc, 200)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(temps, gaps, "b-", lw=2, label=r"Numerical $\Delta(T)$")
        ax.plot(t_fit, amp * np.sqrt(tc - t_fit), "r--", lw=2, label=r"Fit: $A\sqrt{T_c - T}$")
        param_box(ax, [r"$t = 1.0$", r"$a = 1.0$", rf"$U = {U:.1f}$", rf"$\mu = {MU:.1f}$"],
                  loc=case["box"], fontsize=11)
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel(r"Order Parameter $\Delta$")
        ax.grid(True)
        ax.legend(loc="upper right")
        save(fig, f"ch4_gap_vs_temperature_{dim}")


if __name__ == "__main__":
    main()
