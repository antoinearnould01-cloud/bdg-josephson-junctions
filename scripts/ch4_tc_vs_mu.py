"""Chapter 4 - mean-field phase diagram T_c(mu) of the uniform superconductor.

Report figures: "phase diagram Tc vs mu in 1D", "phase diagram Tc vs mu in 2D".
Solves the linearised gap equation (bdg.analytic) on a grid of mu.
Runtime: ~10 s (1D), ~10 min (2D, double integral per point).
"""

import numpy as np
import pandas as pd
from _common import cached, param_box, parse_args, plt, save

from bdg.analytic import critical_temperature_1d, critical_temperature_2d

CASES = {
    "1d": dict(u=-2.0, mu=np.linspace(-2.5, 2.5, 200), tc=critical_temperature_1d),
    "2d": dict(u=-6.0, mu=np.linspace(-5.5, 5.5, 100), tc=critical_temperature_2d),
}


def main():
    args = parse_args(__doc__)
    for dim, case in CASES.items():
        df = cached(f"ch4_tc_vs_mu_{dim}",
                    lambda: pd.DataFrame({"mu": case["mu"],
                                          "Tc": [case["tc"](m, u=case["u"]) for m in case["mu"]]}),
                    args.recompute)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.mu, df.Tc, color="firebrick", lw=2)
        ax.fill_between(df.mu, df.Tc, color="firebrick", alpha=0.1, label="Superconducting region")
        param_box(ax, [r"$t = 1.0$", r"$a = 1.0$", rf"$U = {case['u']:.1f}$"], fontsize=11)
        ax.set_xlabel(r"Chemical Potential $\mu$")
        ax.set_ylabel(r"Critical Temperature $T_c$")
        ax.grid(True)
        ax.legend(loc="center")
        save(fig, f"ch4_tc_vs_mu_{dim}")


if __name__ == "__main__":
    main()
