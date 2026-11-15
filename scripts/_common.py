"""Shared helpers for the figure scripts: paths, data cache, plotting style."""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
FIGURES = ROOT / "figures"

# T_c of the leads used to label temperatures as T / T_c in Chapter 5.
TC_LEADS = 0.24

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--recompute", action="store_true",
                        help="ignore the cached data in data/ and run the simulation again")
    return parser.parse_args()


def cached(name: str, compute, recompute: bool = False) -> pd.DataFrame:
    """Load data/<name>.csv, or run `compute()` (returning a DataFrame) and save it."""
    path = DATA / f"{name}.csv"
    if path.exists() and not recompute:
        return pd.read_csv(path)
    df = compute()
    DATA.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def save(fig, name: str) -> Path:
    """Save to figures/<name>.png."""
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")
    return path


def param_box(ax, lines, loc="upper left", fontsize=11):
    """Small framed box listing the simulation parameters."""
    positions = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, fontsize=fontsize,
            ha=ha, va=va, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))


def scan_critical(points, current_at, warm_start=True, reset_when=None, label=str, **search):
    """Critical point for each item of `points`.

    current_at(point) must return a function phase -> current.
    warm_start  : start each search from the optimum of the previous point.
    reset_when  : optional predicate(point) -> True to restart from pi/2.
    Extra keyword arguments are passed to bdg.find_critical_point.
    Returns a list of CriticalPoint (unreliable maxima are kept, flagged).
    """
    from bdg import find_critical_point

    results, guess = [], np.pi / 2
    for point in points:
        if not warm_start or (reset_when is not None and reset_when(point)):
            guess = np.pi / 2
        t0 = time.perf_counter()
        cp = find_critical_point(current_at(point), phase_guess=guess, **search)
        guess = cp.phase
        flag = "" if cp.reliable else "  (unreliable maximum, discarded)"
        print(f"{label(point)}: phi_max={cp.phase:.4f}, I_max={cp.current:.5e} "
              f"[{time.perf_counter() - t0:.1f}s]{flag}", flush=True)
        results.append(cp)
    return results
