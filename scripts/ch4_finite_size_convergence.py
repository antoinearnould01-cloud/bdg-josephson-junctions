"""Chapter 4 - convergence of the order parameter with the system size N.

Report figure: "Error vs N".
Compares, for a uniform chain at T = 0.001, mu = 0.001:
  * the infinite-system gap (integral gap equation),
  * the gap with periodic boundary conditions (discrete k sum over N points),
  * the self-consistent BdG gap of an open chain of N sites (bulk average).
This chapter uses the convention Delta = -U F (no factor 1/2), as in the report.
Runtime: ~2 min.
"""

import numpy as np
import pandas as pd
from _common import cached, parse_args, plt, save
from scipy.integrate import quad
from scipy.optimize import brentq

T_HOP, MU, TEMP, U = 1.0, 0.001, 0.001, -2.0
SIZES = np.arange(5, 81)


def gap_infinite():
    def equation(delta):
        integrand = lambda k: np.tanh(np.sqrt(delta**2 + (MU + 2 * T_HOP * np.cos(k))**2) / (2 * TEMP)) \
            / np.sqrt(delta**2 + (MU + 2 * T_HOP * np.cos(k))**2)
        return -(U / (2 * np.pi)) * quad(integrand, 0, 2 * np.pi)[0] - 1.0
    return brentq(equation, 1e-10, 10.0)


def gap_periodic(n):
    k = 2 * np.pi * np.arange(n) / n
    def equation(delta):
        e = np.sqrt(delta**2 + (MU + 2 * T_HOP * np.cos(k))**2)
        return -(U / n) * np.sum(np.tanh(e / (2 * TEMP)) / e) - 1.0
    return brentq(equation, 1e-10, 10.0)


def gap_open_chain(n, delta0=0.3, tol=1e-14, eps=1e-15, mixing=0.2, max_iter=1_000_000):
    """Self-consistent BdG for an open chain; returns the mean |Delta| away from the ends."""
    h0 = -T_HOP * (np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)) - MU * np.eye(n)
    delta = delta0 * np.ones(n, dtype=complex)
    for _ in range(max_iter):
        h = np.block([[h0, np.diag(delta)], [np.diag(np.conj(delta)), -h0.conj()]])
        energies, vectors = np.linalg.eigh(h)
        u, v = vectors[:n], vectors[n:]
        f = 1.0 / (np.exp(energies / TEMP) + 1.0)
        new = -U * np.sum(u * np.conj(v) * (1 - 2 * f), axis=1)
        if np.sum(np.abs(new - delta)**2 / (np.abs(delta)**2 + eps)) < tol:
            return np.mean(np.abs(new[3:n - 3]))
        delta = mixing * new + (1 - mixing) * delta
    raise RuntimeError(f"no convergence for N={n}")


def compute():
    return pd.DataFrame({"N": SIZES,
                         "Delta_periodic": [gap_periodic(n) for n in SIZES],
                         "Delta_open": [gap_open_chain(n) for n in SIZES]})


def main():
    args = parse_args(__doc__)
    df = cached("ch4_finite_size_convergence", compute, args.recompute)
    d_inf = gap_infinite()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    ax1.plot(df.N, df.Delta_periodic, "-", lw=2, label="Periodic boundary conditions")
    ax1.plot(df.N, df.Delta_open, "--", lw=2, label="Finite System (BdG)")
    ax1.axhline(d_inf, color="black", ls=":", alpha=0.7, label="Infinite Limit")
    ax1.set_ylabel(r"Order parameter $\Delta$")
    ax1.legend()
    ax1.grid(True)

    ax2.semilogy(df.N, np.abs(df.Delta_periodic - d_inf), "-", lw=2, label="PBC error")
    ax2.semilogy(df.N, np.abs(df.Delta_open - d_inf) / abs(d_inf), "--", lw=2, label="Finite system error")
    ax2.set_xlabel("System size $N$")
    ax2.set_ylabel(r"Log Error $|\Delta_N - \Delta_{\infty}|/|\Delta_{\infty}|$")
    ax2.legend()
    ax2.grid(True, which="both")
    save(fig, "ch4_finite_size_convergence")


if __name__ == "__main__":
    main()
