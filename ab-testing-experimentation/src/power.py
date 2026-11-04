"""Power analysis: sample size and minimum detectable effect, with and without variance reduction.

Before the experiment, variance and the covariate correlation are estimated on historical data
(users observed in a period without the treatment). For a two-sided test at level alpha with
power 1 - beta and equal arms,

    n per arm = 2 (z_{1-alpha/2} + z_{1-beta})^2 sigma^2 / mde^2

CUPED replaces sigma^2 with sigma^2 (1 - rho^2), where rho is the correlation of the metric with
its pre-period value; regression adjustment with sigma^2 (1 - R^2) of the covariate regression.
The analytic sample sizes are then checked by simulating experiments of that size.

Usage:
    python power.py --output ../results [--sims 300]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from estimators import cuped, design, diff_in_means, regression_adjustment
from simulate import draw_experiment, population_truth

METHODS = ["difference in means", "CUPED", "regression adjustment"]


def n_per_arm(mde: float, sigma2: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return 2 * z ** 2 * sigma2 / mde ** 2


def mde(n_arm: float, sigma2: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * np.sqrt(2 * sigma2 / n_arm)


def power_at(n_arm: float, effect: float, sigma2: float, alpha: float = 0.05) -> float:
    z = effect / np.sqrt(2 * sigma2 / n_arm)
    za = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(za - z) + stats.norm.cdf(-za - z))


def historical_variances(n: int = 200_000, seed: int = 5) -> dict:
    """Residual variance of each method's metric, estimated on users without the treatment."""
    h = draw_experiment(n, np.random.default_rng(seed), treat=np.zeros(n, int))
    out = {}
    for metric, pre in [("revenue", h["pre_revenue"]), ("converted", (h["pre_revenue"] > 0).astype(float))]:
        y = h[metric].to_numpy(float)
        rho = np.corrcoef(y, pre)[0, 1]
        X = np.column_stack([np.ones(n), design(h).to_numpy()])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - resid.var() / y.var()
        out[metric] = {"mean": y.mean(), "sigma2": y.var(ddof=1), "rho": rho, "r2": r2,
                       "var": {"difference in means": y.var(ddof=1), "CUPED": y.var(ddof=1) * (1 - rho ** 2),
                               "regression adjustment": y.var(ddof=1) * (1 - r2)}}
    return out


def simulated_power(n_total: int, sims: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rej = {m: 0 for m in METHODS}
    for _ in range(sims):
        df = draw_experiment(n_total, rng)
        t, y = df["treat"].to_numpy(), df["revenue"].to_numpy()
        for e in [diff_in_means(y, t), cuped(y, df["pre_revenue"], t), regression_adjustment(y, design(df), t)]:
            rej[e.method] += e.p < 0.05
    return {m: r / sims for m, r in rej.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="../results")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.output)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    hv = historical_variances()
    truth = population_truth()

    # sample size for a range of relative MDEs
    rows = []
    for metric, v in hv.items():
        for rel in [0.02, 0.03, 0.05, 0.07, 0.10]:
            row = {"metric": metric, "relative MDE": rel}
            for m in METHODS:
                row[f"n total, {m}"] = int(np.ceil(2 * n_per_arm(rel * v["mean"], v["var"][m])))
            rows.append(row)
    sizes = pd.DataFrame(rows)
    sizes.to_csv(out / "power_sample_sizes.csv", index=False)

    # MDE with the 200k users of the main experiment
    mdes = pd.DataFrame([{"metric": metric, "method": m, "relative MDE at 200k users": mde(100_000, v["var"][m]) / v["mean"]}
                         for metric, v in hv.items() for m in METHODS])
    mdes.to_csv(out / "power_mde.csv", index=False)

    # check: n for 80 % power at the true revenue effect, then simulate experiments of that size
    eff, v = truth["ate_revenue"], hv["revenue"]
    check = []
    for i, m in enumerate(METHODS):
        n_total = int(np.ceil(2 * n_per_arm(eff, v["var"][m])))
        sim = simulated_power(n_total, a.sims, a.seed + i)
        check.append({"planned for": m, "n total": n_total, "analytic power": 0.8,
                      **{f"simulated power, {k}": p for k, p in sim.items()}})
    check = pd.DataFrame(check)
    check.to_csv(out / "power_check.csv", index=False)

    # power curves
    n_grid = np.linspace(10_000, 250_000, 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, c in zip(METHODS, ["#7f7f7f", "#1f77b4", "#2ca02c"]):
        ax.plot(n_grid, [power_at(n / 2, eff, v["var"][m]) for n in n_grid], color=c, label=m)
        row = check[check["planned for"] == m].iloc[0]
        ax.plot(row["n total"], row[f"simulated power, {m}"], "o", color=c)
    ax.axhline(0.8, color="k", lw=0.6, ls=":")
    ax.set(xlabel="users in the experiment", ylabel="power",
           title=f"Power to detect the true revenue effect (+{eff / v['mean']:.1%}); dots: simulated")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "figures" / "power_curves.png", dpi=120)

    with open(out / "power.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in d.items()} for k, d in hv.items()}, f, indent=2, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print(json.dumps({k: {kk: round(float(d[kk]), 4) for kk in ["mean", "sigma2", "rho", "r2"]} for k, d in hv.items()}))
        print(sizes, "\n", mdes, "\n", check)


if __name__ == "__main__":
    main()
