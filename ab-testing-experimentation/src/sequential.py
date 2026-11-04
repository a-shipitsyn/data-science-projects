"""Sequential testing: why peeking at a fixed-horizon test inflates false positives, and two fixes.

The experiment runs 14 days and is looked at once a day, using the users enrolled so far
(each user's outcome is taken as known at the look, e.g. a short fixed window after exposure).

    fixed horizon     one test at day 14
    naive peeking     a fixed-horizon test every day, stop at the first p < 0.05
    group sequential  O'Brien-Fleming-type alpha spending (Lan-DeMets), boundaries for 14 looks
                      found by simulating Brownian motion; very strict early, close to 2 at the end
    mSPRT             mixture sequential probability ratio test with a normal prior N(0, tau^2) on
                      the effect; always valid: any number of looks, stop when the likelihood
                      ratio exceeds 1 / alpha

Each method is run on many simulated A/A experiments (no effect: false positive rate) and A/B
experiments (true effect: power and stopping day), with and without CUPED.

Usage:
    python sequential.py --output ../results [--sims 1000 --n 200000]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from simulate import DAYS, draw_experiment, population_truth

ALPHA = 0.05


def obf_spending(t, alpha: float = ALPHA) -> np.ndarray:
    """Two-sided alpha spent by information fraction t: Lan-DeMets O'Brien-Fleming-type function
    alpha*(t) = 2 - 2 Phi(z_{1 - a/2} / sqrt(t)) applied to a = alpha / 2 on each side (as in gsDesign
    and ldbounds)."""
    side = 2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 4) / np.sqrt(np.asarray(t, float)))
    return 2 * side


def obf_boundaries(looks: np.ndarray, alpha: float = ALPHA, paths: int = 1_000_000, seed: int = 0) -> np.ndarray:
    """Two-sided z boundaries at information fractions `looks` (increasing, last = 1).

    The first boundary is exact; later ones are found by simulating Brownian motion: at each look,
    the boundary is set so that the share of paths crossing it for the first time equals the alpha
    spent since the previous look."""
    rng = np.random.default_rng(seed)
    t = np.asarray(looks, float)
    spend = obf_spending(t, alpha)
    inc = np.diff(np.concatenate([[0.0], spend]))
    w = np.cumsum(rng.standard_normal((paths, len(t))) * np.sqrt(np.diff(np.concatenate([[0.0], t]))), axis=1)
    z = np.abs(w / np.sqrt(t))
    bounds = np.empty(len(t))
    bounds[0] = stats.norm.ppf(1 - inc[0] / 2)
    alive = z[:, 0] < bounds[0]
    for k in range(1, len(t)):
        target = int(round(inc[k] * paths))
        za = np.sort(z[alive, k])[::-1]
        bounds[k] = np.inf if target < 1 else za[min(target, len(za)) - 1]
        alive &= z[:, k] < bounds[k]
    return bounds


def msprt_lr(effect, var, tau2):
    """Mixture likelihood ratio of N(0, tau2) alternatives vs no effect, for an estimate with variance var."""
    return np.sqrt(var / (var + tau2)) * np.exp(tau2 * effect ** 2 / (2 * var * (var + tau2)))


def daily_estimates(y, x, day, t, days: int = DAYS) -> dict:
    """Cumulative difference in means and CUPED estimate (with variances) at each daily look.

    y: metric, x: pre-period covariate, day: day of first exposure (1..days), t: assignment 0/1.
    """
    y, x, t = np.asarray(y, float), np.asarray(x, float), np.asarray(t)
    d = np.asarray(day) - 1
    out = {}
    s = {}
    for arm in (0, 1):
        m = t == arm
        for name, v in [("n", np.ones(m.sum())), ("y", y[m]), ("yy", y[m] ** 2), ("x", x[m]), ("xx", x[m] ** 2), ("xy", x[m] * y[m])]:
            s[arm, name] = np.cumsum(np.bincount(d[m], weights=v, minlength=days))
    n = s[0, "n"] + s[1, "n"]
    mx = (s[0, "x"] + s[1, "x"]) / n
    my = (s[0, "y"] + s[1, "y"]) / n
    cov = ((s[0, "xy"] + s[1, "xy"]) / n - mx * my)
    theta = cov / ((s[0, "xx"] + s[1, "xx"]) / n - mx ** 2)
    for label, th in [("dim", np.zeros(days)), ("cuped", theta)]:
        mean, var = {}, {}
        for arm in (0, 1):
            na = s[arm, "n"]
            ym, xm = s[arm, "y"] / na, s[arm, "x"] / na
            vy = s[arm, "yy"] / na - ym ** 2
            vx = s[arm, "xx"] / na - xm ** 2
            cxy = s[arm, "xy"] / na - xm * ym
            mean[arm] = ym - th * (xm - mx)
            var[arm] = (vy - 2 * th * cxy + th ** 2 * vx) / na
        out[label] = (mean[1] - mean[0], var[1] + var[0])
    out["fraction"] = n / n[-1]
    return out


def simulate_paths(sims: int, n: int, null: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    res = {k: (np.zeros((sims, DAYS)), np.zeros((sims, DAYS))) for k in ("dim", "cuped")}
    frac = np.zeros((sims, DAYS))
    for i in range(sims):
        if null:
            df = draw_experiment(n, rng, treat=np.zeros(n, int))
            t = rng.integers(0, 2, n)
        else:
            df = draw_experiment(n, rng)
            t = df["treat"].to_numpy()
        e = daily_estimates(df["revenue"], df["pre_revenue"], df["day"], t)
        for k in ("dim", "cuped"):
            res[k][0][i], res[k][1][i] = e[k]
        frac[i] = e["fraction"]
    return {**res, "fraction": frac}


def decide(effect: np.ndarray, var: np.ndarray, rule: str, bounds=None, tau2=None) -> np.ndarray:
    """Day of the first rejection (1-based) for each simulated experiment, 0 if never rejected."""
    z = np.abs(effect) / np.sqrt(var)
    if rule == "fixed":
        rej = np.zeros_like(z, bool)
        rej[:, -1] = z[:, -1] > stats.norm.ppf(1 - ALPHA / 2)
    elif rule == "peeking":
        rej = z > stats.norm.ppf(1 - ALPHA / 2)
    elif rule == "obf":
        rej = z > bounds[None, :]
    elif rule == "msprt":
        rej = msprt_lr(effect, var, tau2) >= 1 / ALPHA
    first = np.where(rej.any(axis=1), rej.argmax(axis=1) + 1, 0)
    return first


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="../results")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    out = Path(a.output)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    truth = population_truth()
    tau2 = (0.05 * truth["control_revenue_mean"]) ** 2          # prior: effects of about 5 %
    bounds = obf_boundaries(np.arange(1, DAYS + 1) / DAYS)
    aa = simulate_paths(a.sims, a.n, True, a.seed)
    ab = simulate_paths(a.sims, a.n, False, a.seed + 1)

    rows = []
    for est in ("dim", "cuped"):
        for rule, label in [("fixed", "fixed horizon, day 14"), ("peeking", "naive peeking, daily"),
                            ("obf", "group sequential (O'Brien-Fleming), daily"), ("msprt", "mSPRT, daily")]:
            r_aa = decide(*aa[est], rule, bounds, tau2)
            r_ab = decide(*ab[est], rule, bounds, tau2)
            stop = np.where(r_ab > 0, r_ab, DAYS)
            rows.append({
                "estimator": "CUPED" if est == "cuped" else "difference in means", "method": label,
                "false positive rate (A/A)": (r_aa > 0).mean(), "power (A/B)": (r_ab > 0).mean(),
                "mean stopping day when significant": r_ab[r_ab > 0].mean() if (r_ab > 0).any() else np.nan,
                "expected days run (A/B)": stop.mean(),
            })
    res = pd.DataFrame(rows)
    res.to_csv(out / "sequential.csv", index=False)
    pd.DataFrame({"day": np.arange(1, DAYS + 1), "obf_boundary": bounds,
                  "naive": stats.norm.ppf(1 - ALPHA / 2)}).to_csv(out / "sequential_boundaries.csv", index=False)

    # false positive rate of naive peeking vs number of looks (equally spaced, the last on day 14)
    z_aa = np.abs(aa["dim"][0]) / np.sqrt(aa["dim"][1])
    fpr_looks = [(k, (z_aa[:, np.round(np.arange(1, k + 1) * DAYS / k).astype(int) - 1] > 1.96).any(axis=1).mean())
                 for k in range(1, DAYS + 1)]
    pd.DataFrame(fpr_looks, columns=["looks", "false positive rate"]).to_csv(out / "sequential_peeking.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    days = np.arange(1, DAYS + 1)
    zs = aa["dim"][0] / np.sqrt(aa["dim"][1])
    crossed = (np.abs(zs) > 1.96).any(axis=1)
    for i in range(min(200, a.sims)):
        axes[0].plot(days, zs[i], color="#d62728" if crossed[i] else "#bbbbbb", lw=0.6, alpha=0.6)
    for sgn in (1, -1):
        axes[0].plot(days, sgn * np.full(DAYS, 1.96), "k--", lw=1, label="fixed-horizon 1.96" if sgn == 1 else None)
        axes[0].plot(days, sgn * np.minimum(bounds, 6), "b-", lw=1.5, label="O'Brien-Fleming" if sgn == 1 else None)
    axes[0].set(ylim=(-5, 5), xlabel="day", ylabel="z statistic",
                title=f"200 A/A tests: red ones cross 1.96 on some day")
    axes[0].legend(loc="lower left", fontsize=8)
    k, f = zip(*fpr_looks)
    axes[1].plot(k, f, "o-", color="#d62728")
    axes[1].axhline(ALPHA, color="k", ls=":")
    axes[1].set(xlabel="number of looks", ylabel="false positive rate", ylim=(0, None),
                title="Naive peeking: false positives grow with looks")
    fig.tight_layout()
    fig.savefig(out / "figures" / "sequential.png", dpi=120)

    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print("O'Brien-Fleming boundaries:", np.round(bounds, 2))
        print(res)
        print(pd.DataFrame(fpr_looks, columns=["looks", "fpr"]).T)


if __name__ == "__main__":
    main()
