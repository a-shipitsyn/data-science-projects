"""Treatment effect estimators for a randomised experiment, and a Monte Carlo study of them.

    diff_in_means          Welch difference in means
    cuped                  CUPED: outcome adjusted by a pre-period covariate, theta = cov(y, x) / var(x)
    regression_adjustment  Lin (2013): OLS with centred covariates interacted with treatment, HC2 errors
    ratio_delta            difference of ratio metrics (e.g. revenue per session) with delta-method errors
    relative_effect        relative lift with a delta-method interval
    srm_test               sample ratio mismatch: chi-square test of the observed split

Monte Carlo study (python estimators.py): many experiments are drawn from the simulator and each
estimator is compared with the true population effect — bias, standard error, interval coverage,
power, and the sample size saved compared with the plain difference in means.

Usage:
    python estimators.py --output ../results --sims 500 --n 200000
"""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

Z95 = stats.norm.ppf(0.975)


@dataclass
class Estimate:
    effect: float
    se: float
    method: str = ""

    @property
    def ci(self) -> tuple[float, float]:
        return self.effect - Z95 * self.se, self.effect + Z95 * self.se

    @property
    def p(self) -> float:
        return float(2 * stats.norm.sf(abs(self.effect / self.se))) if self.se > 0 else float("nan")

    def as_dict(self) -> dict:
        lo, hi = self.ci
        return {**asdict(self), "ci_low": lo, "ci_high": hi, "p": self.p}


def _split(y, t):
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    return y[t], y[~t]


def diff_in_means(y, t) -> Estimate:
    y1, y0 = _split(y, t)
    return Estimate(y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)), "difference in means")


def cuped_theta(y, x) -> float:
    x = np.asarray(x, float)
    return float(np.cov(y, x)[0, 1] / x.var(ddof=1))


def cuped(y, x, t, theta: float | None = None) -> Estimate:
    """CUPED with a pooled theta. theta can be fixed in advance, e.g. from pre-experiment data."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = cuped_theta(y, x) if theta is None else theta
    e = diff_in_means(y - theta * (x - x.mean()), t)
    return Estimate(e.effect, e.se, "CUPED")


def regression_adjustment(y, X, t) -> Estimate:
    """Lin (2013) estimator: y ~ t + Xc + t:Xc with centred covariates, HC2 standard error of t.

    Unbiased in large samples under randomisation even if the linear model is wrong, and never
    less precise asymptotically than the difference in means.
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    Xc = np.asarray(X, float)
    Xc = Xc - Xc.mean(axis=0)
    D = np.column_stack([np.ones_like(t), t, Xc, Xc * t[:, None]])
    XtX_inv = np.linalg.pinv(D.T @ D)
    beta = XtX_inv @ D.T @ y
    resid = y - D @ beta
    h = np.einsum("ij,jk,ik->i", D, XtX_inv, D)                 # leverage
    meat = (D * (resid ** 2 / (1 - h))[:, None]).T @ D           # HC2
    cov = XtX_inv @ meat @ XtX_inv
    return Estimate(float(beta[1]), float(np.sqrt(cov[1, 1])), "regression adjustment")


def ratio_delta(num, den, t) -> Estimate:
    """Difference of ratio metrics sum(num) / sum(den) between arms, randomised by user.

    Delta method: var(R) = (var(num) - 2 R cov(num, den) + R^2 var(den)) / (n mean(den)^2).
    Treating sessions as independent units would understate the variance.
    """
    num, den, t = np.asarray(num, float), np.asarray(den, float), np.asarray(t).astype(bool)

    def arm(m):
        a, b = num[m], den[m]
        r = a.mean() / b.mean()
        v = (a.var(ddof=1) - 2 * r * np.cov(a, b)[0, 1] + r ** 2 * b.var(ddof=1)) / (len(a) * b.mean() ** 2)
        return r, v

    r1, v1 = arm(t)
    r0, v0 = arm(~t)
    return Estimate(r1 - r0, np.sqrt(v1 + v0), "ratio, delta method")


def relative_effect(y, t, x=None) -> Estimate:
    """Relative lift (mean_t / mean_c - 1) with a delta-method standard error; CUPED-adjusted if x is given."""
    y = np.asarray(y, float)
    if x is not None:
        x = np.asarray(x, float)
        y = y - cuped_theta(y, x) * (x - x.mean())
    y1, y0 = _split(y, t)
    m1, m0 = y1.mean(), y0.mean()
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    se = np.sqrt(v1 / m0 ** 2 + m1 ** 2 * v0 / m0 ** 4)
    return Estimate(m1 / m0 - 1, se, "relative" + (", CUPED" if x is not None else ""))


def srm_test(n_treatment: int, n_control: int, expected_share: float = 0.5) -> float:
    """p-value of the chi-square test that the split matches the planned share."""
    n = n_treatment + n_control
    exp = np.array([expected_share, 1 - expected_share]) * n
    return float(stats.chisquare([n_treatment, n_control], exp).pvalue)


COVARIATES = ["pre_revenue", "pre_sessions", "desktop", "new_user", "tenure_years"]


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric pre-experiment covariates for regression adjustment and uplift models."""
    return pd.DataFrame({
        "pre_revenue": df["pre_revenue"], "pre_sessions": df["pre_sessions"],
        "desktop": (df["device"] == "desktop").astype(float), "new_user": df["new_user"].astype(float),
        "tenure_years": df["tenure_days"] / 365,
    })


def monte_carlo(sims: int, n: int, seed: int = 1, null: bool = False) -> pd.DataFrame:
    """Draw `sims` experiments of `n` users; estimate the revenue effect with every method."""
    from simulate import draw_experiment, draw_users
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(sims):
        df = draw_experiment(n, rng)
        if null:                                  # A/A: redraw outcomes with nobody treated
            from simulate import draw_outcomes
            out = draw_outcomes(df, np.zeros(n, int), rng)
            df[out.columns] = out
        t = df["treat"].to_numpy()
        y = df["revenue"].to_numpy()
        ests = [
            diff_in_means(y, t),
            cuped(y, df["pre_revenue"], t),
            regression_adjustment(y, design(df), t),
            ratio_delta(df["revenue"], df["sessions"], t),
        ]
        for e in ests:
            rows.append({"sim": s, **e.as_dict()})
    return pd.DataFrame(rows)


def summarise(mc: pd.DataFrame, truth: dict, null: bool = False) -> pd.DataFrame:
    true = {"difference in means": truth["ate_revenue"], "CUPED": truth["ate_revenue"],
            "regression adjustment": truth["ate_revenue"],
            "ratio, delta method": truth["ate_revenue"] / truth["sessions_mean"]}
    out = []
    for m, g in mc.groupby("method", sort=False):
        tv = 0.0 if null else true[m]
        out.append({
            "method": m, "true": tv, "mean estimate": g["effect"].mean(), "bias": g["effect"].mean() - tv,
            "sd of estimates": g["effect"].std(), "mean se": g["se"].mean(),
            "coverage 95%": ((g["ci_low"] <= tv) & (tv <= g["ci_high"])).mean(),
            ("false positive rate" if null else "power"): (g["p"] < 0.05).mean(),
        })
    res = pd.DataFrame(out)
    base = res.loc[res["method"] == "difference in means", "sd of estimates"].iloc[0]
    res["variance vs diff in means"] = np.where(res["method"].str.startswith("ratio"), np.nan, (res["sd of estimates"] / base) ** 2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="../results")
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from simulate import population_truth
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    truth = population_truth()
    res = summarise(monte_carlo(a.sims, a.n, a.seed), truth)
    aa = summarise(monte_carlo(a.sims, a.n, a.seed + 1, null=True), truth, null=True)
    res.to_csv(out / "estimators_ab.csv", index=False)
    aa.to_csv(out / "estimators_aa.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print(f"A/B, {a.sims} experiments of {a.n} users\n", res, f"\n\nA/A\n", aa)
    with open(out / "estimators.json", "w") as f:
        json.dump({"sims": a.sims, "n": a.n, "truth": {k: truth[k] for k in ["ate_revenue", "sessions_mean"]}}, f, indent=2)


if __name__ == "__main__":
    main()
