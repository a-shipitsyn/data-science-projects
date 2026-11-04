import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from estimators import (cuped, design, diff_in_means, monte_carlo, ratio_delta, regression_adjustment,
                        relative_effect, srm_test, summarise)
from simulate import population_truth


def test_diff_in_means_matches_welch():
    rng = np.random.default_rng(0)
    y, t = rng.normal(size=3000), rng.integers(0, 2, 3000)
    e = diff_in_means(y, t)
    w = stats.ttest_ind(y[t == 1], y[t == 0], equal_var=False)
    assert np.isclose(e.effect, y[t == 1].mean() - y[t == 0].mean())
    assert abs(e.p - w.pvalue) < 1e-3


def test_regression_adjustment_matches_statsmodels_hc2(experiment):
    df, _ = experiment
    t = (df["group"] == "treatment").to_numpy(float)
    X = design(df).to_numpy(float)
    Xc = X - X.mean(axis=0)
    D = np.column_stack([np.ones_like(t), t, Xc, Xc * t[:, None]])
    ref = sm.OLS(df["revenue"].to_numpy(), D).fit(cov_type="HC2")
    e = regression_adjustment(df["revenue"], design(df), t)
    assert np.isclose(e.effect, ref.params[1])
    assert np.isclose(e.se, ref.bse[1], rtol=1e-6)


def test_variance_reduction(experiment):
    df, _ = experiment
    t = (df["group"] == "treatment").to_numpy()
    raw = diff_in_means(df["revenue"], t)
    cu = cuped(df["revenue"], df["pre_revenue"], t)
    ra = regression_adjustment(df["revenue"], design(df), t)
    assert ra.se < cu.se < raw.se
    assert cu.se < 0.95 * raw.se


def test_cuped_with_independent_covariate_is_plain_difference():
    rng = np.random.default_rng(1)
    y, x, t = rng.normal(size=20000), rng.normal(size=20000), rng.integers(0, 2, 20000)
    assert abs(cuped(y, x, t).se / diff_in_means(y, t).se - 1) < 0.01


def test_ratio_delta_se_matches_bootstrap(experiment):
    df, _ = experiment
    t = (df["group"] == "treatment").to_numpy()
    e = ratio_delta(df["revenue"], df["sessions"], t)
    rng = np.random.default_rng(2)
    num, den = df["revenue"].to_numpy(), df["sessions"].to_numpy()
    boot = []
    for _ in range(200):
        i = rng.integers(0, len(df), len(df))
        boot.append(ratio_delta(num[i], den[i], t[i]).effect)
    assert abs(np.std(boot) / e.se - 1) < 0.15


def test_relative_effect():
    y = np.r_[np.full(1000, 2.0), np.full(1000, 2.2)] + np.random.default_rng(3).normal(0, 0.1, 2000)
    t = np.r_[np.zeros(1000), np.ones(1000)]
    e = relative_effect(y, t)
    assert abs(e.effect - 0.1) < 0.01 and e.p < 1e-6


def test_srm():
    assert srm_test(5000, 5000) == 1.0
    assert srm_test(5200, 4800) < 0.001
    assert srm_test(5050, 4950) > 0.1


def test_monte_carlo_aa_is_calibrated():
    truth = population_truth(200_000)
    res = summarise(monte_carlo(sims=60, n=20_000, seed=5, null=True), truth, null=True)
    assert (res["coverage 95%"] > 0.85).all()
    assert (res["bias"].abs() < 3 * res["sd of estimates"] / np.sqrt(60)).all()
