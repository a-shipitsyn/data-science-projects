import numpy as np
from scipy import stats

from estimators import cuped, diff_in_means
from sequential import daily_estimates, decide, msprt_lr, obf_boundaries


def test_obf_boundaries_match_reference():
    # Lan-DeMets O'Brien-Fleming, two-sided alpha 0.05, five equally spaced looks
    ref = np.array([4.877, 3.357, 2.680, 2.290, 2.031])
    b = obf_boundaries(np.arange(1, 6) / 5, paths=400_000)
    assert np.allclose(b, ref, atol=0.05)


def test_msprt_likelihood_ratio():
    v, tau2 = 1.0, 4.0
    assert msprt_lr(0.0, v, tau2) < 1
    assert msprt_lr(3.0, v, tau2) > msprt_lr(2.0, v, tau2)
    assert np.isclose(msprt_lr(-2.0, v, tau2), msprt_lr(2.0, v, tau2))


def test_daily_estimates_final_look_matches_estimators(experiment):
    df, _ = experiment
    t = (df["group"] == "treatment").to_numpy(int)
    e = daily_estimates(df["revenue"], df["pre_revenue"], df["day"], t)
    raw, cu = diff_in_means(df["revenue"], t), cuped(df["revenue"], df["pre_revenue"], t)
    assert np.isclose(e["dim"][0][-1], raw.effect)
    assert np.isclose(np.sqrt(e["dim"][1][-1]), raw.se, rtol=1e-3)
    assert np.isclose(e["cuped"][0][-1], cu.effect, rtol=1e-6)
    assert np.isclose(np.sqrt(e["cuped"][1][-1]), cu.se, rtol=1e-3)
    assert np.isclose(e["fraction"][-1], 1.0)


def test_peeking_inflates_false_positives_and_corrections_hold():
    """Brownian-motion A/A paths: daily peeking at 1.96 is far above 5 %; OBF and mSPRT are not."""
    rng = np.random.default_rng(0)
    sims, days = 20_000, 14
    n_day = 1000
    inc = rng.normal(0, 1, (sims, days)) * np.sqrt(n_day)
    n = n_day * np.arange(1, days + 1)
    effect = np.cumsum(inc, axis=1) / n            # running mean of unit-variance differences
    var = np.broadcast_to(1.0 / n, effect.shape)
    bounds = obf_boundaries(np.arange(1, days + 1) / days, paths=200_000)
    fpr = {r: (decide(effect, var, r, bounds, tau2=(3 / np.sqrt(n_day)) ** 2) > 0).mean()
           for r in ["fixed", "peeking", "obf", "msprt"]}
    assert abs(fpr["fixed"] - 0.05) < 0.01
    assert fpr["peeking"] > 0.15
    assert fpr["obf"] < 0.06
    assert fpr["msprt"] < 0.05
