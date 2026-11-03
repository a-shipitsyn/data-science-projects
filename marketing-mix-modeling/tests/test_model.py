import numpy as np
import pandas as pd
import pytest

from evaluate import baseline_predict, bootstrap
from model import DECAY, contributions, fit, media_summary, metrics, predict, search, tv_change, validate
from transforms import CHANNELS


@pytest.mark.parametrize("family", ["ols", "nb"])
def test_recovers_truth_with_true_hyperparameters(sim, family):
    df, truth, params = sim
    model = fit(df, params, family)
    summary = media_summary(df, contributions(model, df, params))
    for ch in CHANNELS:
        assert summary[ch]["share"] == pytest.approx(truth["true_contribution_share"][ch], abs=0.02)
    tv = tv_change(model, df, params)
    effect = [tv["by_year"][y]["calls_per_1000_contacts"] for y in ["2021", "2022", "2023"]]
    assert effect[0] > effect[2]
    assert tv["equal_years_p"] < 0.05


def test_media_beats_baseline_on_holdout(sim):
    df, _, params = sim
    n_train = len(df) - 90
    model = fit(df.iloc[:n_train], params)
    test = df.index[n_train:]
    y = df.loc[test, "calls"]
    assert metrics(y, predict(model, df, params)[test])["rmse"] < metrics(y, baseline_predict(df, n_train)[test])["rmse"]


def test_search_stays_in_bounds(sim):
    df, _, _ = sim
    params = search(df.iloc[:400], maxiter=2)
    for ch in CHANNELS:
        assert DECAY[ch][0] <= params[ch]["decay"] <= DECAY[ch][1]


def test_bootstrap_interval_contains_estimate(sim):
    df, _, params = sim
    model = fit(df, params)
    summary = media_summary(df, contributions(model, df, params))
    ci = bootstrap(model, df, params, n=30)
    for ch in CHANNELS:
        assert ci[f"{ch}_share"]["ci_low"] <= summary[ch]["share"] <= ci[f"{ch}_share"]["ci_high"]


def test_validate(sim):
    df, _, _ = sim
    with pytest.raises(ValueError, match="missing columns"):
        validate(df.drop(columns="tv"))
    bad = df.copy()
    bad.loc[3, "tv_spend"] = -1
    bad.loc[5, "calls"] = np.nan
    with pytest.raises(ValueError, match="calls: 1 missing values.*tv_spend: negative values"):
        validate(bad)
    with pytest.raises(ValueError, match="not consecutive"):
        validate(pd.concat([df.iloc[:10], df.iloc[11:20]]))


def test_weibull_search_returns_kernel_params(sim):
    df, _, _ = sim
    params = search(df.iloc[:400], maxiter=2, adstock="weibull")
    for ch in CHANNELS:
        assert {"k", "scale", "half_sat", "shape"} == set(params[ch])
