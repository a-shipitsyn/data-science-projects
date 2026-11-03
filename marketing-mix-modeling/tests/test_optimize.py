import numpy as np
import pytest

from model import contributions, fit
from optimize import incremental_calls, optimize
from transforms import CHANNELS


@pytest.fixture(scope="module")
def spec(sim):
    df, _, params = sim
    return {"params": params, "coefficients": fit(df, params).params.to_dict()}


def test_budget_kept_and_calls_not_lower(sim, spec):
    year = sim[0].iloc[-365:]
    best = optimize(year, spec, 0.5, 2.0)
    spend = np.array([year[f"{ch}_spend"].sum() for ch in CHANNELS])
    assert (best * spend).sum() == pytest.approx(spend.sum(), rel=1e-4)
    assert np.all((best >= 0.5 - 1e-9) & (best <= 2.0 + 1e-9))
    assert incremental_calls(best, year, spec).sum() >= incremental_calls(np.ones(3), year, spec).sum() - 1e-6


def test_incremental_calls_match_model_contributions(sim, spec):
    df, _, params = sim
    model = fit(df, params)
    calls = incremental_calls(np.ones(3), df, spec)
    contrib = contributions(model, df, params)
    # same data and coefficients: outdoor and digital match exactly; TV uses the latest yearly coefficient
    for i, ch in enumerate(CHANNELS):
        if ch != "tv":
            assert calls[i] == pytest.approx(contrib[ch].sum())


def test_within_observed_cap(sim, spec):
    df = sim[0]
    year = df.iloc[-365:]
    high = np.array([max(df[ch].max() / year[ch].max(), 0.5) for ch in CHANNELS])
    best = optimize(year, spec, 0.5, np.minimum(2.0, high))
    for m, ch in zip(best, CHANNELS, strict=True):
        assert m * year[ch].max() <= df[ch].max() * (1 + 1e-6)
