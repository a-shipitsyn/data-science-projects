import numpy as np

from pricing_engine import training
from pricing_engine.evaluate import CostPlusBaseline, metrics
from pricing_engine.features import build_features
from pricing_engine.model import PricingModel
from pricing_engine.schema import parse


def dataset(mock, tenant):
    lme, quotes, truth = mock
    df = build_features([parse(q) for q in quotes[tenant]], lme)
    return df.sort_values("quote_date", kind="stable").reset_index(drop=True), truth[tenant].set_index("quote_id")


def test_model_beats_baseline_and_approaches_noise_floor(mock):
    df, truth = dataset(mock, "tenant_a")
    train, test = training.time_split(df, 0.25)
    pred = PricingModel(seed=0).fit(train).predict(test)
    actual = test["quoted_price_per_kg"]
    m = metrics(actual, pred["price_per_kg"], pred["lower"], pred["upper"])
    floor = metrics(actual, truth.loc[test["quote_id"], "expected_price_per_kg"])["mape"]
    baseline = metrics(actual, CostPlusBaseline().fit(train).predict(test)["price_per_kg"])["mape"]
    assert m["mape"] < 0.6 * baseline
    assert m["mape"] < 3 * floor
    assert 0.8 <= m["coverage"] <= 1.0
    assert (pred["lower"] < pred["price_per_kg"]).all() and (pred["price_per_kg"] < pred["upper"]).all()


def test_outliers_are_detected(mock):
    df, truth = dataset(mock, "tenant_a")
    model = PricingModel(seed=0).fit(df)
    assert model.n_outliers_ >= 0.5 * truth["is_outlier"].sum()


def test_deterministic(mock):
    df, _ = dataset(mock, "tenant_b")
    p1 = PricingModel(seed=3).fit(df).predict(df.head(20))
    p2 = PricingModel(seed=3).fit(df).predict(df.head(20))
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
