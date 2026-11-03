import numpy as np
import pandas as pd
import pytest

from pricing_engine import geometry, market


def test_rectangle_and_tube():
    bar = geometry.profile_features([(0, 0), (40, 0), (40, 20), (0, 20)], [], 20)
    assert bar["area_mm2"] == pytest.approx(800)
    assert bar["perimeter_per_area"] == pytest.approx(120 / 800)
    assert bar["solidity"] == pytest.approx(1)
    tube = geometry.profile_features([(0, 0), (40, 0), (40, 20), (0, 20)], [[(2, 2), (38, 2), (38, 18), (2, 18)]], 2)
    assert tube["area_mm2"] == pytest.approx(800 - 36 * 16)
    assert tube["n_hollows"] == 1 and tube["perimeter_per_area"] > bar["perimeter_per_area"]


def test_concave_profile_has_low_solidity():
    angle = [(0, 0), (50, 0), (50, 3), (3, 3), (3, 50), (0, 50)]
    assert geometry.profile_features(angle, [], 3)["solidity"] < 0.25


def test_hole_larger_than_outer_rejected():
    with pytest.raises(ValueError):
        geometry.profile_features([(0, 0), (1, 0), (1, 1)], [[(0, 0), (5, 0), (5, 5), (0, 5)]], 1)


def test_tolerance_tightness():
    std = geometry.standard_linear_tol(100)
    t = geometry.tolerance_features(std / 4, {"flatness_mm": 0.15}, 100)
    assert t["tight_linear"] == pytest.approx(2)
    assert t["tight_flatness"] == pytest.approx(1)
    assert t["tight_twist"] == 0 and t["n_tight_gdt"] == 1


def test_market_features_do_not_look_ahead():
    days = pd.bdate_range("2024-01-01", periods=120)
    lme = pd.Series(np.arange(120, dtype=float) + 1000, index=days)
    quote_day = days[100]
    f = market.lme_features(lme, [quote_day]).iloc[0]
    assert f["lme_lag1"] == lme.iloc[99]  # previous trading day, not the quote day
    assert f["lme_ma1m"] == pytest.approx(lme.iloc[78:100].mean())


def test_market_history_too_short():
    lme = pd.Series([1000.0] * 10, index=pd.bdate_range("2024-01-01", periods=10))
    with pytest.raises(ValueError):
        market.lme_features(lme, [pd.Timestamp("2024-01-10")])
