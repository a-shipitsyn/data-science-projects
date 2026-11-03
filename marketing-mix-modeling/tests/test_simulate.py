import pandas as pd

from simulate import simulate


def test_deterministic_with_seed():
    a, ta, la = simulate(seed=3)
    b, tb, lb = simulate(seed=3)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(la, lb)
    assert ta == tb


def test_columns_and_values(sim):
    df, truth, _ = sim
    assert (df["calls"] >= 0).all() and df["date"].is_monotonic_increasing
    assert set(truth["true_tv_calls_per_1000_contacts"]) == {"2021", "2022", "2023"}
