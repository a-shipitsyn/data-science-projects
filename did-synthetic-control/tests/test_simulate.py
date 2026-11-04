import numpy as np

from simulate import WEEKS, ramp, simulate_market, simulate_pilot, simulate_staggered


def test_pilot_design(pilot):
    assert pilot.Y.shape == (200, WEEKS)
    assert pilot.treated.sum() == 25
    assert (pilot.attrs.loc[pilot.treated, "urban"] == 1).all()
    assert np.all(pilot.effect[pilot.D == 0] == 0)
    assert np.all(pilot.effect[pilot.D == 1] > 0)


def test_staggered_waves(staggered):
    waves, counts = np.unique(staggered.G, return_counts=True)
    assert dict(zip(waves.tolist(), counts.tolist())) == {30.0: 40, 50.0: 40, 70.0: 40, np.inf: 80}
    # earlier waves gain more
    m = [staggered.effect[(staggered.G == g)][:, -1].mean() for g in (30, 50, 70)]
    assert m[0] > m[1] > m[2]


def test_market_single_treated(market):
    assert market.treated.sum() == 1 and market.Y.shape == (31, WEEKS)


def test_truth_matches_effect_column():
    df, truth = simulate_pilot(seed=5)
    post = df[df["week"] >= df["launch_week"]]
    assert np.isclose(post["true_effect"].mean(), truth["att"], atol=1e-6)


def test_ramp():
    assert ramp(np.array([-1]), 4, "exp")[0] == 0
    assert np.isclose(ramp(np.array([25]), 26, "linear")[0], 1.0)
    assert 0 < ramp(np.array([0]), 4, "exp")[0] < ramp(np.array([10]), 4, "exp")[0] < 1


def test_seeds_reproducible():
    a, _ = simulate_staggered(seed=7)
    b, _ = simulate_staggered(seed=7)
    assert a.equals(b)
