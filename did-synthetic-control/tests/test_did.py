import numpy as np
import pandas as pd
import statsmodels.api as sm

from did import _demean2, callaway_santanna, did_2x2, did_dr, event_study, imputation, twfe, twfe_weights
from panel import Panel


def toy_panel(n=60, T=20, launch=None, effect=0.1, seed=0, trend_gap=0.0):
    """Unit + week effects + noise; constant effect after launch; optional extra trend for treated."""
    rng = np.random.default_rng(seed)
    weeks = np.arange(1, T + 1)
    G = np.full(n, np.inf) if launch is None else launch
    tr = np.isfinite(G)
    Y = rng.normal(0, 1, n)[:, None] + rng.normal(0, 1, T)[None, :] + rng.normal(0, 0.05, (n, T))
    Y += trend_gap * tr[:, None] * weeks[None, :]
    D = weeks[None, :] >= G[:, None]
    Y += effect * D
    return Panel(Y, G, weeks, np.arange(n), pd.DataFrame({"x": tr.astype(float) + rng.normal(0, 0.5, n)}), effect * D)


def test_did_2x2_recovers_constant_effect():
    G = np.where(np.arange(60) < 20, 11.0, np.inf)
    r = did_2x2(toy_panel(launch=G))
    assert abs(r.att - 0.1) < 3 * r.se


def test_twfe_matches_statsmodels_dummies():
    G = np.where(np.arange(30) < 10, 8.0, np.inf)
    p = toy_panel(n=30, T=12, launch=G)
    r = twfe(p)
    n, T = p.Y.shape
    df = pd.DataFrame({"y": p.Y.ravel(), "D": p.D.ravel(), "u": np.repeat(np.arange(n), T), "t": np.tile(np.arange(T), n)})
    X = pd.get_dummies(df[["u", "t"]].astype(str), drop_first=True).astype(float)
    X.insert(0, "D", df["D"])
    ref = sm.OLS(df["y"], sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": df["u"]})
    assert np.isclose(r.att, ref.params["D"])
    assert np.isclose(r.se, ref.bse["D"], rtol=1e-6)


def test_demean2_removes_two_way_effects():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 1)) + rng.normal(size=(1, 7))
    assert np.allclose(_demean2(A), 0)


def test_twfe_weights_sum_to_one_and_go_negative_when_staggered():
    G = np.repeat([5.0, 10.0, 15.0, np.inf], 15)
    p = toy_panel(launch=G)
    w = twfe_weights(p)
    assert np.isclose(w.sum(), 1)
    assert (w < 0).any()
    # TWFE equals the weighted sum of cell effects when parallel trends hold exactly
    noiseless = Panel(p.effect.copy(), p.G, p.weeks, p.units, p.attrs, p.effect)
    assert np.isclose(twfe(noiseless).att, (w * p.effect).sum())


def test_pretrend_test_size_and_power():
    G = np.where(np.arange(60) < 20, 15.0, np.inf)
    null = [event_study(toy_panel(T=30, launch=G, seed=s), leads=8, lags=10)["pretrend_p"] for s in range(200)]
    assert 0.02 < np.mean(np.array(null) < 0.05) < 0.09
    tilt = event_study(toy_panel(T=30, launch=G, trend_gap=0.02), leads=8, lags=10)
    assert tilt["pretrend_p"] < 0.001


def test_event_study_recovers_effect_path():
    G = np.where(np.arange(60) < 20, 15.0, np.inf)
    es = event_study(toy_panel(T=30, launch=G), leads=8, lags=10)
    post = es["k"] >= 0
    assert np.allclose(es["coef"][post], 0.1, atol=0.05)


def test_dr_did_removes_confounded_trend():
    # treated units have x = 1 and trend with x; conditioning on x restores parallel trends
    rng = np.random.default_rng(3)
    n, T = 400, 20
    x = (rng.random(n) < 0.5).astype(float)
    G = np.where((x == 1) & (rng.random(n) < 0.3), 11.0, np.inf)
    weeks = np.arange(1, T + 1)
    Y = rng.normal(0, 1, n)[:, None] + 0.02 * x[:, None] * weeks[None, :] + rng.normal(0, 0.05, (n, T))
    Y += 0.1 * (weeks[None, :] >= G[:, None])
    p = Panel(Y, G, weeks, np.arange(n), pd.DataFrame({"x": x}))
    naive, dr = did_2x2(p), did_dr(p, x[:, None], boot=50)
    assert abs(naive.att - 0.1) > 0.05
    assert abs(dr.att - 0.1) < 3 * dr.se


def test_staggered_estimators_unbiased_where_twfe_is_not():
    # dynamic effects that grow with time since launch: TWFE is biased, CS and imputation are not
    rng = np.random.default_rng(4)
    n, T = 120, 40
    G = np.repeat([10.0, 20.0, 30.0, np.inf], 30)
    weeks = np.arange(1, T + 1)
    e = weeks[None, :] - G[:, None]
    eff = np.where(e >= 0, 0.01 * (e + 1), 0.0)
    Y = rng.normal(0, 1, n)[:, None] + rng.normal(0, 1, T)[None, :] + rng.normal(0, 0.02, (n, T)) + eff
    p = Panel(Y, G, weeks, np.arange(n), pd.DataFrame(index=range(n)), eff)
    truth = eff[p.D == 1].mean()
    assert abs(twfe(p).att - truth) > 0.02
    for r in (callaway_santanna(p, "never", 50), callaway_santanna(p, "notyet", 50), imputation(p, 50)):
        assert abs(r.att - truth) < 0.005, r.method


def test_on_simulated_store_panels(pilot, staggered):
    t_p = pilot.effect[pilot.D == 1].mean()
    assert did_2x2(pilot).att - t_p > 0.005                     # urban trend biases plain DiD upwards
    t_s = staggered.effect[staggered.D == 1].mean()
    assert twfe(staggered).att < t_s - 0.01
