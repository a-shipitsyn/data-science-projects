import numpy as np
import pandas as pd
from scipy.optimize import minimize

from panel import Panel
from synth import asc_fit, did_placebo, sc_fit, sdid_fit, simplex_ls, synth


def test_simplex_ls_matches_slsqp():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 20))
    b = A[:, :3].mean(axis=1) + rng.normal(0, 0.1, 50)
    w = simplex_ls(A, b)
    ref = minimize(lambda x: ((b - A @ x) ** 2).sum(), np.full(20, 0.05), bounds=[(0, 1)] * 20,
                   constraints={"type": "eq", "fun": lambda x: x.sum() - 1}, method="SLSQP")
    assert np.isclose(w.sum(), 1) and (w >= 0).all()
    assert ((b - A @ w) ** 2).sum() <= ref.fun + 1e-6


def factor_panel(J=25, T=60, T0=45, effect=0.05, seed=0):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 0.05, (2, T)), axis=1)
    mu = rng.uniform(0, 2, (J, 2))
    mu[0] = mu[1:4].mean(axis=0)                                      # treated inside the convex hull
    Y = mu @ f + rng.normal(0, 0.005, (J, T))
    Y[0] = mu[1:4].mean(axis=0) @ f + rng.normal(0, 0.005, T)
    weeks = np.arange(1, T + 1)
    G = np.full(J, np.inf)
    G[0] = T0 + 1
    Y[0, weeks >= G[0]] += effect
    return Panel(Y, G, weeks, np.arange(J), pd.DataFrame({"u": range(J)}))


def test_synthetic_control_recovers_effect_when_treated_is_in_hull():
    p = factor_panel()
    for m, tol in [("synthetic control", 0.01), ("augmented synthetic control", 0.01), ("synthetic DiD", 0.02)]:
        r = synth(p, m)
        assert abs(r.att - 0.05) < tol, m          # SDID regularises its weights more, at some bias here
    assert abs(did_placebo(p).att - 0.05) > abs(synth(p, "synthetic control").att - 0.05)


def test_sc_reproduces_exact_combination():
    rng = np.random.default_rng(1)
    Y0 = rng.normal(size=(10, 30)).cumsum(axis=1)
    w_true = np.zeros(10)
    w_true[[2, 5]] = [0.3, 0.7]
    y1 = w_true @ Y0
    pre = np.arange(30) < 20
    w, path = sc_fit(y1, Y0, pre)
    assert np.allclose(w, w_true, atol=1e-3)
    assert np.allclose(asc_fit(y1, Y0, pre), y1, atol=1e-2)


def test_sdid_weights_on_simplex():
    p = factor_panel()
    pre = p.weeks < p.G[0]
    tau, omega, lam = sdid_fit(p.Y[:1], p.Y[1:], pre)
    assert np.isclose(omega.sum(), 1) and np.isclose(lam.sum(), 1)
    assert (omega >= 0).all() and (lam >= 0).all()


def test_market_scenario(market):
    truth = market.effect[market.D == 1].mean()
    asc = synth(market, "augmented synthetic control")
    assert abs(asc.att - truth) < 0.015
    assert abs(did_placebo(market).att - truth) > abs(asc.att - truth)
