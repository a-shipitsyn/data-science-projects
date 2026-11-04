import numpy as np

from uplift import aipw_scores, features, fit_models, ipw_policy_gain, qini_curve


class Const:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def test_qini_perfect_ranking_beats_random():
    rng = np.random.default_rng(0)
    n = 20000
    tau = rng.normal(0, 1, n)
    t = rng.integers(0, 2, n)
    y = rng.normal(0, 1, n) + t * tau
    grid = np.linspace(0.05, 1, 20)
    best, rand = qini_curve(tau, t, y, grid), qini_curve(rng.normal(size=n), t, y, grid)
    assert best[np.argmin(abs(grid - 0.5))] > rand[np.argmin(abs(grid - 0.5))] + 0.2
    assert np.isclose(best[-1], rand[-1])


def test_aipw_with_constant_models_is_ipw():
    rng = np.random.default_rng(1)
    t, y = rng.integers(0, 2, 5000), rng.normal(size=5000)
    psi = aipw_scores(Const(0.0), Const(0.0), np.zeros((5000, 1)), t, y)
    assert np.isclose(psi.mean(), np.mean(y * t / 0.5 - y * (1 - t) / 0.5))


def test_policy_gain_of_treat_all_is_zero():
    rng = np.random.default_rng(2)
    t, y = rng.integers(0, 2, 1000), rng.normal(size=1000)
    g, se = ipw_policy_gain(np.ones(1000, int), t, y)
    assert g == 0 and se == 0


def test_models_find_the_harmed_segment(experiment):
    df, _ = experiment
    X = features(df).to_numpy(float)
    t = (df["group"] == "treatment").to_numpy(int)
    models, cf = fit_models(X, t, df["revenue"].to_numpy(float))
    tau_cf = np.asarray(models["causal forest"](X)).ravel()
    harmed = ((df["device"] == "desktop") & (df["tenure_days"] > 730)).to_numpy()
    assert np.isfinite(tau_cf).all() and tau_cf.shape == (len(df),)
    assert tau_cf[harmed].mean() < tau_cf[~harmed].mean()
