import numpy as np

from simulate import HIDDEN, draw_outcomes, draw_users, segment, simulate


def test_columns_and_split(experiment):
    df, truth = experiment
    assert len(df) == 40_000
    assert not set(HIDDEN) & set(df.columns)
    assert set(df["group"]) == {"treatment", "control"}
    assert abs((df["group"] == "treatment").mean() - 0.5) < 0.01
    assert df["day"].between(1, 14).all()
    assert (df["sessions"] >= 1).all() and (df["revenue"] >= 0).all()
    assert ((df["revenue"] > 0) == (df["converted"] == 1)).all()


def test_true_effects_match_outcomes():
    """The stored per-user effects equal the difference of expected outcomes under both arms."""
    rng = np.random.default_rng(0)
    u = draw_users(400_000, rng)
    y1 = draw_outcomes(u, np.ones(len(u), int), rng)
    y0 = draw_outcomes(u, np.zeros(len(u), int), rng)
    d = y1["revenue"] - y0["revenue"]
    se = d.std() / np.sqrt(len(d))
    assert abs(d.mean() - u["true_revenue_effect"].mean()) < 4 * se
    dc = y1["converted"] - y0["converted"]
    assert abs(dc.mean() - u["true_conversion_effect"].mean()) < 4 * dc.std() / np.sqrt(len(dc))


def test_harmed_segment(experiment):
    df, truth = experiment
    eff = truth["population"]["segment_revenue_effect"]
    assert eff["desktop / >2y"] < 0
    assert all(v >= 0 for k, v in eff.items() if k != "desktop / >2y")
    seg = segment(df["device"], df["new_user"], df["tenure_days"])
    assert set(seg) == set(eff)


def test_srm_bug_drops_only_mobile_treatment_non_buyers(experiment_srm):
    bug, _ = experiment_srm
    full, _ = simulate(n=200_000, seed=11)
    lost = full[~full["user_id"].isin(bug["user_id"])]
    assert 0 < len(lost) < 0.05 * len(full)
    assert (lost["group"] == "treatment").all()
    assert (lost["device"] == "mobile").all()
    assert (lost["orders"] == 0).all()
