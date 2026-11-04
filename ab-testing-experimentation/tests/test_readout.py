import numpy as np
import pandas as pd

from readout import readout

ARGS = dict(metrics=["revenue", "converted"], covariates={"revenue": "pre_revenue", "converted": "pre_revenue"},
            ratios=["revenue/sessions"], guardrails=["sessions"], segments=["device", "tenure_group"], day="day")


def test_readout_sections(experiment):
    df, _ = experiment
    report, s = readout(df, **ARGS)
    for h in ["Sample ratio check", "Metrics", "Ratio metrics", "Guardrails", "Sequential check", "Segments", "Decision"]:
        assert h in report
    assert s["srm_p"] > 0.001
    worst = min(s["segments"], key=lambda r: r["effect"])
    assert worst["segment"] == "desktop / >2y"


def test_readout_stops_on_srm(experiment_srm):
    df, _ = experiment_srm
    report, s = readout(df, **ARGS)
    assert s["srm_p"] < 0.001
    assert s["decision"].startswith("Investigate")
    assert "device = mobile" in report


def test_readout_ships_clear_win():
    rng = np.random.default_rng(0)
    n = 20000
    t = rng.integers(0, 2, n)
    pre = rng.gamma(2, 10, n)
    df = pd.DataFrame({"group": np.where(t == 1, "treatment", "control"), "pre": pre,
                       "y": pre + rng.normal(0, 5, n) + 1.0 * t})
    _, s = readout(df, metrics=["y"], covariates={"y": "pre"})
    assert s["decision"] == "Ship"
    assert s["metrics"]["y"]["p"] < 0.001
