from impact import readout
from simulate import simulate_market, simulate_pilot, simulate_staggered


def test_pilot_readout_prefers_dr_did():
    df, truth = simulate_pilot(seed=1)
    report, s = readout(df, "store", covariates=["urban", "size_index", "region"], boot=50, placebos=20)
    assert s["design"] == "single date"
    assert s["recommended"] == "doubly robust DiD"
    lo, hi = s["ci"]
    assert lo <= truth["att"] <= hi
    for h in ["Design", "Pre-trends", "Estimates", "Recommendation"]:
        assert h in report


def test_staggered_readout_warns_about_twfe():
    df, _ = simulate_staggered(seed=2)
    report, s = readout(df, "store", boot=50)
    assert s["design"] == "staggered"
    assert "negative weight" in report and "do not use it here" in report


def test_market_readout_uses_placebos():
    df, _ = simulate_market(seed=3)
    report, s = readout(df, "region", placebos=20)
    assert s["design"] == "single unit"
    assert "(placebo)" in report
