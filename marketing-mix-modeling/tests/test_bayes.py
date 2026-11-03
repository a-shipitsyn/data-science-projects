import numpy as np
import pandas as pd

from bayes import build, inputs, lagged
from model import controls, scales, tv_years
from transforms import CHANNELS, adstock


def test_lagged_matrix_reproduces_adstock():
    x = np.random.default_rng(0).random(60)
    a = lagged(x, 60) @ 0.6 ** np.arange(60)
    np.testing.assert_allclose(a, adstock(x, 0.6))


def test_model_builds_with_lift_tests(sim):
    df, _, _ = sim
    df = df.iloc[:400]
    lift = pd.DataFrame({"channel": ["ooh"], "start": [df["date"].iloc[100]], "end": [df["date"].iloc[130]],
                         "lift": [100.0], "se": [15.0]})
    years, scale = tv_years(df), scales(df)
    names = list(controls(df).columns.drop("const"))
    m = build(df, inputs(df, years, scale, [], lift), years, lift, names, [])
    logp = m.compile_logp()(m.initial_point())
    assert np.isfinite(logp)
    assert {"decay", "half_sat", "shape", "beta", "beta_tv"} <= {v.name for v in m.free_RVs}
    assert len(CHANNELS) == m["decay"].eval().shape[0]
