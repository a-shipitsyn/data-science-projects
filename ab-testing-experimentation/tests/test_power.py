import numpy as np

from power import historical_variances, mde, n_per_arm, power_at


def test_sample_size_formula():
    assert np.isclose(n_per_arm(0.1, 1.0), 2 * (1.959964 + 0.841621) ** 2 / 0.01, rtol=1e-5)
    n = n_per_arm(0.05, 4.0)
    assert np.isclose(mde(n, 4.0), 0.05)
    assert np.isclose(power_at(n, 0.05, 4.0), 0.8, atol=1e-3)


def test_variance_reduction_lowers_sample_size():
    hv = historical_variances(n=60_000, seed=1)["revenue"]
    assert 0.3 < hv["rho"] < 0.5
    v = hv["var"]
    assert v["regression adjustment"] < v["CUPED"] < v["difference in means"]
    saving = 1 - n_per_arm(1, v["CUPED"]) / n_per_arm(1, v["difference in means"])
    assert np.isclose(saving, hv["rho"] ** 2)
