import numpy as np
import pytest

from transforms import adstock, carryover, hill, weibull_adstock, weibull_kernel


def test_adstock_carries_over():
    np.testing.assert_allclose(adstock(np.array([1.0, 0, 0]), 0.5), [1, 0.5, 0.25])
    np.testing.assert_allclose(adstock(np.array([1.0, 1, 1]), 0.0), [1, 1, 1])


def test_hill_shape():
    assert hill(np.array([2.0]), half_sat=2.0, shape=1.5)[0] == pytest.approx(0.5)
    x = np.linspace(0, 100, 50)
    y = hill(x, half_sat=10, shape=2)
    assert y[0] == 0 and np.all(np.diff(y) >= 0) and y[-1] < 1


def test_weibull_kernel_delayed_peak():
    w = weibull_kernel(k=3.0, scale=5.0)
    assert w.max() == pytest.approx(1.0) and 2 <= np.argmax(w) <= 5
    assert np.argmax(weibull_kernel(k=0.8, scale=3.0)) == 0


def test_carryover_geometric_and_weibull():
    assert carryover({"decay": 0.5}) == pytest.approx(2.0)
    x = np.ones(200)
    p = {"k": 2.0, "scale": 4.0}
    assert weibull_adstock(x, **p)[-1] == pytest.approx(carryover(p))
