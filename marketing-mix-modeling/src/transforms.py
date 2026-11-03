"""Media transforms used in marketing mix modelling.

adstock          geometric carry-over: a_t = x_t + decay * a_{t-1}; the effect peaks on the
                 day of exposure and fades
weibull_adstock  carry-over with a flexible shape: weights follow a Weibull density over the
                 lag (scaled to a maximum of 1), so the effect can peak days after exposure
                 (shape k > 1), as for awareness-building media
hill             saturation: diminishing returns of exposure, 0 at 0, 0.5 at half_sat, -> 1
"""
import numpy as np

CHANNELS = ["tv", "ooh", "digital"]
MAX_LAG = 28


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.empty(len(x))
    carry = 0.0
    for i, value in enumerate(x):
        carry = value + decay * carry
        out[i] = carry
    return out


def weibull_kernel(k: float, scale: float, max_lag: int = MAX_LAG) -> np.ndarray:
    lag = np.arange(max_lag) + 0.5
    w = (k / scale) * (lag / scale) ** (k - 1) * np.exp(-((lag / scale) ** k))
    return w / w.max()


def weibull_adstock(x: np.ndarray, k: float, scale: float, max_lag: int = MAX_LAG) -> np.ndarray:
    return np.convolve(x, weibull_kernel(k, scale, max_lag))[:len(x)]


def carryover(params: dict) -> float:
    """Steady-state multiplier of the adstock: a constant exposure x gives adstock x * carryover."""
    if "decay" in params:
        return 1 / (1 - params["decay"])
    return float(weibull_kernel(params["k"], params["scale"]).sum())


def hill(x: np.ndarray, half_sat: float, shape: float) -> np.ndarray:
    x = np.maximum(x, 0)
    return x**shape / (x**shape + half_sat**shape)


def media_effect(x: np.ndarray, half_sat: float, shape: float, decay: float | None = None, k: float | None = None,
                 scale: float | None = None) -> np.ndarray:
    """Adstock (geometric with `decay`, or Weibull with `k`, `scale`), then saturation: values in [0, 1)."""
    a = adstock(x, decay) if decay is not None else weibull_adstock(x, k, scale)
    return hill(a, half_sat, shape)
