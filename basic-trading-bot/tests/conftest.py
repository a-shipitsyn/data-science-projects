import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def candles(close, start="2026-01-01", freq="1h") -> pd.DataFrame:
    """OHLCV frame from close prices: open = previous close, high / low 0.1 % around."""
    close = np.asarray(close, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range(start, periods=len(close), freq=freq, name="timestamp")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001,
                         "low": np.minimum(open_, close) * 0.999, "close": close, "volume": 1.0}, index=idx)


@pytest.fixture
def wave():
    """Trending sine wave: several moving-average crossovers."""
    t = np.arange(600)
    return candles(100 + 10 * np.sin(t / 25) + t * 0.01)
