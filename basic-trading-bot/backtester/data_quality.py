import pandas as pd


def find_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Missing candles: rows where the step to the next timestamp is longer than the usual candle.

    Returns one row per gap with its start, end and the number of missing candles.
    """
    if len(df) < 2:
        return pd.DataFrame(columns=["after", "before", "missing"])
    times = df.index.to_series()
    step = times.diff().median()
    diff = times.diff()
    gaps = diff[diff > step]
    return pd.DataFrame({
        "after": (gaps.index - gaps).to_numpy(),
        "before": gaps.index.to_numpy(),
        "missing": (gaps / step - 1).round().astype(int).to_numpy(),
    })


def report_gaps(df: pd.DataFrame, name: str = "data") -> int:
    """Print a short warning about missing candles; returns the number of missing candles."""
    gaps = find_gaps(df)
    missing = int(gaps["missing"].sum()) if len(gaps) else 0
    if missing:
        print(f"⚠️ {name}: {missing} missing candles in {len(gaps)} gaps, e.g. after {gaps['after'].iloc[0]}")
    return missing
