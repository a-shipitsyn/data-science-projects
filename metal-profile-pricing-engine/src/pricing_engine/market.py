"""Aluminium market price (LME cash, USD/t) and its time-series features.

Features are computed as of the day *before* the quote date, so a quote never sees the
price of its own day or later.
"""
import numpy as np
import pandas as pd

MONTH = 22    # trading days
QUARTER = 66


def load_lme(path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.set_index("date")["price_usd_t"].sort_index()
    if s.isna().any() or (s <= 0).any():
        raise ValueError("market price series has missing or non-positive values")
    return s


def lme_features(lme: pd.Series, dates) -> pd.DataFrame:
    """Lag, moving averages, 1-month return and volatility as of each date (exclusive)."""
    log_ret = np.log(lme).diff()
    table = pd.DataFrame({
        "lme_lag1": lme,
        "lme_ma1m": lme.rolling(MONTH).mean(),
        "lme_ma3m": lme.rolling(QUARTER).mean(),
        "lme_ret1m": lme / lme.shift(MONTH) - 1,
        "lme_vol1m": log_ret.rolling(MONTH).std() * np.sqrt(252),
    })
    # value known at the end of day d is used for quotes from day d+1
    table.index = (table.index + pd.Timedelta(days=1)).astype("datetime64[ns]")
    query = pd.DataFrame({"quote_date": pd.to_datetime(pd.Series(dates)).astype("datetime64[ns]")})
    order = query["quote_date"].argsort(kind="stable")
    merged = pd.merge_asof(query.iloc[order], table, left_on="quote_date", right_index=True)
    merged.index = query.index[order]
    out = merged.sort_index().drop(columns="quote_date")
    if out.isna().any().any():
        raise ValueError("market history too short for some quote dates")
    return out.reset_index(drop=True)
