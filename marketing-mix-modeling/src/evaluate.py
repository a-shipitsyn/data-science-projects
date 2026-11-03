"""Model evaluation: baseline without media, rolling-origin cross-validation, bootstrap intervals.

baseline       OLS on the controls only (trend, seasonality, holidays, weekdays, currency
               shock) — shows how much the media variables add.
rolling CV     expanding training window, the next `horizon` days as test; hyperparameters
               are searched again in every fold, so the test days never influence them.
bootstrap      moving-block bootstrap of the residuals (blocks keep autocorrelation), model
               refitted with fixed hyperparameters: 95 % intervals for contribution shares,
               ROI and TV effectiveness per year. Hyperparameter uncertainty is not included.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from model import Results, contributions, controls, fit, fit_design, media_summary, metrics, predict, search, tv_change
from transforms import CHANNELS


def baseline_predict(df: pd.DataFrame, n_train: int) -> pd.Series:
    """Controls-only OLS fitted on the first n_train days, predictions for all days."""
    X = controls(df)
    model = sm.OLS(df["calls"].iloc[:n_train], X.iloc[:n_train]).fit()
    return pd.Series(np.asarray(X @ model.params), index=df.index)


def rolling_cv(df: pd.DataFrame, folds: int = 3, horizon: int = 90, family: str = "ols", seed: int = 0,
               maxiter: int = 200, adstock: str = "geometric") -> pd.DataFrame:
    rows = []
    for k in range(folds):
        end = len(df) - (folds - k - 1) * horizon
        hist = df.iloc[:end]
        n_train = end - horizon
        train = hist.iloc[:n_train]
        params = search(train, seed, maxiter, adstock)
        model = fit(train, params, family)
        test = hist.index[n_train:]
        pred = predict(model, hist, params)
        base = baseline_predict(hist, n_train)
        y = hist.loc[test, "calls"]
        rows.append({"fold": k + 1, "test_start": f"{hist.loc[test[0], 'date']:%Y-%m-%d}",
                     **{f"mmm_{m}": v for m, v in metrics(y, pred[test]).items()},
                     **{f"baseline_{m}": v for m, v in metrics(y, base[test]).items()}})
    return pd.DataFrame(rows)


def bootstrap(model: Results, df: pd.DataFrame, params: dict, family: str = "ols", n: int = 200, block: int = 28,
              seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names, index=df.index)
    fitted = pd.Series(np.asarray(X @ model.params), index=df.index)
    resid = (df["calls"] - fitted).to_numpy()
    starts_needed = int(np.ceil(len(df) / block))
    draws = []
    for _ in range(n):
        starts = rng.integers(0, len(df) - block, starts_needed)
        r = np.concatenate([resid[s:s + block] for s in starts])[:len(df)]
        y = (fitted + r).clip(lower=0)
        m = fit_design(y, X, family, alpha=getattr(model, "alpha", None))
        m.tv_years = model.tv_years
        summary = media_summary(df, contributions(m, df, params))
        tv = tv_change(m, df, params)["by_year"] if len(model.tv_years) > 1 else {}
        draws.append({**{f"{ch}_share": summary[ch]["share"] for ch in CHANNELS},
                      **{f"{ch}_calls_per_1000": summary[ch]["calls_per_1000"] for ch in CHANNELS},
                      **{f"tv_{y}_calls_per_1000_contacts": v["calls_per_1000_contacts"] for y, v in tv.items()}})
    draws = pd.DataFrame(draws)
    return {c: {"ci_low": float(draws[c].quantile(0.025)), "ci_high": float(draws[c].quantile(0.975))}
            for c in draws.columns}
