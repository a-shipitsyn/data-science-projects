"""Marketing mix model: features, hyperparameter search, fit and model statistics.

Features: controls (trend, seasonality index, holidays, weekdays, currency shock = positive
7-day log return of the exchange rate) + media effects (adstock -> Hill saturation per
channel; geometric adstock, or Weibull adstock that allows a delayed peak). TV has a separate coefficient per year to measure the change of its effect;
later years without own estimate use the latest one.

Fit: OLS, or a negative binomial GLM with identity link (additive effects, count noise whose
variance grows with the mean). Days with extreme residuals (> 4 robust standard deviations)
get a dummy each. Standard errors are HAC (Newey-West).
"""
import warnings
from itertools import pairwise

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import differential_evolution
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from transforms import CHANNELS, carryover, media_effect

Results = RegressionResultsWrapper | GLMResultsWrapper

# Search ranges. Decay per channel type follows common practice (e.g. the Robyn guidelines):
# TV carries over longest, outdoor less, digital the least. Without such prior knowledge an
# always-on channel can absorb part of the baseline. half_sat is relative to the typical
# steady-state exposure of the channel.
DECAY = {"tv": (0.3, 0.8), "ooh": (0.1, 0.4), "digital": (0.0, 0.3)}
HALF_SAT = (0.3, 2.0)
SHAPE = (0.5, 3.0)
WEIBULL_K = (0.5, 5.0)          # k > 1: the effect peaks days after exposure
WEIBULL_SCALE = (0.5, 14.0)     # days
REQUIRED = ["date", "calls", *CHANNELS, *[f"{ch}_spend" for ch in CHANNELS], "season", "fx_rate", "holiday"]
WEEKDAYS = ["tue", "wed", "thu", "fri", "sat", "sun"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, missing and negative values, daily dates; raise ValueError listing all problems."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    errors = [f"{c}: {n} missing values" for c, n in df[REQUIRED].isna().sum().items() if n]
    for c in ["calls", *CHANNELS, *[f"{ch}_spend" for ch in CHANNELS], "fx_rate"]:
        if (df[c] < 0).any():
            errors.append(f"{c}: negative values")
    if not (df["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        errors.append("date: not consecutive days")
    if errors:
        raise ValueError("invalid input: " + "; ".join(errors))
    return df


def controls(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({"const": 1.0, "trend": np.arange(len(df)) / 365, "season": df["season"],
                      "holiday": df["holiday"]}, index=df.index)
    X["fx_shock"] = np.log(df["fx_rate"]).diff(7).fillna(0).clip(lower=0)
    for k, day in enumerate(WEEKDAYS, start=1):
        X[day] = (df["date"].dt.day_of_week == k).astype(int)
    return X


def scales(df: pd.DataFrame) -> dict:
    """Typical exposure of each channel, so that half-saturation is searched on a relative scale."""
    return {ch: float(df.loc[df[ch] > 0, ch].mean()) for ch in CHANNELS}


def tv_years(df: pd.DataFrame) -> list[int]:
    """Years with TV activity: each gets its own TV coefficient."""
    return sorted(df.loc[df["tv"] > 0, "date"].dt.year.unique().tolist())


def media(df: pd.DataFrame, params: dict, years: list[int] | None = None) -> pd.DataFrame:
    """Media effects; with `years` TV is split by year, other years mapped to the latest earlier one."""
    X = pd.DataFrame(index=df.index)
    for ch in CHANNELS:
        effect = media_effect(df[ch].to_numpy(dtype=float), **params[ch])
        if ch == "tv" and years:
            year = df["date"].dt.year.to_numpy()
            assigned = np.array(years)[np.clip(np.searchsorted(years, year, side="right") - 1, 0, None)]
            for y in years:
                X[f"tv_{y}"] = effect * (assigned == y)
        else:
            X[ch] = effect
    return X


def design(df: pd.DataFrame, params: dict, years: list[int] | None = None) -> pd.DataFrame:
    return controls(df).join(media(df, params, years))


def decode(z: np.ndarray, scale: dict, adstock: str = "geometric") -> dict:
    """Vector of hyperparameters -> {channel: {decay | k, scale; half_sat, shape}};
    half_sat is given relative to the channel's typical steady-state adstock."""
    out = {}
    size = 3 if adstock == "geometric" else 4
    for i, ch in enumerate(CHANNELS):
        v = z[size * i:size * (i + 1)]
        p = {"decay": v[0]} if adstock == "geometric" else {"k": v[0], "scale": v[1]}
        p |= {"half_sat": v[-2] * scale[ch] * carryover(p), "shape": v[-1]}
        out[ch] = p
    return out


def bounds(adstock: str = "geometric") -> list[tuple]:
    if adstock == "geometric":
        return [b for ch in CHANNELS for b in (DECAY[ch], HALF_SAT, SHAPE)]
    return [b for _ in CHANNELS for b in (WEIBULL_K, WEIBULL_SCALE, HALF_SAT, SHAPE)]


def search(df: pd.DataFrame, seed: int = 0, maxiter: int = 200, adstock: str = "geometric") -> dict:
    """Hyperparameters minimising the least-squares error, negative media effects penalised."""
    scale, years = scales(df), tv_years(df)
    y = df["calls"].to_numpy(dtype=float)

    def loss(z: np.ndarray) -> float:
        X = design(df, decode(z, scale, adstock), years)
        beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
        resid = y - X.to_numpy() @ beta
        media_cols = [i for i, c in enumerate(X.columns) if c.startswith(tuple(CHANNELS))]
        return resid @ resid + 1e3 * np.sum(np.clip(-beta[media_cols], 0, None))

    res = differential_evolution(loss, bounds(adstock), seed=seed, tol=1e-6, maxiter=maxiter, polish=True)
    return {ch: {k: round(float(v), 4) for k, v in p.items()} for ch, p in decode(res.x, scale, adstock).items()}


def outlier_dummies(df: pd.DataFrame, X: pd.DataFrame, z: float = 4.0) -> pd.DataFrame:
    resid = sm.OLS(df["calls"], X).fit().resid
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    X = X.copy()
    for i in df.index[np.abs(resid) > z * mad]:
        X[f"outlier_{df.loc[i, 'date']:%Y-%m-%d}"] = (df.index == i).astype(int)
    return X


def fit_design(y: pd.Series, X: pd.DataFrame, family: str = "ols", alpha: float | None = None) -> Results:
    """Fit on a ready design matrix. For 'nb', alpha (NB dispersion) is estimated by moments if not given."""
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    if family == "ols":
        return ols
    if alpha is None:
        mu = ols.fittedvalues.clip(lower=0.5)
        alpha = max(float(np.mean((ols.resid**2 - mu) / mu**2)), 1e-4)
    with warnings.catch_warnings():  # identity link is intended: effects are additive, as in the OLS model
        warnings.simplefilter("ignore", sm.tools.sm_exceptions.DomainWarning)
        glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha, link=sm.families.links.Identity()))
    model = glm.fit(start_params=ols.params, cov_type="HAC", cov_kwds={"maxlags": 7})
    model.alpha = alpha
    return model


def fit(df: pd.DataFrame, params: dict, family: str = "ols", years: list[int] | None = None) -> Results:
    """Fit with outlier dummies; `years` are the TV years (default: years with TV in df)."""
    years = years or tv_years(df)
    model = fit_design(df["calls"], outlier_dummies(df, design(df, params, years)), family)
    model.tv_years = years
    return model


def predict(model: Results, df: pd.DataFrame, params: dict) -> pd.Series:
    """Prediction without outlier dummies (they only absorb known anomalies in training)."""
    X = design(df, params, model.tv_years)
    return pd.Series(np.asarray(X @ model.params[X.columns]), index=df.index)


def residuals(model: Results) -> pd.Series:
    """Raw residuals for OLS, Pearson residuals for the GLM (should be homoscedastic)."""
    return model.resid_pearson if isinstance(model.model, sm.GLM) \
        else model.resid


def metrics(y: pd.Series, pred: pd.Series) -> dict:
    e = y - pred
    return {"r2": float(1 - (e**2).sum() / ((y - y.mean()) ** 2).sum()), "rmse": float(np.sqrt((e**2).mean())),
            "mape": float((e.abs() / y.clip(lower=1)).mean())}


def diagnostics(model: Results) -> dict:
    resid = residuals(model)
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    X = X.loc[:, (X.std() > 0) & ~X.columns.str.startswith("outlier_")]
    vif = {c: variance_inflation_factor(X.to_numpy(), i) for i, c in enumerate(X.columns) if c != "const"}
    return {
        "durbin_watson": float(durbin_watson(resid)),
        "ljung_box_p_lag7": float(acorr_ljungbox(resid, lags=[7])["lb_pvalue"].iloc[0]),
        "breusch_pagan_p": float(het_breuschpagan(resid, model.model.exog)[1]),
        "jarque_bera_p": float(jarque_bera(resid)[1]),
        "max_vif": float(max(vif.values())), "max_vif_variable": max(vif, key=vif.get),
    }


def tv_change(model: Results, df: pd.DataFrame, params: dict) -> dict:
    """TV effectiveness per year: incremental calls per 1000 contacts (95 % CI from the
    coefficient), and a Wald test that the TV coefficient is equal in all years."""
    X = design(df, params, model.tv_years)
    cols = [f"tv_{y}" for y in model.tv_years]
    ci = model.conf_int()
    out = {}
    for c in cols:
        contacts = df.loc[X[c] > 0, "tv"].sum() or np.nan
        per_unit = 1000 * X[c].sum() / contacts
        out[c[3:]] = {"calls_per_1000_contacts": model.params[c] * per_unit,
                      "ci_low": ci.loc[c, 0] * per_unit, "ci_high": ci.loc[c, 1] * per_unit}
    p = model.wald_test(", ".join(f"{a} = {b}" for a, b in pairwise(cols)), scalar=True).pvalue if len(cols) > 1 \
        else np.nan
    return {"by_year": out, "equal_years_p": float(p)}


def contributions(model: Results, df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Daily contribution of each driver (coefficient x feature), TV years merged."""
    X = design(df, params, model.tv_years)
    c = X * model.params[X.columns]
    tv_cols = [col for col in c.columns if col.startswith("tv_")]
    c["tv"] = c[tv_cols].sum(axis=1)
    c["weekday"] = c[WEEKDAYS].sum(axis=1)
    return c.drop(columns=tv_cols + WEEKDAYS)


def media_summary(df: pd.DataFrame, contrib: pd.DataFrame) -> dict:
    """Per channel: share of all calls, incremental calls per 1000 of spend, cost per incremental call."""
    out = {}
    for ch in CHANNELS:
        calls, spend = contrib[ch].sum(), df[f"{ch}_spend"].sum()
        out[ch] = {"share": calls / df["calls"].sum(), "calls_per_1000": 1000 * calls / spend,
                   "cost_per_call": spend / calls if calls > 0 else np.inf}
    return out
