"""Bayesian marketing mix model (PyMC), optionally calibrated with lift tests.

Same structure as model.py — geometric adstock and Hill saturation per channel, TV effect
per year, controls, outlier dummies — but all parameters, including the media
hyperparameters, are estimated jointly with priors:
    decay        Beta per channel type, centred on the usual ranges (TV 0.3-0.8,
                 outdoor 0.1-0.4, digital 0-0.3)
    half_sat     LogNormal around the channel's typical steady-state exposure
    shape        Gamma, mean 1.5
    media coef   HalfNormal (effects are non-negative)
    controls     Normal on standardised controls
    likelihood   negative binomial on daily calls, mean = sum of additive effects
Lift tests (channel, period, measured incremental calls, standard error) add a likelihood
term: the model's incremental calls of the channel over the period ~ Normal(measured, se).
This is how experiments calibrate an MMM when channels are hard to separate.

Intervals are 95 % posterior intervals and include the uncertainty of the hyperparameters.

Usage:
    python bayes.py --input ../data/mock_daily.csv --output ../results/bayes \
                    [--lift-tests ../data/lift_tests.csv] [--holdout 90] [--truth ../data/truth.json]
"""
import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from model import controls, metrics, outlier_dummies, scales, tv_years, validate
from transforms import CHANNELS, MAX_LAG

DECAY_PRIOR = {"tv": (5.5, 4.5), "ooh": (2.5, 7.5), "digital": (1.5, 8.5)}   # Beta(a, b), means 0.55 / 0.25 / 0.15


def lagged(x: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Matrix with column l = x shifted by l days (zeros before the start)."""
    out = np.zeros((len(x), max_lag))
    for lag in range(max_lag):
        out[lag:, lag] = x[:len(x) - lag]
    return out


def inputs(df: pd.DataFrame, years: list[int], scale: dict, outliers: list[str], lift: pd.DataFrame | None) -> dict:
    """Numeric inputs of the model for a data frame (training data or the full history)."""
    C = controls(df).drop(columns="const")
    year = df["date"].dt.year.to_numpy()
    tv_year = np.clip(np.searchsorted(years, year, side="right") - 1, 0, None)
    dates = df["date"].dt.strftime("%Y-%m-%d")
    data = {"controls": C.to_numpy(dtype=float), "tv_year_idx": tv_year,
            "outliers": np.column_stack([(dates == d).to_numpy(dtype=float) for d in outliers])
            if outliers else np.zeros((len(df), 0))}
    for ch in CHANNELS:
        data[f"lag_{ch}"] = lagged(df[ch].to_numpy(dtype=float) / scale[ch])
    if lift is not None:
        data["lift_mask"] = np.array([((df["date"] >= r.start) & (df["date"] <= r.end)).to_numpy(dtype=float)
                                      for r in lift.itertuples()])
    return data


def build(df: pd.DataFrame, data: dict, years: list[int], lift: pd.DataFrame | None,
          control_names: list[str], outliers: list[str]) -> pm.Model:
    y = df["calls"].to_numpy(dtype=float)
    c_mean, c_std = data["controls"].mean(axis=0), data["controls"].std(axis=0) + 1e-9
    coords = {"channel": CHANNELS, "control": control_names, "tv_year": [str(y) for y in years],
              "outlier": outliers, "lag": np.arange(MAX_LAG)}
    with pm.Model(coords=coords) as m:
        C = pm.Data("controls", data["controls"])
        tv_year = pm.Data("tv_year_idx", data["tv_year_idx"])
        O = pm.Data("outliers", data["outliers"])
        intercept = pm.Normal("intercept", y.mean(), 20)
        gamma = pm.Normal("gamma", 0, 10, dims="control")
        base = intercept + pt.dot((C - c_mean) / c_std, gamma)
        out_coef = pm.Normal("outlier_coef", 0, 50, dims="outlier")

        decay = pm.Beta("decay", [DECAY_PRIOR[ch][0] for ch in CHANNELS], [DECAY_PRIOR[ch][1] for ch in CHANNELS],
                        dims="channel")
        half_sat = pm.LogNormal("half_sat", 0, 0.5, dims="channel")
        shape = pm.Gamma("shape", mu=1.5, sigma=0.75, dims="channel")
        beta_tv = pm.HalfNormal("beta_tv", 30, dims="tv_year")
        beta = pm.HalfNormal("beta", 30, dims="channel")

        contrib = {}
        for i, ch in enumerate(CHANNELS):
            lag = pm.Data(f"lag_{ch}", data[f"lag_{ch}"])
            a = pt.dot(lag, decay[i] ** pt.arange(MAX_LAG)) * (1 - decay[i])   # steady state of typical exposure = 1
            sat = 1 / (1 + (half_sat[i] / (a + 1e-6)) ** shape[i])               # Hill, finite gradient at a = 0
            coef = beta_tv[tv_year] if ch == "tv" else beta[i]
            contrib[ch] = pm.Deterministic(f"contrib_{ch}", coef * sat)
        mu = pm.Deterministic("mu", base + sum(contrib.values()) + pt.dot(O, out_coef))

        alpha = pm.HalfNormal("alpha", 100)
        pm.NegativeBinomial("calls", mu=pt.maximum(mu, 0.5), alpha=alpha, observed=y, shape=mu.shape)
        if lift is not None:
            mask = pm.Data("lift_mask", data["lift_mask"])
            measured = pt.stack([pt.dot(mask[j], contrib[ch]) for j, ch in enumerate(lift["channel"])])
            pm.Normal("lift", mu=measured, sigma=lift["se"].to_numpy(dtype=float), observed=lift["lift"].to_numpy())
    return m


def sample(m: pm.Model, draws: int, chains: int, seed: int) -> az.InferenceData:
    with m:
        return pm.sample(draws=draws, tune=draws, chains=chains, target_accept=0.9, random_seed=seed,
                         progressbar=False)


def interval(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "ci_low": float(np.quantile(x, 0.025)), "ci_high": float(np.quantile(x, 0.975))}


def summarize(idata: az.InferenceData, df: pd.DataFrame, years: list[int], scale: dict) -> dict:
    post = idata.posterior.stack(sample=("chain", "draw"))
    out = {"decay": {ch: interval(post["decay"].sel(channel=ch).values) for ch in CHANNELS}, "media": {}}
    for ch in CHANNELS:
        calls = post[f"contrib_{ch}"].sum("contrib_" + ch + "_dim_0").values
        out["media"][ch] = {"share": interval(calls / df["calls"].sum()),
                            "calls_per_1000": interval(1000 * calls / df[f"{ch}_spend"].sum())}
    tv = post["contrib_tv"].values
    year = df["date"].dt.year.to_numpy()
    out["tv_calls_per_1000_contacts"] = {
        str(y): interval(1000 * tv[year == y].sum(axis=0) / df.loc[year == y, "tv"].sum()) for y in years}
    summary = az.summary(idata, var_names=["decay", "half_sat", "shape", "beta", "beta_tv", "gamma", "alpha"])
    out["convergence"] = {"max_r_hat": float(summary["r_hat"].max()), "min_ess_bulk": float(summary["ess_bulk"].min()),
                          "divergences": int(idata.sample_stats["diverging"].sum())}
    return out


def posterior_draws(idata: az.InferenceData, years: list[int], scale: dict, n: int, seed: int) -> list[dict]:
    """Draws in the format of model.json (params with absolute half_sat, coefficients) for optimize.py."""
    post = idata.posterior.stack(sample=("chain", "draw"))
    idx = np.random.default_rng(seed).choice(post.sizes["sample"], n, replace=False)
    draws = []
    for i in idx:
        p = post.isel(sample=i)
        params, coefs = {}, {}
        for ch in CHANNELS:
            d = float(p["decay"].sel(channel=ch))
            params[ch] = {"decay": d, "half_sat": float(p["half_sat"].sel(channel=ch)) * scale[ch] / (1 - d),
                          "shape": float(p["shape"].sel(channel=ch))}
            if ch != "tv":
                coefs[ch] = float(p["beta"].sel(channel=ch))
        coefs |= {f"tv_{y}": float(p["beta_tv"].sel(tv_year=str(y))) for y in years}
        draws.append({"params": params, "coefficients": coefs})
    return draws


def fit(df: pd.DataFrame, lift: pd.DataFrame | None, draws: int, chains: int, seed: int) -> tuple:
    years, scale = tv_years(df), scales(df)
    outliers = [c[8:] for c in outlier_dummies(df, controls(df)).columns if c.startswith("outlier_")]
    names = list(controls(df).columns.drop("const"))
    data = inputs(df, years, scale, outliers, lift)
    m = build(df, data, years, lift, names, outliers)
    return m, sample(m, draws, chains, seed), years, scale, outliers


def main() -> None:
    ap = argparse.ArgumentParser(description="Bayesian marketing mix model with optional lift-test calibration")
    ap.add_argument("--input", required=True, help="daily CSV: date, calls, media, spend and control columns")
    ap.add_argument("--output", required=True, help="directory for results")
    ap.add_argument("--lift-tests", help="CSV: channel, start, end, lift, se")
    ap.add_argument("--holdout", type=int, default=90, help="days at the end for validation, 0 = skip")
    ap.add_argument("--draws", type=int, default=1000, help="posterior draws per chain (and as many tuning steps)")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--truth", help="ground truth JSON from simulate.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        df = validate(pd.read_csv(args.input, parse_dates=["date"]))
        lift = pd.read_csv(args.lift_tests, parse_dates=["start", "end"]) if args.lift_tests else None
    except (OSError, ValueError) as err:
        sys.exit(f"{err}")
    result = {"lift_tests": args.lift_tests is not None}

    if args.holdout:
        n_train = len(df) - args.holdout
        train = df.iloc[:n_train]
        lift_train = lift[lift["end"] < train["date"].iloc[-1]] if lift is not None else None
        m, idata, years, scale, outliers = fit(train, lift_train if lift_train is not None and len(lift_train) else None,
                                              args.draws, args.chains, args.seed)
        with m:
            pm.set_data(inputs(df, years, scale, outliers, None))
            mu = pm.sample_posterior_predictive(idata, var_names=["mu"], progressbar=False).posterior_predictive["mu"]
        pred = pd.Series(mu.mean(("chain", "draw")).values, index=df.index)
        test = df.index[n_train:]
        result["holdout"] = metrics(df.loc[test, "calls"], pred[test])
        result["train"] = metrics(train["calls"], pred[train.index])

    m, idata, years, scale, outliers = fit(df, lift, args.draws, args.chains, args.seed)
    result |= summarize(idata, df, years, scale)
    (out / "bayes.json").write_text(json.dumps(result, indent=2))
    (out / "posterior_draws.json").write_text(json.dumps(posterior_draws(idata, years, scale, 200, args.seed)))

    print(f"lift tests: {result['lift_tests']}, convergence: {result['convergence']}")
    if args.holdout:
        print(pd.DataFrame({"train": result["train"], "holdout": result["holdout"]}).round(3).to_string())
    rows = []
    for ch in CHANNELS:
        for k in ["share", "calls_per_1000"]:
            rows.append([f"{ch} {k}", *result["media"][ch][k].values()])
        rows.append([f"{ch} decay", *result["decay"][ch].values()])
    for y, v in result["tv_calls_per_1000_contacts"].items():
        rows.append([f"tv calls per 1000 contacts {y}", *v.values()])
    table = pd.DataFrame(rows, columns=["parameter", "mean", "ci_low", "ci_high"])
    if args.truth:
        truth = json.loads(Path(args.truth).read_text())
        true = {f"{ch} share": truth["true_contribution_share"][ch] for ch in CHANNELS}
        true |= {f"{ch} calls_per_1000": truth["true_roi_calls_per_1000"][ch] for ch in CHANNELS}
        true |= {f"{ch} decay": truth["media"][ch]["decay"] for ch in CHANNELS}
        true |= {f"tv calls per 1000 contacts {y}": v for y, v in truth["true_tv_calls_per_1000_contacts"].items()}
        table.insert(1, "true", table["parameter"].map(true))
    table.to_csv(out / "bayes_summary.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
