"""Mock data for marketing mix modelling with a known ground truth.

Daily calls from potential buyers, driven by
    baseline     level + trend + monthly seasonality + weekday + holidays
    media        TV (flights of a few weeks), outdoor (monthly), digital (always on, with bursts);
                 each with adstock and Hill saturation; TV effectiveness declines every year
    shock        demand pull-forward after a sharp currency depreciation (7-day log return)
plus negative binomial noise and a few one-day outliers.
Media exposure is in thousands of contacts (OTS), spend in currency units at a fixed
cost per thousand contacts (CPM).

Lift tests: experiments that measure a channel's incremental calls over one month (e.g. by
switching it off in test regions), simulated as the true value plus noise with a 15 %
standard error — two for outdoor, one for digital. They are used to calibrate the Bayesian
model (bayes.py).

Usage:
    python simulate.py --output ../data/mock_daily.csv --truth ../data/truth.json \
                       --lift-tests ../data/lift_tests.csv [--seed 7]
"""
import argparse
import json

import numpy as np
import pandas as pd

from transforms import CHANNELS, media_effect

TRUTH = {
    "base": 25.0, "trend_per_year": 2.0, "season": 8.0, "weekend": -9.0, "holiday": -15.0,
    "fx_shock": 150.0,
    "media": {
        "tv": {"decay": 0.6, "half_sat": 20000.0, "shape": 2.0, "coef_by_year": {"2021": 30.0, "2022": 24.0, "2023": 18.0}},
        "ooh": {"decay": 0.3, "half_sat": 2000.0, "shape": 1.5, "coef": 12.0},
        "digital": {"decay": 0.15, "half_sat": 900.0, "shape": 1.2, "coef": 10.0},
    },
    "cpm": {"tv": 4.0, "ooh": 2.0, "digital": 6.0},
    "dispersion": 50.0,
}
SEASON_BY_MONTH = [0.05, 0.0, 0.75, 0.5, 0.15, 0.35, 0.1, 0.35, 0.95, 1.0, 0.65, 0.85]
TV_FLIGHTS = [("2021-02-15", 21), ("2021-09-06", 14), ("2022-03-14", 21), ("2022-10-03", 21),
              ("2023-03-06", 21), ("2023-09-18", 28)]


def simulate(seed: int = 7) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    n = len(dates)
    df = pd.DataFrame({"date": dates})

    # media exposure, thousands of contacts
    tv = np.zeros(n)
    for start, days in TV_FLIGHTS:
        i = dates.get_loc(pd.Timestamp(start))
        tv[i:i + days] = rng.uniform(8000, 16000) * rng.uniform(0.8, 1.2, days)
    months = dates.to_period("M")
    month_ids = pd.factorize(months)[0]
    ooh_month = np.where(rng.random(month_ids.max() + 1) < 0.5, rng.uniform(30000, 90000, month_ids.max() + 1), 0)
    ooh = ooh_month[month_ids] / dates.days_in_month
    burst = np.isin(month_ids, rng.choice(month_ids.max() + 1, 8, replace=False))
    digital = np.clip(rng.normal(400, 120, n) * np.where(burst, 2.5, 1.0), 0, None)
    df["tv"], df["ooh"], df["digital"] = tv.round(), ooh.round(1), digital.round()
    for ch in CHANNELS:
        df[f"{ch}_spend"] = (df[ch] * TRUTH["cpm"][ch]).round()

    # controls
    df["season"] = np.array(SEASON_BY_MONTH)[dates.month - 1]
    shock = np.zeros(n)
    i = dates.get_loc(pd.Timestamp("2022-11-20"))
    shock[i:i + 10] = 0.03                             # ~30 % depreciation over ten days
    df["fx_rate"] = (60 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, n) + shock))).round(3)

    # expected calls
    t = TRUTH
    fx_ret7 = np.log(df["fx_rate"]).diff(7).fillna(0).clip(lower=0)
    holiday = ((dates.month == 1) & (dates.day <= 8)) | ((dates.month == 5) & (dates.day <= 3))
    df["holiday"] = holiday.astype(int)
    baseline = (t["base"] + t["trend_per_year"] * np.arange(n) / 365 + t["season"] * df["season"]
                + t["weekend"] * (dates.day_of_week >= 5) + t["holiday"] * holiday + t["fx_shock"] * fx_ret7)
    contrib = {}
    for ch in CHANNELS:
        m = t["media"][ch]
        effect = media_effect(df[ch].to_numpy(), decay=m["decay"], half_sat=m["half_sat"], shape=m["shape"])
        coef = np.array([m["coef_by_year"][str(y)] for y in dates.year]) if ch == "tv" else m["coef"]
        contrib[ch] = coef * effect
    mean = np.clip(baseline + sum(contrib.values()), 0.5, None)

    # negative binomial noise, a few one-day outliers
    calls = rng.poisson(rng.gamma(t["dispersion"], mean / t["dispersion"]))
    outliers = rng.choice(n, 5, replace=False)
    calls[outliers] = calls[outliers] * 3
    df.insert(1, "calls", calls)

    truth = json.loads(json.dumps(TRUTH))
    truth["true_contribution_share"] = {ch: round(contrib[ch].sum() / mean.sum(), 4) for ch in CHANNELS}
    years = dates.year.astype(str)
    truth["true_tv_calls_per_1000_contacts"] = {
        y: round(1000 * contrib["tv"][years == y].sum() / df.loc[years == y, "tv"].sum(), 4) for y in np.unique(years)}
    truth["true_roi_calls_per_1000"] = {ch: round(1000 * contrib[ch].sum() / df[f"{ch}_spend"].sum(), 4)
                                        for ch in CHANNELS}
    return df, truth, lift_tests(df, contrib, np.random.default_rng(seed + 1))


def lift_tests(df: pd.DataFrame, contrib: dict, rng: np.random.Generator, se: float = 0.15) -> pd.DataFrame:
    """Measured incremental calls of a channel over a month: true value with relative noise `se`."""
    month = df["date"].dt.to_period("M")
    rows = []
    for ch, year, count in [("ooh", 2022, 1), ("ooh", 2023, 1), ("digital", 2023, 1)]:
        active = df.loc[(df[ch] > 0) & (df["date"].dt.year == year)]
        busiest = active.groupby(month[active.index])[ch].sum().nlargest(count).index
        for m in busiest:
            true = float(contrib[ch][(month == m).to_numpy()].sum())
            rows.append({"channel": ch, "start": m.start_time.date(), "end": m.end_time.date(),
                         "lift": round(true * (1 + se * rng.standard_normal()), 1), "se": round(se * true, 1),
                         "true_lift": round(true, 1)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate mock marketing mix data with a known ground truth")
    ap.add_argument("--output", default="../data/mock_daily.csv")
    ap.add_argument("--truth", default="../data/truth.json")
    ap.add_argument("--lift-tests", default="../data/lift_tests.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df, truth, lift = simulate(args.seed)
    df.to_csv(args.output, index=False)
    lift.drop(columns="true_lift").to_csv(args.lift_tests, index=False)
    truth["true_lift"] = lift["true_lift"].tolist()
    with open(args.truth, "w") as f:
        json.dump(truth, f, indent=2)
    print(f"{len(df)} days: {args.output}, ground truth: {args.truth}, lift tests: {args.lift_tests}")


if __name__ == "__main__":
    main()
