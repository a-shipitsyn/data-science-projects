"""Weekly store and region panels with a known launch effect, for three quasi-experiments.

A retail chain launches a new in-store service (e.g. click-and-collect lockers). Stores were
not randomised, so the effect has to be estimated from sales before and after the launch.

Log weekly sales of store i in week t without the launch:
    log y_it = size_i + chain trend_t + amplitude_i x seasonality_t + loading_i x region shock_r(i),t
               + urban growth x t (urban stores only) + AR(1) noise
The chain trend and region shocks are random walks; urban stores grow about 3 % a year faster.
With the launch, log sales rise by the true effect of that store and week (column true_effect),
so every estimate can be compared with the truth.

Scenarios:
    pilot       25 urban stores launch in week 78 of 104; the effect ramps up to +5 % in about a
                month. Treated stores are urban, so they would have grown faster anyway:
                parallel trends hold only within urban stores.
    staggered   all 200 stores are randomised into launch waves: 40 in week 30, 40 in week 50,
                40 in week 70, and 80 not launched within the data. The effect builds up over half
                a year, to +8 %, +5 % and +3 % for the three waves (early adopters gain more).
    market      one region launches in week 80 and 30 other regions are possible controls.
                Region sales follow three latent factors; the treated region is more exposed to
                the growth factor than the average region, so plain DiD is biased.

Usage:
    python simulate.py --scenario pilot --output ../data/pilot.csv.gz [--seed 1]
    python simulate.py --all --truth ../data/truth.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WEEKS = 104
N_STORES = 200
N_REGIONS = 8
URBAN_GROWTH = 0.0006                 # extra log growth per week for urban stores (~3 % a year)
PILOT = {"n_treated": 25, "launch": 78, "effect": 0.05, "ramp_weeks": 4}
STAGGERED = {"waves": {30: 40, 50: 40, 70: 40}, "effect": {30: 0.08, 50: 0.05, 70: 0.03}, "ramp_weeks": 26}
MARKET = {"n_regions": 31, "launch": 80, "effect": 0.04, "ramp_weeks": 3}


def _noise(rng, shape, rho=0.5, sd=0.03):
    e = rng.normal(0, sd, shape)
    for t in range(1, shape[1]):
        e[:, t] += rho * e[:, t - 1]
    return e


def _season(t):
    return np.sin(2 * np.pi * t / 52) + 0.5 * np.sin(4 * np.pi * t / 52)


def store_baseline(n: int, rng: np.random.Generator) -> tuple[np.ndarray, pd.DataFrame]:
    """Log sales without the launch (n x WEEKS) and store attributes."""
    t = np.arange(WEEKS)
    region = rng.integers(0, N_REGIONS, n)
    urban = rng.random(n) < 0.4
    size = rng.normal(np.log(500), 0.35, n)                    # log weekly sales, thousand SEK
    chain = 0.0008 * t + np.cumsum(rng.normal(0, 0.004, WEEKS))
    region_shock = np.cumsum(rng.normal(0, 0.005, (N_REGIONS, WEEKS)), axis=1)
    loading = rng.normal(1, 0.25, n)
    amp = rng.normal(0.08, 0.02, n)
    urban_slope = urban * rng.uniform(0.7, 1.3, n) * URBAN_GROWTH
    y0 = (size[:, None] + chain[None, :] + amp[:, None] * _season(t)[None, :]
          + loading[:, None] * region_shock[region] + urban_slope[:, None] * t[None, :] + _noise(rng, (n, WEEKS)))
    attrs = pd.DataFrame({"store": np.arange(1, n + 1), "region": region + 1, "urban": urban.astype(int),
                          "size_index": np.round(np.exp(size - np.log(500)), 3)})
    return y0, attrs


def ramp(e: np.ndarray, weeks: float, kind: str) -> np.ndarray:
    """Share of the full effect reached e weeks after launch (0 before launch)."""
    e = np.asarray(e, float)
    if kind == "exp":
        r = 1 - np.exp(-(e + 1) / weeks)
    else:
        r = np.minimum((e + 1) / weeks, 1.0)
    return np.where(e >= 0, r, 0.0)


def _long(y, effect, launch, attrs, unit="store") -> pd.DataFrame:
    n = y.shape[0]
    df = pd.DataFrame({
        unit: np.repeat(attrs[unit].to_numpy(), WEEKS), "week": np.tile(np.arange(1, WEEKS + 1), n),
        "sales": np.round(np.exp(y), 2).ravel(), "true_effect": np.round(effect, 6).ravel(),
    })
    df = df.merge(attrs.assign(launch_week=launch), on=unit)
    cols = [unit, "week"] + [c for c in attrs.columns if c != unit] + ["launch_week", "sales", "true_effect"]
    return df[cols]


def simulate_pilot(seed: int = 1, n: int = N_STORES) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    y0, attrs = store_baseline(n, rng)
    p = PILOT
    treated = rng.choice(np.flatnonzero(attrs["urban"] == 1), p["n_treated"], replace=False)
    launch = np.full(n, np.nan)
    launch[treated] = p["launch"]
    e = np.arange(1, WEEKS + 1)[None, :] - launch[:, None]               # weeks since launch (NaN never)
    theta = rng.normal(1, 0.3, n)                                       # store-level effect multiplier
    effect = np.nan_to_num(p["effect"] * theta[:, None] * ramp(e, p["ramp_weeks"], "exp"))
    df = _long(y0 + effect, effect, launch, attrs)
    return df, _truth(df, "store")


def simulate_staggered(seed: int = 2, n: int = N_STORES) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    y0, attrs = store_baseline(n, rng)
    p = STAGGERED
    order = rng.permutation(n)
    launch = np.full(n, np.nan)
    start = 0
    for week, k in p["waves"].items():
        launch[order[start:start + k]] = week
        start += k
    e = np.arange(1, WEEKS + 1)[None, :] - launch[:, None]
    theta = rng.normal(1, 0.3, n)
    full = np.array([p["effect"].get(int(w), 0.0) if not np.isnan(w) else 0.0 for w in launch])
    effect = np.nan_to_num(full[:, None] * theta[:, None] * ramp(e, p["ramp_weeks"], "linear"))
    df = _long(y0 + effect, effect, launch, attrs)
    return df, _truth(df, "store")


def simulate_market(seed: int = 3) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    p = MARKET
    J = p["n_regions"]
    t = np.arange(WEEKS)
    f = np.vstack([0.002 * t + np.cumsum(rng.normal(0, 0.004, WEEKS)),       # growth
                   0.03 * np.sin(2 * np.pi * t / 26 + rng.uniform(0, 2 * np.pi)),  # promotion cycle
                   np.cumsum(rng.normal(0, 0.006, WEEKS))])                # local economy
    mu = rng.uniform(0, 2, (J, 3))
    mu[0] = [1.7, 1.2, 0.6]                                                # treated: high growth exposure
    level = rng.normal(np.log(20000), 0.3, J)
    amp = rng.normal(0.06, 0.015, J)
    y0 = level[:, None] + mu @ f + amp[:, None] * _season(t)[None, :] + _noise(rng, (J, WEEKS), sd=0.008)
    launch = np.full(J, np.nan)
    launch[0] = p["launch"]
    e = np.arange(1, WEEKS + 1)[None, :] - launch[:, None]
    effect = np.nan_to_num(p["effect"] * ramp(e, p["ramp_weeks"], "exp"))
    attrs = pd.DataFrame({"region": np.arange(1, J + 1)})
    df = _long(y0 + effect, effect, launch, attrs, unit="region")
    return df, _truth(df, "region")


def _truth(df: pd.DataFrame, unit: str) -> dict:
    post = df[df["week"] >= df["launch_week"]]
    by_e = post.assign(e=post["week"] - post["launch_week"]).groupby("e")["true_effect"].mean()
    out = {"att": float(post["true_effect"].mean()),
           "att_by_event_week": {int(k): float(v) for k, v in by_e.items()}}
    if df["launch_week"].nunique() > 1:
        out["att_by_wave"] = {int(k): float(v) for k, v in post.groupby("launch_week")["true_effect"].mean().items()}
    return out


SCENARIOS = {"pilot": simulate_pilot, "staggered": simulate_staggered, "market": simulate_market}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS))
    ap.add_argument("--all", action="store_true", help="write all three scenarios to ../data")
    ap.add_argument("--output")
    ap.add_argument("--truth")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    names = list(SCENARIOS) if a.all else [a.scenario]
    truth = {}
    for name in names:
        kw = {} if a.seed is None else {"seed": a.seed}
        df, tr = SCENARIOS[name](**kw)
        path = a.output if (a.output and not a.all) else f"../data/{name}.csv.gz"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        truth[name] = tr
        print(f"{name}: {df.iloc[:, 0].nunique()} units x {WEEKS} weeks -> {path}; true ATT {tr['att']:.4f} log points")
    if a.truth:
        with open(a.truth, "w") as f:
            json.dump(truth, f, indent=2)


if __name__ == "__main__":
    main()
