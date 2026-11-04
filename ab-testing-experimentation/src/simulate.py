"""Mock online experiment with a known ground truth.

An e-commerce app tests a new checkout flow. Users are randomised 50/50 on their first visit
during a 14-day experiment and followed for 14 days after it.

Each user has
    device       mobile / desktop
    country      SE, NO, DK, FI (order value differs by country)
    tenure_days  days since sign-up; about a quarter are new users (< 30 days)
    activity     latent shopping intensity, Gamma distributed (not observed)
    aov          the user's typical order value, log-normal (not observed)

Orders in a 14-day window ~ Poisson(mu), log mu = intercept + log(activity) + device + tenure.
Pre-period (8 weeks before the experiment): orders ~ Poisson(4 mu) — the CUPED covariate.
Sessions ~ Poisson, driven by the same activity; the treatment does not change them.
Revenue = orders x aov x noise.

The treatment multiplies mu by exp(delta), with delta depending on the user:
    +0.10 on mobile, +0.12 for new users, -0.35 for desktop users with more than two years of tenure.
Because mu and aov are known, the true per-user effects are exact:
    revenue        (exp(delta) - 1) * mu * aov
    conversion     exp(-mu) - exp(-mu * exp(delta))
and are stored with the data (columns true_*), so every estimate can be checked against them.
Population values (from expected values over 2 million simulated users) are in truth.json.

With --srm-bug a logging bug is added: treatment users on mobile who do not buy are lost
from the data with probability 0.05 (e.g. the new checkout page crashes before the exposure
event is logged) — a sample ratio mismatch that biases the naive effect upwards.

Usage:
    python simulate.py --output ../data/experiment.csv.gz --truth ../data/truth.json [--n 200000] [--seed 11]
    python simulate.py --output ../data/experiment_srm.csv.gz --srm-bug
"""
import argparse
import json

import numpy as np
import pandas as pd

COUNTRIES = np.array(["SE", "NO", "DK", "FI"])
COUNTRY_P = [0.45, 0.20, 0.20, 0.15]
COUNTRY_AOV = np.array([1.0, 1.25, 1.1, 0.9])   # order value multiplier, same order as COUNTRIES
DAYS = 14
TRUTH_PARAMS = {
    "intercept": -2.3, "desktop": 0.25, "tenure_1y": 0.30, "new_user": -0.40,
    "activity_shape": 0.3, "aov_log_mean": 6.0, "aov_log_sd": 0.45, "revenue_noise_sd": 0.30,
    "effect": {"mobile": 0.10, "new_user": 0.12, "desktop_tenure_2y": -0.35},
    "srm_bug_drop": 0.05,
}


def segment(device, new_user, tenure_days) -> np.ndarray:
    """Device x tenure segment label, e.g. 'mobile / new' or 'desktop / >2y'."""
    tenure = np.where(np.asarray(new_user) == 1, "new", np.where(np.asarray(tenure_days) > 730, ">2y", "30d-2y"))
    return np.char.add(np.char.add(np.asarray(device, dtype=str), " / "), tenure)


def draw_users(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Users with covariates, latent rates and true per-user effects (no outcomes yet)."""
    p = TRUTH_PARAMS
    desktop = rng.random(n) >= 0.6
    c = rng.choice(len(COUNTRIES), n, p=COUNTRY_P)
    new = rng.random(n) < 0.25
    tenure = np.where(new, rng.integers(0, 30, n), 30 + rng.exponential(500, n)).astype(int)
    activity = rng.gamma(p["activity_shape"], 1 / p["activity_shape"], n)
    mu = np.exp(p["intercept"] + np.log(activity) + p["desktop"] * desktop
                + p["tenure_1y"] * (tenure > 365) + p["new_user"] * new)
    aov = np.exp(rng.normal(p["aov_log_mean"], p["aov_log_sd"], n)) * COUNTRY_AOV[c]
    e = p["effect"]
    delta = e["mobile"] * ~desktop + e["new_user"] * new + e["desktop_tenure_2y"] * (desktop & (tenure > 730))
    return pd.DataFrame({
        "device": np.where(desktop, "desktop", "mobile"), "country": COUNTRIES[c],
        "tenure_days": tenure, "new_user": new.astype(int),
        "mu": mu, "aov": aov, "delta": delta, "activity": activity,
        "true_revenue_effect": (np.exp(delta) - 1) * mu * aov,
        "true_conversion_effect": np.exp(-mu) - np.exp(-mu * np.exp(delta)),
    })


def draw_outcomes(u: pd.DataFrame, treat: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Pre-period and experiment-period outcomes for users `u` with assignment `treat` (0/1)."""
    sd = TRUTH_PARAMS["revenue_noise_sd"]
    n = len(u)
    mu, aov = u["mu"].to_numpy(), u["aov"].to_numpy()
    noise = lambda: np.exp(rng.normal(-sd ** 2 / 2, sd, n))   # mean-one order value noise
    pre_orders = rng.poisson(4 * mu)
    orders = rng.poisson(mu * np.exp(u["delta"].to_numpy() * treat))
    sess_rate = 1.5 + 4 * u["activity"].to_numpy()
    return pd.DataFrame({
        "pre_sessions": rng.poisson(4 * sess_rate),
        "pre_revenue": (pre_orders * aov * noise()).round(2),
        "sessions": rng.poisson(sess_rate) + 1,                 # at least the exposure session
        "orders": orders,
        "converted": (orders > 0).astype(int),
        "revenue": (orders * aov * noise()).round(2),
    })


def draw_experiment(n: int, rng: np.random.Generator, treat: np.ndarray | None = None) -> pd.DataFrame:
    """Users, 50/50 assignment, outcomes and day of first exposure, with the latent columns kept."""
    u = draw_users(n, rng)
    treat = rng.integers(0, 2, n) if treat is None else treat
    df = pd.concat([u, draw_outcomes(u, treat, rng)], axis=1)
    df["treat"] = treat
    df["day"] = rng.integers(1, DAYS + 1, n)
    return df


def population_truth(n: int = 2_000_000, seed: int = 0) -> dict:
    """Population average effects and control means, from expected values of a large draw."""
    u = draw_users(n, np.random.default_rng(seed))
    seg = segment(u["device"], u["new_user"], u["tenure_days"])
    return {
        "ate_revenue": float(u["true_revenue_effect"].mean()),
        "ate_conversion": float(u["true_conversion_effect"].mean()),
        "control_revenue_mean": float((u["mu"] * u["aov"]).mean()),
        "control_conversion_mean": float((1 - np.exp(-u["mu"])).mean()),
        "sessions_mean": 1 + 1.5 + 4 * float(u["activity"].mean()),
        "segment_share": {k: float(v) for k, v in pd.Series(seg).value_counts(normalize=True).sort_index().items()},
        "segment_revenue_effect": {k: float(v) for k, v in u["true_revenue_effect"].groupby(seg).mean().sort_index().items()},
    }


HIDDEN = ["mu", "aov", "delta", "activity", "treat"]


def simulate(n: int = 200_000, seed: int = 11, srm_bug: bool = False) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    df = draw_experiment(n, rng)
    treat = df["treat"].to_numpy()
    df.insert(0, "user_id", np.arange(1, n + 1))
    df.insert(1, "group", np.where(treat == 1, "treatment", "control"))
    df.insert(2, "day", df.pop("day"))
    df.insert(6, "tenure_group", np.where(df["new_user"] == 1, "new", np.where(df["tenure_days"] > 730, ">2y", "30d-2y")))
    if srm_bug:
        lost = (treat == 1) & (df["device"] == "mobile") & (df["orders"] == 0) & (rng.random(n) < TRUTH_PARAMS["srm_bug_drop"])
        df = df[~lost].reset_index(drop=True)
    pop = population_truth()
    seg = segment(df["device"], df["new_user"], df["tenure_days"])
    truth = {
        "params": TRUTH_PARAMS, "n_users": int(n), "seed": seed, "srm_bug": srm_bug,
        "sample_ate_revenue": float(df["true_revenue_effect"].mean()),
        "sample_ate_conversion": float(df["true_conversion_effect"].mean()),
        "sample_segment_revenue_effect": {k: float(v) for k, v in df["true_revenue_effect"].groupby(seg).mean().sort_index().items()},
        "population": pop,
        "ate_revenue_relative": pop["ate_revenue"] / pop["control_revenue_mean"],
        "ate_conversion_relative": pop["ate_conversion"] / pop["control_conversion_mean"],
    }
    return df.drop(columns=HIDDEN), truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="../data/experiment.csv.gz")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--srm-bug", action="store_true")
    a = ap.parse_args()
    df, truth = simulate(a.n, a.seed, a.srm_bug)
    df.to_csv(a.output, index=False, float_format="%.6g")
    if a.truth:
        with open(a.truth, "w") as f:
            json.dump(truth, f, indent=2)
    pop = truth["population"]
    print(f"{len(df)} users -> {a.output}; true revenue effect {pop['ate_revenue']:.2f} per user "
          f"({truth['ate_revenue_relative']:.1%}), conversion {pop['ate_conversion']:+.4f} ({truth['ate_conversion_relative']:.1%})")


if __name__ == "__main__":
    main()
