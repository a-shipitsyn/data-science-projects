"""Simulated debtor snapshot with a known payment mechanism.

Debtors (amounts in arbitrary currency units):
    age            truncated normal, 18-85
    income         log-normal, rises to a peak around 50 years; missing for 25 % of the
                   unemployed and 5 % of the employed, so missingness carries signal
    principal      log-normal; interest accrues with the case age, balance adds fees
    external_debt  zero for a third of debtors, otherwise log-normal, grows with income
    past payments  driven by an unobserved willingness to pay

Future payments: a monthly hazard of the first payment,
    logit = base + w_willingness * willingness + w_income * log(income / median)
            + w_unemployed * unemployed + w_debt_burden * log1p(external_debt / income)
            + w_case_age * case_age_years + w_age * ((age - 45) / 10) ** 2
summed over the 3 / 3 / 6 months of the horizons. Willingness is not in the data and is
seen only through past payments, so no model can reach the true probabilities. After the
first payment the debtor keeps paying in each later interval with probability 0.7;
amounts are gamma distributed around a share of the balance and of the income.

--snapshot later shifts the population a year on (older cases, 5 % higher incomes,
lower base hazard), to try out validate_time.py and monitor.py.
The true parameters, payment rates and the ROC-AUC of the true probabilities (an upper
bound for any model) are written to a JSON file next to the data.

Usage:
    python simulate.py --output ../data/mock_debtors.csv [--n 8000] [--seed 0]
                       [--snapshot earlier|later] [--truth ../data/truth.json]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

INTERVALS = {"3m": 3, "6m": 3, "12m": 6}
WEIGHTS = {"willingness": 1.2, "income": 0.5, "unemployed": -0.7, "debt_burden": -0.6,
           "case_age": -0.15, "age": -0.12}
BASE = {"earlier": -3.8, "later": -4.0}
MEDIAN_INCOME = 250_000


def logistic(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def make_debtors(n: int = 8000, seed: int = 0, snapshot: str = "earlier") -> tuple[pd.DataFrame, dict]:
    """Debtor table with inputs and targets, and the truth needed to check models."""
    rng = np.random.default_rng(seed)
    later = snapshot == "later"

    age = np.clip(rng.normal(42, 13, n), 18, 85).round()
    unemployed = rng.random(n) < 0.18
    log_income = (np.log(MEDIAN_INCOME) - 0.0012 * (age - 50) ** 2 + rng.normal(0, 0.55, n)
                  - 1.2 * unemployed + 0.05 * later)
    income = np.exp(log_income)
    income_missing = rng.random(n) < np.where(unemployed, 0.25, 0.05)

    case_age = rng.gamma(2, 14, n) + 12 * later
    principal = np.exp(rng.normal(np.log(9000), 1.0, n))
    rate = rng.choice([8, 12, 18, 24, 29], n, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    interest = principal * rate / 100 * case_age / 12 * rng.uniform(0.6, 1.0, n)
    fees = rng.gamma(1.5, 300, n)
    has_external = rng.random(n) < 0.67
    external_debt = np.where(has_external, np.exp(rng.normal(np.log(0.4 * income), 1.0)), 0)

    willingness = rng.normal(0, 1, n)
    past_payer = rng.random(n) < logistic(-1.3 + 1.1 * willingness - 0.6 * unemployed)
    past_12m = np.where(past_payer, rng.gamma(1.5, 0.03 * principal + 400), 0)
    recent = rng.uniform(0, 1, n)
    past_6m = past_12m * np.where(rng.random(n) < 0.6, recent ** 0.5, 0)
    past_3m = past_6m * np.where(rng.random(n) < 0.6, recent ** 0.5, 0)

    logit = (BASE[snapshot]
             + WEIGHTS["willingness"] * willingness
             + WEIGHTS["income"] * (log_income - np.log(MEDIAN_INCOME))
             + WEIGHTS["unemployed"] * unemployed
             + WEIGHTS["debt_burden"] * np.log1p(external_debt / income)
             + WEIGHTS["case_age"] * case_age / 12
             + WEIGHTS["age"] * ((age - 45) / 10) ** 2)
    monthly = logistic(logit)

    df = pd.DataFrame({
        "id": np.arange(n),
        "case_age_months": case_age.round(),
        "interest_rate": rate,
        "principal": principal.round(),
        "principal_interest": (principal + interest).round(),
        "balance": (principal + interest + fees).round(),
        "age": age,
        "income": np.where(income_missing, np.nan, income.round(-2)),
        "external_debt": external_debt.round(-2),
        "paid_amount_past_3m": past_3m.round(),
        "paid_amount_past_6m": past_6m.round(),
        "paid_amount_past_12m": past_12m.round(),
    })

    balance = df["balance"].to_numpy()
    paid, amount, not_paid_yet, true_proba = np.zeros(n, bool), np.zeros(n), np.ones(n), {}
    for h, months in INTERVALS.items():
        first = ~paid & (rng.random(n) < 1 - (1 - monthly) ** months)
        again = paid & (rng.random(n) < 0.7)
        size = 0.04 * balance * months + 0.003 * income * months
        amount += np.where(first | again, rng.gamma(2, size / 2), 0)
        amount = np.minimum(amount, balance)
        paid |= first
        not_paid_yet *= (1 - monthly) ** months
        true_proba[h] = 1 - not_paid_yet
        df[f"paid_next_{h}"] = paid.astype(int)
        df[f"amount_next_{h}"] = amount.round()

    truth = {
        "snapshot": snapshot, "n": n, "seed": seed,
        "base_logit": BASE[snapshot], "weights": WEIGHTS,
        "payment_rate": {h: round(df[f"paid_next_{h}"].mean(), 4) for h in INTERVALS},
        "true_proba_roc_auc": {h: round(roc_auc_score(df[f"paid_next_{h}"], true_proba[h]), 3)
                               for h in INTERVALS},
    }
    return df, truth


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulated debtor snapshot with a known mechanism")
    ap.add_argument("--output", required=True, help="CSV to write")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", choices=["earlier", "later"], default="earlier")
    ap.add_argument("--truth", help="JSON with the true parameters")
    args = ap.parse_args()

    df, truth = make_debtors(args.n, args.seed, args.snapshot)
    df.to_csv(args.output, index=False)
    print(f"{args.output}: {len(df)} debtors, payment rate {truth['payment_rate']}")
    print(f"ROC-AUC of the true probabilities: {truth['true_proba_roc_auc']}")
    if args.truth:
        with open(args.truth, "w") as f:
            json.dump(truth, f, indent=2)


if __name__ == "__main__":
    main()
