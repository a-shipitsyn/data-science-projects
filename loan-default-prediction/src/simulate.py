"""Simulate a table of short-term point-of-sale loans with a known default mechanism.

Each loan is due after 14 days, followed by a 7-day reminder period. The default
probability is a logistic function of the borrower's history, debt, new credit, an
expiring card, the merchant and the issue month, plus an unobserved reliability that
shows only through past payments. True parameters are written to truth.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {
    'electronics': ['phones', 'computers', 'audio'],
    'fashion': ['clothing', 'shoes', 'accessories'],
    'home': ['furniture', 'kitchen', 'garden'],
    'leisure': ['sports', 'travel', 'gaming'],
}
# merchant effects on the default log-odds
GROUP_EFFECT = {'electronics': 0.25, 'fashion': -0.10, 'home': -0.25, 'leisure': 0.10}
CATEGORY_EFFECT = {
    'phones': 0.20, 'computers': -0.10, 'audio': 0.0,
    'clothing': 0.10, 'shoes': -0.15, 'accessories': 0.20,
    'furniture': -0.20, 'kitchen': 0.0, 'garden': 0.10,
    'sports': -0.10, 'travel': 0.30, 'gaming': 0.25,
}
COEF = {
    'reliability': -0.70,                 # unobserved
    'log1p_failed_6m': 0.80,
    'log1p_debt_to_repaid': 0.50,         # existing debt / (repaid in 1 year + 100)
    'log1p_history_days': -0.22,
    'new_credit_to_amount_14d': 0.40,
    'card_expires_before_due': 1.20,
    'log_amount': 0.20,
    'month_index': 0.03,                  # drift: risk rises over the period
}
START, MONTHS = '2024-01-01', 18


def simulate(n: int = 40_000, seed: int = 0,
             target_rate: float = 0.08) -> tuple[pd.DataFrame, np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    days = pd.date_range(START, periods=MONTHS, freq='MS')
    end = days[-1] + pd.offsets.MonthEnd(0)
    span = (end - days[0]).days + 1
    # volume grows over the period
    offset = np.floor(span * rng.beta(1.5, 1, n)).astype(int)
    issue = days[0] + pd.to_timedelta(np.sort(offset), unit='D')
    month_index = np.asarray((issue.year - days[0].year) * 12 + issue.month - days[0].month)

    # borrower: new customers have no history
    new = rng.random(n) < 0.12
    history = np.where(new, 0, np.round(np.exp(rng.uniform(np.log(30), np.log(2500), n))))
    reliability = rng.normal(size=n)
    rate = rng.lognormal(np.log(0.9), 0.7, n)               # purchases per month
    ticket = rng.lognormal(np.log(60), 0.7, n)              # typical purchase amount
    fail_prob = 1 / (1 + np.exp(-(-2.8 - 0.9 * reliability)))

    def window(days_: float) -> np.ndarray:
        # expected purchases in a past window, limited by the length of the history
        return rate * np.minimum(history, days_) / 30

    def increments(edges: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Purchases and failed payments in consecutive windows, then cumulative sums."""
        paid, failed, prev = [], [], 0
        for d in edges:
            k = rng.poisson(window(d) - window(prev))
            f = rng.binomial(k, fail_prob)
            paid.append(k - f)
            failed.append(f)
            prev = d
        return np.cumsum(paid, axis=0), np.cumsum(failed, axis=0)

    edges = [14, 30, 91, 182, 365]
    paid, failed = increments(edges)
    repaid = np.cumsum([
        np.round(p * ticket * rng.lognormal(0, 0.3, n)) for p in np.diff(paid, axis=0, prepend=0)
    ], axis=0)

    active = rng.poisson(0.4 * rate + 0.6 * (fail_prob > 0.15))
    debt = np.round(active * ticket * rng.lognormal(0, 0.5, n))
    new_7 = np.where(rng.random(n) < 0.15, np.round(ticket * rng.lognormal(0, 0.6, n)), 0)
    new_14 = new_7 + np.where(rng.random(n) < 0.10, np.round(ticket * rng.lognormal(0, 0.6, n)), 0)
    amount = np.round(ticket * rng.lognormal(0.2, 0.6, n)) + 5

    # card expiry: first day of the month; a few cards expire before the due date
    months_left = np.where(rng.random(n) < 0.03, 0, rng.integers(1, 60, n))
    expiry = (issue.to_period('M') + months_left).to_timestamp()
    expires_before_due = np.asarray(expiry < issue + pd.Timedelta(days=14)).astype(int)

    groups = list(GROUPS)
    group = rng.choice(groups, n, p=[0.30, 0.35, 0.20, 0.15])
    category = np.array([rng.choice(GROUPS[g]) for g in group])

    x = {
        'reliability': reliability,
        'log1p_failed_6m': np.log1p(failed[3]),
        'log1p_debt_to_repaid': np.log1p(debt / (repaid[4] + 100)),
        'log1p_history_days': np.log1p(history),
        'new_credit_to_amount_14d': np.minimum(new_14 / amount, 5),
        'card_expires_before_due': expires_before_due,
        'log_amount': np.log(amount),
        'month_index': month_index,
    }
    eta = sum(COEF[k] * x[k] for k in COEF)
    eta = eta + np.vectorize(GROUP_EFFECT.get)(group) + np.vectorize(CATEGORY_EFFECT.get)(category)
    intercept = _intercept(eta, target_rate)
    p = 1 / (1 + np.exp(-(intercept + eta)))
    default = rng.random(n) < p

    # outstanding balances: defaulters owe part of the loan after the reminder period;
    # some payers are late but pay during the reminder period
    late = default | (rng.random(n) < 0.10)
    outstanding_due = np.where(late, np.round(amount * rng.uniform(0.3, 1, n)), 0)
    outstanding_after = np.where(default, np.round(outstanding_due * rng.uniform(0.4, 1, n)), 0)

    df = pd.DataFrame({
        'issue_date': issue.strftime('%Y-%m-%d'),
        'amount': amount.astype(int),
        'card_exp_month': expiry.month.astype(float),
        'card_exp_year': expiry.year.astype(float),
        'existing_debt': debt,
        'active_loans': active,
        'history_days': history.astype(int),
        'new_credit_7d': new_7.astype(int),
        'new_credit_14d': new_14.astype(int),
        'paid_on_time_3m': paid[2],
        'paid_on_time_6m': paid[3],
        'failed_payments_3m': failed[2],
        'failed_payments_6m': failed[3],
        'failed_payments_1y': failed[4],
        'repaid_14d': repaid[0].astype(int),
        'repaid_1m': repaid[1].astype(int),
        'repaid_3m': repaid[2].astype(int),
        'repaid_6m': repaid[3].astype(int),
        'repaid_1y': repaid[4].astype(int),
        'merchant_group': group,
        'merchant_category': category,
        'outstanding_due': outstanding_due.astype(int),
        'outstanding_after_grace': outstanding_after.astype(int),
    })
    # missing values: debt more often unknown for new customers, card expiry rarely
    miss_debt = rng.random(n) < np.where(new, 0.20, 0.03)
    df.loc[miss_debt, 'existing_debt'] = np.nan
    miss_card = rng.random(n) < 0.02
    df.loc[miss_card, ['card_exp_month', 'card_exp_year']] = np.nan

    truth = {
        'seed': seed,
        'n': n,
        'start': START,
        'months': MONTHS,
        'intercept': round(float(intercept), 4),
        'coefficients': COEF,
        'group_effect': GROUP_EFFECT,
        'category_effect': CATEGORY_EFFECT,
        'default_rate': round(float(default.mean()), 4),
        'mean_true_probability': round(float(p.mean()), 4),
    }
    return df, p, truth


def _intercept(eta: np.ndarray, rate: float) -> float:
    """Intercept giving the target mean default probability (bisection)."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(mid + eta)))) > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='../data/mock_loans.csv')
    parser.add_argument('--truth', default='../data/truth.json')
    parser.add_argument('--n', type=int, default=40_000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    df, p, truth = simulate(args.n, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    # true probabilities, for the oracle row of the evaluation
    pd.Series(p.round(6), name='true_probability').to_csv(
        Path(args.output).with_name('true_probability.csv'), index=False)
    Path(args.truth).write_text(json.dumps(truth, indent=2) + '\n')
    print(f'{len(df)} loans, default rate {df.outstanding_after_grace.gt(0).mean():.3f} '
          f'-> {args.output}')


if __name__ == '__main__':
    main()
