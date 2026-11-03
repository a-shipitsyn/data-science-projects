"""Permutation importance of each input column on the out-of-time holdout.

The model is trained on the development period. Each raw input column (with every
feature derived from it) is shuffled in the holdout, and the loss of quality is the
importance: increase in log-loss (threshold-free, proper) and drop in PR-AUC.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

import schema
from evaluate import load, time_split
from models import MODELS, CalibratedClassifier, build

# raw columns shuffled together: the card expiry is one input
GROUPS = {c: [c] for c in schema.INPUTS if c not in schema.CARD}
GROUPS['card_expiry'] = schema.CARD


def permutation_importance(model: CalibratedClassifier, df: pd.DataFrame, repeats: int = 5,
                           seed: int = 0) -> pd.DataFrame:
    y = df[schema.TARGET].to_numpy()
    p = model.predict_proba(df)[:, 1]
    base_ll, base_pr = log_loss(y, p), average_precision_score(y, p)
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in GROUPS.items():
        ll, pr = [], []
        for _ in range(repeats):
            shuffled = df.copy()
            order = rng.permutation(len(df))
            shuffled[cols] = df[cols].to_numpy()[order]
            q = model.predict_proba(shuffled)[:, 1]
            ll.append(log_loss(y, q) - base_ll)
            pr.append(base_pr - average_precision_score(y, q))
        rows.append({'input': name, 'log_loss_increase': np.mean(ll),
                     'log_loss_increase_2sd': 2 * np.std(ll, ddof=1),
                     'pr_auc_drop': np.mean(pr), 'pr_auc_drop_2sd': 2 * np.std(pr, ddof=1)})
    return pd.DataFrame(rows).sort_values('log_loss_increase', ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='../data/mock_loans.csv')
    parser.add_argument('--output', default='../results/importance.csv')
    parser.add_argument('--model', default='hgb', choices=MODELS)
    parser.add_argument('--holdout-months', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dev, test = time_split(load(args.input), args.holdout_months)
    model = build(args.model, args.seed).fit(dev, dev[schema.TARGET])
    out = permutation_importance(model, test, args.repeats, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
