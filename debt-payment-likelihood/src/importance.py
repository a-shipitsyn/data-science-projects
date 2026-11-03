"""Permutation importance of the input columns for the payment likelihood model.

Each input column is shuffled in a held-out test set, so the derived flags, wealth and
ratio features change with it; importance = mean drop of ROC-AUC per horizon.

Usage:
    python importance.py --input debtors.csv [--model gb|xgb] [--repeats 10]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import NUMERIC
from model import TARGETS, PaymentModel
from schema import read_validated


def permutation_importance(model: PaymentModel, X: pd.DataFrame, Y: pd.DataFrame, n_repeats: int = 10,
                           seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = model.predict_proba(X)
    base = {t: roc_auc_score(Y[t], base[t]) for t in TARGETS}
    rows = []
    for col in NUMERIC:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            proba = model.predict_proba(Xp)
            drops.append([base[t] - roc_auc_score(Y[t], proba[t]) for t in TARGETS])
        rows.append([col, *np.mean(drops, axis=0)])
    return pd.DataFrame(rows, columns=["column", *TARGETS]).sort_values(TARGETS[-1], ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Permutation importance of input columns")
    ap.add_argument("--input", required=True, help="CSV with input columns and paid_next_* targets")
    ap.add_argument("--model", choices=["gb", "xgb"], default="gb")
    ap.add_argument("--balance", choices=["none", "weight", "down"], default="none")
    ap.add_argument("--horizons", choices=["chained", "independent"], default="chained")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    df = read_validated(args.input, targets=True)
    X, Y = df.drop(columns=TARGETS), df[TARGETS]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y[TARGETS[-1]],
                                                        random_state=42)
    model = PaymentModel(model=args.model, balance=args.balance, horizons=args.horizons).fit(X_train, Y_train)
    print("Drop of ROC-AUC when the column is shuffled (30 % hold-out):")
    print(permutation_importance(model, X_test, Y_test, args.repeats).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
