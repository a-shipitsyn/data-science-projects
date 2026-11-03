"""Expected amount paid within 3, 6 and 12 months: P(payment) * E[amount | payment], see model.py.

Evaluation with stratified cross-validation. Debtors are ranked by expected amount and by
payment probability alone; for both rankings the share of the total amount paid that comes
from the top 10 % / 20 % debtors is reported (amount_top10, amount_top20 vs proba_top10,
proba_top20). calibration = predicted total / actual total amount.

Usage:
    python recovery.py --input debtors.csv [--model gb|xgb] [--balance none|weight|down]
                       [--horizons chained|independent] [--output recovery.joblib]
"""
import argparse

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from model import AMOUNTS, TARGETS, RecoveryModel
from schema import read_validated
from train import attach_metadata, capture, summarize


def cross_validate(X: pd.DataFrame, Y: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                   **model_args) -> pd.DataFrame:
    rows = []
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, Y[TARGETS[-1]])
    for fold, (train, test) in enumerate(folds):
        model = RecoveryModel(**model_args).fit(X.iloc[train], Y.iloc[train])
        expected = model.predict(X.iloc[test])
        proba = model.payment.predict_proba(X.iloc[test])
        for target, amount in zip(TARGETS, AMOUNTS):
            y = Y[amount].iloc[test]
            rows.append({
                "fold": fold, "target": amount,
                "amount_top10": capture(y, expected[amount], 0.1),
                "proba_top10": capture(y, proba[target], 0.1),
                "amount_top20": capture(y, expected[amount], 0.2),
                "proba_top20": capture(y, proba[target], 0.2),
                "calibration": expected[amount].sum() / y.sum(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expected recovery models with cross-validation")
    ap.add_argument("--input", required=True, help="CSV with input columns, paid_next_* and amount_next_* targets")
    ap.add_argument("--model", choices=["gb", "xgb"], default="gb", help="gradient boosting implementation")
    ap.add_argument("--balance", choices=["none", "weight", "down"], default="none")
    ap.add_argument("--horizons", choices=["chained", "independent"], default="chained")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", help="save the model refitted on all data (joblib)")
    args = ap.parse_args()

    df = read_validated(args.input, targets=True, amounts=True)
    X, Y = df.drop(columns=TARGETS + AMOUNTS), df[TARGETS + AMOUNTS]
    model_args = {"model": args.model, "balance": args.balance, "horizons": args.horizons}

    scores = cross_validate(X, Y, n_splits=args.folds, **model_args)
    print(f"{model_args}, {args.folds}-fold stratified CV (± 2 std):")
    print(summarize(scores).to_string())

    if args.output:
        model = RecoveryModel(**model_args).fit(X, Y)
        joblib.dump(attach_metadata(model, X, model_args, scores), args.output)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
