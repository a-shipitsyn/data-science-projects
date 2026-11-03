"""Batch scoring of a debtor snapshot with a saved model.

Output, one row per debtor, sorted by priority (highest first):
    <id>, paid_next_3m, paid_next_6m, paid_next_12m   probability of a payment within the horizon
    amount_next_3m, amount_next_6m, amount_next_12m   expected amount (recovery.py models only)
    rank                                             1 = contact first; by the 12-month expected
                                                     amount if available, otherwise by probability

Usage:
    python predict.py --model model.joblib --input snapshot.csv --output scores.csv [--id-column id]
"""
import argparse

import joblib
import pandas as pd

from model import AMOUNTS, TARGETS, PaymentModel, RecoveryModel
from schema import read_validated


def score(model: PaymentModel | RecoveryModel, X: pd.DataFrame) -> pd.DataFrame:
    out = model.payment.predict_proba(X) if hasattr(model, "payment") else model.predict_proba(X)
    if hasattr(model, "regressors_"):
        out = out.join(model.predict(X))
    priority = out[AMOUNTS[-1]] if AMOUNTS[-1] in out else out[TARGETS[-1]]
    out["rank"] = priority.rank(ascending=False, method="first").astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a debtor snapshot")
    ap.add_argument("--model", required=True, help="model saved by train.py or recovery.py")
    ap.add_argument("--input", required=True, help="snapshot CSV with input columns")
    ap.add_argument("--output", required=True, help="scores CSV")
    ap.add_argument("--id-column", default="id", help="debtor id column, copied to the output if present")
    args = ap.parse_args()

    model = joblib.load(args.model)
    X = read_validated(args.input)
    scores = score(model, X)
    if args.id_column in X.columns:
        scores.insert(0, args.id_column, X[args.id_column])
    scores.sort_values("rank").to_csv(args.output, index=False)
    print(f"scored {len(scores)} debtors with a model trained at {model.metadata_['trained_at']}: {args.output}")


if __name__ == "__main__":
    main()
