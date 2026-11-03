"""Out-of-time validation: train on one snapshot, evaluate on a later one.

Cross-validation on a single snapshot shows how the model generalises to other debtors;
this shows how it holds up when payment behaviour changes over time, which is what
matters before deployment. Metrics are the same as in train.py.

Usage:
    python validate_time.py --train snapshot_2023.csv --test snapshot_2024.csv [--model gb|xgb]
"""
import argparse

from model import TARGETS, PaymentModel
from schema import read_validated
from train import score


def main() -> None:
    ap = argparse.ArgumentParser(description="Train on an earlier snapshot, test on a later one")
    ap.add_argument("--train", required=True, help="earlier snapshot CSV with input columns and targets")
    ap.add_argument("--test", required=True, help="later snapshot CSV with input columns and targets")
    ap.add_argument("--model", choices=["gb", "xgb"], default="gb")
    ap.add_argument("--balance", choices=["none", "weight", "down"], default="none")
    ap.add_argument("--horizons", choices=["chained", "independent"], default="chained")
    args = ap.parse_args()

    train = read_validated(args.train, targets=True)
    test = read_validated(args.test, targets=True)
    model = PaymentModel(model=args.model, balance=args.balance, horizons=args.horizons)
    model.fit(train.drop(columns=TARGETS), train[TARGETS])
    proba = model.predict_proba(test.drop(columns=TARGETS))
    print("Out-of-time scores on the later snapshot:")
    print(score(test[TARGETS], proba).set_index("target").round(3).to_string())


if __name__ == "__main__":
    main()
