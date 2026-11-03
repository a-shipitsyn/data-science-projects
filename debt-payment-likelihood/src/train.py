"""Payment likelihood within 3, 6 and 12 months, see model.py.

Evaluation with stratified cross-validation (stratified by the 12-month target). Metrics:
    roc_auc, pr_auc  ranking quality; PR-AUC is the informative one for a rare positive class
    brier            mean squared error of the predicted probability
    calibration      mean predicted probability / observed payment rate, 1 = calibrated
    top10, top20     share of all payers among the 10 % / 20 % debtors with the highest
                     probability: what a collection team with limited capacity would reach
    inconsistent     share of debtors with P(3m) > P(6m) or P(6m) > P(12m)
With --output the model is refitted on all data and saved with joblib, together with
metadata: training time, settings, library versions, CV scores and the reference profile
of the inputs used by monitor.py.

Usage:
    python train.py --input debtors.csv [--model gb|xgb] [--balance none|weight|down]
                    [--horizons chained|independent] [--output model.joblib]
"""
import argparse
from datetime import UTC, datetime
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model import TARGETS, PaymentModel, RecoveryModel
from monitor import profile
from schema import read_validated


def capture(y: pd.Series, score: pd.Series, top: float) -> float:
    """Share of all positives among the `top` fraction of rows with the highest score."""
    n = round(len(y) * top)
    return np.asarray(y)[np.argsort(-np.asarray(score))[:n]].sum() / np.sum(y)


def score(Y: pd.DataFrame, proba: pd.DataFrame) -> pd.DataFrame:
    """Metrics per target for predicted probabilities (DataFrame with one column per target)."""
    inconsistent = (proba.diff(axis=1).iloc[:, 1:] < -1e-12).any(axis=1).mean()
    rows = []
    for target in TARGETS:
        y, p = Y[target], proba[target]
        rows.append({
            "target": target,
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "calibration": p.mean() / y.mean(),
            "top10": capture(y, p, 0.1),
            "top20": capture(y, p, 0.2),
            "inconsistent": inconsistent,
        })
    return pd.DataFrame(rows)


def cross_validate(X: pd.DataFrame, Y: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                   **model_args) -> pd.DataFrame:
    scores = []
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, Y[TARGETS[-1]])
    for fold, (train, test) in enumerate(folds):
        proba = PaymentModel(**model_args).fit(X.iloc[train], Y.iloc[train]).predict_proba(X.iloc[test])
        scores.append(score(Y.iloc[test], proba).assign(fold=fold))
    return pd.concat(scores, ignore_index=True)


def attach_metadata(model: PaymentModel | RecoveryModel, X: pd.DataFrame, model_args: dict,
                    scores: pd.DataFrame) -> PaymentModel | RecoveryModel:
    """Store what is needed to trace and monitor a saved model."""
    model.metadata_ = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_args": model_args,
        "versions": {pkg: version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "xgboost"]},
        "cv_scores": scores.drop(columns="fold").groupby("target", sort=False).mean().to_dict("index"),
        "profile": profile(X),
        "mean_proba": getattr(model, "payment", model).predict_proba(X).mean().to_dict(),
    }
    return model


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """mean ± 2 std over folds for each target and metric."""
    metrics = scores.drop(columns="fold").groupby("target", sort=False)
    return metrics.mean().round(3).astype(str) + " ± " + (2 * metrics.std()).round(3).astype(str)


def main() -> None:
    ap = argparse.ArgumentParser(description="Payment likelihood models with cross-validation")
    ap.add_argument("--input", required=True, help="CSV with input columns and paid_next_* targets")
    ap.add_argument("--model", choices=["gb", "xgb"], default="gb", help="gradient boosting implementation")
    ap.add_argument("--balance", choices=["none", "weight", "down"], default="none",
                    help="re-balancing of the classes in training")
    ap.add_argument("--horizons", choices=["chained", "independent"], default="chained")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", help="save the model refitted on all data (joblib)")
    args = ap.parse_args()

    df = read_validated(args.input, targets=True)
    X, Y = df.drop(columns=TARGETS), df[TARGETS]
    model_args = {"model": args.model, "balance": args.balance, "horizons": args.horizons}

    scores = cross_validate(X, Y, n_splits=args.folds, **model_args)
    print(f"{model_args}, {args.folds}-fold stratified CV (± 2 std):")
    print(summarize(scores).to_string())

    if args.output:
        model = PaymentModel(**model_args).fit(X, Y)
        joblib.dump(attach_metadata(model, X, model_args, scores), args.output)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
