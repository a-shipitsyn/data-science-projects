"""Data drift check: compare a new snapshot with the training data.

The reference profile (decile bins and missing share per input column) is stored in the
model at training time, so the training data is not needed. Drift is measured with the
population stability index, PSI = sum((new - ref) * ln(new / ref)) over the bins:
    < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift (retrain / investigate).
The mean predicted probability is compared with the one on the training data as well.

Usage:
    python monitor.py --model model.joblib --input snapshot.csv [--output drift.csv]
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from features import NUMERIC
from schema import read_validated

EPS = 1e-4
LEVELS = [(0.25, "large shift"), (0.1, "moderate shift"), (-np.inf, "stable")]


def profile(X: pd.DataFrame, n_bins: int = 10) -> dict:
    """Reference profile of the input columns: bin edges, bin shares, missing share."""
    out = {}
    for col in NUMERIC:
        values = X[col].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
        out[col] = {"edges": edges, "shares": _shares(values, edges), "missing": X[col].isna().mean()}
    return out


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi(ref: np.ndarray, new: np.ndarray) -> float:
    ref, new = np.clip(ref, EPS, None), np.clip(new, EPS, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


def drift(reference: dict, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, ref in reference.items():
        values = X[col].dropna().to_numpy(dtype=float)
        value = psi(ref["shares"], _shares(values, ref["edges"]))
        rows.append({"column": col, "psi": round(value, 4),
                     "missing_ref": round(ref["missing"], 4), "missing_new": round(X[col].isna().mean(), 4),
                     "status": next(label for limit, label in LEVELS if value > limit)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Drift of a new snapshot against the training data")
    ap.add_argument("--model", required=True, help="model saved by train.py or recovery.py")
    ap.add_argument("--input", required=True, help="new snapshot CSV with input columns")
    ap.add_argument("--output", help="save the drift table as CSV")
    args = ap.parse_args()

    model = joblib.load(args.model)
    meta = model.metadata_
    X = read_validated(args.input)
    table = drift(meta["profile"], X)
    print(table.to_string(index=False))

    payment = getattr(model, "payment", model)
    mean_new = payment.predict_proba(X).mean()
    print("\nMean predicted probability, training -> new snapshot:")
    for target, value in mean_new.items():
        print(f"  {target}: {meta['mean_proba'][target]:.3f} -> {value:.3f}")
    if args.output:
        table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
