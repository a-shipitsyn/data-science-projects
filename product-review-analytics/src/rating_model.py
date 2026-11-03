"""Predict the star rating (1-5) from the review text and structured features.

Word TF-IDF (1-2 grams) on title + text, plus structured features, into a class-weighted logistic regression.
C is tuned by nested CV. Compared with a majority-class baseline and feature subsets in stratified
5-fold CV; metrics are the mean ± 2 std across folds.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_reviews

SEED = 42
NUMERIC = ["log_helpful", "n_words", "n_exclaim", "upper_ratio"]


def build_model(text=True, numeric=(), categorical=(), C=2.0):
    """Logistic regression on any mix of TF-IDF text, numeric and categorical columns."""
    parts = []
    if text:
        parts.append(("words", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                                               strip_accents="unicode"), "review"))
    if numeric:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                      list(numeric)))
    if categorical:
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                           OneHotEncoder(handle_unknown="ignore")), list(categorical)))
    return make_pipeline(
        ColumnTransformer(parts),
        LogisticRegression(C=C, class_weight="balanced", max_iter=3000),
    )


def features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC + ["category"]].copy()
    X["review"] = df["title"] + ". " + df["text"]
    X["recommend"] = df["recommend"].map({1.0: "yes", 0.0: "no"})
    return X


def evaluate(y, pred) -> dict:
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "mae_stars": mean_absolute_error(y, pred),
        "f1_1_star": f1_score(y, pred, labels=[1], average="macro", zero_division=0),
    }


def with_tuning(model, c_grid: list[float], seed: int = SEED):
    """Tune C of a logistic regression pipeline with an inner 3-fold CV on the training fold (nested CV)."""
    if isinstance(model, DummyClassifier):
        return model
    if len(c_grid) == 1:
        return clone(model).set_params(logisticregression__C=c_grid[0])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(model, {"logisticregression__C": c_grid}, scoring="f1_macro", cv=inner, n_jobs=-1)


def cross_validate_model(model, X, y, cv, c_grid):
    """Out-of-fold predictions, metrics per outer fold and the C chosen in each fold."""
    pred = np.empty_like(y)
    folds, chosen = [], []
    for train, test in cv.split(X, y):
        fitted = clone(with_tuning(model, c_grid)).fit(X.iloc[train], y[train])
        pred[test] = fitted.predict(X.iloc[test])
        folds.append(evaluate(y[test], pred[test]))
        if isinstance(fitted, GridSearchCV):
            chosen.append(fitted.best_params_["logisticregression__C"])
    return pred, pd.DataFrame(folds), chosen


def summarise(name: str, folds: pd.DataFrame, chosen: list) -> dict:
    """Mean and 2 standard deviations across the outer folds."""
    row = {"model": name}
    for metric in folds.columns:
        row[metric] = folds[metric].mean()
        row[f"{metric}_2std"] = 2 * folds[metric].std()
    row["C_per_fold"] = " ".join(f"{c:g}" for c in chosen)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw review CSV")
    parser.add_argument("--output", required=True, help="folder for metrics and the confusion matrix")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-grid", default="0.5,2,8",
                        help="values of C tuned in the inner CV; one value switches tuning off")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = load_reviews(args.input)
    X, y = features(df), df["rating"].to_numpy()
    c_grid = [float(c) for c in args.c_grid.split(",")]

    models = {
        "majority class": DummyClassifier(strategy="most_frequent"),
        "text only": build_model(),
        "text + review stats + category": build_model(numeric=NUMERIC, categorical=["category"]),
        "text + review stats + category + recommend": build_model(numeric=NUMERIC,
                                                                  categorical=["category", "recommend"]),
    }
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    rows, predictions = [], {}
    for name, model in models.items():
        pred, folds, chosen = cross_validate_model(model, X, y, cv, c_grid)
        predictions[name] = pred
        rows.append(summarise(name, folds, chosen))
        print(f"{name}: " + ", ".join(f"{m} {folds[m].mean():.3f} ± {2 * folds[m].std():.3f}" for m in folds.columns)
              + (f", C per fold {rows[-1]['C_per_fold']}" if chosen else ""))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "rating_model_metrics.csv", index=False)

    best = metrics.loc[metrics["macro_f1"].idxmax(), "model"]
    print(f"\nBest by macro-F1: {best}")
    print(classification_report(y, predictions[best], digits=3))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, predictions[best], normalize="true", values_format=".2f",
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Out-of-fold predictions, share of each true rating\n{best}", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)

    off_by_one = np.mean(np.abs(predictions[best] - y) <= 1)
    print(f"Predictions within one star: {off_by_one:.1%}")


if __name__ == "__main__":
    main()
