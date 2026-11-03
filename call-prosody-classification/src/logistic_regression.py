"""Logistic regression with scaling and PCA:
    - evaluation with cross-validation, F1-score because of target imbalance
    - feature importance with permutation method

Usage:
    python logistic_regression.py --input clean_data.csv
"""
import argparse

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(n_components=15):
    steps = [("scaler", StandardScaler())]
    if n_components:
        steps.append(("pca", PCA(n_components=n_components)))
    steps.append(("estimator", LogisticRegression(class_weight="balanced", max_iter=1000)))
    return Pipeline(steps)


def evaluate(pipe, X, y, scoring="f1", cv=4):
    scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv)
    print("\nCross validation score:")
    print(f"\t{scoring}: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})\n")
    return scores


def feature_importance(X, y, n_repeats=30):
    """Permutation importance of a model on original (not PCA) features.
    Not enough data for a train-test split, so the full data set is used."""
    pipe = make_pipeline(n_components=None)
    pipe.fit(X, y)
    perm = permutation_importance(pipe, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return pd.DataFrame({
        "Feature": X.columns,
        "Importance Mean": perm.importances_mean,
        "Importance Std": perm.importances_std,
    }).sort_values(by="Importance Mean", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Logistic regression on clean call features")
    ap.add_argument("--input", default="clean_data.csv", help="clean data CSV from preprocess.py")
    ap.add_argument("--top", type=int, default=20, help="number of important features to print")
    args = ap.parse_args()

    X = pd.read_csv(args.input)
    y = X.pop("label")

    pipe = make_pipeline()
    evaluate(pipe, X, y)

    # train final model on all the data
    pipe.fit(X, y)

    print("Permutation importance:")
    print(feature_importance(X, y).head(args.top))


if __name__ == "__main__":
    main()
