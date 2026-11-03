"""Data preprocessing:
    - parse speaker and ratio features from JSON strings and stack them into a flat table
    - delete columns with most NA and impute the rest

Input: CSV from audio_features.py with columns speaker_a, speaker_b, ratio[, label],
each a JSON object {"full": {...}, "start": {...}, "middle": {...}, "end": {...}}.
Output columns are named <source>__<region>__<feature>, e.g. speaker_a__end__pitch_mean.

Usage:
    python preprocess.py --input features.csv --output clean_data.csv
"""
import argparse
import json

import pandas as pd
from sklearn.impute import KNNImputer

SOURCES = ["speaker_a", "speaker_b", "ratio"]
REGIONS = ["full", "start", "middle", "end"]
MAX_NA_RATIO = 0.8


def flatten_regions(s):
    """Convert JSON string with per-region features into a flat dictionary.

    Input:  '{"full": {"pitch_mean": ..}, "start": {..}, "middle": {..}, "end": {..}}'
    Output: {'full__pitch_mean': .., 'start__pitch_mean': .., ..}
    """
    d = json.loads(s)
    out = dict()
    for k in d["full"].keys():
        for region in REGIONS:
            out[region + "__" + k] = d[region][k] if d.get(region) else None
    return out


def parse_features(df):
    """Expand every JSON column into flat feature columns."""
    parts = []
    for source in SOURCES:
        part = pd.DataFrame([flatten_regions(s) for s in df[source]])
        part.columns = [source + "__" + k for k in part.columns]
        parts.append(part)
    X = pd.concat(parts, axis=1).astype(float)
    y = df["label"] if "label" in df else None
    return X, y


def clean_features(X):
    """Delete columns with most NA and impute the rest with KNN."""
    na_ratio = X.isna().sum() / X.shape[0]
    X = X.drop(columns=list(na_ratio.index[na_ratio > MAX_NA_RATIO]))
    imputer = KNNImputer(n_neighbors=3)
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


def preprocess_data(df):
    X, y = parse_features(df)
    X = clean_features(X)
    return X, y


def main():
    ap = argparse.ArgumentParser(description="Flatten and impute call features")
    ap.add_argument("--input", default="features.csv", help="feature CSV from audio_features.py")
    ap.add_argument("--output", default="clean_data.csv", help="clean data CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Rows: {df.shape[0]}")
    if "label" in df:
        n_pos = (df["label"] == 1).sum()
        print(f"Purchases: {n_pos} (1:{df.shape[0] / max(n_pos, 1):.1f})")

    X, y = preprocess_data(df)
    clean_data = X.copy()
    if y is not None:
        clean_data["label"] = y
    clean_data.to_csv(args.output, index=False)
    print(f"Saved {X.shape[1]} features to {args.output}")


if __name__ == "__main__":
    main()
