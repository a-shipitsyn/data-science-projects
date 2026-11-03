"""Load the review CSV and convert it to a clean table with generic column names."""

import argparse
import re

import numpy as np
import pandas as pd

COLUMNS = {
    "asins": "product_id",
    "name": "product_name",
    "categories": "categories",
    "reviews.date": "date",
    "reviews.doRecommend": "recommend",
    "reviews.numHelpful": "helpful_votes",
    "reviews.rating": "rating",
    "reviews.title": "title",
    "reviews.text": "text",
}


CATEGORY_RULES = [
    ("smart speaker", r"\becho\b|virtual assistant|speaker"),
    ("streaming", r"fire tv|streaming|tvs & home theater"),
    ("accessory", r"charger|adapter|\bcover\b|\bcase\b|cable"),
    ("e-reader", r"e-?reader|paperwhite|voyage|oasis"),
    ("tablet", r"tablet"),
]


def clean_name(name: str) -> str:
    """Product names may hold the same name twice, joined by a line break, and trailing commas."""
    return re.split(r"[\r\n]", str(name))[0].strip(" ,")


def coarse_category(text: str) -> str:
    """Coarse product category from the product name, or from the category list if the name is missing.

    The keyword found first wins: "Charger for Fire Tablets" is an accessory,
    "Kids Edition Tablet with Case" is a tablet.
    """
    text = text.lower()
    hits = [(m.start(), label) for label, pattern in CATEGORY_RULES if (m := re.search(pattern, text))]
    return min(hits)[1] if hits else "other"


def load_reviews(path: str) -> pd.DataFrame:
    """Read the raw CSV, keep review fields only (no user names or URLs) and add simple features."""
    raw = pd.read_csv(path, usecols=list(COLUMNS), low_memory=False)
    df = raw.rename(columns=COLUMNS)

    df = df.dropna(subset=["rating", "text", "product_id"])
    df["rating"] = df["rating"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()

    # Product name is missing for some rows: take the most common name of the same product id
    df["product_name"] = df["product_name"].dropna().map(clean_name)
    names = df.dropna(subset=["product_name"]).groupby("product_id")["product_name"].agg(lambda s: s.mode()[0])
    df["product_name"] = df["product_id"].map(names)
    df["category"] = df["product_name"].fillna(df["categories"]).map(coarse_category)
    df["product_name"] = df["product_name"].fillna(df["product_id"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["recommend"] = df["recommend"].map({True: 1.0, False: 0.0, "TRUE": 1.0, "FALSE": 0.0})
    df["helpful_votes"] = df["helpful_votes"].fillna(0).clip(lower=0)

    df["n_words"] = df["text"].str.split().str.len()
    df["n_exclaim"] = df["text"].str.count("!")
    df["upper_ratio"] = df["text"].str.count(r"[A-Z]") / df["text"].str.len().clip(lower=1)
    df["year"] = df["date"].dt.year
    df["log_helpful"] = np.log1p(df["helpful_votes"])
    return df.drop(columns=["categories"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw review CSV")
    args = parser.parse_args()

    df = load_reviews(args.input)
    print(df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
