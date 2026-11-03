"""Synthetic reviews in the raw CSV layout, so the tests run without the dataset."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PRODUCTS = [
    ("P1", "Reader Pro e-reader, 6 inch glare-free display", "eBook Readers,Electronics"),
    ("P2", "Home Speaker with voice assistant", "Smart Speakers,Electronics"),
    ("P3", "Kids Tablet, 7 inch, with Kid-Proof Case", "Tablets,Electronics"),
]
TEXTS = {
    5: ["Love it! The screen is great and the battery lasts for weeks.",
        "Easy to set up, my daughter loves it. Great gift!",
        "Excellent sound and the voice assistant is very handy."],
    4: ["Good value for the price, apps work fine.",
        "Nice design and light, but storage is small."],
    3: ["It is ok. Performance is slow sometimes.",
        "Average tablet, the battery drains fast."],
    2: ["Wifi connection keeps dropping, not happy.",
        "Screen has glare and it freezes a lot."],
    1: ["Stopped working after a week, returned it for a refund.",
        "Terrible, customer service was no help at all."],
}


def make_raw_reviews(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ratings = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.08, 0.08, 0.14, 0.3, 0.4])
    rows = []
    for i, rating in enumerate(ratings):
        pid, name, categories = PRODUCTS[i % len(PRODUCTS)]
        text = rng.choice(TEXTS[rating]) + f" Review number {i}."
        rows.append({
            "id": f"r{i}", "name": name if i % 5 else np.nan, "asins": pid, "brand": "Brand",
            "categories": categories, "reviews.date": f"2016-{1 + i % 12:02d}-15T00:00:00.000Z",
            "reviews.doRecommend": rating >= 3, "reviews.numHelpful": int(rng.poisson(1)),
            "reviews.rating": float(rating), "reviews.text": text, "reviews.title": "Title",
            "reviews.username": f"user{i}", "reviews.sourceURLs": "http://example.com",
        })
    raw = pd.DataFrame(rows)
    raw.loc[0, "reviews.rating"] = np.nan  # dropped by the loader
    raw.loc[1, "reviews.text"] = np.nan  # dropped by the loader
    return raw


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "reviews.csv"
    make_raw_reviews().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def reviews(raw_csv):
    from data import load_reviews

    return load_reviews(raw_csv)
