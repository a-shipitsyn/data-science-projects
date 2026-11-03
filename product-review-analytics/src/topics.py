"""Extract the most common topics from the reviews with keyword extraction.

1. Key phrases: the bigrams and trigrams that appear in the most reviews.
2. Topics: NMF on word TF-IDF of the review text; each topic is described by its top keywords,
   and its size is the share of reviews where it is the dominant topic.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

from data import load_reviews

SEED = 42
# Words that appear in almost every review and say nothing about the topic
EXTRA_STOP_WORDS = {"amazon", "product", "bought", "buy", "purchased", "purchase", "really", "just", "use", "got",
                    "like", "love", "great", "good", "nice", "works", "work", "well", "does", "did", "would", "one"}
STOP_WORDS = list(ENGLISH_STOP_WORDS | EXTRA_STOP_WORDS)


def key_phrases(texts: pd.Series, top: int = 15) -> pd.DataFrame:
    """Bigrams and trigrams ranked by the share of reviews that contain them.

    Stop words stay inside phrases ("easy to use"), but a phrase may not start or end with one.
    """
    vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=20, binary=True)
    counts = vectorizer.fit_transform(texts)
    phrases = vectorizer.get_feature_names_out()
    stop = set(STOP_WORDS)
    keep = np.array([p.split()[0] not in stop and p.split()[-1] not in stop for p in phrases])
    share = np.asarray(counts.mean(axis=0)).ravel()
    result = pd.DataFrame({"phrase": phrases[keep], "share_of_reviews": share[keep]})
    return result.sort_values("share_of_reviews", ascending=False).head(top).reset_index(drop=True)


def nmf_topics(texts: pd.Series, n_topics: int = 5, n_words: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    """NMF topics with their top keywords; also returns the dominant topic of each review."""
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, min_df=10, max_df=0.5, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=SEED, max_iter=500)
    weights = nmf.fit_transform(tfidf)
    vocab = vectorizer.get_feature_names_out()

    dominant = np.where(weights.max(axis=1) > 0, weights.argmax(axis=1), -1)
    rows = []
    for k, component in enumerate(nmf.components_):
        rows.append({
            "topic": k + 1,
            "share_of_reviews": np.mean(dominant == k),
            "keywords": ", ".join(vocab[component.argsort()[::-1][:n_words]]),
        })
    topics = pd.DataFrame(rows).sort_values("share_of_reviews", ascending=False)
    return topics, dominant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw review CSV")
    parser.add_argument("--output", help="CSV for the topic table")
    parser.add_argument("--topics", type=int, default=5)
    args = parser.parse_args()

    df = load_reviews(args.input)
    texts = df["title"] + ". " + df["text"]

    print("Most common key phrases:")
    print(key_phrases(texts).to_string(index=False, float_format="{:.1%}".format))

    topics, dominant = nmf_topics(texts, args.topics)
    df["topic"] = dominant + 1
    topics["mean_rating"] = topics["topic"].map(df.groupby("topic")["rating"].mean())
    print("\nTopics (NMF):")
    with pd.option_context("display.max_colwidth", 120):
        print(topics.to_string(index=False, float_format="{:.2f}".format))
    if args.output:
        topics.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
