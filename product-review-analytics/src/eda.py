"""Exploratory analysis: key numbers, figures and the factors that influence the rating."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from data import load_reviews
from themes import theme_flags

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
SEED = 42

# "recommend" is left out: it is the reviewer's own verdict, a restatement of the rating rather than a cause
FACTORS = ["n_words", "n_exclaim", "upper_ratio", "log_helpful", "year", "category"]


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y" if ax.get_xlabel() != "importance" else "x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rating_distribution(df, path):
    share = df["rating"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(share.index.astype(str), share.values, color=BLUE, width=0.6)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in share.values], color=INK, padding=2)
    ax.set_xlabel("stars")
    ax.set_ylabel("share of reviews")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style(ax, "Rating distribution")
    save(fig, path)


def plot_monthly(df, path, min_reviews=50):
    """Two panels with a shared time axis: review volume and mean rating per month."""
    monthly = df.dropna(subset=["date"]).set_index("date").resample("MS")["rating"].agg(["size", "mean"])
    monthly = monthly[monthly["size"] >= min_reviews]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    top.bar(monthly.index, monthly["size"], width=20, color=BLUE)
    top.set_ylabel("reviews")
    style(top, "Reviews per month")
    bottom.plot(monthly.index, monthly["mean"], color=BLUE, linewidth=2, marker="o", markersize=4)
    bottom.set_ylabel("mean stars")
    style(bottom, f"Mean rating per month (months with ≥ {min_reviews} reviews)")
    save(fig, path)


def plot_by_category(df, path):
    stats = df.groupby("category")["rating"].agg(["size", "mean", lambda r: (r <= 3).mean()])
    stats.columns = ["reviews", "mean", "low_share"]
    stats = stats.sort_values("low_share")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(stats.index, stats["low_share"], color=BLUE, height=0.6)
    ax.bar_label(bars, labels=[f"{s:.1%}  (n={n:,})" for s, n in zip(stats["low_share"], stats["reviews"])],
                 color=INK, padding=3)
    ax.set_xlabel("share of 1–3 star reviews")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, stats["low_share"].max() * 1.5)
    style(ax, "Share of 1–3 star reviews by product category")
    save(fig, path)
    return stats


def plot_length(df, path):
    median_words = df.groupby("rating")["n_words"].median()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(median_words.index.astype(str), median_words.values, color=BLUE, width=0.6)
    ax.bar_label(bars, fmt="%.0f", color=INK, padding=2)
    ax.set_xlabel("stars")
    ax.set_ylabel("median words per review")
    style(ax, "Review length by rating")
    save(fig, path)


def rating_factors(df):
    """Permutation importance of structured features and review themes in a gradient boosting model.

    Importance is measured on a held-out 25 % as the drop in R². The direction column is the
    difference in mean rating between reviews above and below the feature's median
    (or with vs without the theme), so the reader sees which way a factor pushes the rating.
    """
    X = pd.concat([df[FACTORS], theme_flags(df["title"] + ". " + df["text"])], axis=1)
    X["category"] = X["category"].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(X, df["rating"], test_size=0.25, random_state=SEED,
                                                        stratify=df["rating"])
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED)
    model.fit(X_train, y_train)
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=SEED, scoring="r2")

    rows = []
    for i, col in enumerate(X.columns):
        if col == "category":
            means = df.groupby("category")["rating"].mean()
            direction = f"best {means.idxmax()} {means.max():.2f}, worst {means.idxmin()} {means.min():.2f}"
        else:
            values = X[col].astype(float)
            high = values > values.median() if values.nunique() > 2 else values == 1
            direction = f"{df.loc[high, 'rating'].mean() - df.loc[~high & values.notna(), 'rating'].mean():+.2f} stars"
        rows.append({"factor": col, "importance": result.importances_mean[i],
                     "std": result.importances_std[i], "direction": direction})
    return pd.DataFrame(rows).sort_values("importance", ascending=False), model.score(X_test, y_test)


def plot_factors(factors, path, top=10):
    top_factors = factors.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top_factors["factor"], top_factors["importance"], xerr=top_factors["std"], color=BLUE, height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.set_xlabel("importance")
    style(ax, "Factors behind the rating (drop in R² when shuffled)")
    save(fig, path)


def insights(df, by_category):
    """Headline numbers for the report."""
    rec = df.dropna(subset=["recommend"])
    corr = df[["n_words", "rating"]].corr(method="spearman").iloc[0, 1]
    first, last = df["year"].min(), df["year"].max()
    by_year = df[df["year"].between(2015, 2017)].groupby("year")["rating"].mean()
    return [
        (f"Reviews: {len(df):,} for {df['product_id'].nunique()} products; {df['year'].between(2015, 2017).mean():.1%} "
        f"written in 2015-2017 (range {first:.0f}-{last:.0f})."),
        (f"Ratings are very positive: {(df['rating'] == 5).mean():.0%} five stars, mean {df['rating'].mean():.2f}, "
        f"only {(df['rating'] <= 2).mean():.1%} one or two stars."),
        (f"Recommendation matches the rating: mean {rec.loc[rec['recommend'] == 1, 'rating'].mean():.2f} stars when "
        f"recommended vs {rec.loc[rec['recommend'] == 0, 'rating'].mean():.2f} when not "
        f"({(rec['recommend'] == 0).mean():.1%} not recommended)."),
        (f"Unhappy reviewers write more: median {df.loc[df['rating'] <= 2, 'n_words'].median():.0f} words for 1-2 stars "
        f"vs {df.loc[df['rating'] == 5, 'n_words'].median():.0f} for 5 stars (Spearman {corr:+.2f})."),
        "Share of 1-3 star reviews by category: "
        + ", ".join(f"{c} {s:.1%}" for c, s in by_category["low_share"].items()) + ".",
        "Mean rating by year: " + ", ".join(f"{y:.0f} {m:.2f}" for y, m in by_year.items()) + ".",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw review CSV")
    parser.add_argument("--output", required=True, help="folder for figures and tables")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = load_reviews(args.input)

    plot_rating_distribution(df, out / "rating_distribution.png")
    plot_monthly(df, out / "monthly_trend.png")
    by_category = plot_by_category(df, out / "low_ratings_by_category.png")
    plot_length(df, out / "length_by_rating.png")

    factors, r2 = rating_factors(df)
    factors.to_csv(out / "rating_factors.csv", index=False)
    plot_factors(factors, out / "rating_factors.png")

    lines = insights(df, by_category)
    (out / "insights.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nFactor model R² on held-out reviews: {r2:.2f}")
    print(factors.head(10).to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
