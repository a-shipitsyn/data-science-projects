"""Exploratory analysis: descriptive statistics, correlations and plots.

Figures:
    calls_media.png   daily calls (weekly mean line) with TV flights, outdoor months and
                      digital bursts marked
    patterns.png      mean calls by weekday and by month, distribution of daily calls

Usage:
    python eda.py --input ../data/mock_daily.csv --output ../results
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from transforms import CHANNELS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def periods(active: pd.Series, dates: pd.Series) -> list[tuple]:
    """Start and end dates of consecutive runs where `active` is true."""
    runs = (active != active.shift()).cumsum()[active]
    return [(dates[idx].min(), dates[idx].max()) for idx in runs.groupby(runs).groups.values()]


def plot_calls_media(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["calls"], lw=0.5, color="grey", label="calls")
    ax.plot(df["date"], df["calls"].rolling(7, center=True).mean(), color="black", lw=1.2, label="7-day mean")
    digital_burst = df["digital"].rolling(7, center=True, min_periods=1).mean() > 1.6 * df["digital"].median()
    for (name, active, color) in [("TV", df["tv"] > 0, "tab:orange"), ("outdoor", df["ooh"] > 0, "tab:green"),
                                  ("digital burst", digital_burst, "tab:red")]:
        for i, (start, end) in enumerate(periods(active, df["date"])):
            ax.axvspan(start, end, color=color, alpha=0.15 if name != "TV" else 0.3, label=name if i == 0 else None)
    ax.set_ylabel("calls per day")
    ax.legend(loc="upper left", ncol=5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_patterns(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    df.groupby(df["date"].dt.day_name().str[:3], sort=False)["calls"].mean().plot.bar(ax=axes[0], color="tab:blue")
    axes[0].set_title("mean calls by weekday")
    df.groupby(df["date"].dt.month)["calls"].mean().plot.bar(ax=axes[1], color="tab:blue")
    axes[1].set_title("mean calls by month")
    df["calls"].plot.hist(bins=40, ax=axes[2], color="tab:blue")
    axes[2].set_title("daily calls")
    for ax in axes:
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory analysis of marketing mix data")
    ap.add_argument("--input", required=True, help="daily CSV: date, calls, media, spend and control columns")
    ap.add_argument("--output", required=True, help="directory for figures")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, parse_dates=["date"])

    print(df.drop(columns="date").describe().round(1).T.to_string())
    print("\nshare of days with exposure:", {ch: round((df[ch] > 0).mean(), 3) for ch in CHANNELS})
    print("total spend share:", (df[[f"{ch}_spend" for ch in CHANNELS]].sum()
                                 / df[[f"{ch}_spend" for ch in CHANNELS]].sum().sum()).round(3).to_dict())
    print("\ncorrelation with calls:")
    print(df[["calls", *CHANNELS, "season", "fx_rate", "holiday"]].corr()["calls"].drop("calls").round(3).to_string())
    plot_calls_media(df, out / "calls_media.png")
    plot_patterns(df, out / "patterns.png")


if __name__ == "__main__":
    main()
