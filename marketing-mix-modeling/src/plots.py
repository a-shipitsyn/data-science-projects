"""Figures of the marketing mix model."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from transforms import CHANNELS, carryover, hill

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_fit(df: pd.DataFrame, pred: pd.Series, holdout_start: pd.Timestamp, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["calls"], lw=0.8, color="grey", label="actual")
    ax.plot(df["date"], pred, lw=1.2, color="tab:blue", label="model")
    ax.axvspan(holdout_start, df["date"].iloc[-1], color="tab:orange", alpha=0.15, label="hold-out")
    ax.set_ylabel("calls per day")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_decomposition(df: pd.DataFrame, contrib: pd.DataFrame, path: Path) -> None:
    weekly = contrib.assign(date=df["date"]).set_index("date").resample("W").mean()
    base = weekly.drop(columns=CHANNELS).sum(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(weekly.index, base, *[weekly[ch] for ch in CHANNELS], labels=["baseline", *CHANNELS], alpha=0.8)
    ax.plot(df.set_index("date")["calls"].resample("W").mean(), color="black", lw=1, label="actual")
    ax.set_ylabel("calls per day, weekly mean")
    ax.legend(loc="upper left", ncol=5)
    _save(fig, path)


def plot_response(df: pd.DataFrame, params: dict, coefficients: pd.Series, path: Path) -> None:
    """Incremental calls per day for a constant daily spend (steady-state adstock); TV with its latest coefficient."""
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(12, 3.5))
    last_tv = [c for c in coefficients.index if c.startswith("tv_")][-1]
    for ax, ch in zip(axes, CHANNELS, strict=True):
        p = params[ch]
        cpm = df[f"{ch}_spend"].sum() / df[ch].sum()
        typical = df.loc[df[ch] > 0, f"{ch}_spend"].mean()
        spend = np.linspace(0, 3 * typical, 200)
        steady = spend / cpm * carryover(p)
        ax.plot(spend, coefficients[last_tv if ch == "tv" else ch] * hill(steady, p["half_sat"], p["shape"]))
        ax.axvline(typical, ls="--", color="grey")
        ax.set_title(ch)
        ax.set_xlabel("daily spend")
    axes[0].set_ylabel("incremental calls per day")
    _save(fig, path)


def plot_residuals(resid: pd.Series, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(resid.to_numpy(), lw=0.6)
    axes[0].set_title("residuals")
    sm.graphics.tsa.plot_acf(resid, lags=28, ax=axes[1])
    sm.qqplot(resid, line="s", ax=axes[2])
    axes[2].set_title("Q-Q")
    _save(fig, path)
