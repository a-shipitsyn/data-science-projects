"""Train all models on the training genes, tune them on validation genes, report on test genes.

Masked examples have 5-30 known bases on each side (drawn independently). Metrics: accuracy with
a 95 % interval, log-loss, recall per base and accuracy by the shorter context length.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnn_model import CNNModel
from genome import BASES, load_splits
from kmer_model import KmerModel
from linear_model import LinearModel
from windows import sample_examples

BLUE, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"
CONTEXT_BINS = [(5, 9), (10, 19), (20, 30)]


def log_loss(proba, y):
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, None))))


def metrics(proba, y, lengths) -> dict:
    pred = proba.argmax(axis=1)
    correct = pred == y
    acc = correct.mean()
    row = {"accuracy": acc, "ci95": 1.96 * np.sqrt(acc * (1 - acc) / len(y)), "log_loss": log_loss(proba, y)}
    for code, base in enumerate(BASES):
        row[f"recall_{base}"] = correct[y == code].mean()
    shortest = lengths.min(axis=1)
    for lo, hi in CONTEXT_BINS:
        row[f"acc_context_{lo}_{hi}"] = correct[(shortest >= lo) & (shortest <= hi)].mean()
    return row


def tune_kmer(model, val):
    """Highest order and smoothing chosen by validation log-loss."""
    rows = []
    for max_k in model.orders:
        for alpha in (2, 4, 8, 16, 32):
            proba = model.predict_proba(val[0], val[2], max_k=max_k, alpha=alpha)
            rows.append({"max_k": max_k, "alpha": alpha, "accuracy": (proba.argmax(1) == val[1]).mean(),
                         "log_loss": log_loss(proba, val[1])})
    table = pd.DataFrame(rows)
    return table.loc[table["log_loss"].idxmin()], table


def plot_by_context(results: pd.DataFrame, path: Path):
    """Small multiples, one panel per context bin, so each bar's value can be read directly."""
    labels = [f"{lo}–{hi}" for lo, hi in CONTEXT_BINS]
    models = results.index[::-1]
    fig, axes = plt.subplots(1, len(CONTEXT_BINS), figsize=(11, 3.6), sharey=True, sharex=True)
    upper = max(0.55, results[[f"acc_context_{lo}_{hi}" for lo, hi in CONTEXT_BINS]].to_numpy().max() + 0.06)
    for ax, (lo, hi), label in zip(axes, CONTEXT_BINS, labels):
        values = results.loc[models, f"acc_context_{lo}_{hi}"]
        bars = ax.barh(models, values, color=BLUE, height=0.6)
        ax.bar_label(bars, fmt="%.3f", color=INK, padding=3, fontsize=8)
        ax.set_xlim(0.2, upper)
        ax.set_title(f"shorter side {label} bases", loc="left", color=INK, fontsize=10)
        ax.set_facecolor(SURFACE)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("test accuracy", color=MUTED, fontsize=8)
    fig.suptitle("Accuracy by context length", x=0.01, ha="left", color=INK)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="FASTA file")
    parser.add_argument("--output", required=True, help="folder for metrics and figures")
    parser.add_argument("--n-val", type=int, default=100_000)
    parser.add_argument("--n-test", type=int, default=200_000)
    parser.add_argument("--n-linear", type=int, default=1_000_000, help="training examples for logistic regression")
    parser.add_argument("--cnn-steps", type=int, default=4000)
    parser.add_argument("--cnn-weights", help="load CNN weights from this file instead of training")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    splits = load_splits(args.input, seed=args.seed)
    print({name: f"{len(g.lengths)} sequences, {g.n_bases / 1e6:.1f}M bases" for name, g in splits.items()},
          f"({time.time() - t0:.0f} s)")
    val = sample_examples(splits["val"], args.n_val, seed=args.seed + 11)
    test = sample_examples(splits["test"], args.n_test, seed=args.seed + 12)

    probas, fit_seconds, settings = {}, {}, {}

    prior = np.bincount(splits["train"].bases[splits["train"].bases < 4], minlength=4) / splits["train"].n_bases
    probas["Base frequencies"] = np.tile(prior, (len(test[0]), 1))

    t = time.time()
    kmer = KmerModel().fit(splits["train"])
    fit_seconds["k-mer, k ≤ 5"] = fit_seconds["k-mer, k ≤ 12"] = time.time() - t
    best, kmer_tuning = tune_kmer(kmer, val)
    kmer_tuning.to_csv(out / "kmer_tuning.csv", index=False)
    settings["k-mer"] = {"max_k": int(best["max_k"]), "alpha": float(best["alpha"]), "contexts": kmer.n_contexts}
    probas["k-mer, k ≤ 5"] = kmer.predict_proba(test[0], test[2], max_k=5, alpha=best["alpha"])
    probas["k-mer, k ≤ 12"] = kmer.predict_proba(test[0], test[2], max_k=int(best["max_k"]), alpha=best["alpha"])
    print(f"k-mer: fitted in {fit_seconds['k-mer, k ≤ 12']:.0f} s, {settings['k-mer']}")

    train = sample_examples(splits["train"], args.n_linear, seed=args.seed + 1)
    for name, kmers in [("Logistic regression, positions", False), ("Logistic regression, positions + k-mers", True)]:
        scores = {}
        for C in (0.1, 1.0):
            t = time.time()
            model = LinearModel(kmers=kmers, C=C).fit(*train)
            scores[C] = (log_loss(model.predict_proba(val[0], val[2]), val[1]), model, time.time() - t)
        C = min(scores, key=lambda c: scores[c][0])
        probas[name] = scores[C][1].predict_proba(test[0], test[2])
        fit_seconds[name] = scores[C][2]
        settings[name] = {"C": C, "train_examples": args.n_linear}
        print(f"{name}: C={C}, fitted in {fit_seconds[name]:.0f} s")

    cnn = CNNModel(steps=args.cnn_steps, seed=args.seed)
    if args.cnn_weights:
        cnn.load(args.cnn_weights)
        # keep the training details recorded when these weights were trained
        previous = out / "settings.json"
        if previous.exists() and "CNN" in json.loads(previous.read_text()):
            settings["CNN"] = json.loads(previous.read_text())["CNN"]
            fit_seconds["CNN"] = settings["CNN"].get("fit_seconds")
        print(f"CNN: loaded {args.cnn_weights}")
    else:
        t = time.time()
        cnn.fit(splits["train"], val=(val[0], val[1]))
        fit_seconds["CNN"] = time.time() - t
        cnn.save(out / "cnn_weights.pt")
        settings["CNN"] = {"steps": args.cnn_steps, "batch": cnn.batch, "fit_seconds": fit_seconds["CNN"],
                           "val_log_loss": cnn.history}
        print(f"CNN: trained in {fit_seconds['CNN']:.0f} s")
    probas["CNN"] = cnn.predict_proba(test[0])

    # Blend of the two best families: weight chosen on validation
    kmer_val = kmer.predict_proba(val[0], val[2], max_k=int(best["max_k"]), alpha=best["alpha"])
    cnn_val = cnn.predict_proba(val[0])
    weights = np.linspace(0, 1, 11)
    blend_loss = [log_loss(w * kmer_val + (1 - w) * cnn_val, val[1]) for w in weights]
    w = float(weights[int(np.argmin(blend_loss))])
    settings["k-mer + CNN"] = {"kmer_weight": w}
    probas["k-mer + CNN"] = w * probas["k-mer, k ≤ 12"] + (1 - w) * probas["CNN"]

    results = pd.DataFrame({name: metrics(p, test[1], test[2]) for name, p in probas.items()}).T
    results["fit_seconds"] = pd.Series(fit_seconds)
    results.to_csv(out / "metrics.csv", float_format="%.4f")
    (out / "settings.json").write_text(json.dumps(settings, indent=2, default=float))
    plot_by_context(results, out / "accuracy_by_context.png")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results[["accuracy", "ci95", "log_loss", "acc_context_5_9", "acc_context_10_19",
                       "acc_context_20_30"]].round(4))
    print(f"Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
