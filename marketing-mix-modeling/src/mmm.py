"""Marketing mix model: fit, validate, test, decompose (see model.py and evaluate.py).

1. Hyperparameters (adstock decay, Hill half-saturation and shape per channel) are searched
   on the training period, the model is fitted (OLS or negative binomial GLM).
2. Validation on the last --holdout days against a baseline without media; optional
   rolling-origin cross-validation (--cv-folds); assumption tests on training residuals.
3. Refit on all data: TV effectiveness per year, contribution and ROI per channel with
   block-bootstrap 95 % intervals (--bootstrap), response curves, plots.
   With --truth the estimates are compared with the simulation's ground truth.

Usage:
    python mmm.py --input ../data/mock_daily.csv --output ../results [--family ols|nb]
                  [--adstock geometric|weibull] [--cv-folds 3] [--bootstrap 200] [--truth ../data/truth.json]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from evaluate import baseline_predict, bootstrap, rolling_cv
from model import (
    contributions,
    diagnostics,
    fit,
    media_summary,
    metrics,
    predict,
    residuals,
    search,
    tv_change,
    validate,
)
from plots import plot_decomposition, plot_fit, plot_residuals, plot_response
from transforms import CHANNELS


def compare_truth(truth: dict, params: dict, tv: dict, summary: dict) -> pd.DataFrame:
    rows = []
    for ch in CHANNELS:
        if "decay" in params[ch]:
            rows.append([f"{ch} decay", truth["media"][ch]["decay"], params[ch]["decay"]])
        rows.append([f"{ch} contribution share", truth["true_contribution_share"][ch], summary[ch]["share"]])
        rows.append([f"{ch} calls per 1000 spend", truth["true_roi_calls_per_1000"][ch], summary[ch]["calls_per_1000"]])
    for year, value in truth["true_tv_calls_per_1000_contacts"].items():
        rows.append([f"tv calls per 1000 contacts {year}", value, tv["by_year"][year]["calls_per_1000_contacts"]])
    return pd.DataFrame(rows, columns=["parameter", "true", "estimated"]).round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit and evaluate a marketing mix model")
    ap.add_argument("--input", required=True, help="daily CSV: date, calls, media, spend and control columns")
    ap.add_argument("--output", required=True, help="directory for results and figures")
    ap.add_argument("--family", choices=["ols", "nb"], default="ols", help="OLS or negative binomial GLM")
    ap.add_argument("--adstock", choices=["geometric", "weibull"], default="geometric",
                    help="carry-over shape: geometric decay, or Weibull with a possible delayed peak")
    ap.add_argument("--holdout", type=int, default=90, help="days at the end used for validation")
    ap.add_argument("--cv-folds", type=int, default=0, help="rolling-origin CV folds of --holdout days, 0 = off")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap samples for intervals, 0 = off")
    ap.add_argument("--truth", help="ground truth JSON from simulate.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        df = validate(pd.read_csv(args.input, parse_dates=["date"]))
    except (OSError, ValueError) as err:
        sys.exit(f"{args.input}: {err}")
    n_train = len(df) - args.holdout
    train, test = df.iloc[:n_train], df.index[n_train:]

    # 1-2: hyperparameters, hold-out validation, tests
    params = search(train, args.seed, adstock=args.adstock)
    model = fit(train, params, args.family)
    pred = predict(model, df, params)
    base = baseline_predict(df, n_train)
    result = {
        "family": args.family, "adstock": args.adstock, "params": params,
        "train": metrics(train["calls"], pred[train.index]),
        "holdout": metrics(df.loc[test, "calls"], pred[test]),
        "baseline_holdout": metrics(df.loc[test, "calls"], base[test]),
        "diagnostics": diagnostics(model),
        "outlier_days": [c[8:] for c in model.params.index if c.startswith("outlier_")],
    }
    plot_fit(df, pred, df.loc[test[0], "date"], out / "fit.png")
    plot_residuals(residuals(model), out / "residuals.png")
    if args.cv_folds:
        cv = rolling_cv(df, args.cv_folds, args.holdout, args.family, args.seed, adstock=args.adstock)
        cv.to_csv(out / "cv.csv", index=False)
        result["cv"] = cv.drop(columns=["fold", "test_start"]).agg(["mean", "std"]).to_dict()

    # 3: refit on all data
    model = fit(df, params, args.family)
    contrib = contributions(model, df, params)
    result |= {"tv_change": tv_change(model, df, params), "media": media_summary(df, contrib),
               "coefficients": model.params.to_dict(),
               "cpm": {ch: df[f"{ch}_spend"].sum() / df[ch].sum() for ch in CHANNELS}}
    if args.bootstrap:
        result["bootstrap_ci"] = bootstrap(model, df, params, args.family, args.bootstrap, seed=args.seed)
    plot_decomposition(df, contrib, out / "decomposition.png")
    plot_response(df, params, model.params, out / "response_curves.png")
    (out / "summary.txt").write_text(str(model.summary()))
    (out / "model.json").write_text(json.dumps(result, indent=2, default=float))

    print(f"{args.family}, {args.adstock} adstock, hyperparameters: {json.dumps(params)}")
    print(pd.DataFrame({k: result[k] for k in ["train", "holdout", "baseline_holdout"]}).round(3).to_string())
    if args.cv_folds:
        print(f"\nrolling CV, {args.cv_folds} folds of {args.holdout} days:")
        print(cv.round(3).to_string(index=False))
    print("\ndiagnostics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["diagnostics"].items()})
    print("outlier days (train):", result["outlier_days"])
    print("\nTV effectiveness by year (incremental calls per 1000 contacts):")
    print(pd.DataFrame(result["tv_change"]["by_year"]).T.round(2).to_string())
    print(f"equal across years, Wald p = {result['tv_change']['equal_years_p']:.4f}")
    print("\nmedia:")
    print(pd.DataFrame(result["media"]).round(3).to_string())
    if args.bootstrap:
        print(f"\nbootstrap 95 % intervals ({args.bootstrap} samples):")
        print(pd.DataFrame(result["bootstrap_ci"]).T.round(3).to_string())
    if args.truth:
        truth = json.loads(Path(args.truth).read_text())
        table = compare_truth(truth, params, result["tv_change"], result["media"])
        table.to_csv(out / "truth_vs_estimate.csv", index=False)
        print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
