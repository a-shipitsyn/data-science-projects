"""Command line: python -m pricing_engine <command> [options]

    gen-key                                print a new master key
    simulate  --output data                write mock market data and quotes
    init      --market data/lme.csv        load shared market data into the store
    tenant    --tenant tenant_a            register a tenant, print its API key (once)
    ingest    --tenant t --input f.jsonl   validate and store historical quotes
    train     --tenant t                   tune, evaluate on a time hold-out, register model
    evaluate  --tenant t                   hold-out report with baselines and error analysis
    feedback  --tenant t --input f.jsonl   add new quote outcomes (continuous learning)
    retrain   --tenant t                   champion / challenger retraining
    serve                                  run the API

The store root is `--store` or $PRICING_STORE (default ./store); the master key is
$PRICING_MASTER_KEY.
"""
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from . import simulate, training
from .evaluate import metrics, order_value
from .market import load_lme
from .security import TenantDirectory, new_master_key
from .storage import TenantStore


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(report: dict) -> str:
    """Hold-out table and error analysis as text."""
    rows = [("model", report["model"]), ("baseline: linear", report["baseline_linear"]),
            ("baseline: cost-plus", report["baseline_cost_plus"])]
    if "noise_floor" in report:
        rows.append(("true expected price", report["noise_floor"]))
    tr, te = report["train"], report["test"]
    lines = [(f"train {tr['start']}..{tr['end']} (n={tr['n']}, {report['outliers_dropped']} outliers dropped), "
              f"test {te['start']}..{te['end']} (n={te['n']})"),
             f"{'':22}{'MAPE':>7}{'MdAPE':>7}{'WAPE':>7}{'RMSE':>7}{'R²':>7}{'bias':>7}{'cover':>7}{'width':>7}"]
    for name, m in rows:
        cov = f"{m['coverage']:7.2f}{m['interval_width']:7.2f}" if "coverage" in m else ""
        lines.append(f"{name:22}{m['mape']:7.3f}{m['mdape']:7.3f}{m['wape']:7.3f}{m['rmse']:7.2f}{m['r2']:7.3f}"
                     f"{m['bias']:+7.3f}{cov}")
    for group, table in report["errors"].items():
        lines.append(f"\nby {group}:")
        lines += [f"  {r[group]!s:16} n={r['n']:<4} MAPE={r['mape']:.3f} bias={r['bias']:+.3f}"
                  + (f" coverage={r['coverage']:.2f}" if "coverage" in r else "") for r in table]
    return "\n".join(lines)


def noise_floor(report, store, lme, truth_path) -> dict:
    """Error of the simulation's expected price on the same test quotes: the best possible."""
    truth = pd.read_csv(truth_path).set_index("quote_id")
    df = training.load_dataset(store, lme, start=report["test"]["start"])
    test = df[df["quote_id"].isin(report["test_ids"])]
    return metrics(test["quoted_price_per_kg"], truth.loc[test["quote_id"], "expected_price_per_kg"],
                   order_value=order_value(test))


def main(argv=None):
    p = argparse.ArgumentParser(prog="pricing_engine", description="Metal profile pricing engine")
    p.add_argument("--store", default=os.environ.get("PRICING_STORE", "store"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("gen-key")
    s = sub.add_parser("simulate")
    s.add_argument("--output", default="data")
    s.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("init")
    s.add_argument("--market", required=True)
    s = sub.add_parser("tenant")
    s.add_argument("--tenant", required=True)
    for name in ["ingest", "feedback"]:
        s = sub.add_parser(name)
        s.add_argument("--tenant", required=True)
        s.add_argument("--input", required=True)
    for name in ["train", "evaluate"]:
        s = sub.add_parser(name)
        s.add_argument("--tenant", required=True)
        s.add_argument("--trials", type=int, default=40)
        s.add_argument("--target", choices=["margin", "log"], default="margin")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--output", help="write the report as JSON here")
        s.add_argument("--truth", help="simulated truth (truth_prices.csv): adds the noise floor to the report")
    s = sub.add_parser("retrain")
    s.add_argument("--tenant", required=True)
    s.add_argument("--force", action="store_true")
    s.add_argument("--trials", type=int, default=0, help="re-tune hyperparameters with this many trials")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    root = Path(args.store)

    if args.command == "gen-key":
        print(new_master_key())
    elif args.command == "simulate":
        simulate.main(["--output", args.output, "--seed", str(args.seed)])
    elif args.command == "init":
        load_lme(args.market)  # validate before copying
        (root / "market").mkdir(parents=True, exist_ok=True)
        shutil.copy(args.market, root / "market" / "lme.csv")
        print(f"market data loaded into {root / 'market'}")
    elif args.command == "tenant":
        key = TenantDirectory(root).create(args.tenant)
        TenantStore(root, args.tenant).close()
        print(f"tenant {args.tenant} registered; API key (shown once): {key}")
    elif args.command == "serve":
        import uvicorn

        from .api import create_app
        uvicorn.run(create_app(root), host=args.host, port=args.port)
    else:
        if args.tenant not in TenantDirectory(root).tenants():
            p.error(f"unknown tenant {args.tenant!r}; register it with the 'tenant' command")
        store = TenantStore(root, args.tenant)
        lme = load_lme(root / "market" / "lme.csv")
        if args.command == "ingest":
            r = store.add_quotes(read_jsonl(args.input), source=Path(args.input).name)
            print(json.dumps(r, indent=2))
        elif args.command == "feedback":
            print(json.dumps(training.feedback(store, lme, read_jsonl(args.input)), indent=2))
        elif args.command == "retrain":
            print(json.dumps(training.retrain(store, lme, args.force, args.trials), indent=2, default=str))
        else:
            if args.command == "train":
                report = training.train_tenant(store, lme, args.trials, args.target, seed=args.seed)
                print(f"registered model v{report['version']} (tuning CV MAPE {report['tuning_cv_mape']:.3f})")
            else:
                df = training.load_dataset(store, lme)
                train, test = training.time_split(df, 0.2)
                params, _ = training.tune(train, args.trials, args.target, seed=args.seed)
                report, _ = training.holdout_report(train, test, params, args.target, args.seed)
            if args.truth:
                report["noise_floor"] = noise_floor(report, store, lme, args.truth)
            print(summary(report))
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        store.close()


if __name__ == "__main__":
    main()
