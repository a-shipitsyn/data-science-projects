"""Train the final model and save it with metadata for the API and monitoring."""
import argparse
import json
import pickle
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import schema
from evaluate import load
from features import selected_features
from models import MODELS, build
from monitor import profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='../data/mock_loans.csv')
    parser.add_argument('--model', default='hgb', choices=MODELS)
    parser.add_argument('--params', default=None,
                        help='JSON file with hyperparameters, e.g. ../results/tuned_params.json')
    parser.add_argument('--before', default=None,
                        help='train only on loans issued before this date')
    parser.add_argument('--output', default='../models/model.pkl')
    parser.add_argument('--scores', default='../results/cv_summary.csv',
                        help='CV summary to store with the model, if it exists')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    df = load(args.input)
    dates = pd.to_datetime(df[schema.DATE])
    if args.before:
        df, dates = df[dates < pd.Timestamp(args.before)], dates[dates < pd.Timestamp(args.before)]
    params = json.loads(Path(args.params).read_text()) if args.params else None
    model = build(args.model, args.seed, params=params).fit(df, df[schema.TARGET])

    scores = None
    if Path(args.scores).exists():
        summary = pd.read_csv(args.scores).set_index('model')
        if args.model in summary.index:
            scores = summary.loc[args.model].round(4).to_dict()

    meta = {
        'model': args.model,
        'params': params,
        'trained_at': datetime.now(UTC).isoformat(timespec='seconds'),
        'seed': args.seed,
        'n_loans': len(df),
        'issue_dates': [str(dates.min().date()), str(dates.max().date())],
        'default_rate': round(float(df[schema.TARGET].mean()), 4),
        'threshold': model.threshold_,
        'features': selected_features(model.estimator_),
        'cv_scores': scores,
        'versions': {'python': platform.python_version(),
                     **{p: version(p) for p in ['scikit-learn', 'numpy', 'pandas', 'torch']}},
        'profile': profile(df),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({'model': model, 'meta': meta}, f)
    print(f"{args.model}: {len(df)} loans, threshold {model.threshold_:.4f}, "
          f"{len(meta['features'])} features -> {args.output}")


if __name__ == '__main__':
    main()
