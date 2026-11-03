"""Ablation of three choices, with gradient boosting on the same folds and seeds.

1. weights:  class-balanced vs unweighted training; raw, isotonic or sigmoid (Platt)
             calibrated scores; fixed 0.5 cut-off vs the imbalanced-accuracy threshold
             (chosen on the held-out 20 % in all other variants);
2. encoding: merchant default rates computed in sample vs cross-fitted;
3. split:    random split of all loans vs out-of-time holdout of the last months.
"""
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import schema
from evaluate import load, time_split
from metrics import evaluate
from models import CalibratedClassifier, build

COLS = ['imbalanced_accuracy', 'roc_auc', 'pr_auc', 'brier_skill', 'log_loss', 'ece',
        'mean_predicted', 'observed_rate', 'flagged']
WEIGHTS: dict[str, dict[str, Any]] = {
    'balanced, raw, cut-off 0.5': {'name': 'hgb_balanced', 'calibration': None, 'threshold': 0.5},
    'balanced, raw': {'name': 'hgb_balanced', 'calibration': None},
    'balanced, isotonic': {'name': 'hgb_balanced', 'calibration': 'isotonic'},
    'balanced, sigmoid': {'name': 'hgb_balanced'},
    'unweighted, raw': {'name': 'hgb', 'calibration': None},
    'unweighted, isotonic': {'name': 'hgb', 'calibration': 'isotonic'},
    'unweighted, sigmoid': {'name': 'hgb'},
}


def score(model: CalibratedClassifier, df: pd.DataFrame, **extra: object) -> dict:
    p = model.predict_proba(df)[:, 1]
    return {**extra, **{k: v for k, v in evaluate(df[schema.TARGET], p, model.threshold_).items()
                        if k in COLS}}


def run(df: pd.DataFrame, seeds: list[int], folds: int, months: int) -> pd.DataFrame:
    dev, test = time_split(df, months)
    rows = []
    for seed in seeds:
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(dev, dev[schema.TARGET])):
            train, valid = dev.iloc[tr], dev.iloc[te]
            ids = {'seed': seed, 'fold': fold}
            for variant, kw in WEIGHTS.items():
                kw = {**kw}
                m = build(kw.pop('name'), seed, **kw).fit(train, train[schema.TARGET])
                rows.append(score(m, valid, study='weights', variant=variant, part='test', **ids))
            for enc in ['in_sample', 'cross_fitted']:
                m = build('hgb', seed, encoder=enc).fit(train, train[schema.TARGET])
                rows.append(score(m, train, study='encoding', variant=enc, part='train', **ids))
                rows.append(score(m, valid, study='encoding', variant=enc, part='test', **ids))
            print(f'seed {seed} fold {fold} done', flush=True)

        # random 80/20 split of all months vs training on the earlier months only
        train, valid = train_test_split(df, test_size=len(test) / len(df),
                                        stratify=df[schema.TARGET], random_state=seed)
        m = build('hgb', seed).fit(train, train[schema.TARGET])
        rows.append(score(m, valid, study='split', variant='random', part='test', seed=seed))
        m = build('hgb', seed).fit(dev, dev[schema.TARGET])
        rows.append(score(m, test, study='split', variant='out-of-time', part='test', seed=seed))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='../data/mock_loans.csv')
    parser.add_argument('--output', default='../results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--holdout-months', type=int, default=3)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = run(load(args.input), args.seeds, args.folds, args.holdout_months)
    rows.to_csv(out / 'ablation.csv', index=False)
    g = rows.groupby(['study', 'variant', 'part'], sort=False)[COLS]
    summary = g.mean().round(4).join(g.std().mul(2).round(4).add_suffix('_2sd'))
    summary.to_csv(out / 'ablation_summary.csv')
    pd.set_option('display.width', 200)
    print(summary[COLS].round(3).to_string())


if __name__ == '__main__':
    main()
