"""Compare models on the same folds and seeds, then on an out-of-time holdout.

The last months form the holdout; the earlier months (development period) are used for
repeated stratified k-fold CV. Model selection uses imbalanced accuracy; differences to
the original model are tested with the Nadeau–Bengio corrected resampled t-test.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import schema
from metrics import best_threshold, corrected_ttest, decision_table, evaluate, reliability
from models import MODELS, build

REFERENCE = 'hgb_balanced'
SUMMARY = ['imbalanced_accuracy', 'tpr', 'tnr', 'balanced_accuracy', 'g_mean', 'precision',
           'npv', 'f1', 'mcc', 'kappa', 'roc_auc', 'gini', 'ks', 'pr_auc', 'pr_auc_baseline',
           'brier', 'brier_skill', 'log_loss', 'ece', 'mean_predicted', 'observed_rate']


def load(path: str, true_prob: str | None = None) -> pd.DataFrame:
    df = schema.add_target(schema.validate(pd.read_csv(path), outcomes=True))
    if true_prob:
        df['true_probability'] = pd.read_csv(true_prob)['true_probability'].to_numpy()
    return df


def time_split(df: pd.DataFrame, months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development period and the last `months` months as holdout."""
    month = pd.to_datetime(df[schema.DATE]).dt.to_period('M')
    start = month.max() - months + 1
    return df[month < start].reset_index(drop=True), df[month >= start].reset_index(drop=True)


def oracle(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    """The simulated true probabilities: an upper bound for any model."""
    t = best_threshold(train[schema.TARGET], train['true_probability'])
    return evaluate(test[schema.TARGET], test['true_probability'], t)


def cross_validate(dev: pd.DataFrame, names: list[str], seeds: list[int],
                   folds: int) -> pd.DataFrame:
    rows = []
    y = dev[schema.TARGET].to_numpy()
    for seed in seeds:
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(dev, y)):
            train, test = dev.iloc[tr], dev.iloc[te]
            for name in names:
                start = time.perf_counter()
                model = build(name, seed).fit(train, train[schema.TARGET])
                seconds = time.perf_counter() - start
                p = model.predict_proba(test)[:, 1]
                rows.append({'model': name, 'seed': seed, 'fold': fold, 'fit_seconds': seconds,
                             **evaluate(test[schema.TARGET], p, model.threshold_)})
                print(f"seed {seed} fold {fold} {name:13s} "
                      f"imacc {rows[-1]['imbalanced_accuracy']:.3f}", flush=True)
            if 'true_probability' in dev:
                rows.append({'model': 'oracle', 'seed': seed, 'fold': fold,
                             **oracle(train, test)})
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and two standard deviations over folds × seeds; fit time relative to fastest."""
    g = folds.groupby('model', sort=False)
    out = g[SUMMARY].mean().add_suffix('_mean').join(g[SUMMARY].std().mul(2).add_suffix('_2sd'))
    fit = g['fit_seconds'].mean()
    out['fit_time_ratio'] = fit / fit[fit > 0].min()
    return out.reset_index()


def compare(folds: pd.DataFrame, n_train: float, n_test: float, reference: str = REFERENCE,
            metrics=('imbalanced_accuracy', 'roc_auc', 'pr_auc', 'log_loss')):
    """Paired differences to the reference model over the same folds × seeds."""
    rows = []
    ref = folds[folds.model == reference].set_index(['seed', 'fold'])
    for name in folds.model.unique():
        if name in (reference, 'oracle'):
            continue
        other = folds[folds.model == name].set_index(['seed', 'fold'])
        for m in metrics:
            diff, t, p = corrected_ttest(other[m] - ref[m], n_train, n_test)
            rows.append({'model': name, 'vs': reference, 'metric': m, 'mean_diff': diff,
                         't': t, 'p_value': p})
    return pd.DataFrame(rows)


def holdout(dev: pd.DataFrame, test: pd.DataFrame, names: list[str],
            seed: int = 0) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, probs = [], {}
    for name in names:
        model = build(name, seed).fit(dev, dev[schema.TARGET])
        probs[name] = model.predict_proba(test)[:, 1]
        rows.append({'model': name, **evaluate(test[schema.TARGET], probs[name], model.threshold_)})
    if 'true_probability' in dev:
        rows.append({'model': 'oracle', **oracle(dev, test)})
    return pd.DataFrame(rows), probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='../data/mock_loans.csv')
    parser.add_argument('--true-prob', default=None,
                        help='simulated true probabilities, for the oracle row')
    parser.add_argument('--output', default='../results')
    parser.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--holdout-months', type=int, default=3)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dev, test = time_split(load(args.input, args.true_prob), args.holdout_months)
    print(f'development {len(dev)} loans, holdout {len(test)} loans')

    folds = cross_validate(dev, args.models, args.seeds, args.folds)
    folds.to_csv(out / 'cv_folds.csv', index=False)
    summary = summarise(folds)
    summary.to_csv(out / 'cv_summary.csv', index=False)
    if REFERENCE in args.models:
        n_test = len(dev) / args.folds
        compare(folds, len(dev) - n_test, n_test).to_csv(out / 'cv_tests.csv', index=False)

    # the model with the best CV imbalanced accuracy gets the decision table
    ranked = summary[summary.model != 'oracle'].sort_values('imbalanced_accuracy_mean')
    best = ranked.model.iloc[-1]
    scores, probs = holdout(dev, test, args.models)
    scores.to_csv(out / 'holdout.csv', index=False)
    y, loss = test[schema.TARGET], test['outstanding_after_grace']
    decision_table(y, probs[best], loss).assign(model=best).to_csv(
        out / 'decision_table.csv', index=False)
    reliability(y, probs[best]).assign(model=best).to_csv(out / 'reliability.csv', index=False)

    cols = ['imbalanced_accuracy_mean', 'imbalanced_accuracy_2sd', 'roc_auc_mean', 'pr_auc_mean',
            'brier_skill_mean', 'ece_mean', 'fit_time_ratio']
    print(summary.set_index('model')[cols].round(3).to_string())
    print(f'selected: {best}')
    print(scores.set_index('model')[['imbalanced_accuracy', 'roc_auc', 'pr_auc', 'brier_skill',
                                     'mean_predicted', 'observed_rate']].round(3).to_string())


if __name__ == '__main__':
    main()
