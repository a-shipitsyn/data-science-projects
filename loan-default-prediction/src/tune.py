"""Random search over gradient-boosting hyperparameters.

Candidates are scored by imbalanced accuracy in stratified k-fold CV on the development
period (each candidate calibrated and thresholded inside its training folds). The best
CV score is optimistic, since it is selected on the same folds, so the tuned and the
default model are then compared on the untouched out-of-time holdout.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

import schema
from evaluate import load, time_split
from metrics import evaluate, imbalanced_accuracy
from models import build

SPACE = {
    'learning_rate': loguniform(0.02, 0.3),
    'max_depth': [2, 3, 4, 5, 6, None],
    'max_leaf_nodes': randint(4, 32),
    'min_samples_leaf': randint(10, 300),
    'l2_regularization': loguniform(1e-3, 10),
    'max_iter': [100, 200, 400],
}
PREFIX = 'estimator__model__'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='../data/mock_loans.csv')
    parser.add_argument('--output', default='../results')
    parser.add_argument('--model', default='hgb', choices=['hgb', 'hgb_balanced'])
    parser.add_argument('--n-iter', type=int, default=30)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--holdout-months', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dev, test = time_split(load(args.input), args.holdout_months)
    y = dev[schema.TARGET]
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    scorer = make_scorer(imbalanced_accuracy)

    search = RandomizedSearchCV(
        build(args.model, args.seed), {PREFIX + k: v for k, v in SPACE.items()},
        n_iter=args.n_iter, scoring=scorer, cv=cv, refit=False,
        random_state=args.seed, verbose=1)
    search.fit(dev, y)
    results = pd.DataFrame(search.cv_results_)
    results.columns = [c.replace('param_' + PREFIX, '') for c in results.columns]
    keep = list(SPACE) + ['mean_test_score', 'std_test_score', 'mean_fit_time']
    results[keep].sort_values('mean_test_score', ascending=False).to_csv(
        out / 'tuning.csv', index=False)

    best = {k.replace(PREFIX, ''): v for k, v in search.best_params_.items()}
    best = {k: (v.item() if hasattr(v, 'item') else v) for k, v in best.items()}
    (out / 'tuned_params.json').write_text(json.dumps(best, indent=2) + '\n')
    default_cv = cross_val_score(build(args.model, args.seed), dev, y, scoring=scorer, cv=cv)

    rows = []
    for label, params in [('default', None), ('tuned', best)]:
        model = build(args.model, args.seed, params=params).fit(dev, y)
        p = model.predict_proba(test)[:, 1]
        cv_mean = default_cv.mean() if params is None else search.best_score_
        rows.append({'config': label, 'cv_imbalanced_accuracy': cv_mean,
                     **evaluate(test[schema.TARGET], p, model.threshold_)})
    holdout = pd.DataFrame(rows)
    holdout.to_csv(out / 'tuning_holdout.csv', index=False)

    print('best parameters:', best)
    print(holdout.set_index('config')[['cv_imbalanced_accuracy', 'imbalanced_accuracy',
                                       'roc_auc', 'pr_auc', 'brier_skill', 'log_loss']]
          .round(4).to_string())


if __name__ == '__main__':
    main()
