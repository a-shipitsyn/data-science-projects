"""Two-stage recommender: retrieve candidates, then rerank them with gradient boosting.

Stage 1 retrieves the two-tower top `n_tower` listings for every business, plus the EASE top
`n_ease` for businesses with history. Stage 2 reranks the candidates with LightGBM LambdaRank on
features a dot product cannot express: rent against budget, city match, the business's history
by property type, city and landlord, listing popularity and age, and the retrieval scores.

The ranker learns on the validation window from stage-1 models trained before it; stage 1 is
then retrained before the test window, so the test window is never used for training.
Evaluates on the test window and saves the serving model (stage 1 retrained on all events).
"""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

# PyTorch and LightGBM each bring an OpenMP runtime: on macOS importing lightgbm first crashes
# and LightGBM with several threads hangs, so torch is imported first and the ranker uses 1 thread
import torch

# isort: split
import lightgbm as lgb

from baselines import EASE
from data import build_split, load
from metrics import evaluate
from two_tower import TwoTower

FEATURES = ['tower_score', 'tower_rank', 'ease_score', 'ease_rank', 'popularity', 'item_age',
            'city_match', 'log_rent_budget', 'log_area', 'log_footfall', 'segment', 'property_type',
            'history_len', 'history_type_share', 'history_city_share', 'history_rent_diff',
            'history_landlord']
CATEGORICAL = ['segment', 'property_type']
NO_RANK = 1000  # rank given to a listing the source did not retrieve


def one_hot(codes, n):
    out = np.zeros((len(codes), n))
    out[np.arange(len(codes)), codes] = 1
    return out


class TwoStage:
    def __init__(self, tower_epochs=2, ease_reg=100.0, n_tower=200, n_ease=100, seed=0):
        self.tower_epochs, self.ease_reg = tower_epochs, ease_reg
        self.n_tower, self.n_ease, self.seed = n_tower, n_ease, seed

    # stage 1
    def fit_retrieval(self, split, log=print):
        self.split = split
        self.tower = TwoTower(epochs=self.tower_epochs, seed=self.seed).fit(split, log=log)
        self.ease = EASE(self.ease_reg).fit(split)
        self._history_stats()
        return self

    def _history_stats(self):
        """Per-business shares of history by type, city and landlord, mean log rent, popularity."""
        s = self.split
        X = (s.X > 0).astype(np.float64)
        self.history_len = np.asarray(X.sum(axis=1)).ravel()
        n = np.maximum(self.history_len, 1)[:, None]
        self.type_code = pd.Categorical(s.items['property_type']).codes
        self.city_code = pd.Categorical(s.items['city'], categories=sorted(s.items['city'].unique())).codes
        self.landlord_code = pd.Categorical(s.items['landlord_id']).codes
        self.type_share = X @ one_hot(self.type_code, self.type_code.max() + 1) / n
        self.city_share = X @ one_hot(self.city_code, self.city_code.max() + 1) / n
        self.landlord_count = X @ one_hot(self.landlord_code, self.landlord_code.max() + 1)
        self.log_rent = np.log(s.items['rent'].values)
        self.history_rent = np.where(self.history_len > 0, (X @ self.log_rent) / n.ravel(), np.nan)
        recent = s.train[s.train['ts'] >= s.cutoff - 30][['u', 'i']].drop_duplicates()
        self.popularity = np.bincount(recent['i'], minlength=s.n_items)
        user_city = pd.Categorical(s.users['city'], categories=sorted(s.items['city'].unique())).codes
        self.user_city = user_city
        self.segment = pd.Categorical(s.users['segment']).codes

    def candidates(self, users, allowed):
        """Candidate pairs from both sources; `allowed` is a (n_users, n_items) bool mask."""
        tower = np.where(allowed, self.tower.score(users), -np.inf)
        ease = np.where(allowed, self.ease.score(users), -np.inf)
        warm = self.history_len[users] > 0
        rows = []
        for r, u in enumerate(users):
            t_top = np.argsort(-tower[r])[:self.n_tower]
            e_top = np.argsort(-ease[r])[:self.n_ease] if warm[r] else np.array([], dtype=int)
            items = np.union1d(t_top, e_top)
            items = items[np.isfinite(tower[r, items])]
            t_rank = np.full(self.split.n_items, NO_RANK)
            t_rank[t_top] = np.arange(len(t_top))
            e_rank = np.full(self.split.n_items, NO_RANK)
            e_rank[e_top] = np.arange(len(e_top))
            rows.append(pd.DataFrame({'u': u, 'i': items, 'tower_score': tower[r, items],
                                      'tower_rank': t_rank[items],
                                      'ease_score': np.where(warm[r], ease[r, items], np.nan),
                                      'ease_rank': e_rank[items]}))
        return pd.concat(rows, ignore_index=True)

    def features(self, c):
        s, u, i = self.split, c['u'].values, c['i'].values
        items = s.items
        f = c.copy()
        f['popularity'] = self.popularity[i]
        f['item_age'] = s.cutoff - items['publish_day'].values[i]
        f['city_match'] = (self.user_city[u] == self.city_code[i]).astype(int)
        f['log_rent_budget'] = self.log_rent[i] - np.log(s.users['budget'].values[u])
        f['log_area'] = np.log(items['area_m2'].values[i])
        f['log_footfall'] = np.log(items['footfall'].values[i])
        f['segment'] = self.segment[u]
        f['property_type'] = self.type_code[i]
        f['history_len'] = self.history_len[u]
        f['history_type_share'] = self.type_share[u, self.type_code[i]]
        f['history_city_share'] = self.city_share[u, self.city_code[i]]
        f['history_rent_diff'] = self.log_rent[i] - self.history_rent[u]
        f['history_landlord'] = self.landlord_count[u, self.landlord_code[i]]
        return f

    def allowed(self, users, split=None):
        """Listings available in the split's window that the business has not seen before."""
        split = split or self.split
        mask = np.tile(split.candidates, (len(users), 1))
        rows = split.X[users]
        mask[np.repeat(np.arange(len(users)), np.diff(rows.indptr)), rows.indices] = False
        return mask

    # stage 2
    def fit_ranker(self, log=print):
        """Learn the ranker on the window of the split stage 1 was trained on."""
        s = self.split
        users = np.array(sorted(s.targets))
        f = self.features(self.candidates(users, self.allowed(users)))
        targets = {u: set(t) for u, t in s.targets.items()}
        f['label'] = [int(i in targets[u]) for u, i in zip(f['u'], f['i'])]
        f = f[f.groupby('u')['label'].transform('max') > 0]  # queries without a hit teach nothing
        rng = np.random.default_rng(self.seed)
        stop_users = set(rng.choice(f['u'].unique(), f['u'].nunique() // 5, replace=False))
        is_stop = f['u'].isin(stop_users)
        fit, stop = f[~is_stop], f[is_stop]
        self.ranker = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.05,
                                     num_leaves=31, min_child_samples=50, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.8, n_jobs=1,
                                     random_state=self.seed, verbose=-1)
        self.ranker.fit(fit[FEATURES], fit['label'], group=fit.groupby('u', sort=False).size().values,
                        eval_set=[(stop[FEATURES], stop['label'])],
                        eval_group=[stop.groupby('u', sort=False).size().values], eval_at=[10],
                        categorical_feature=CATEGORICAL,
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        log(f'ranker: {len(fit)} pairs, {fit["label"].sum()} positives, '
            f'best iteration {self.ranker.best_iteration_}')
        return self

    def rank(self, users, allowed):
        """Reranked candidate pairs with column 'score'."""
        f = self.features(self.candidates(users, allowed))
        f['score'] = self.ranker.predict(f[FEATURES], num_iteration=self.ranker.best_iteration_)
        return f

    def score(self, users):
        """Dense scores for evaluation: reranked candidates first, everything else below."""
        f = self.rank(users, self.allowed(users))
        out = np.full((len(users), self.split.n_items), -1e9)
        row = pd.Series(np.arange(len(users)), index=users)
        out[row[f['u']].values, f['i'].values] = f['score'].values
        return out

    def candidate_recall(self, users):
        """Share of target listings that stage 1 retrieves (the ceiling for the ranker)."""
        c = self.candidates(users, self.allowed(users))
        got = c.groupby('u')['i'].apply(set)
        hits = [len(got.get(u, set()) & set(self.split.targets[u])) / len(self.split.targets[u])
                for u in users]
        return float(np.mean(hits))

    def save(self, model_dir, meta):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.tower.save(model_dir / 'two_tower.pt')
        self.ranker.booster_.save_model(model_dir / 'ranker.txt',
                                        num_iteration=self.ranker.best_iteration_)
        params = {k: getattr(self, k) for k in ('tower_epochs', 'ease_reg', 'n_tower', 'n_ease', 'seed')}
        (model_dir / 'meta.json').write_text(json.dumps({**meta, 'params': params}, indent=2))

    @classmethod
    def load(cls, model_dir, split):
        """Restore a saved pipeline; EASE is refitted from `split` (closed form, deterministic)."""
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / 'meta.json').read_text())
        model = cls(**meta['params'])
        model.split = split
        model.tower = TwoTower.load(model_dir / 'two_tower.pt', split)
        model.ease = EASE(model.ease_reg).fit(split)
        model._history_stats()
        booster = lgb.Booster(model_file=str(model_dir / 'ranker.txt'))
        model.ranker = _BoosterRanker(booster)
        return model, meta


class _BoosterRanker:
    """A loaded LightGBM booster with the predict() interface of LGBMRanker."""

    def __init__(self, booster):
        self.booster_, self.best_iteration_ = booster, None

    def predict(self, X, num_iteration=None):
        return self.booster_.predict(X, num_iteration=num_iteration)


def quiet(_):
    pass


def bootstrap_ci(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile(rng.choice(values, (n, len(values))).mean(axis=1), [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', default='../data')
    parser.add_argument('--output', default='../results')
    parser.add_argument('--model-dir', default='../models')
    parser.add_argument('--settings', help="run.py's settings.json: two-tower epochs, EASE reg")
    parser.add_argument('--val-start', type=float, default=240)
    parser.add_argument('--test-start', type=float, default=300)
    parser.add_argument('--end', type=float, default=365)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2, help='CPU threads for PyTorch')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    params = {'seed': args.seed}
    if args.settings:
        settings = json.loads(Path(args.settings).read_text())
        params.update(tower_epochs=settings['Two-tower']['epochs'], ease_reg=settings['EASE']['reg'])
    users, items, events = load(args.data)
    val_split = build_split(users, items, events, args.val_start, args.test_start)
    test_split = build_split(users, items, events, args.test_start, args.end)

    model = TwoStage(**params).fit_retrieval(val_split, log=quiet).fit_ranker()
    ranker = model.ranker
    model.fit_retrieval(test_split, log=quiet)
    importance = pd.Series(ranker.booster_.feature_importance('gain'), index=FEATURES)

    rows = []
    for segment, warm in [('returning', True), ('new_users', False)]:
        seg_users = test_split.eval_users(warm=warm)
        for name, score in [('Two-tower', model.tower.score), ('Two-stage', model.score)]:
            res = evaluate(score, test_split, seg_users, per_user=True)
            lo, hi = bootstrap_ci(res.pop('per_user_ndcg@10'), seed=args.seed)
            rows.append({'model': name, 'segment': segment, **res, 'ndcg@10_lo': lo, 'ndcg@10_hi': hi})
        rows[-1]['candidate_recall'] = model.candidate_recall(seg_users)
    results = pd.DataFrame(rows)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / 'two_stage.csv', index=False)
    importance.sort_values(ascending=False).to_csv(out / 'ranker_importance.csv', header=['gain'])
    cols = ['model', 'segment', 'recall@10', 'ndcg@10', 'ndcg@10_lo', 'ndcg@10_hi', 'recall@50',
            'coverage@10', 'new_recall@10', 'candidate_recall']
    print(results[cols].round(3).to_string(index=False))
    print('\nranker feature importance (gain share):')
    print((importance / importance.sum()).sort_values(ascending=False).round(3).to_string())

    # serving model: stage 1 on all events, ranker as learned on the validation window
    all_split = build_split(users, items, events, args.end, args.end)
    model.fit_retrieval(all_split, log=quiet)
    model.ranker = ranker
    meta = {'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'data': str(Path(args.data).resolve().name),
            'retrieval_cutoff': args.end, 'ranker_window': [args.val_start, args.test_start],
            'features': FEATURES, 'test_metrics': results.drop(columns=['ndcg@10_lo', 'ndcg@10_hi'])
            .round(4).to_dict(orient='records'),
            'versions': {'python': platform.python_version(),
                         **{p: version(p) for p in ('torch', 'lightgbm', 'numpy', 'pandas')}}}
    model.save(args.model_dir, meta)
    print(f'\nserving model saved to {args.model_dir}')


if __name__ == '__main__':
    main()
