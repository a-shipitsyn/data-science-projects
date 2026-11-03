"""Tune, train and compare all recommenders on a time-based split.

Validation phase: train on events before --val-start, choose hyperparameters (and epochs for
neural models) by NDCG@10 of returning businesses in [--val-start, --test-start).
Test phase: retrain each model with its chosen settings on events before --test-start and
evaluate on [--test-start, --end): returning businesses, new businesses (no history) and new
listings (published in the test window).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baselines import EASE, ItemKNN, Oracle, Popularity
from data import build_split, load
from lightgcn import LightGCN
from metrics import evaluate
from mf import BPR, IALS
from sasrec import SASRec
from two_tower import TwoTower

# name -> (class, grid of settings to try on the validation window)
MODELS = {
    'Popularity': (Popularity, [{'window': 7}, {'window': 30}]),
    'Item-KNN': (ItemKNN, [{'k': 50}, {'k': 200}]),
    'EASE': (EASE, [{'reg': 30.0}, {'reg': 100.0}, {'reg': 300.0}, {'reg': 1000.0}]),
    'iALS': (IALS, [{'reg': 1.0, 'alpha': 5.0}, {'reg': 10.0, 'alpha': 5.0},
                    {'reg': 1.0, 'alpha': 20.0}]),
    'BPR': (BPR, [{}]),
    'LightGCN': (LightGCN, [{'epochs': 100}]),
    'SASRec': (SASRec, [{'epochs': 60}]),
    'Two-tower': (TwoTower, [{}]),
}
NEURAL = {'BPR', 'LightGCN', 'SASRec', 'Two-tower'}
COLD_START = {'Popularity', 'Two-tower', 'Oracle'}  # can rank for businesses without history


def fit(model, split, val=None):
    if isinstance(model, (BPR, LightGCN, SASRec, TwoTower)):
        return model.fit(split, val=val, log=lambda s: print(s, flush=True))
    return model.fit(split)


def bootstrap_ci(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n, len(values))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def tune(name, cls, grid, val_split, seed):
    warm = val_split.eval_users(warm=True)
    best = None
    for params in grid:
        model = cls(**params, seed=seed) if name in NEURAL or name == 'iALS' else cls(**params)
        fit(model, val_split, val_split if name in NEURAL else None)
        ndcg = evaluate(model.score, val_split, warm, ks=(10,))['ndcg@10']
        print(f'{name} {params} val ndcg@10 {ndcg:.4f}', flush=True)
        if best is None or ndcg > best[1]:
            if name in NEURAL:
                params = {**params, 'epochs': model.best_epoch}
            best = (params, ndcg)
    return best


def plot(results, path):
    """Three panels: NDCG@10 of returning businesses, of new businesses, recall@10 on new listings."""
    panels = [('returning', 'ndcg@10', 'Returning businesses\nNDCG@10'),
              ('new_users', 'ndcg@10', 'New businesses\nNDCG@10'),
              ('returning', 'new_recall@10', 'New listings\nRecall@10')]
    names = list(dict.fromkeys(results['model']))
    fig, axes = plt.subplots(1, 3, figsize=(10, 0.42 * len(names) + 1.2), sharey=True)
    fig.patch.set_facecolor('#fcfcfb')
    for ax, (segment, metric, title) in zip(axes, panels):
        sub = results[results['segment'] == segment].set_index('model')
        y = np.arange(len(names))
        vals = [sub[metric].get(n, np.nan) for n in names]
        colors = ['#898781' if n == 'Oracle' else '#2a78d6' for n in names]
        ax.barh(y, np.nan_to_num(vals), height=0.6, color=colors)
        for yi, v in zip(y, vals):
            label = '–' if np.isnan(v) else f'{v:.2f}'
            ax.text((0 if np.isnan(v) else v) + 0.01, yi, label, va='center', fontsize=8,
                    color='#52514e')
        ax.set_title(title, fontsize=10, color='#0b0b0b', loc='left')
        ax.set_facecolor('#fcfcfb')
        ax.set_xlim(0, max(0.6, np.nanmax(vals) * 1.2))
        ax.xaxis.grid(True, color='#e1e0d9', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors='#898781', labelsize=8, length=0)
        for side in ('top', 'right', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.spines['left'].set_color('#c3c2b7')
    axes[0].set_yticks(np.arange(len(names)), names, fontsize=9, color='#0b0b0b')
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', default='../data')
    parser.add_argument('--output', default='../results')
    parser.add_argument('--val-start', type=float, default=240)
    parser.add_argument('--test-start', type=float, default=300)
    parser.add_argument('--end', type=float, default=365)
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    parser.add_argument('--truth', help='truth.npz of the simulation: adds the oracle')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2, help='CPU threads for PyTorch')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    users, items, events = load(args.data)
    val_split = build_split(users, items, events, args.val_start, args.test_start)
    test_split = build_split(users, items, events, args.test_start, args.end)
    segments = {'returning': test_split.eval_users(warm=True),
                'new_users': test_split.eval_users(warm=False)}
    print(f'test: {len(segments["returning"])} returning, {len(segments["new_users"])} new '
          f'businesses, {test_split.new_items[test_split.candidates].sum()} new of '
          f'{test_split.candidates.sum()} available listings', flush=True)

    rows, chosen = [], {}
    runs = [(name, *MODELS[name]) for name in args.models]
    if args.truth:
        runs.append(('Oracle', None, None))
    for name, cls, grid in runs:
        start = time.time()
        if name == 'Oracle':
            model, params = Oracle(dict(np.load(args.truth))), {}
        else:
            params, _ = tune(name, cls, grid, val_split, args.seed)
            seeded = name in NEURAL or name == 'iALS'
            model = cls(**params, seed=args.seed) if seeded else cls(**params)
        fit(model, test_split)
        chosen[name] = params
        for segment, seg_users in segments.items():
            if segment == 'new_users' and name not in COLD_START:
                continue
            res = evaluate(model.score, test_split, seg_users, per_user=True)
            lo, hi = bootstrap_ci(res.pop('per_user_ndcg@10'), seed=args.seed)
            rows.append({'model': name, 'segment': segment, **res,
                         'ndcg@10_lo': lo, 'ndcg@10_hi': hi,
                         'seconds': round(time.time() - start)})
        print(f'{name} done: {params}', flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out / 'results.csv', index=False)
    (out / 'settings.json').write_text(json.dumps(chosen, indent=2))
    cols = ['model', 'recall@10', 'ndcg@10', 'ndcg@10_lo', 'ndcg@10_hi', 'recall@50',
            'coverage@10', 'new_recall@10']
    for segment in segments:
        print(f'\n{segment}:\n' + results.loc[results['segment'] == segment, cols]
              .round(3).to_string(index=False))
    plot(results, out / 'comparison.png')


if __name__ == '__main__':
    main()
