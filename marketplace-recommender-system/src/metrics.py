"""Ranking metrics: Recall@K, NDCG@K, catalogue coverage and recall on new listings.

Recall@K is normalised by min(K, number of targets), so a user with many targets can reach 1.
Listings seen before the cutoff and listings not available in the window are never ranked.
"""

import numpy as np


def top_k(scores, k):
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def evaluate(score_fn, split, users, ks=(10, 50), batch_size=1024, per_user=False):
    """Score `users` with score_fn(user_indices) -> (n, n_items) array and average the metrics.

    With per_user=True the result also holds the array of each user's NDCG at the first K.
    """
    kmax = max(ks)
    discount = 1 / np.log2(np.arange(2, kmax + 2))
    sums = {f'{m}@{k}': 0.0 for k in ks for m in ('recall', 'ndcg')}
    new_hits, new_total = 0, 0
    recommended = set()
    user_ndcg = []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.asarray(score_fn(batch), dtype=np.float64).copy()
        scores[:, ~split.candidates] = -np.inf
        rows = split.X[batch]
        scores[np.repeat(np.arange(len(batch)), np.diff(rows.indptr)), rows.indices] = -np.inf
        ranked = top_k(scores, kmax)
        for u, rec in zip(batch, ranked):
            target = split.targets[u]
            hit = np.isin(rec, target)
            for k in ks:
                n = min(k, len(target))
                sums[f'recall@{k}'] += hit[:k].sum() / n
                ndcg = (hit[:k] * discount[:k]).sum() / discount[:n].sum()
                sums[f'ndcg@{k}'] += ndcg
                if k == ks[0]:
                    user_ndcg.append(ndcg)
            new_target = target[split.new_items[target]]
            new_hits += np.isin(rec[:ks[0]], new_target).sum()
            new_total += min(ks[0], len(new_target))
            recommended.update(rec[:ks[0]].tolist())
    out = {name: s / len(users) for name, s in sums.items()}
    out[f'coverage@{ks[0]}'] = len(recommended) / split.candidates.sum()
    out[f'new_recall@{ks[0]}'] = new_hits / new_total if new_total else np.nan
    if per_user:
        out[f'per_user_ndcg@{ks[0]}'] = np.array(user_ndcg)
    return out
