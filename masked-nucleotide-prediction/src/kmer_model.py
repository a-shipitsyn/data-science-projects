"""k-mer context model: base frequencies between k left and k right neighbours, for several k.

Short contexts (k <= DENSE_MAX_K) are counted into dense tables in one parallel numba pass.
Longer contexts are too many for a dense table (k = 10 has 4 ** 20 = 10 ** 12 possible values)
but only a few million occur, so they are stored as sorted arrays (parallel bucket sort) and
looked up by binary search.

Each order is smoothed towards the one below it:

    p_0 = overall base frequencies
    p_k = (counts_k + alpha * p_previous) / (n_k + alpha)

A context never seen in training (n_k = 0) keeps the estimate of the shorter context; a
frequent one is dominated by its own counts. Each example uses only the orders k that fit
into the shorter of its two context lengths.
"""

import numpy as np

from counting import context_codes, count_all_parallel, count_sparse_parallel, lookup_sparse
from genome import Genome
from windows import CENTRE

DENSE_MAX_K = 5
ORDERS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)


class KmerModel:
    def __init__(self, orders: tuple[int, ...] = ORDERS, alpha: float = 8.0):
        self.orders, self.alpha = tuple(sorted(orders)), alpha
        self.dense: dict[int, np.ndarray] = {}
        self.sparse: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.prior = np.full(4, 0.25)

    def fit(self, genome: Genome) -> "KmerModel":
        dense_orders = [k for k in self.orders if k <= DENSE_MAX_K]
        if dense_orders:
            tables = count_all_parallel(genome.bases, max(dense_orders))
            self.dense = {k: tables[k - 1] for k in dense_orders}
        for k in self.orders:
            if k > DENSE_MAX_K:
                self.sparse[k] = count_sparse_parallel(genome.bases, k)
        totals = np.bincount(genome.bases[genome.bases < 4], minlength=4).astype(float)
        self.prior = totals / totals.sum()
        return self

    def counts(self, k: int, codes: np.ndarray) -> np.ndarray:
        if k in self.dense:
            return self.dense[k][codes].astype(np.float64)
        return lookup_sparse(*self.sparse[k], codes)

    def predict_proba(self, windows: np.ndarray, lengths: np.ndarray, max_k: int | None = None,
                      alpha: float | None = None) -> np.ndarray:
        """Probabilities of A, C, G, T; max_k and alpha can be changed for tuning without refitting."""
        alpha = self.alpha if alpha is None else alpha
        shortest = lengths.min(axis=1)
        proba = np.tile(self.prior, (len(windows), 1))
        for k in self.orders:
            if max_k is not None and k > max_k:
                break
            rows = np.flatnonzero(shortest >= k)
            counts = self.counts(k, context_codes(windows[rows], CENTRE, k))
            proba[rows] = (counts + alpha * proba[rows]) / (counts.sum(axis=1, keepdims=True) + alpha)
        return proba

    @property
    def n_contexts(self) -> dict[int, int]:
        return {k: int((t.sum(axis=1) > 0).sum()) for k, t in self.dense.items()} | \
               {k: len(c) for k, (c, _) in self.sparse.items()}
