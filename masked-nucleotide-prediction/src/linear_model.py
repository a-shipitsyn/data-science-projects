"""Multinomial logistic regression on sparse one-hot features of the context window.

Two feature sets:
- "positions": the base at each of the 60 context positions, one-hot (unknown positions are all zero);
- "positions + k-mers": also the joint code of the 1, 2 and 3 nearest neighbours on each side,
  one-hot, so the linear model can use interactions close to the gap.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from counting import context_codes
from genome import SEP
from windows import CENTRE, WIDTH

KMER_ORDERS = (1, 2, 3)


def features(windows: np.ndarray, lengths: np.ndarray, kmers: bool) -> sp.csr_matrix:
    n = len(windows)
    known = windows != SEP
    rows, cols = np.nonzero(known)
    blocks = [sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols * 4 + windows[rows, cols])),
                            shape=(n, WIDTH * 4))]
    if kmers:
        shortest = lengths.min(axis=1)
        for k in KMER_ORDERS:
            size = 4 ** (2 * k)
            present = np.flatnonzero(shortest >= k)
            codes = context_codes(windows[present], CENTRE, k)
            blocks.append(sp.csr_matrix((np.ones(len(present), dtype=np.float32), (present, codes)), shape=(n, size)))
    return sp.hstack(blocks, format="csr")


class LinearModel:
    def __init__(self, kmers: bool = True, C: float = 1.0):
        self.kmers, self.C = kmers, C
        self.model = LogisticRegression(C=C, max_iter=1000)

    def fit(self, windows: np.ndarray, targets: np.ndarray, lengths: np.ndarray) -> "LinearModel":
        self.model.fit(features(windows, lengths, self.kmers), targets)
        return self

    def predict_proba(self, windows: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features(windows, lengths, self.kmers))
