"""Non-neural recommenders: recent popularity, item-KNN, EASE, and the oracle on the true utility.

Each model has fit(split) and score(user_indices) -> (n, n_items) array.
"""

import numpy as np

from simulate import true_utility


class Popularity:
    """Number of businesses that interacted with a listing in the last `window` days."""

    def __init__(self, window=30):
        self.window = window

    def fit(self, split):
        recent = split.train[split.train['ts'] >= split.cutoff - self.window]
        pop = np.zeros(split.n_items)
        pairs = recent[['u', 'i']].drop_duplicates()
        np.add.at(pop, pairs['i'].values, 1)
        self.pop = pop
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))


class ItemKNN:
    """Cosine similarity between listings' interaction columns, `k` nearest neighbours kept."""

    def __init__(self, k=100):
        self.k = k

    def fit(self, split):
        X = (split.X > 0).astype(np.float32)
        self.X = X
        norms = np.sqrt(np.asarray(X.sum(axis=0))).ravel() + 1e-9
        S = (X.T @ X).toarray() / np.outer(norms, norms)
        np.fill_diagonal(S, 0)
        drop = np.argpartition(-S, self.k, axis=1)[:, self.k:]
        np.put_along_axis(S, drop, 0, axis=1)
        self.S = S
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.S)


class EASE:
    """Embarrassingly shallow autoencoder (Steck, 2019): closed-form item-item weights."""

    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, split):
        X = (split.X > 0).astype(np.float64)
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0)
        self.X, self.B = X, B
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.B)


class Oracle:
    """Ranks by the true static utility of the simulation: a reference, not a model."""

    def __init__(self, truth):
        self.truth = truth

    def fit(self, split):
        self.split = split
        return self

    def score(self, users):
        return true_utility(self.split.users.iloc[users], self.split.items,
                            {**self.truth, 'user_vec': self.truth['user_vec'][users],
                             'second_city': self.truth['second_city'][users]})
