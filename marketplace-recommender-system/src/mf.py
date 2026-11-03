"""Matrix factorisation for implicit feedback: iALS (numpy) and BPR (PyTorch)."""

import numpy as np
import torch
from torch import nn

from training import batches, set_seed, train


class IALS:
    """Implicit ALS (Hu, Koren, Volinsky, 2008): confidence 1 + alpha * event weight."""

    def __init__(self, factors=64, reg=1.0, alpha=5.0, iterations=15, seed=0):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    @staticmethod
    def _solve(C, other, reg):
        """One half-step: solve each row's weighted least squares against the fixed factors."""
        k = other.shape[1]
        gram = other.T @ other + reg * np.eye(k)
        out = np.zeros((C.shape[0], k))
        for r in range(C.shape[0]):
            idx = C.indices[C.indptr[r]:C.indptr[r + 1]]
            if len(idx) == 0:
                continue
            conf = C.data[C.indptr[r]:C.indptr[r + 1]]
            Y = other[idx]
            A = gram + (Y.T * (conf - 1)) @ Y
            out[r] = np.linalg.solve(A, Y.T @ conf)
        return out

    def fit(self, split):
        rng = np.random.default_rng(self.seed)
        C = split.X.copy().astype(np.float64)
        C.data = 1 + self.alpha * C.data
        Ct = C.T.tocsr()
        self.U = rng.normal(0, 0.01, (split.n_users, self.factors))
        self.V = rng.normal(0, 0.01, (split.n_items, self.factors))
        for _ in range(self.iterations):
            self.U = self._solve(C, self.V, self.reg)
            self.V = self._solve(Ct, self.U, self.reg)
        return self

    def score(self, users):
        return self.U[users] @ self.V.T


class _BPRNet(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)


class BPR:
    """Bayesian personalised ranking (Rendle et al., 2009): positive above a random negative."""

    def __init__(self, dim=64, lr=5e-3, weight_decay=1e-5, batch_size=2048, epochs=60, seed=0):
        self.dim, self.lr, self.weight_decay = dim, lr, weight_decay
        self.batch_size, self.epochs, self.seed = batch_size, epochs, seed

    def fit(self, split, val=None, log=print):
        set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        coo = split.X.tocoo()
        users, pos = torch.as_tensor(coo.row), torch.as_tensor(coo.col)
        self.net = _BPRNet(split.n_users, split.n_items, self.dim)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def run_epoch():
            self.net.train()
            total = 0.0
            for b in batches(len(users), self.batch_size, rng):
                u, i = users[b], pos[b]
                j = torch.randint(split.n_items, (len(b),))
                eu = self.net.user(u)
                diff = (eu * (self.net.item(i) - self.net.item(j))).sum(-1)
                loss = -nn.functional.logsigmoid(diff).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            return total / len(users)

        return train(self, run_epoch, self.epochs, val, log=log)

    @torch.no_grad()
    def score(self, users):
        u = self.net.user(torch.as_tensor(users))
        return (u @ self.net.item.weight.T).numpy()
