"""LightGCN (He et al., 2020): embeddings smoothed over the business-listing graph.

Each layer averages neighbours' embeddings with symmetric normalisation and no weights or
non-linearities; the final embedding is the mean over layers. Trained with BPR loss.
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from training import batches, set_seed, train


def normalised_adjacency(X):
    """Symmetrically normalised adjacency of the bipartite graph as a torch sparse tensor."""
    B = (X > 0).astype(np.float32)
    A = sp.bmat([[None, B], [B.T, None]]).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    d = np.power(deg, -0.5, out=np.zeros_like(deg), where=deg > 0)
    A = sp.diags(d) @ A @ sp.diags(d)
    coo = A.tocoo()
    idx = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(coo.data, dtype=torch.float32),
                                   A.shape).coalesce()


class _Net(nn.Module):
    def __init__(self, n_users, n_items, dim, layers, adj):
        super().__init__()
        self.n_users, self.layers, self.adj = n_users, layers, adj
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self):
        x = self.emb.weight
        out = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.adj, x)
            out.append(x)
        h = torch.stack(out).mean(0)
        return h[:self.n_users], h[self.n_users:]


class LightGCN:
    def __init__(self, dim=64, layers=3, lr=5e-3, reg=1e-4, batch_size=2048, epochs=100, seed=0):
        self.dim, self.layers, self.lr, self.reg = dim, layers, lr, reg
        self.batch_size, self.epochs, self.seed = batch_size, epochs, seed

    def fit(self, split, val=None, log=print):
        set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = _Net(split.n_users, split.n_items, self.dim, self.layers,
                        normalised_adjacency(split.X))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        coo = split.X.tocoo()
        users, pos = torch.as_tensor(coo.row), torch.as_tensor(coo.col)

        def run_epoch():
            self.net.train()
            total = 0.0
            for b in batches(len(users), self.batch_size, rng):
                u, i = users[b], pos[b]
                j = torch.randint(split.n_items, (len(b),))
                hu, hi = self.net()
                diff = (hu[u] * (hi[i] - hi[j])).sum(-1)
                ego = self.net.emb.weight
                l2 = (ego[u].pow(2).sum() + ego[self.net.n_users + i].pow(2).sum()
                      + ego[self.net.n_users + j].pow(2).sum()) / len(b)
                loss = -nn.functional.logsigmoid(diff).mean() + self.reg * l2
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self._cache = None
            return total / len(users)

        self._cache = None
        train(self, run_epoch, self.epochs, val, eval_every=5, log=log)
        self._cache = None  # best weights may have been restored
        return self

    @torch.no_grad()
    def score(self, users):
        if self._cache is None:
            self._cache = self.net()
        hu, hi = self._cache
        return (hu[torch.as_tensor(users)] @ hi.T).numpy()
