"""Two-tower retrieval model: business and listing encoders with ID embeddings and features.

Trained with in-batch sampled softmax and logQ correction (Yi et al., 2019). ID embeddings are
dropped at random during training, so the towers also learn to work from features alone; new
businesses and new listings, which have no trained ID, are scored from their features.
"""

import numpy as np
import torch
from torch import nn

from data import item_features, user_features
from training import batches, set_seed, train


class Tower(nn.Module):
    def __init__(self, n_ids, cat_sizes, n_num, dim, hidden):
        super().__init__()
        self.ids = nn.Embedding(n_ids, dim)
        self.cats = nn.ModuleList(nn.Embedding(n, min(16, n)) for n in cat_sizes)
        width = dim + sum(min(16, n) for n in cat_sizes) + n_num
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        nn.init.normal_(self.ids.weight, std=0.05)

    def forward(self, ids, keep_id, cats, nums):
        parts = [self.ids(ids) * keep_id[:, None]]
        parts += [emb(cats[:, j]) for j, emb in enumerate(self.cats)]
        parts.append(nums)
        return nn.functional.normalize(self.mlp(torch.cat(parts, dim=1)), dim=1)


class _Net(nn.Module):
    def __init__(self, n_users, n_items, ucat_sizes, icat_sizes, dim, hidden):
        super().__init__()
        self.user = Tower(n_users, ucat_sizes, 2, dim, hidden)
        self.item = Tower(n_items, icat_sizes, 3, dim, hidden)


class TwoTower:
    def __init__(self, dim=64, hidden=128, temperature=0.05, id_dropout=0.3, lr=2e-3,
                 weight_decay=0.0, batch_size=1024, epochs=40, seed=0):
        self.dim, self.hidden, self.temperature, self.id_dropout = dim, hidden, temperature, id_dropout
        self.weight_decay = weight_decay
        self.lr, self.batch_size, self.epochs, self.seed = lr, batch_size, epochs, seed

    def _prepare(self, split):
        """Feature tensors and a fresh network for the users and items of `split`."""
        ucat, unum = user_features(split.users)
        icat, inum = item_features(split.items)
        self.ucat, self.unum = torch.as_tensor(ucat), torch.as_tensor(unum)
        self.icat, self.inum = torch.as_tensor(icat), torch.as_tensor(inum)
        self.user_known = torch.as_tensor(np.diff(split.X.indptr) > 0, dtype=torch.float32)
        self.item_known = torch.as_tensor(np.diff(split.X.tocsc().indptr) > 0, dtype=torch.float32)
        self.net = _Net(split.n_users, split.n_items, (ucat.max(0) + 1).tolist(),
                        (icat.max(0) + 1).tolist(), self.dim, self.hidden)

    def fit(self, split, val=None, log=print):
        set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._prepare(split)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        coo = split.X.tocoo()
        users, items = torch.as_tensor(coo.row), torch.as_tensor(coo.col)
        freq = np.bincount(coo.col, minlength=split.n_items) / len(coo.col)
        log_q = torch.as_tensor(np.log(freq + 1e-9), dtype=torch.float32)

        def run_epoch():
            self.net.train()
            total = 0.0
            for b in batches(len(users), self.batch_size, rng):
                u, i = users[b], items[b]
                keep_u = (torch.rand(len(b)) > self.id_dropout).float()
                keep_i = (torch.rand(len(b)) > self.id_dropout).float()
                eu = self.net.user(u, keep_u, self.ucat[u], self.unum[u])
                ei = self.net.item(i, keep_i, self.icat[i], self.inum[i])
                logits = eu @ ei.T / self.temperature - log_q[i][None, :]
                same = i[:, None] == i[None, :]  # the same listing twice in a batch is not a negative
                logits = logits.masked_fill(same & ~torch.eye(len(b), dtype=torch.bool), -1e9)
                loss = nn.functional.cross_entropy(logits, torch.arange(len(b)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            return total / len(users)

        return train(self, run_epoch, self.epochs, val, eval_every=1, log=log)

    def save(self, path):
        params = {k: getattr(self, k) for k in ('dim', 'hidden', 'temperature', 'id_dropout', 'lr',
                                                 'weight_decay', 'batch_size', 'epochs', 'seed')}
        torch.save({'params': params, 'state': self.net.state_dict()}, path)

    @classmethod
    def load(cls, path, split):
        """Restore a saved model; `split` must have the same users, items and cutoff as in training."""
        ckpt = torch.load(path)
        model = cls(**ckpt['params'])
        model._prepare(split)
        model.net.load_state_dict(ckpt['state'])
        model.net.eval()
        return model

    @torch.no_grad()
    def item_vectors(self):
        ids = torch.arange(len(self.item_known))
        return self.net.item(ids, self.item_known, self.icat, self.inum)

    @torch.no_grad()
    def score(self, users):
        u = torch.as_tensor(users)
        eu = self.net.user(u, self.user_known[u], self.ucat[u], self.unum[u])
        return (eu @ self.item_vectors().T).numpy()
