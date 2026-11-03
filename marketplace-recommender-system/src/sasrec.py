"""SASRec (Kang and McAuley, 2018): a causal transformer over each business's listing sequence.

Trained with full softmax cross-entropy over all listings instead of the original sampled
binary loss, which is known to work better (Klenitskiy and Vasilev, 2023). The next listing is
predicted from the last position of the sequence before the cutoff.
"""

import numpy as np
import torch
from torch import nn

from training import batches, set_seed, train


class _Net(nn.Module):
    def __init__(self, n_items, dim, max_len, layers, heads, dropout):
        super().__init__()
        self.heads = heads
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)  # 0 = padding
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.items.weight, std=0.05)
        nn.init.normal_(self.pos.weight, std=0.05)
        self.register_buffer('causal', torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1))

    def forward(self, seq):
        L = seq.shape[1]
        pad = seq == 0
        x = self.items(seq) * self.items.embedding_dim ** 0.5 + self.pos.weight[None, :L]
        x = self.drop(x) * ~pad[..., None]
        # causal + padding mask; every position may attend to itself, so no row is fully masked
        mask = (self.causal[None, :L, :L] | pad[:, None, :]) & ~torch.eye(L, dtype=torch.bool)
        h = self.encoder(x, mask=mask.repeat_interleave(self.heads, 0))
        return self.norm(h)

    def logits(self, h):
        return h @ self.items.weight[1:].T


def left_pad(seqs, length):
    """Last `length` items of each sequence, item index + 1, left-padded with 0."""
    out = np.zeros((len(seqs), length), dtype=np.int64)
    for r, s in enumerate(seqs):
        s = s[-length:]
        if len(s):
            out[r, -len(s):] = s + 1
    return out


class SASRec:
    def __init__(self, dim=64, max_len=50, layers=2, heads=2, dropout=0.2, lr=1e-3,
                 batch_size=128, epochs=100, seed=0):
        self.dim, self.max_len, self.layers, self.heads, self.dropout = dim, max_len, layers, heads, dropout
        self.lr, self.batch_size, self.epochs, self.seed = lr, batch_size, epochs, seed

    def fit(self, split, val=None, log=print):
        set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.sequences = split.sequences
        train_seqs = [s for s in split.sequences if len(s) >= 2]
        padded = torch.as_tensor(left_pad(train_seqs, self.max_len + 1))
        self.net = _Net(split.n_items, self.dim, self.max_len, self.layers, self.heads, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        def run_epoch():
            self.net.train()
            total, count = 0.0, 0
            for b in batches(len(padded), self.batch_size, rng):
                inp, target = padded[b, :-1], padded[b, 1:]
                mask = (inp > 0) & (target > 0)
                logits = self.net.logits(self.net(inp)[mask])
                loss = nn.functional.cross_entropy(logits, target[mask] - 1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * mask.sum().item()
                count += mask.sum().item()
            return total / count

        return train(self, run_epoch, self.epochs, val, eval_every=5, log=log)

    @torch.no_grad()
    def score(self, users):
        seq = torch.as_tensor(left_pad([self.sequences[u] for u in users], self.max_len))
        h = self.net(seq)[:, -1]
        return self.net.logits(h).numpy()
