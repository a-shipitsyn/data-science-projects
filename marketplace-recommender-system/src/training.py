"""Shared training loop for the PyTorch models: early stopping on validation NDCG@10."""

import copy

import numpy as np
import torch

from metrics import evaluate


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model, run_epoch, epochs, val=None, eval_every=2, patience=3, log=print):
    """Run `epochs` epochs; with a validation split keep the best weights and record best_epoch.

    run_epoch() trains one epoch and returns the mean loss. model.net is the torch module and
    model.score(users) scores users of the split the model was fitted on.
    """
    best, best_state, best_epoch, bad = -np.inf, None, epochs, 0
    for epoch in range(1, epochs + 1):
        loss = run_epoch()
        if val is None or epoch % eval_every:
            continue
        model.net.eval()
        ndcg = evaluate(model.score, val, val.eval_users(warm=True), ks=(10,))['ndcg@10']
        log(f'  epoch {epoch:3d}  loss {loss:.4f}  val ndcg@10 {ndcg:.4f}')
        if ndcg > best:
            best, best_state, best_epoch, bad = ndcg, copy.deepcopy(model.net.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.net.load_state_dict(best_state)
    model.best_epoch = best_epoch
    model.net.eval()
    return model


def batches(n, batch_size, rng):
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]
