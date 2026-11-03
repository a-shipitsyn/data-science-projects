"""Classifiers compared on the same folds, and the calibration + threshold wrapper.

All models sit behind the same feature pipeline. Each is wrapped in
CalibratedClassifier: fitted on 80 % of the training rows, its scores are mapped to
probabilities by Platt scaling on the other 20 %, where the decision threshold that
maximises imbalanced accuracy is also chosen.
"""
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.distributions import Bernoulli

from features import make_pipeline
from metrics import best_threshold

torch.set_num_threads(1)

MODELS = ['logreg', 'hgb_balanced', 'hgb', 'mlp', 'rl_linear', 'rl_deep']


def mlp(in_dim: int, hidden: Sequence[int], dropout: Sequence[float]) -> nn.Sequential:
    layers = []
    for units, rate in zip(hidden, dropout):
        layers += [nn.Linear(in_dim, units), nn.BatchNorm1d(units), nn.ReLU(), nn.Dropout(rate)]
        in_dim = units
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def rewards(actions: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """+pos_weight / +1 for a correct default / repaid decision, the negative when wrong,
    minus the batch mean as a baseline to reduce the variance of the policy gradient."""
    r = torch.where(y == 1, pos_weight, 1.0) * torch.where(actions == y, 1.0, -1.0)
    return r - r.mean()


class TorchClassifier(ClassifierMixin, BaseEstimator):
    """Feed-forward network trained either with weighted binary cross-entropy
    (loss='bce') or as a REINFORCE policy that samples approve/decline actions and is
    rewarded for correct decisions (loss='reinforce')."""

    def __init__(self, hidden: Sequence[int] = (256, 48), dropout: Sequence[float] = (0.36, 0.11),
                 loss: str = 'bce', lr: float = 2e-3, epochs: int = 30, batch_size: int = 128,
                 random_state: int = 0):
        self.hidden = hidden
        self.dropout = dropout
        self.loss = loss
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> 'TorchClassifier':
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1)
        self.classes_ = np.array([0, 1])
        pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
        self.net_ = mlp(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.net_.train()
        for _ in range(self.epochs):
            order = torch.from_numpy(rng.permutation(len(X)))
            for start in range(0, len(X), self.batch_size):
                idx = order[start:start + self.batch_size]
                if len(idx) < 2:                      # BatchNorm needs two rows
                    continue
                logits = self.net_(X[idx])
                if self.loss == 'bce':
                    loss = bce(logits, y[idx])
                else:
                    policy = Bernoulli(logits=logits)
                    actions = policy.sample()
                    loss = -(policy.log_prob(actions) * rewards(actions, y[idx], pos_weight)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net_.eval()
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(np.asarray(X), dtype=torch.float32)))
        p = p.numpy().ravel().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model(name: str, seed: int = 0) -> BaseEstimator:
    """Estimator for a model name (without the feature pipeline)."""
    if name == 'logreg':
        return LogisticRegression(max_iter=1000)
    if name == 'hgb_balanced':          # the original model and hyperparameters
        return HistGradientBoostingClassifier(
            class_weight='balanced', l2_regularization=0.5, max_depth=3, max_leaf_nodes=5,
            random_state=seed)
    if name == 'hgb':                   # same, trained on the true class ratio
        return HistGradientBoostingClassifier(
            l2_regularization=0.5, max_depth=3, max_leaf_nodes=5, random_state=seed)
    if name == 'mlp':                   # architecture from the original Optuna search
        return TorchClassifier(hidden=(256, 48), dropout=(0.36, 0.11), loss='bce',
                               lr=2e-3, epochs=30, random_state=seed)
    if name == 'rl_linear':
        return TorchClassifier(hidden=(), dropout=(), loss='reinforce', lr=1e-2,
                               epochs=15, random_state=seed)
    if name == 'rl_deep':               # architecture from the original Optuna search
        return TorchClassifier(hidden=(96, 67, 102), dropout=(0.30, 0.11, 0.37),
                               loss='reinforce', lr=1e-3, epochs=20, random_state=seed)
    raise ValueError(f'unknown model {name!r}')


class CalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on part of the data, calibrate its scores on the held-out part
    and choose the imbalanced-accuracy threshold there.

    calibration: 'sigmoid' (Platt scaling: logistic regression on the logit of the
    score; keeps the ranking), 'isotonic', or None for the raw scores.
    threshold=<float> fixes the cut-off instead of choosing it.
    """

    def __init__(self, estimator: BaseEstimator, calibration: str | None = 'sigmoid',
                 threshold: float | None = None, val_size: float = 0.2, random_state: int = 0):
        self.estimator = estimator
        self.calibration = calibration
        self.threshold = threshold
        self.val_size = val_size
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> 'CalibratedClassifier':
        y = np.asarray(y)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=self.random_state)
        self.estimator_ = clone(self.estimator).fit(X_fit, y_fit)
        self.classes_ = self.estimator_.classes_
        score = self.estimator_.predict_proba(X_val)[:, 1]
        if self.calibration == 'sigmoid':
            self.calibrator_ = LogisticRegression(C=1e6).fit(_logit(score), y_val)
        elif self.calibration == 'isotonic':
            self.calibrator_ = IsotonicRegression(out_of_bounds='clip').fit(score, y_val)
        elif self.calibration is None:
            self.calibrator_ = None
        else:
            raise ValueError(f'unknown calibration {self.calibration!r}')
        p = self._calibrate(score)
        self.threshold_ = (best_threshold(y_val, p) if self.threshold is None
                           else float(self.threshold))
        return self

    def _calibrate(self, score: np.ndarray) -> np.ndarray:
        if self.calibration == 'sigmoid':
            return self.calibrator_.predict_proba(_logit(score))[:, 1]
        if self.calibration == 'isotonic':
            return self.calibrator_.predict(score)
        return score

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self._calibrate(self.estimator_.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def build(name: str, seed: int = 0, calibration: str | None = 'sigmoid',
          threshold: float | None = None, encoder: str = 'cross_fitted',
          params: dict | None = None) -> CalibratedClassifier:
    """Feature pipeline + model, wrapped with calibration and threshold choice.

    params: hyperparameters overriding the model defaults (e.g. from tune.py).
    """
    model = make_model(name, seed)
    if params:
        model.set_params(**params)
    pipe = make_pipeline(model, encoder=encoder, seed=seed)
    return CalibratedClassifier(pipe, calibration=calibration, threshold=threshold,
                                random_state=seed)
