"""Evaluation metrics: imbalanced accuracy, standard classification and probability
metrics, threshold choice, decision table and the corrected resampled t-test.

Imbalanced accuracy is defined in A. Shipitsyn, *Statistical Learning with Imbalanced
Data*, MSc thesis, Linköping University, 2017 (DiVA diva2:1119403).
"""
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

EPS = 1e-6


def class_accuracies(y: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Accuracy within each true class (recall per class)."""
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    return np.array([np.mean(y_pred[y == c] == c) for c in np.unique(y)])


def imbalanced_accuracy(y: ArrayLike, y_pred: ArrayLike,
                        weights: tuple[float, float] = (0.67, 0.33)) -> float:
    """Weighted combination of the worst class accuracy and the geometric mean of all
    class accuracies: sqrt((w1·worst)² + (w2·gmean)²) / sqrt(w1² + w2²), in [0, 1].
    """
    acc = class_accuracies(y, y_pred)
    worst = acc.min()
    gmean = np.prod(acc) ** (1 / len(acc))
    w1, w2 = weights
    return float(np.sqrt((w1 * worst) ** 2 + (w2 * gmean) ** 2) / np.sqrt(w1 ** 2 + w2 ** 2))


def best_threshold(y: ArrayLike, p: ArrayLike, grid: int = 400) -> float:
    """Probability cut-off maximising imbalanced accuracy (predict default if p >= t)."""
    y, p = np.asarray(y), np.asarray(p)
    cuts = np.unique(p)
    if len(cuts) > grid:
        cuts = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
    scores = [imbalanced_accuracy(y, (p >= t).astype(int)) for t in cuts]
    return float(cuts[int(np.argmax(scores))])


def expected_calibration_error(y: ArrayLike, p: ArrayLike, bins: int = 10) -> float:
    """Mean |observed rate − mean predicted| over equal-count bins, weighted by size."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, len(edges) - 2)
    err = 0.0
    for b in np.unique(idx):
        m = idx == b
        err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def reliability(y: ArrayLike, p: ArrayLike, bins: int = 10) -> pd.DataFrame:
    """Reliability curve: mean predicted vs observed default rate in equal-count bins."""
    df = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(p)})
    df['bin'] = pd.qcut(df['p'].rank(method='first'), bins, labels=False)
    out = df.groupby('bin').agg(n=('y', 'size'), mean_predicted=('p', 'mean'),
                                observed=('y', 'mean'))
    return out.reset_index()


def evaluate(y: ArrayLike, p: ArrayLike, threshold: float) -> dict[str, float]:
    """All metrics for probabilities p and the decision p >= threshold."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y_pred = (p >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    tpr, tnr = tp / (tp + fn), tn / (tn + fp)
    fpr, tpr_curve, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    base = y.mean()
    brier = brier_score_loss(y, p)
    return {
        'imbalanced_accuracy': imbalanced_accuracy(y, y_pred),
        'tpr': tpr,
        'tnr': tnr,
        'balanced_accuracy': (tpr + tnr) / 2,
        'g_mean': np.sqrt(tpr * tnr),
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'npv': tn / (tn + fn) if tn + fn else 0.0,
        'f1': f1_score(y, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y, y_pred),
        'kappa': cohen_kappa_score(y, y_pred),
        'flagged': y_pred.mean(),
        'roc_auc': auc,
        'gini': 2 * auc - 1,
        'ks': float(np.max(tpr_curve - fpr)),
        'pr_auc': average_precision_score(y, p),
        'pr_auc_baseline': base,
        'brier': brier,
        'brier_skill': 1 - brier / (base * (1 - base)),
        'log_loss': log_loss(y, p, labels=[0, 1]),
        'ece': expected_calibration_error(y, p),
        'mean_predicted': p.mean(),
        'observed_rate': base,
        'threshold': threshold,
    }


def decision_table(y: ArrayLike, p: ArrayLike, loss: ArrayLike,
                   approvals: tuple[float, ...] = (0.80, 0.90, 0.95)) -> pd.DataFrame:
    """Approve the loans with the lowest predicted default probability.

    For each approval rate: the probability cut-off, imbalanced accuracy of the
    approve/decline decision, default rate among approved, share of defaults declined,
    share of good loans declined, and loss among approved relative to approving all.
    """
    y, p, loss = np.asarray(y), np.asarray(p), np.asarray(loss, dtype=float)
    rows = []
    for a in approvals:
        cut = np.quantile(p, a)
        declined = p > cut
        rows.append({
            'approval_rate': a,
            'pd_cutoff': cut,
            'imbalanced_accuracy': imbalanced_accuracy(y, declined.astype(int)),
            'default_rate_approved': y[~declined].mean(),
            'defaults_declined': declined[y == 1].mean(),
            'good_declined': declined[y == 0].mean(),
            'loss_vs_approve_all': loss[~declined].sum() / loss.sum(),
        })
    rows.append({'approval_rate': 1.0, 'pd_cutoff': np.nan, 'imbalanced_accuracy': np.nan,
                 'default_rate_approved': y.mean(), 'defaults_declined': 0.0,
                 'good_declined': 0.0, 'loss_vs_approve_all': 1.0})
    return pd.DataFrame(rows)


def corrected_ttest(diffs: ArrayLike, n_train: float,
                    n_test: float) -> tuple[float, float, float]:
    """Nadeau–Bengio corrected resampled t-test for repeated k-fold differences.

    Returns (mean difference, t statistic, two-sided p-value).
    """
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    if var == 0:
        return float(d.mean()), np.nan, np.nan
    t = d.mean() / np.sqrt(var)
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), k - 1))
