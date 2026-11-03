import numpy as np
import pytest

from metrics import (
    best_threshold,
    corrected_ttest,
    decision_table,
    evaluate,
    expected_calibration_error,
    imbalanced_accuracy,
)


def test_imbalanced_accuracy_formula():
    y = np.array([0] * 8 + [1] * 2)
    y_pred = np.array([0] * 6 + [1] * 2 + [1, 0])        # TNR 0.75, TPR 0.5
    worst, gmean = 0.5, np.sqrt(0.75 * 0.5)
    expected = np.sqrt((0.67 * worst) ** 2 + (0.33 * gmean) ** 2) / np.sqrt(0.67 ** 2 + 0.33 ** 2)
    assert imbalanced_accuracy(y, y_pred) == pytest.approx(expected)


def test_imbalanced_accuracy_bounds():
    y = np.array([0, 0, 0, 1])
    assert imbalanced_accuracy(y, y) == pytest.approx(1)
    assert imbalanced_accuracy(y, np.zeros(4)) == pytest.approx(0)   # majority vote fails


def test_imbalanced_accuracy_multiclass():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert imbalanced_accuracy(y, np.array([0, 0, 1, 1, 2, 0])) < 1


def test_best_threshold_separates_classes():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.2, 0.3, 0.6, 0.7])
    t = best_threshold(y, p)
    assert 0.3 < t <= 0.6


def test_calibrated_probabilities_have_small_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.3, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, p * 2) > 0.1


def test_evaluate_consistency():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < p).astype(int)
    r = evaluate(y, p, 0.5)
    assert r['gini'] == pytest.approx(2 * r['roc_auc'] - 1)
    assert r['balanced_accuracy'] == pytest.approx((r['tpr'] + r['tnr']) / 2)
    assert 0 < r['ks'] <= 1 and r['brier_skill'] > 0


def test_decision_table_monotone():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 0.3, 5000)
    y = (rng.random(5000) < p).astype(int)
    t = decision_table(y, p, loss=y * 100.0)
    assert t['default_rate_approved'].is_monotonic_increasing
    assert t['loss_vs_approve_all'].iloc[-1] == 1


def test_corrected_ttest_wider_than_naive():
    d = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
    _, t, p = corrected_ttest(d, n_train=800, n_test=200)
    naive_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(t) < naive_t and 0 < p < 1
