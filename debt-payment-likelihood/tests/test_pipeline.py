"""End to end: train with metadata, score a snapshot, check drift."""
import pandas as pd
from conftest import make_debtors

from model import TARGETS, PaymentModel, RecoveryModel
from monitor import drift, psi
from predict import score
from train import attach_metadata, cross_validate


def test_cross_validate_and_score(debtors):
    X, Y = debtors.drop(columns=TARGETS), debtors[TARGETS]
    scores = cross_validate(X, Y, n_splits=3)
    assert scores.groupby("target")["roc_auc"].mean().min() > 0.6
    model = attach_metadata(PaymentModel().fit(X, Y), X, {}, scores)
    out = score(model, X)
    assert list(out.columns) == TARGETS + ["rank"]
    assert sorted(out["rank"]) == list(range(1, len(X) + 1))


def test_recovery_ranking(debtors):
    X = debtors.drop(columns=TARGETS)
    model = RecoveryModel().fit(X, debtors)
    out = score(model, X)
    top = out.sort_values("rank").iloc[0]
    assert top["amount_next_12m"] == out["amount_next_12m"].max()


def test_drift_detects_shift(debtors):
    X = debtors.drop(columns=TARGETS)
    scores = cross_validate(X, debtors[TARGETS], n_splits=2)
    model = attach_metadata(PaymentModel().fit(X, debtors[TARGETS]), X, {}, scores)
    same = drift(model.metadata_["profile"], make_debtors(seed=1))
    assert (same["status"] == "stable").all()
    shifted = make_debtors(seed=2)
    shifted["external_debt"] *= 3
    table = drift(model.metadata_["profile"], shifted).set_index("column")
    assert table.loc["external_debt", "status"] == "large shift"


def test_psi_zero_for_same_distribution():
    assert psi(pd.Series([0.5, 0.5]), pd.Series([0.5, 0.5])) == 0
