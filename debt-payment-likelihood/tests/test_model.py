import joblib
import numpy as np
import pytest

from model import TARGETS, PaymentModel, RecoveryModel, correct_prior


def test_correct_prior():
    # balanced 50/50 training (beta = share of non-payers kept) maps 0.5 back to the true rate
    rate = 0.1
    beta = rate / (1 - rate)
    assert correct_prior(0.5, beta) == pytest.approx(rate)
    assert correct_prior(0.5, 1.0) == pytest.approx(0.5)


@pytest.mark.parametrize("balance", ["none", "weight", "down"])
def test_probabilities_calibrated(debtors, balance):
    X, Y = debtors.drop(columns=TARGETS), debtors[TARGETS]
    proba = PaymentModel(balance=balance).fit(X, Y).predict_proba(X)
    assert proba.to_numpy().min() >= 0 and proba.to_numpy().max() <= 1
    # the prior correction after re-balancing is approximate on small samples
    tolerance = 0.1 if balance == "none" else 0.5
    for t in TARGETS:
        assert proba[t].mean() == pytest.approx(Y[t].mean(), rel=tolerance)


@pytest.mark.parametrize("model", ["gb", "xgb"])
def test_chained_horizons_consistent(debtors, model):
    X, Y = debtors.drop(columns=TARGETS), debtors[TARGETS]
    proba = PaymentModel(model=model, horizons="chained").fit(X, Y).predict_proba(X).to_numpy()
    assert np.all(np.diff(proba, axis=1) >= -1e-12)


def test_recovery_non_negative(debtors):
    X, Y = debtors.drop(columns=TARGETS), debtors
    expected = RecoveryModel().fit(X, Y).predict(X)
    assert (expected.to_numpy() >= 0).all()


def test_save_load(debtors, tmp_path):
    X, Y = debtors.drop(columns=TARGETS), debtors[TARGETS]
    model = PaymentModel().fit(X, Y)
    joblib.dump(model, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
