import numpy as np
import pandas as pd
import pytest

import schema
from features import DateFeatures, InSampleTargetEncoder, selected_features
from models import TorchClassifier, build


def test_days_to_card_expiry():
    row = pd.DataFrame([{c: 0 for c in schema.INPUTS}])
    row[schema.DATE] = '2025-03-15'
    row['card_exp_month'], row['card_exp_year'] = 4.0, 2025.0
    row['merchant_group'] = row['merchant_category'] = 'x'
    out = DateFeatures().transform(row)
    assert out['days_to_card_expiry'].iloc[0] == 17
    assert 'card_exp_year' not in out and schema.DATE not in out


def test_in_sample_encoder_unseen_category():
    X = pd.DataFrame({'g': ['a', 'a', 'b', 'b']})
    enc = InSampleTargetEncoder().fit(X, [1, 0, 0, 0])
    assert enc.transform(pd.DataFrame({'g': ['a', 'z']})).ravel().tolist() == [0.5, 0.25]


@pytest.mark.parametrize('encoder', ['cross_fitted', 'in_sample'])
def test_pipeline_handles_unseen_merchant(loans, encoder):
    model = build('hgb', encoder=encoder).fit(loans, loans[schema.TARGET])
    new = loans.head(3).copy()
    new['merchant_category'] = 'never_seen'
    p = model.predict_proba(new)[:, 1]
    assert len(p) == 3 and np.all((p >= 0) & (p <= 1))


def test_calibrated_model(loans):
    model = build('logreg').fit(loans, loans[schema.TARGET])
    p = model.predict_proba(loans)[:, 1]
    assert np.all((p >= 0) & (p <= 1))
    assert abs(p.mean() - loans[schema.TARGET].mean()) < 0.02     # calibration keeps the mean
    assert set(model.predict(loans)) <= {0, 1}
    assert 0 < model.threshold_ < 1


def test_targets_and_outcomes_not_used_as_features(loans):
    model = build('logreg').fit(loans, loans[schema.TARGET])
    names = selected_features(model.estimator_)
    assert not set(names) & set(schema.OUTCOMES + [schema.TARGET])


@pytest.mark.parametrize('loss', ['bce', 'reinforce'])
def test_torch_classifier_deterministic(loss):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + rng.normal(size=400) > 1).astype(int)
    a = TorchClassifier(hidden=(8,), dropout=(0.1,), loss=loss, epochs=3).fit(X, y)
    b = TorchClassifier(hidden=(8,), dropout=(0.1,), loss=loss, epochs=3).fit(X, y)
    pa, pb = a.predict_proba(X)[:, 1], b.predict_proba(X)[:, 1]
    assert np.allclose(pa, pb)
    assert np.corrcoef(pa, X[:, 0])[0, 1] > 0      # learns the direction of the signal
