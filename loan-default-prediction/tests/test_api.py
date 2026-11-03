import pickle

import pytest
from fastapi.testclient import TestClient

import schema
from models import build


@pytest.fixture(scope='module')
def client(loans, tmp_path_factory):
    path = tmp_path_factory.mktemp('model') / 'model.pkl'
    model = build('hgb').fit(loans, loans[schema.TARGET])
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'meta': {'model': 'hgb'}}, f)
    import api
    api.MODEL_PATH = str(path)
    with TestClient(api.app) as c:
        yield c


def test_predict(client):
    import api
    r = client.post('/predict', json=api.EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body['default_probability'] <= 1
    assert body['decision'] in {'approve', 'decline'}


def test_risk_level_parameter(client):
    import api
    strict = client.post('/predict', params={'max_pd': 0.0001}, json=api.EXAMPLE).json()
    lenient = client.post('/predict', params={'max_pd': 0.9999}, json=api.EXAMPLE).json()
    assert strict['decision'] == 'decline' and lenient['decision'] == 'approve'


def test_missing_optional_fields(client):
    import api
    body = {k: v for k, v in api.EXAMPLE.items() if k not in schema.OPTIONAL}
    assert client.post('/predict', json=body).status_code == 200


def test_rejects_invalid_input(client):
    import api
    assert client.post('/predict', json={**api.EXAMPLE, 'amount': -1}).status_code == 422
    nested = {**api.EXAMPLE, 'failed_payments_3m': 5, 'failed_payments_6m': 1}
    r = client.post('/predict', json=nested)
    assert r.status_code == 422 and 'failed_payments_3m' in r.text
