import copy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pricing_engine import simulate, training
from pricing_engine.api import create_app
from pricing_engine.security import TenantDirectory
from pricing_engine.storage import TenantStore


@pytest.fixture(scope="module")
def client(module_store_root, mock):
    store_root = module_store_root
    directory = TenantDirectory(store_root)
    keys = {}
    for tenant in ["tenant_a", "tenant_b"]:
        keys[tenant] = directory.create(tenant)
        store = TenantStore(store_root, tenant)
        store.add_quotes(mock[1][tenant], "test")
        training.train_tenant(store, mock[0], n_trials=2)
        store.close()
    return TestClient(create_app(store_root)), keys


def spec_of(quote):
    return {k: v for k, v in quote.items() if k not in ("quote_id", "quoted_price_per_kg")}


def test_requires_api_key(client, mock):
    c, _ = client
    spec = spec_of(mock[1]["tenant_a"][0])
    assert c.post("/quote", json=spec).status_code == 401
    assert c.post("/quote", json=spec, headers={"X-API-Key": "nope"}).status_code == 401


def test_quote_both_schema_versions(client, mock):
    c, keys = client
    for tenant in ["tenant_a", "tenant_b"]:
        r = c.post("/quote", json=spec_of(mock[1][tenant][0]), headers={"X-API-Key": keys[tenant]})
        assert r.status_code == 200, r.text
        q = r.json()
        lo, hi = q["interval_90"]
        assert lo < q["price_per_kg"] < hi and q["order_total"] > 0


def test_tenants_get_their_own_model(client, mock):
    c, keys = client
    spec = spec_of(mock[1]["tenant_b"][0])
    prices = {t: c.post("/quote", json=spec, headers={"X-API-Key": k}).json()["price_per_kg"] for t, k in keys.items()}
    assert prices["tenant_a"] != prices["tenant_b"]


def test_invalid_spec_is_422(client, mock):
    c, keys = client
    spec = spec_of(mock[1]["tenant_b"][0])
    spec["material"] = {"alloy": "unobtainium"}
    assert c.post("/quote", json=spec, headers={"X-API-Key": keys["tenant_b"]}).status_code == 422


def test_feedback_monitoring_and_models(client, mock):
    c, keys = client
    h = {"X-API-Key": keys["tenant_a"]}
    rng = np.random.default_rng(5)
    cfg = simulate.TRUTH["tenants"]["tenant_a"]
    new, _, _ = simulate.simulate_quotes(rng, mock[0], "tenant_a", cfg, n=30, start="2025-07-01", end="2025-08-29",
                                         id_prefix="N")
    wrong = copy.deepcopy(new[0]) | {"quote_id": "TYPO", "quoted_price_per_kg": 999.0}
    r = c.post("/feedback", json=new + [wrong], headers=h).json()
    assert r["inserted"] == 30 and r["rejected"] == 1
    status = c.get("/monitoring", headers=h).json()
    assert status["feedback_n"] == 30 and "recent_mape" in status
    models = c.get("/models", headers=h).json()
    assert len(models) == 1 and models[0]["champion"]
