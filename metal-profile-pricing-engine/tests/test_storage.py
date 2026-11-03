import copy

import pytest
from cryptography.fernet import InvalidToken

from pricing_engine.security import TenantDirectory, check_tenant
from pricing_engine.storage import TenantStore


def test_ingest_validates_and_deduplicates(store_root, mock):
    store = TenantStore(store_root, "tenant_a")
    raws = mock[1]["tenant_a"][:50]
    bad = copy.deepcopy(raws[0]) | {"quote_id": "BAD"}
    bad["manufacturing"]["quantity_pcs"] = -5
    r = store.add_quotes(raws + [bad], "test")
    assert (r["inserted"], r["rejected"]) == (50, 1)
    assert store.add_quotes(raws, "test")["duplicates"] == 50
    assert [q.schema_version for q in store.load_records()] == [2] * 50  # v1 upgraded on read


def test_date_range_retrieval(store_root, mock):
    store = TenantStore(store_root, "tenant_b")
    store.add_quotes(mock[1]["tenant_b"], "test")
    recs = store.load_records("2024-01-01", "2024-03-31")
    assert recs and all("2024-01-01" <= r.quote_date.isoformat() <= "2024-03-31" for r in recs)


def test_records_encrypted_at_rest(store_root, mock):
    store = TenantStore(store_root, "tenant_a")
    store.add_quotes(mock[1]["tenant_a"][:5], "test")
    raw = (store.dir / "quotes.db").read_bytes()
    assert mock[1]["tenant_a"][0]["customer"]["customer_id"].encode() not in raw


def test_tenant_keys_isolate_models(store_root):
    a, b = TenantStore(store_root, "tenant_a"), TenantStore(store_root, "tenant_b")
    v = a.models.save({"weights": [1, 2, 3]}, {"note": "test"})
    a.models.promote(v)
    assert a.models.load()[0] == {"weights": [1, 2, 3]}
    stolen = (a.models.path / "v0001" / "model.enc").read_bytes()
    with pytest.raises(InvalidToken):
        b.cipher.decrypt(stolen)


def test_tampered_model_is_not_unpickled(store_root):
    a = TenantStore(store_root, "tenant_a")
    a.models.promote(a.models.save({"x": 1}, {}))
    f = a.models.path / "v0001" / "model.enc"
    blob = bytearray(f.read_bytes())
    blob[60] ^= 1
    f.write_bytes(bytes(blob))
    with pytest.raises(InvalidToken):
        a.models.load()


def test_api_keys(store_root):
    d = TenantDirectory(store_root)
    key_a, key_b = d.create("tenant_a"), d.create("tenant_b")
    assert d.resolve(key_a) == "tenant_a" and d.resolve(key_b) == "tenant_b"
    assert d.resolve("wrong") is None and d.resolve(None) is None
    assert key_a not in (store_root / "tenants.json").read_text()


@pytest.mark.parametrize("tenant", ["../x", "Tenant", "", "a/b", "x" * 33])
def test_unsafe_tenant_ids_rejected(tenant):
    with pytest.raises(ValueError):
        check_tenant(tenant)
