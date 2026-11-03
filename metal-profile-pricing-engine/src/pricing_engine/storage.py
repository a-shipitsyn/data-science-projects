"""Per-tenant storage: quotes in an SQLite file per tenant, models in a versioned registry.

Layout (directories 0700, files 0600):
    <root>/tenants.json                API-key hashes
    <root>/market/lme.csv              shared public market data
    <root>/tenants/<tenant>/quotes.db  quotes, rejected records, prediction log
    <root>/tenants/<tenant>/models/    v0001/model.enc, v0001/metadata.json, champion.json

Isolation: one database file and one encryption key per tenant — no query can mix tenants,
and a copied file is unreadable without that tenant's key. Quote records are stored as
received (any schema version), encrypted; only the id, date and version stay in clear
for indexed retrieval. Records are upgraded to the current schema on read.
"""
import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import joblib

from . import schema
from .geometry import profile_features
from .security import check_tenant, tenant_cipher

DDL = """
CREATE TABLE IF NOT EXISTS quotes (
    quote_id TEXT PRIMARY KEY, quote_date TEXT NOT NULL, schema_version INTEGER NOT NULL,
    record BLOB NOT NULL, source TEXT NOT NULL, ingested_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS quotes_date ON quotes (quote_date);
CREATE TABLE IF NOT EXISTS rejected (
    id INTEGER PRIMARY KEY AUTOINCREMENT, record BLOB NOT NULL, reason TEXT NOT NULL,
    source TEXT NOT NULL, received_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions (
    quote_id TEXT NOT NULL, model_version INTEGER NOT NULL, predicted REAL NOT NULL, lower REAL NOT NULL,
    upper REAL NOT NULL, actual REAL, created_at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


class TenantStore:
    def __init__(self, root, tenant: str, key: bytes | None = None):
        self.tenant = check_tenant(tenant)
        self.dir = _private_dir(_private_dir(Path(root) / "tenants") / tenant)
        self.cipher = tenant_cipher(tenant, key)
        db = self.dir / "quotes.db"
        self.conn = sqlite3.connect(db)
        db.chmod(0o600)
        self.conn.executescript(DDL)
        self.models = ModelRegistry(self.dir / "models", self.cipher)

    def _enc(self, obj) -> bytes:
        return self.cipher.encrypt(json.dumps(obj, default=str).encode())

    def _dec(self, blob):
        return json.loads(self.cipher.decrypt(blob))

    def add_quotes(self, raws: list[dict], source: str) -> dict:
        """Validate and insert quotes; invalid ones go to `rejected`, duplicates are ignored."""
        valid, rejected = [], []
        for raw in raws:
            try:
                rec = schema.parse(raw)
                profile_features(rec.geometry.cross_section.outer, rec.geometry.cross_section.holes,
                                 rec.geometry.min_wall_mm)
                valid.append((raw, rec))
            except ValueError as e:
                rejected.append((raw, schema.short_error(e)))
        now = _now()
        with self.conn:
            before = self.count()
            self.conn.executemany(
                "INSERT OR IGNORE INTO quotes VALUES (?, ?, ?, ?, ?, ?)",
                [(r.quote_id, r.quote_date.isoformat(), raw.get("schema_version", 1), self._enc(raw), source, now)
                 for raw, r in valid])
            inserted = self.count() - before
        self.reject(rejected, source)
        return {"inserted": inserted, "duplicates": len(valid) - inserted, "rejected": len(rejected),
                "reasons": [reason for _, reason in rejected]}

    def reject(self, rejected: list[tuple[dict, str]], source: str):
        """Keep rejected records (encrypted) with the reason, for review."""
        now = _now()
        with self.conn:
            self.conn.executemany("INSERT INTO rejected (record, reason, source, received_at) VALUES (?, ?, ?, ?)",
                                  [(self._enc(raw), reason, source, now) for raw, reason in rejected])

    def count(self, since: str | None = None) -> int:
        q, args = "SELECT COUNT(*) FROM quotes", ()
        if since:
            q, args = q + " WHERE ingested_at > ?", (since,)
        return self.conn.execute(q, args).fetchone()[0]

    def load_records(self, start: str | None = None, end: str | None = None) -> list:
        """Quotes in a date range, upgraded to the current schema, in date order."""
        q = "SELECT record FROM quotes WHERE quote_date >= ? AND quote_date <= ? ORDER BY quote_date, quote_id"
        rows = self.conn.execute(q, (start or "0000", end or "9999")).fetchall()
        return [schema.parse(self._dec(blob)) for (blob,) in rows]

    def log_prediction(self, quote_id, model_version, pred, actual=None):
        with self.conn:
            self.conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (quote_id, model_version, pred["price_per_kg"], pred["lower"], pred["upper"],
                               actual, _now()))

    def recent_feedback(self, n: int, model_version: int) -> list[tuple]:
        """Latest predictions of a model with a known outcome: (predicted, lower, upper, actual)."""
        return self.conn.execute(
            "SELECT predicted, lower, upper, actual FROM predictions WHERE actual IS NOT NULL "
            "AND model_version = ? ORDER BY rowid DESC LIMIT ?", (model_version, n)).fetchall()

    def close(self):
        self.conn.close()


class ModelRegistry:
    """Versioned, encrypted models of one tenant with a champion pointer.

    Models are pickled (joblib) and encrypted with the tenant key. Fernet authenticates the
    ciphertext, so a tampered or foreign file fails to decrypt before anything is
    unpickled — unpickling untrusted bytes would allow code execution.
    """

    def __init__(self, path: Path, cipher):
        self.path, self.cipher = path, cipher

    def versions(self) -> list[int]:
        if not self.path.exists():
            return []
        return sorted(int(p.name[1:]) for p in self.path.glob("v[0-9]*") if (p / "model.enc").exists())

    def save(self, model, metadata: dict) -> int:
        version = (self.versions() or [0])[-1] + 1
        d = _private_dir(_private_dir(self.path) / f"v{version:04d}")
        buf = io.BytesIO()
        joblib.dump(model, buf)
        (d / "model.enc").write_bytes(self.cipher.encrypt(buf.getvalue()))
        (d / "metadata.json").write_text(json.dumps({"version": version, "created_at": _now(), **metadata},
                                                    indent=2, default=str))
        for f in d.iterdir():
            f.chmod(0o600)
        return version

    def metadata(self, version: int) -> dict:
        return json.loads((self.path / f"v{version:04d}" / "metadata.json").read_text())

    def load(self, version: int | None = None):
        version = version or self.champion()
        if version is None:
            raise LookupError("no trained model for this tenant")
        blob = self.cipher.decrypt((self.path / f"v{version:04d}" / "model.enc").read_bytes())
        return joblib.load(io.BytesIO(blob)), self.metadata(version)

    def champion(self) -> int | None:
        p = self.path / "champion.json"
        return json.loads(p.read_text())["version"] if p.exists() else None

    def promote(self, version: int):
        if version not in self.versions():
            raise LookupError(f"model version {version} does not exist")
        p = self.path / "champion.json"
        p.write_text(json.dumps({"version": version, "promoted_at": _now()}))
        p.chmod(0o600)
