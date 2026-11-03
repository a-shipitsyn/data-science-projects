"""Tenant identifiers, API keys and per-tenant encryption keys.

- The master key comes from the environment (`PRICING_MASTER_KEY`), never from files in
  the store; in production it would live in a secrets manager / KMS.
- Each tenant gets its own data key derived from the master key with HKDF, so one
  tenant's key cannot decrypt another tenant's data or models.
- API keys are random 256-bit tokens shown once; only their SHA-256 hash is stored and
  compared in constant time. The tenant is resolved from the key, never from the request.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

TENANT_RE = re.compile(r"^[a-z0-9_]{1,32}$")
MASTER_KEY_ENV = "PRICING_MASTER_KEY"


def check_tenant(tenant: str) -> str:
    """Tenant ids become directory names: allow only a safe pattern (no path traversal)."""
    if not TENANT_RE.fullmatch(tenant or ""):
        raise ValueError("tenant id must match [a-z0-9_]{1,32}")
    return tenant


def master_key() -> bytes:
    key = os.environ.get(MASTER_KEY_ENV)
    if not key:
        raise RuntimeError(f"set {MASTER_KEY_ENV} (create one with: python -m pricing_engine gen-key)")
    raw = base64.urlsafe_b64decode(key)
    if len(raw) != 32:
        raise RuntimeError(f"{MASTER_KEY_ENV} must be 32 url-safe base64 bytes")
    return raw


def tenant_cipher(tenant: str, key: bytes | None = None) -> Fernet:
    """Authenticated encryption (AES-128-CBC + HMAC-SHA256) with the tenant's derived key."""
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"pricing-engine/tenant/" + check_tenant(tenant).encode())
    return Fernet(base64.urlsafe_b64encode(hkdf.derive(key or master_key())))


def new_master_key() -> str:
    return Fernet.generate_key().decode()


def _hash(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class TenantDirectory:
    """Tenant ids and API-key hashes, stored in `<root>/tenants.json`."""

    def __init__(self, root):
        self.path = Path(root) / "tenants.json"

    def _read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def create(self, tenant: str) -> str:
        """Register a tenant (or rotate its key) and return the new API key."""
        tenants = self._read()
        api_key = secrets.token_urlsafe(32)
        tenants[check_tenant(tenant)] = _hash(api_key)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.write_text(json.dumps(tenants, indent=2))
        self.path.chmod(0o600)
        return api_key

    def resolve(self, api_key: str | None) -> str | None:
        """Tenant of an API key, or None."""
        if not api_key:
            return None
        h = _hash(api_key)
        found = None
        for tenant, stored in self._read().items():
            if hmac.compare_digest(stored, h):
                found = tenant
        return found

    def tenants(self) -> list[str]:
        return sorted(self._read())
