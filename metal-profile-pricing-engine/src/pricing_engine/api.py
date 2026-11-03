"""HTTP API. Every request carries `X-API-Key`; the tenant is resolved from the key, so
a caller can only reach its own data and models.

    POST /quote        product spec (schema v1 or v2) → final price, 90 % interval
    POST /feedback     list of quotes with their final price → validated and stored
    GET  /monitoring   drift status and retraining triggers
    POST /retrain      champion / challenger retraining
    GET  /models       model versions and their metadata
"""
import logging
import os
import time
from pathlib import Path
from typing import Annotated

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from . import training
from .market import load_lme
from .schema import SpecV1, SpecV2, upgrade
from .security import TenantDirectory
from .storage import TenantStore

log = logging.getLogger("pricing_engine.api")
Spec = Annotated[SpecV1 | SpecV2, Field(discriminator="schema_version")]
MAX_BATCH = 1000


def create_app(root: str | Path | None = None) -> FastAPI:
    root = Path(root or os.environ.get("PRICING_STORE", "store"))
    app = FastAPI(title="Metal profile pricing engine", version="1.0")
    directory = TenantDirectory(root)
    lme_path = root / "market" / "lme.csv"

    def market():
        try:
            return load_lme(lme_path)
        except FileNotFoundError:
            raise HTTPException(503, "market data not loaded")

    def tenant_store(x_api_key: Annotated[str | None, Header()] = None):
        tenant = directory.resolve(x_api_key)
        if tenant is None:
            raise HTTPException(401, "invalid or missing API key")
        store = TenantStore(root, tenant)
        try:
            yield store
        finally:
            store.close()

    Store = Annotated[TenantStore, Depends(tenant_store)]

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        log.info("%s %s %d %.0f ms", request.method, request.url.path, response.status_code,
                 1000 * (time.perf_counter() - t0))
        return response

    @app.exception_handler(LookupError)
    async def not_found(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/quote")
    def quote(spec: Spec, store: Store):
        return training.predict_one(store, market(), upgrade(spec))

    @app.post("/feedback")
    def feedback(quotes: Annotated[list[dict], Body(max_length=MAX_BATCH)], store: Store):
        return training.feedback(store, market(), quotes)

    @app.get("/monitoring")
    def monitoring(store: Store):
        return training.monitor(store)

    @app.post("/retrain")
    def retrain(store: Store, force: bool = False):
        r = training.retrain(store, market(), force=force)
        return {k: r.get(k) for k in ["promoted", "version", "status", "window", "champion", "challenger", "reason"]
                if k in r}

    @app.get("/models")
    def models(store: Store):
        champion = store.models.champion()
        out = []
        for v in store.models.versions():
            m = store.models.metadata(v)
            out.append({"version": v, "champion": v == champion, "created_at": m["created_at"],
                        "n_quotes": m["n_quotes"], "data_end": m["data_end"], "holdout": m["holdout"]})
        return out

    return app
