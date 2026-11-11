"""
main.py — Chronos Studio API entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000

Design notes:
  * Binds 0.0.0.0 so the workspace preview proxy can reach it.
  * CORS is permissive in dev (configurable) because the UI is served from a
    different origin than the API in every deployment we care about.
  * Every ChronosError becomes a uniform JSON body; unexpected exceptions
    become a 500 with a plain-English message and a collapsed trace.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import session as session_api
from app.core.config import get_settings, optional_features
from app.core.errors import ChronosError, as_error_payload
from app.core.schemas import SCHEMA_VERSION

logging.basicConfig(
    level=logging.DEBUG if get_settings().debug else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("chronos")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "A free, beginner-friendly, web-based time-series analysis workbench — "
        "'JASP for time series'. Upload a series, get guided preprocessing advice, "
        "decomposition, stationarity verdicts, ACF/PACF, model suggestions, "
        "diagnostics, forecasts, accuracy evaluation and a manuscript-ready report."
    ),
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.allow_all_origins else settings.allowed_origins,
    allow_credentials=not settings.allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Process-Time-Ms", "X-Session-Id", "X-Report-Kind", "X-Report-Filename",
        "X-Report-Verdict", "X-Report-Ready", "X-Report-Score", "X-Report-Scope",
        "X-Report-Theme", "X-Report-Data-Hash",
    ],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = int((time.perf_counter() - t0) * 1000)
    response.headers["X-Process-Time-Ms"] = str(ms)
    if ms > 3000:
        log.warning("slow request %s %s took %dms", request.method, request.url.path, ms)
    return response


@app.exception_handler(ChronosError)
async def chronos_error_handler(request: Request, exc: ChronosError) -> JSONResponse:
    log.info("handled error %s at %s: %s", exc.code, request.url.path, exc.plain)
    return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error at %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": as_error_payload(exc)})


# --------------------------------------------------------------------------- #
# Routers. Later parts append here — nothing else in this file changes.
# --------------------------------------------------------------------------- #
API_PREFIX = "/api/v1"
app.include_router(session_api.router, prefix=API_PREFIX)

DEFERRED_ROUTERS: list[str] = [
    # Part 1
    "app.api.datasets",
    # Part 2
    "app.api.eda",
    # Parts 3-7
    "app.api.preprocess", "app.api.decompose", "app.api.tests_stat", "app.api.correlate",
    "app.api.gate",
    # Parts 8-12
    "app.api.models",
    "app.api.select",
    # Part 13
    # (selection + diagnostics live in app.api.select)
    # Part 14
    "app.api.forecast",
    # Part 15
    "app.api.metrics",
    # Part 16 — plot/table catalogue, per-item exports and report-insert basket.
    # Part 18 consumes it; `app.api.catalog` does not own whole-report exports.
    "app.api.catalog",
    # Part 17 — grounded assistant (offline-first; live Gemini is optional).
    "app.api.ai",
    # Part 18 — whole-report builders, archive, scorecard and S15 finalization.
    "app.api.export",
    # Part 19 — demo mode, seed studies and cancellable background jobs.
    "app.api.demo",
]
for _mod in DEFERRED_ROUTERS:
    try:  # pragma: no cover - becomes live in later parts
        _m = __import__(_mod, fromlist=["router"])
        app.include_router(_m.router, prefix=API_PREFIX)
        log.info("mounted router %s", _mod)
    except ModuleNotFoundError as _exc:
        # Only a *missing module* means "not built yet". An ImportError from inside
        # a built router is a real bug and must not be swallowed by this loop.
        if _exc.name != _mod:
            raise
        log.debug("router %s not built yet (deferred)", _mod)


@app.get("/", include_in_schema=False)
def root() -> dict[str, object]:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "schema_version": SCHEMA_VERSION,
        "docs": "/docs",
        "api": API_PREFIX,
        "ai_configured": settings.has_gemini_key(),
        "optional_features": optional_features(),
    }


__all__ = ["app"]
