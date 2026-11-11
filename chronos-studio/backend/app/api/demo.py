"""
api/demo.py — Part 19 demo mode, seed studies and cancellable background jobs.

Two things the CRAFT prompt asks for at the end (§8 Part 19, §5.15) live here:

* **Demo mode / seed data.** Each scenario is a complete, narrated "example
  analysis" that replays a bundled seed dataset through the whole S0→S15
  pipeline. A non-programmer clicks one button and gets a fully populated,
  explorable study — every stage, result and report is real (it is computed by
  the same endpoints a manual run uses), not a mocked screenshot.
* **Heavy work as background jobs with progress + cancel.** The replay runs on
  a worker thread. `start` returns immediately; `status` reports labelled
  progress, per-step timings and errors; `cancel` stops between steps. The UI
  never blocks. CV grids and model fits are exactly the heavy steps in the
  recipe, so this is the same machinery §5.15 asks for.

Privacy: jobs live only in this process's memory, are swept after one hour, and
hold no user data (only a session id and step labels). Sessions themselves are
TTL-bounded and hard-deletable — see `core/session.py` and `COPY.privacy`.
"""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.session import get_store

router = APIRouter(prefix="/demo", tags=["demo"])


# --------------------------------------------------------------------------- #
# Scenarios — the seed studies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemoScenario:
    id: str
    title: str
    example_id: str
    blurb: str
    teaches: str
    models: tuple[str, ...]
    horizon: int
    folds: int


SCENARIOS: tuple[DemoScenario, ...] = (
    DemoScenario(
        id="seasonal_demand",
        title="Seasonal demand (AirPassengers)",
        example_id="airpassengers",
        blurb=(
            "144 monthly airline passengers, 1949–1960. Trend plus a seasonal "
            "swing that grows with the level — the classic case for a log/Box–Cox "
            "transform, seasonal differencing and multiplicative ETS."
        ),
        teaches=(
            "Why two stationarity tests must agree, why MAPE is not the only "
            "error metric, and why a seasonal naive benchmark is mandatory."
        ),
        models=("ets", "arima", "drift", "mean"),
        horizon=12,
        folds=3,
    ),
    DemoScenario(
        id="marketing_mix",
        title="Traffic & marketing spend",
        example_id="traffic_spend",
        blurb=(
            "15 years of monthly store traffic with a marketing-spend column. "
            "Adds an exogenous-regressor branch and a correlation/CCF study."
        ),
        teaches=(
            "How an exogenous scenario path is built, how rolling-origin CV "
            "keeps the test window untouched, and how residual adequacy gates "
            "a forecast."
        ),
        models=("ets", "arima", "arimax", "drift", "mean"),
        horizon=6,
        folds=3,
    ),
    DemoScenario(
        id="intermittent_parts",
        title="Intermittent spare-parts demand",
        example_id="spare_parts",
        blurb=(
            "A spare-parts series that is 72% zeros. Ordinary MAPE and Gaussian "
            "intervals are meaningless here; the app switches to Croston-style "
            "and count-aware methods."
        ),
        teaches=(
            "Why the metric-selection policy bans MAPE on zero-heavy data, why "
            "a discrete predictive distribution is needed, and why the naive "
            "benchmark still has to be fitted."
        ),
        models=("croston", "naive", "mean"),
        horizon=6,
        folds=3,
    ),
)

SCENARIO_BY_ID = {s.id: s for s in SCENARIOS}


@dataclass(frozen=True)
class DemoStep:
    """One labelled call in the replay. Pure data: the worker and the tests
    share this definition, so a demo can never drift from what is tested."""

    label: str
    stage: str
    method: Literal["GET", "POST"]
    path: str
    body: dict[str, Any] | None = None


def build_steps(scenario_id: str) -> list[DemoStep]:
    """The exact S0→S15 replay for a scenario (§8 Part 19: 'replay a full study')."""
    sc = SCENARIO_BY_ID.get(scenario_id)
    if sc is None:
        raise KeyError(scenario_id)
    models = list(sc.models)
    h = sc.horizon
    f = sc.folds
    return [
        DemoStep("Load the seed dataset", "S0", "POST", "/api/v1/data/example",
                 {"example_id": sc.example_id}),
        DemoStep("Describe the series", "S1", "POST", "/api/v1/analysis/profile", {}),
        DemoStep("Ask the preprocessing advisor", "S2", "POST", "/api/v1/analysis/advice", {}),
        DemoStep("Decompose trend, season and remainder", "S4", "POST",
                 "/api/v1/analysis/decompose", {"method": "stl"}),
        DemoStep("Run the stationarity battery", "S5", "POST", "/api/v1/analysis/tests", {}),
        # Dependency order matters: S6 needs S5, S7 needs S5+S6 and S8 needs S7
        # (core/session.py::STAGE_DEPENDENCIES). Running S7 after S8 would mark
        # the gate stale, so the replay follows the same order a human must.
        DemoStep("Correlate and cross-correlate", "S6", "GET", "/api/v1/analysis/correlate"),
        DemoStep("Rank the next-step actions", "S7", "GET", "/api/v1/analysis/next"),
        DemoStep("Check the readiness gate", "S8", "POST", "/api/v1/analysis/gate", {}),
        DemoStep("Suggest eligible models", "S9", "POST", "/api/v1/analysis/suggest", {}),
        DemoStep("Read the model suggestions", "S9", "GET", "/api/v1/analysis/suggest"),
        DemoStep("Collect the computed facts", "S9", "GET", "/api/v1/analysis/facts"),
        DemoStep("Fit candidates and the mandatory baselines", "S10", "POST",
                 "/api/v1/analysis/fit", {"model_ids": models}),
        DemoStep(f"Rolling-origin CV ({f} folds, horizon {h})", "S11", "POST",
                 "/api/v1/analysis/select",
                 {"horizon": h, "folds": f, "models": models}),
        DemoStep("Residual adequacy diagnostics", "S12", "POST",
                 "/api/v1/analysis/diagnose", {}),
        DemoStep(f"Forecast {h} steps with intervals", "S13", "POST",
                 "/api/v1/analysis/forecast", {"horizon": h}),
        # MIN_PATHS (ts/forecast/intervals.py) is 2000; fewer simulation paths
        # are refused with HTTP 422, so the seed study uses the exact floor.
        DemoStep("Out-of-sample evaluation", "S14", "POST", "/api/v1/analysis/evaluate",
                 {"horizon": h, "folds": f, "n_paths": 2_000, "reps": 1}),
        DemoStep("Finalize the manuscript-readiness report", "S15", "POST",
                 "/api/v1/export/finalize", {}),
    ]


# --------------------------------------------------------------------------- #
# Job registry
# --------------------------------------------------------------------------- #
@dataclass
class DemoJob:
    job_id: str
    scenario_id: str
    session_id: str
    steps: list[DemoStep] = field(default_factory=list)
    state: Literal["queued", "running", "complete", "cancelled", "failed"] = "queued"
    done: int = 0
    current: str = ""
    finished: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict[str, Any]:
        now = self.finished_at or datetime.now(timezone.utc)
        return {
            "job_id": self.job_id,
            "scenario_id": self.scenario_id,
            "session_id": self.session_id,
            "state": self.state,
            "done": self.done,
            "total": len(self.steps),
            "current": self.current,
            "percent": round(100.0 * self.done / max(1, len(self.steps)), 1),
            "labels": [s.label for s in self.steps],
            "finished": list(self.finished),
            "errors": list(self.errors),
            "elapsed_s": round(max(0.0, (now - self.created_at).total_seconds()), 2),
        }


_JOBS: dict[str, DemoJob] = {}
_LOCK = threading.Lock()
_IDS = itertools.count(1)
_JOB_TTL = timedelta(hours=1)


def _sweep() -> None:
    """Bounded in-memory registry: no persistence beyond the session TTL."""
    cutoff = datetime.now(timezone.utc) - _JOB_TTL
    with _LOCK:
        for jid in [k for k, v in _JOBS.items() if v.created_at < cutoff]:
            _JOBS.pop(jid, None)


def _worker(job: DemoJob) -> None:
    """Replay the recipe on a worker thread. Never raises into the server."""
    # Imported lazily: importing app.main at module load time would create a
    # circular import (main imports this router).
    from starlette.testclient import TestClient

    from app.main import app

    with _LOCK:
        job.state = "running"
        job.started_at = datetime.now(timezone.utc)
    headers = {"X-Session-Id": job.session_id}
    try:
        with TestClient(app) as client:
            for step in job.steps:
                if job.cancel.is_set():
                    break
                with _LOCK:
                    job.current = step.label
                t0 = time.perf_counter()
                ok, err = True, ""
                try:
                    kwargs: dict[str, Any] = {"headers": headers}
                    if step.method == "POST":
                        kwargs["json"] = step.body or {}
                    response = client.request(step.method, step.path, **kwargs)
                    if response.status_code >= 400:
                        ok = False
                        err = f"HTTP {response.status_code}: {response.text[:180]}"
                except Exception as exc:  # a demo must never take the server down
                    ok = False
                    err = f"{type(exc).__name__}: {exc}"[:200]
                record = {
                    "label": step.label,
                    "stage": step.stage,
                    "status": "ok" if ok else "error",
                    "seconds": round(time.perf_counter() - t0, 3),
                    "error": err,
                }
                with _LOCK:
                    job.finished.append(record)
                    job.done += 1
                    if err:
                        job.errors.append(f"{step.label}: {err}")
    except Exception as exc:  # pragma: no cover - defensive
        with _LOCK:
            job.errors.append(f"worker: {type(exc).__name__}: {exc}"[:200])
    finally:
        with _LOCK:
            if job.cancel.is_set():
                job.state = "cancelled"
            elif job.errors:
                job.state = "failed"
            else:
                job.state = "complete"
            job.current = ""
            job.finished_at = datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
class StartRequest(BaseModel):
    scenario_id: str = Field(default="seasonal_demand")


class StatusResponse(BaseModel):
    job_id: str
    scenario_id: str
    session_id: str
    state: str
    done: int
    total: int
    current: str
    percent: float
    labels: list[str]
    finished: list[dict[str, Any]]
    errors: list[str]
    elapsed_s: float


@router.get("/scenarios")
def list_scenarios() -> dict[str, Any]:
    """Metadata for the UI, including the exact ordered step labels."""
    return {
        "scenarios": [
            {
                "id": s.id,
                "title": s.title,
                "example_id": s.example_id,
                "blurb": s.blurb,
                "teaches": s.teaches,
                "models": list(s.models),
                "horizon": s.horizon,
                "n_steps": len(build_steps(s.id)),
                "steps": [st.label for st in build_steps(s.id)],
                "stages": [st.stage for st in build_steps(s.id)],
            }
            for s in SCENARIOS
        ]
    }


@router.post("/start", response_model=StatusResponse)
def start(request: StartRequest | None = None) -> StatusResponse:
    """Create a fresh session and start its replay in the background."""
    _sweep()
    sid = (request.scenario_id if request else None) or "seasonal_demand"
    if sid not in SCENARIO_BY_ID:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown demo scenario '{sid}'. Choose one of {sorted(SCENARIO_BY_ID)}.",
        )
    session_id = get_store().create().id
    job = DemoJob(
        job_id=f"demo-{next(_IDS):04d}-{uuid.uuid4().hex[:8]}",
        scenario_id=sid,
        session_id=session_id,
        steps=build_steps(sid),
    )
    with _LOCK:
        _JOBS[job.job_id] = job
    threading.Thread(target=_worker, args=(job,), name=f"chronos-{job.job_id}",
                     daemon=True).start()
    return StatusResponse(**job.public())


@router.get("/status", response_model=StatusResponse)
def status(job_id: str) -> StatusResponse:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="That demo job is no longer available.")
        return StatusResponse(**job.public())


@router.post("/cancel", response_model=StatusResponse)
def cancel(job_id: str) -> StatusResponse:
    """Cooperative cancel: the current step finishes, then the thread stops."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="That demo job is no longer available.")
        if job.state in {"queued", "running"}:
            job.cancel.set()
        return StatusResponse(**job.public())


@router.delete("/jobs/{job_id}")
def forget(job_id: str) -> dict[str, Any]:
    """Explicit delete-my-demo (privacy). Does not delete the analysis session."""
    with _LOCK:
        removed = _JOBS.pop(job_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail="That demo job is no longer available.")
    return {"deleted": job_id, "remaining": len(_JOBS)}


__all__ = [
    "SCENARIOS", "DemoScenario", "DemoStep", "build_steps",
    "list_scenarios", "start", "status", "cancel", "router",
]
