"""
core/session.py — in-memory session store and the stage-gating rules.

Part 0 ships the state machine and the store; the stage *handlers* arrive in
later parts. Nothing here performs statistics.

Two deliberate design points:

1. `require_stage()` is the only gate. Later parts call it instead of
   hand-rolling "have you done X yet?" checks, so the dependency map lives in
   exactly one place (`STAGE_DEPENDENCIES`).
2. Re-running a stage marks downstream stages `stale` (see
   `PipelineState.mark`). Results are keyed by (stage, series_version) so a
   stale result is never silently reused after a preprocessing change.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from app.core.config import get_settings, optional_features
from app.core.errors import SessionNotFoundError, StageOrderError
from app.core.schemas import (
    AnyResult,
    PipelineState,
    STAGE_ORDER,
    SessionData,
    Stage,
    StageState,
)
from app.core.seeds import seed_everything

# --------------------------------------------------------------------------- #
# Which stages must be complete before a given stage may run.               #
# Stage order alone is not enough: e.g. S7 needs both S5 and S6, and S9     #
# needs the S8 gate to have issued a verdict.                              #
# --------------------------------------------------------------------------- #
STAGE_DEPENDENCIES: dict[Stage, tuple[Stage, ...]] = {
    Stage.S0_INGEST: (),
    Stage.S1_PROFILE: (Stage.S0_INGEST,),
    Stage.S2_ADVISE_PREP: (Stage.S1_PROFILE,),
    # S3 is the *display* of the prepare step, and its UI reads the S2 advice. Applying a
    # treatment programmatically only requires the profile (enforced in the endpoint).
    Stage.S3_PREPARE: (Stage.S2_ADVISE_PREP,),
    Stage.S4_DECOMPOSE: (Stage.S1_PROFILE,),
    Stage.S5_TEST_STATION: (Stage.S1_PROFILE,),
    Stage.S6_CORRELATE: (Stage.S5_TEST_STATION,),
    Stage.S7_ADVISE_NEXT: (Stage.S5_TEST_STATION, Stage.S6_CORRELATE),
    Stage.S8_GATE: (Stage.S5_TEST_STATION, Stage.S7_ADVISE_NEXT),
    Stage.S9_SUGGEST_MODEL: (Stage.S8_GATE,),
    Stage.S10_FIT: (Stage.S9_SUGGEST_MODEL,),
    Stage.S11_SELECT: (Stage.S10_FIT,),
    Stage.S12_DIAGNOSE: (Stage.S11_SELECT,),
    Stage.S13_FORECAST: (Stage.S11_SELECT,),
    Stage.S14_EVALUATE: (Stage.S13_FORECAST,),
    Stage.S15_FINALIZE: (Stage.S14_EVALUATE, Stage.S12_DIAGNOSE),
}


class Session:
    """One analysis. Holds state, results and the fitted objects (server-side only)."""

    def __init__(self, session_id: str | None = None, seed: int | None = None) -> None:
        settings = get_settings()
        self.id = session_id or uuid.uuid4().hex
        self.seed = seed if seed is not None else settings.default_seed
        seed_everything(self.seed)

        self.state = PipelineState(session_id=self.id, seed=self.seed)
        self.state.library_versions = _library_versions()
        self.state.current_stage = Stage.S0_INGEST

        # results[stage][result_id] -> AnyResult
        self.results: dict[Stage, dict[str, AnyResult]] = defaultdict(dict)
        # advisor reports per stage
        self.advice: dict[Stage, Any] = {}
        # opaque server-side objects: time series, fitted models, figure data
        self.objects: dict[str, Any] = {}
        # (stage, series_version) cache so stale results are detectable
        self.result_version: dict[str, int] = {}

        self.created_at = datetime.now(timezone.utc)
        self.last_seen = self.created_at
        self.ai_requests = 0

    # -- lifecycle ---------------------------------------------------------- #
    def touch(self) -> None:
        self.last_seen = datetime.now(timezone.utc)

    def age(self) -> timedelta:
        return datetime.now(timezone.utc) - self.last_seen

    # -- stage gating ------------------------------------------------------- #
    def require_stage(self, stage: Stage, allow_stale: bool = False) -> None:
        """
        Raise StageOrderError unless every dependency of `stage` has been run.

        `allow_stale` exists for steps that *modify* the series: applying a second
        treatment must not be blocked by the fact that the first one correctly marked
        the upstream analysis stale. The distinction that matters is "never run" versus
        "run, and now out of date" — the first is a mistake, the second is the pipeline
        working as designed.
        """
        ok = {"complete", "stale"} if allow_stale else {"complete"}
        missing = [
            dep
            for dep in STAGE_DEPENDENCIES[stage]
            if self.state.stages[dep.value].status not in ok
        ]
        if missing:
            raise StageOrderError(
                plain=(
                    f"'{stage.value}' needs {', '.join(m.value for m in missing)} "
                    "to be finished first."
                ),
                fix=(
                    "Use the pipeline stepper to complete "
                    f"{', '.join(m.value for m in missing)} before continuing."
                ),
                context={"stage": stage.value, "missing": [m.value for m in missing]},
            )

    def mark_stage(self, stage: Stage, status: str, message: str = "") -> None:
        self.state.mark(stage, status, message)  # type: ignore[arg-type]
        self.state.current_stage = stage
        self.state.updated_at = datetime.now(timezone.utc)

    # -- results ------------------------------------------------------------ #
    def add_result(self, result: AnyResult) -> None:
        """Store a result and record its series version so staleness is detectable."""
        stage = _as_stage(result.stage)
        self.results[stage][result.id] = result
        self.result_version[result.id] = self.state.series_version
        if result.id not in self.state.result_ids:
            self.state.result_ids.append(result.id)
        st: StageState = self.state.stages[stage.value]
        st.n_results = len(self.results[stage])
        for cav in result.caveats:
            if cav not in self.state.caveats:
                self.state.caveats.append(cav)

    def get_result(self, result_id: str) -> AnyResult | None:
        # Snapshot the outer mapping: a concurrent finalization may add a new
        # stage bucket while another request is reading stored results.
        for stage_results in list(self.results.values()):
            if result_id in stage_results:
                return stage_results[result_id]
        return None

    def results_for(self, stage: Stage) -> list[AnyResult]:
        return list(self.results.get(stage, {}).values())

    def all_results(self) -> Iterable[AnyResult]:
        for stage_results in list(self.results.values()):
            yield from list(stage_results.values())

    def export_state(self) -> dict[str, Any]:
        """Everything needed to render the app after a reload."""
        return {
            "session_id": self.id,
            "created_at": self.created_at.isoformat(),
            "seed": self.seed,
            "pipeline": self.state.model_dump(mode="json"),
            "progress": self.state.progress(),
            "dataset": self.state.dataset.model_dump(mode="json") if self.state.dataset else None,
            "preprocessing": [p.model_dump(mode="json") for p in self.state.preprocessing],
            "series_version": self.state.series_version,
            "optional_features": optional_features(),
            "ai_configured": get_settings().has_gemini_key(),
        }


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #
class SessionStore:
    """
    Thread-safe, in-memory, TTL-bounded. Deliberately boring.

    v1 has no database (§4.1). If you later add Redis, this class is the only
    thing that changes — every caller goes through `get_store()`.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._last_sweep = time.time()

    # -- CRUD --------------------------------------------------------------- #
    def create(self, seed: int | None = None) -> Session:
        with self._lock:
            self._sweep()
            if len(self._sessions) >= get_settings().max_sessions:
                self._evict_oldest()
            s = Session(seed=seed)
            self._sessions[s.id] = s
            return s

    def get(self, session_id: str) -> Session:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                raise SessionNotFoundError()
            if s.age() > timedelta(hours=get_settings().session_ttl_hours):
                del self._sessions[session_id]
                raise SessionNotFoundError()
            s.touch()
            return s

    def has(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def delete(self, session_id: str) -> bool:
        """The 'delete my session' privacy button (§5.15)."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- housekeeping ------------------------------------------------------- #
    def _sweep(self) -> None:
        now = time.time()
        if now - self._last_sweep < 60:
            return
        self._last_sweep = now
        ttl = timedelta(hours=get_settings().session_ttl_hours)
        expired = [sid for sid, s in self._sessions.items() if s.age() > ttl]
        for sid in expired:
            del self._sessions[sid]

    def _evict_oldest(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.items(), key=lambda kv: kv[1].last_seen)
        del self._sessions[oldest[0]]


_STORE: SessionStore | None = None


def get_store() -> SessionStore:
    global _STORE
    if _STORE is None:
        _STORE = SessionStore()
    return _STORE


def _as_stage(stage: Stage | str) -> Stage:
    """
    Results are serialised with `use_enum_values=True`, so `result.stage` arrives
    as a plain string. One conversion point, used everywhere, beats sprinkling
    isinstance checks through the codebase.
    """
    return stage if isinstance(stage, Stage) else Stage(stage)


def _library_versions() -> dict[str, str]:
    """Recorded with every session so a report can state what produced it."""
    versions: dict[str, str] = {}
    for mod in ("numpy", "pandas", "scipy", "statsmodels"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - optional
            versions[mod] = "absent"
    try:
        import app

        versions["chronos_studio"] = app.__version__
    except Exception:  # pragma: no cover
        versions["chronos_studio"] = "dev"
    return versions


__all__ = [
    "_as_stage",
    "Session",
    "SessionStore",
    "get_store",
    "STAGE_DEPENDENCIES",
    "STAGE_ORDER",
]
