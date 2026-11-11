"""
api/preprocess.py — S2 (advise) and S3 (apply) endpoints.

  POST /analysis/advice            run the S2 advisor, store the report
  GET  /analysis/advice            read it back (a page reload costs nothing)
  POST /preprocess/preview         run one candidate operation, do NOT commit
  POST /preprocess/apply           commit it, versioned and undoable
  POST /preprocess/undo            step back one version
  POST /preprocess/redo            re-apply the last undone version (Part 19)
  GET  /preprocess/state           the applied steps + the current series

The split matters: the preview endpoint executes the *same* function the apply
endpoint does, so what the user sees before committing is exactly what happens.
A preview never touches the session; only `apply` mutates state and bumps the
series version, which is what marks S1/downstream stages stale (§4.2).
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import session_dep
from app.core.errors import ChronosError
from app.core.schemas import (
    AnalysisResponse,
    DataClass,
    PreprocessingRecord,
    Stage,
)
from app.core.session import Session
from app.ts.preprocess.advise import PrepAdvice, advise_preprocessing
from app.ts.preprocess.ops import REGISTRY, STEP_OF, apply_op

router = APIRouter(tags=["analysis"])

#: Where the untouched, as-loaded series lives, so undo can always reach version 0.
ORIGINAL = "series_original"
CURRENT = "series"
#: Part 19: undone records, newest last. A fresh `apply` branches history and
#: clears this stack, exactly like undo/redo in an editor.
REDO = "preprocess_redo"


# --------------------------------------------------------------------------- #
# S2 — the advisor
# --------------------------------------------------------------------------- #
class AdviceRequest(BaseModel):
    audience: str = "beginner"


class AdviceResponse(BaseModel):
    stage: str
    headline: str
    confidence: str
    next_stage_ready: bool
    blockers: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    facts_used: dict[str, Any]
    narrative: str
    results: list[dict[str, Any]]
    caveats: list[str]
    duration_ms: int


def _require_series(session: Session) -> pd.Series:
    series: pd.Series | None = session.objects.get(CURRENT)
    if session.state.dataset is None or series is None:
        raise ChronosError(
            plain="There is no data loaded in this session yet.",
            fix="Load a file or an example dataset in Step 0 first.",
        )
    return series


def _require_applyable(session: Session) -> None:
    """
    Applying a treatment needs the data and a completed profile — not the advisor.

    S3's *UI* lives behind S2 (its suggestions come from there), but the operation
    itself is a data change the user may make deliberately. The only thing that must
    never be true is "no data" or "no profile", because both would make the applied
    step unrecordable.
    """
    if session.state.stages[Stage.S1_PROFILE.value].status not in {"complete", "stale"}:
        raise ChronosError(
            plain="The series has not been profiled yet, so a treatment cannot be recorded against it.",
            fix="Run Step 1 (Explore) first — it takes a second and gives every later step its baseline.",
            code="preprocess.no_profile",
        )


@router.post("/analysis/advice", response_model=AdviceResponse)
def run_advice(
    body: AdviceRequest | None = None,
    session: Session = Depends(session_dep),
) -> AdviceResponse:
    """Run S2. Idempotent: re-running replaces the advice rather than stacking it."""
    t0 = time.perf_counter()
    series = _require_series(session)
    session.require_stage(Stage.S2_ADVISE_PREP)

    ds = session.state.dataset
    assert ds is not None
    eda_facts: dict[str, Any] = session.objects.get("eda_facts") or {}
    if not eda_facts:
        raise ChronosError(
            plain="Step 1 has not been run yet, so there is nothing to base advice on.",
            fix="Open Step 1 (Explore) and let it finish, then come back.",
            code="advice.no_profile",
        )

    # Guard the one error that would be invisible: advising on a profile of a series the
    # user has since transformed.
    facts_version = session.objects.get("eda_series_version")
    if facts_version is not None and facts_version != session.state.series_version:
        raise ChronosError(
            plain="Step 1 describes an earlier version of the series.",
            fix="Re-run Step 1 (Explore) so the advice is about the data you have now.",
            code="advice.stale_profile",
        )

    applied = tuple(p.step for p in session.state.preprocessing)
    advice: PrepAdvice = advise_preprocessing(
        series,
        eda_facts=eda_facts,
        name=ds.name,
        frequency_label=ds.frequency.label,
        period=ds.frequency.period,
        data_class=DataClass(ds.data_class),
        applied_ops=applied,
        audience=(body.audience if body else "beginner"),
    )

    session.results[Stage.S2_ADVISE_PREP] = {}
    for result in advice.results:
        session.add_result(result)
    session.advice[Stage.S2_ADVISE_PREP] = advice.report
    session.objects["prep_facts"] = advice.facts
    session.mark_stage(
        Stage.S2_ADVISE_PREP, "complete",
        f"{len(advice.report.recommendations)} suggestions, {len(advice.report.blockers)} blockers",
    )

    return AdviceResponse(
        stage=Stage.S2_ADVISE_PREP.value,
        headline=advice.report.headline,
        confidence=_conf(advice.report.confidence),
        next_stage_ready=advice.report.next_stage_ready,
        blockers=[b.model_dump(mode="json") for b in advice.report.blockers],
        recommendations=[r.model_dump(mode="json") for r in advice.report.recommendations],
        facts_used=advice.facts,
        narrative=advice.report.narrative,
        results=[r.model_dump(mode="json") for r in advice.results],
        caveats=_caveats(advice),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/analysis/advice")
def read_advice(session: Session = Depends(session_dep)) -> dict[str, Any]:
    report = session.advice.get(Stage.S2_ADVISE_PREP)
    if report is None:
        return {"stage": Stage.S2_ADVISE_PREP.value, "status": "not_started",
                "headline": "", "recommendations": [], "blockers": [],
                "facts_used": {}, "results": []}
    return {
        "stage": Stage.S2_ADVISE_PREP.value,
        "status": "complete",
        "headline": report.headline,
        "confidence": _conf(report.confidence),
        "next_stage_ready": report.next_stage_ready,
        "recommendations": [r.model_dump(mode="json") for r in report.recommendations],
        "blockers": [r.model_dump(mode="json") for r in report.blockers],
        "facts_used": report.facts_used,
        "narrative": report.narrative,
        "results": [r.model_dump(mode="json") for r in session.results_for(Stage.S2_ADVISE_PREP)],
    }


def _conf(c: Any) -> str:
    return c.value if hasattr(c, "value") else str(c)


def _caveats(advice: PrepAdvice) -> list[str]:
    out = []
    if not advice.report.next_stage_ready:
        out.append(
            "There are open decisions above. You can still continue, but the analysis will "
            "carry them as caveats."
        )
    return out


# --------------------------------------------------------------------------- #
# S3 — preview and apply
# --------------------------------------------------------------------------- #
class OpRequest(BaseModel):
    op: str = Field(..., description="Key in ts/preprocess/ops.REGISTRY")
    params: dict[str, Any] = Field(default_factory=dict)
    rule_id: str | None = Field(None, description="The advisor rule this implements")
    reason: str = Field("", description="Why the operation is being applied, in the user's words")


class PreviewResponse(BaseModel):
    op: str
    step: str
    method: str
    plain: str
    n_before: int
    n_after: int
    changed: bool
    preview: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
    reversible: bool = True


@router.post("/preprocess/preview", response_model=PreviewResponse)
def preview_op(body: OpRequest, session: Session = Depends(session_dep)) -> PreviewResponse:
    """Execute a candidate operation on a COPY. The session is not modified."""
    series = _require_series(session)
    if body.op not in REGISTRY:
        raise ChronosError(
            plain=f"'{body.op}' is not a preprocessing operation this app knows.",
            fix="Choose one of the operations offered in Step 2.",
            code="preprocess.unknown_op",
        )

    params = dict(body.params)
    # sensible, explicit defaults so a preview can never run with a hidden choice
    if "period" not in params:
        params["period"] = session.state.dataset.frequency.period if session.state.dataset else None
    try:
        result = apply_op(body.op, series, params)
    except Exception as exc:
        raise ChronosError(
            plain="That operation could not be applied to this series.",
            fix="Try a different treatment; the data has not been changed.",
            technical=f"{type(exc).__name__}: {exc}",
            code="preprocess.failed",
        ) from exc

    preview = [
        {"time": str(t), "value": (None if pd.isna(v) else float(v))}
        for t, v in result.series.items()
    ][:2000]

    return PreviewResponse(
        op=body.op,
        step=STEP_OF.get(body.op, "transform"),
        method=result.method,
        plain=result.plain,
        n_before=int(series.notna().sum()),
        n_after=int(result.series.notna().sum()),
        changed=not _identical(series, result.series),
        preview=preview,
        warnings=result.warnings,
        reversible=result.reversible,
    )


@router.post("/preprocess/apply", response_model=AnalysisResponse)
def apply_operation(body: OpRequest, session: Session = Depends(session_dep)) -> AnalysisResponse:
    """
    Commit an operation. Versioned, recorded with its reason, and reversible.

    Bumping the series version is what marks S1 stale — the EDA descriptions no
    longer describe this data, and the UI must say so rather than quietly reusing them.
    """
    t0 = time.perf_counter()
    series = _require_series(session)
    if body.op not in REGISTRY:
        raise ChronosError(
            plain=f"'{body.op}' is not a preprocessing operation this app knows.",
            fix="Choose one of the operations offered in Step 2.",
            code="preprocess.unknown_op",
        )
    # Applying a treatment needs a loaded series and a completed S1 — it does *not*
    # need the advisor to have run. Advice is a recommendation; treating it as a
    # prerequisite would mean a user who knows what they want cannot proceed, and it
    # would make the second treatment in a chain fail whenever the first one correctly
    # marked the upstream stages stale.
    _require_applyable(session)
    session.objects.setdefault(ORIGINAL, series.copy())
    # A new treatment creates a new branch: the undone history no longer applies.
    session.objects[REDO] = []

    params = dict(body.params)
    if "period" not in params:
        params["period"] = session.state.dataset.frequency.period if session.state.dataset else None

    result = apply_op(body.op, series, params)
    session.objects[CURRENT] = result.series

    record = PreprocessingRecord(
        version=len(session.state.preprocessing) + 1,
        step=STEP_OF.get(body.op, "transform"),      # type: ignore[arg-type]
        method=result.method,
        params=params,
        reason=body.reason or _reason_for(body.rule_id, session),
        advisor_rule_id=body.rule_id,
        reversible=result.reversible,
    )
    session.state.preprocessing.append(record)
    session.state.series_version += 1
    # `mark_stage` marks every later stage stale for us; what it cannot know is that the
    # *advice* which produced this op was computed on the series as it was before, so S2
    # is explicitly invalidated too (it re-runs, and its rules see `already_applied`).
    session.mark_stage(Stage.S3_PREPARE, "complete", result.plain[:180])
    if Stage.S2_ADVISE_PREP.value in session.state.stages:
        st = session.state.stages[Stage.S2_ADVISE_PREP.value]
        if st.status == "complete":
            st.status = "stale"
            st.message = "The series changed; re-run to refresh the advice."

    return AnalysisResponse(
        stage=Stage.S3_PREPARE,
        advisor=None,
        results=[],
        pipeline=session.state.progress(),
        caveats=result.warnings,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.post("/preprocess/undo", response_model=AnalysisResponse)
def undo(session: Session = Depends(session_dep)) -> AnalysisResponse:
    """Step back one version. Undo is the reason every op is pure."""
    if not session.state.preprocessing:
        raise ChronosError(
            plain="There is nothing to undo — no preprocessing has been applied.",
            fix="Apply a treatment in Step 2 first.",
            code="preprocess.nothing_to_undo",
        )
    dropped = session.state.preprocessing.pop()
    # Remember it so `redo` can put it back verbatim (Part 19).
    session.objects.setdefault(REDO, []).append(dropped)
    # Replay the remaining chain from the original series: cheap, and impossible to
    # get wrong (no reverse-operation bookkeeping).
    original: pd.Series | None = session.objects.get(ORIGINAL)
    if original is None:
        raise ChronosError(
            plain="The original series is no longer in memory, so this step cannot be undone.",
            fix="Reload the file in Step 0 to start from the untouched data.",
            code="preprocess.no_original",
        )
    series = original.copy()
    for rec in session.state.preprocessing:
        series = apply_op(_op_for(rec.method, rec.step), series, rec.params).series
    session.objects[CURRENT] = series
    session.state.series_version += 1
    if Stage.S2_ADVISE_PREP.value in session.state.stages:
        st = session.state.stages[Stage.S2_ADVISE_PREP.value]
        if st.status == "complete":
            st.status = "stale"
            st.message = "The series changed; re-run to refresh the advice."
    session.mark_stage(Stage.S3_PREPARE, "complete" if session.state.preprocessing else "in_progress",
                       f"{len(session.state.preprocessing)} treatment(s) applied")

    return AnalysisResponse(
        stage=Stage.S3_PREPARE,
        advisor=None,
        results=[],
        pipeline=session.state.progress(),
        caveats=[f"Undid '{dropped.method}' (version {dropped.version})."],
        duration_ms=0,
    )


@router.post("/preprocess/redo", response_model=AnalysisResponse)
def redo(session: Session = Depends(session_dep)) -> AnalysisResponse:
    """Re-apply the most recently undone treatment (Part 19 undo/redo)."""
    stack: list[PreprocessingRecord] = session.objects.get(REDO) or []
    if not stack:
        raise ChronosError(
            plain="There is nothing to redo — undo a treatment first.",
            fix="Use Undo on a preprocessing step, then Redo to put it back.",
            code="preprocess.nothing_to_redo",
        )
    series = _require_series(session)
    rec = stack.pop()
    result = apply_op(_op_for(rec.method, rec.step), series, rec.params)
    session.objects[CURRENT] = result.series
    session.state.preprocessing.append(PreprocessingRecord(
        version=len(session.state.preprocessing) + 1,
        step=rec.step,                                  # type: ignore[arg-type]
        method=rec.method,
        params=dict(rec.params),
        reason=rec.reason,
        advisor_rule_id=rec.advisor_rule_id,
        reversible=rec.reversible,
    ))
    session.state.series_version += 1
    if Stage.S2_ADVISE_PREP.value in session.state.stages:
        st = session.state.stages[Stage.S2_ADVISE_PREP.value]
        if st.status == "complete":
            st.status = "stale"
            st.message = "The series changed; re-run to refresh the advice."
    session.mark_stage(Stage.S3_PREPARE, "complete",
                       f"{len(session.state.preprocessing)} treatment(s) applied")
    return AnalysisResponse(
        stage=Stage.S3_PREPARE,
        advisor=None,
        results=[],
        pipeline=session.state.progress(),
        caveats=[f"Redid '{rec.method}' (version {len(session.state.preprocessing)})."],
        duration_ms=0,
    )


@router.get("/preprocess/state")
def preprocess_state(session: Session = Depends(session_dep)) -> dict[str, Any]:
    series: pd.Series | None = session.objects.get(CURRENT)
    return {
        "steps": [p.model_dump(mode="json") for p in session.state.preprocessing],
        "series_version": session.state.series_version,
        "n": int(series.notna().sum()) if series is not None else 0,
        "n_missing": int(series.isna().sum()) if series is not None else 0,
        "start": str(series.index.min()) if series is not None and len(series) else None,
        "end": str(series.index.max()) if series is not None and len(series) else None,
        "can_undo": bool(session.state.preprocessing),
        "can_redo": bool(session.objects.get(REDO)),
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _identical(a: pd.Series, b: pd.Series) -> bool:
    if len(a) != len(b) or not a.index.equals(b.index):
        return False
    va, vb = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return bool(np.allclose(va, vb, equal_nan=True))


def _reason_for(rule_id: str | None, session: Session) -> str:
    """Attach the recommendation's own words to the record, so the audit trail is readable."""
    if not rule_id:
        return "Applied by the user."
    report = session.advice.get(Stage.S2_ADVISE_PREP)
    if report is not None:
        for rule in list(report.recommendations) + list(report.blockers):
            if rule.id == rule_id:
                return f"{rule.recommend} — {rule.because}"
    return f"Applied by the user (rule {rule_id})."


def _op_for(method: str, step: str) -> str:
    """Map a recorded method back to a registry key, for undo replay."""
    for key, fn in REGISTRY.items():
        if key.endswith(method) or fn.__name__ == method:
            return key
    return {
        "impute": "impute.seasonal", "transform": "transform.none", "outlier": "outlier.flag",
        "smooth": "smooth.ma", "trim": "trim", "aggregate": "aggregate",
    }.get(step, "transform.none")
