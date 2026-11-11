"""
api/session.py — session lifecycle endpoints.

Deliberately thin: no statistics, no file parsing. Part 1 adds the dataset
endpoints in `api/datasets.py`; both mount under /api/v1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Header
from pydantic import BaseModel

from app.api.deps import session_dep
from app.core.config import get_settings, optional_features
from app.core.schemas import PipelineState, STAGE_LABELS, STAGE_ORDER, HealthResponse
from app.core.session import STAGE_DEPENDENCIES, get_store

router = APIRouter(tags=["session"])


class NewSessionResponse(BaseModel):
    session_id: str
    seed: int
    pipeline: PipelineState
    optional_features: dict[str, bool]
    ai_configured: bool


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    feats = optional_features()
    return HealthResponse(
        status="ok",
        version=get_settings().version,
        stages=len(STAGE_ORDER),
        ai_configured=get_settings().has_gemini_key(),
        optional_features=feats,
    )


@router.get("/pipeline", tags=["meta"])
def pipeline_map() -> dict[str, Any]:
    """
    The stage map, so the frontend stepper never hardcodes the pipeline.
    This is the single place that knows Part N's dependency rules.
    """
    return {
        "stages": [
            {
                "code": s.value,
                "label": STAGE_LABELS[s],
                "order": i,
                "depends_on": [d.value for d in STAGE_DEPENDENCIES[s]],
            }
            for i, s in enumerate(STAGE_ORDER)
        ]
    }


@router.post("/sessions", response_model=NewSessionResponse, status_code=201)
def create_session(seed: int | None = None) -> NewSessionResponse:
    s = get_store().create(seed=seed)
    return NewSessionResponse(
        session_id=s.id,
        seed=s.seed,
        pipeline=s.state,
        optional_features=optional_features(),
        ai_configured=get_settings().has_gemini_key(),
    )


@router.get("/sessions/{session_id}", tags=["session"])
def read_session(session_id: str) -> dict[str, Any]:
    s = get_store().get(session_id)
    return s.export_state()


@router.delete("/sessions/{session_id}", tags=["session"])
def delete_session(session_id: str) -> dict[str, Any]:
    """Privacy control: hard-delete this analysis from memory (§5.15)."""
    ok = get_store().delete(session_id)
    return {"deleted": ok, "session_id": session_id}


@router.get("/sessions/{session_id}/results", tags=["session"])
def all_results(session_id: str) -> dict[str, Any]:
    """Everything computed so far, grouped by stage, in one call."""
    s = get_store().get(session_id)
    return {
        stage.value: [r.model_dump(mode="json") for r in s.results_for(stage)]
        for stage in STAGE_ORDER
    }


@router.post("/sessions/{session_id}/results/{result_id}/explain", tags=["session"])
def explain_result(
    session_id: str,
    result_id: str,
    x_session_id: str = Header(..., alias="X-Session-Id"),
) -> dict[str, Any]:
    """
    Part 0 stub. Part 17 wires this to the AI assistant; until then it returns
    the deterministic fields of the result itself, so the UI works with no key.
    """
    s = get_store().get(session_id)
    r = s.get_result(result_id)
    if r is None:
        return {"found": False, "result_id": result_id}
    # Part 17 wires the old stub to the grounded offline generator. It still
    # needs no key, SDK or network call, and never mutates the analysis.
    from types import SimpleNamespace

    from app.ai.digest import build_digest
    from app.ai.grounding import validate_grounding
    from app.ai.narrative import generate_offline
    from app.ai.schemas import PRIVACY_NOTE

    panel_id = str(getattr(r.stage, "value", r.stage))
    digest = build_digest(s, panel_id=panel_id, focus_result_id=result_id)
    answer = generate_offline(digest, SimpleNamespace(
        action="explain",
        result_id=result_id,
        panel_id=panel_id,
        audience="beginner",
        eli15=False,
        question=f"Explain {r.title}",
    ))
    return {
        "found": True,
        "result_id": result_id,
        "title": r.title,
        "plain": r.plain,
        "why": r.why,
        "technical": r.technical,
        "caveats": r.caveats,
        "offline_answer": answer.text,
        "citations": answer.citations,
        "grounded": validate_grounding(answer.text, digest).grounded,
        "digest_result_count": len(digest.result_ids),
        "ai_available": get_settings().has_gemini_key(),
        "privacy_note": PRIVACY_NOTE,
        "note": "Offline explanation. Enable live AI in Settings for a tailored answer.",
    }


@router.post("/sessions/{session_id}/reset", tags=["session"])
def reset_pipeline(session_id: str, from_stage: str | None = None) -> dict[str, Any]:
    """
    Walk the pipeline backwards. Everything at or after `from_stage` becomes
    stale; results are kept but flagged, never silently reused.
    """
    s = get_store().get(session_id)
    start_idx = 0
    if from_stage:
        for i, st in enumerate(STAGE_ORDER):
            if st.value == from_stage:
                start_idx = i
                break
    for st in STAGE_ORDER[start_idx:]:
        s.state.stages[st.value].status = "stale"
    s.state.current_stage = STAGE_ORDER[start_idx]
    s.state.updated_at = datetime.now(timezone.utc)
    return {"reset_from": STAGE_ORDER[start_idx].value, "pipeline": s.state.progress()}


__all__ = ["router"]
