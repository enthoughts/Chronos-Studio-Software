"""api/export.py — Part 18 whole-report, archive and finalization endpoints.

The Part 16 catalogue remains the owner of per-item downloads. This router
consumes that catalogue, the stored results and the pipeline state to make the
whole-report deliverables and to issue the S15 readiness verdict.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel, ConfigDict

from app.api.deps import session_dep
from app.charts.theme import ThemeMode
from app.core.errors import ChronosError
from app.core.schemas import Stage
from app.core.session import Session
from app.ts.report.all import render_all
from app.ts.report.archive import render_archive
from app.ts.report.bundle import (
    EXPORT_SPECS,
    ReportBundle,
    build_bundle,
    build_status_response,
    export_spec,
    normalise_mode,
)
from app.ts.report.docx import render_docx
from app.ts.report.figures import render_figures
from app.ts.report.html import render_html
from app.ts.report.latex import render_latex
from app.ts.report.pdf import render_pdf
from app.ts.report.schemas import ReportKind, ReportScope, ReportStatus
from app.ts.report.scorecard import finalize_session
from app.ts.report.script import render_python, render_r
from app.ts.report.xlsx import render_xlsx

router = APIRouter(prefix="/export", tags=["export"])


class ExportBuildError(ChronosError):
    code = "export.build_failed"
    http_status = 500
    plain = "The report could not be built from the stored results."
    fix = "Try the one-file HTML export, or download the .chronos report archive and report the error."


class ExportUnknownKind(ChronosError):
    code = "export.unknown_kind"
    http_status = 404
    plain = "That export format is not available."
    fix = "Choose one of the export buttons shown in the Ready to write up? panel."


class FinalizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "complete"
    stage: str = "S15"
    stage_status: str
    scorecard_verdict: str
    score: int
    max_score: int
    result_ids: list[str]


BUILDERS: dict[ReportKind, Callable[[ReportBundle], bytes]] = {
    "html": render_html,
    "docx": render_docx,
    "pdf": render_pdf,
    "latex": render_latex,
    "xlsx": render_xlsx,
    "python": render_python,
    "r": render_r,
    "chronos": render_archive,
    "figures": render_figures,
    "all": render_all,
}


@router.get("/status", response_model=ReportStatus)
def read_status(
    mode: ThemeMode = Query("print", description="House theme for figures."),
    scope: ReportScope = Query("all", description="All catalogue items or the selected report basket."),
    session: Session = Depends(session_dep),
) -> ReportStatus:
    """The ten weighted checks, exact verdict, next panels and export catalogue."""
    return build_status_response(session, mode=normalise_mode(mode), scope=scope)


@router.post("/finalize", response_model=FinalizeResponse)
def finalize(
    mode: ThemeMode = Query("print"),
    scope: ReportScope = Query("all"),
    session: Session = Depends(session_dep),
) -> FinalizeResponse:
    """User-confirmed S15 finalization: store three result cards and mark the stage."""
    session.require_stage(Stage.S15_FINALIZE, allow_stale=True)
    scorecard = finalize_session(session)
    # Keep the response consistent with a fresh read of the same session.
    status = build_status_response(session, mode=normalise_mode(mode), scope=scope)
    return FinalizeResponse(
        stage_status=status.stage_status,
        scorecard_verdict=scorecard.verdict,
        score=scorecard.score,
        max_score=scorecard.max_score,
        result_ids=["report.readiness", "report.scorecard", "report.next_steps"],
    )


@router.get("/{kind}")
def download(
    kind: ReportKind,
    mode: ThemeMode = Query("print", description="House theme for figures."),
    scope: ReportScope = Query("all", description="All catalogue items or the selected report basket."),
    download: bool = Query(False, description="Force an attachment for the inline HTML report."),
    session: Session = Depends(session_dep),
) -> Response:
    """Build one export from stored results. HTML is inline by default for sharing."""
    if kind not in BUILDERS:
        raise ExportUnknownKind(context={"kind": kind})
    theme = normalise_mode(mode)
    bundle = build_bundle(session, mode=theme, scope=scope)
    spec = export_spec(kind)
    try:
        data = BUILDERS[kind](bundle)
    except ChronosError:
        raise
    except Exception as exc:
        raise ExportBuildError(
            context={"kind": kind, "scope": scope},
            technical=f"{type(exc).__name__}: {exc}",
        ) from exc
    if not data:
        raise ExportBuildError(context={"kind": kind, "scope": scope}, technical="empty file")
    disposition = "inline" if kind == "html" and not download else "attachment"
    headers = {
        "Content-Disposition": f'{disposition}; filename="{spec.filename}"',
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "X-Report-Kind": kind,
        "X-Report-Filename": spec.filename,
        # HTTP header values must be latin-1 safe; the exact em-dash verdict is
        # in the JSON status and inside every document, not in a header.
        "X-Report-Verdict": bundle.scorecard.status,
        "X-Report-Ready": "true" if bundle.scorecard.ready else "false",
        "X-Report-Score": f"{bundle.scorecard.score}/{bundle.scorecard.max_score}",
        "X-Report-Scope": scope,
        "X-Report-Theme": theme,
        "X-Report-Data-Hash": bundle.data_hash,
    }
    return Response(content=data, media_type=spec.media_type, headers=headers)


@router.get("", include_in_schema=False)
def export_index(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """A discoverable list; the status endpoint is the primary UI payload."""
    return {
        "stage": "S15",
        "exports": [x.model_dump(mode="json") for x in EXPORT_SPECS],
        "status_endpoint": "/api/v1/export/status",
        "finalize_endpoint": "/api/v1/export/finalize",
        "session_dep": "X-Session-Id",
    }


__all__ = ["router"]
