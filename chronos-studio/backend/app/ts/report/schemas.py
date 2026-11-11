"""Pydantic contracts for the Part 18 manuscript-readiness scorecard."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.schemas import SCHEMA_VERSION

ReportStatusName = Literal["ready", "nearly", "not_ready"]
ReportKind = Literal[
    "html", "docx", "pdf", "latex", "xlsx", "python", "r", "chronos", "figures", "all"
]
ReportScope = Literal["all", "selected"]

EXPORT_KINDS: tuple[ReportKind, ...] = (
    "html", "docx", "pdf", "latex", "xlsx", "python", "r", "chronos", "figures", "all",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportItem(_Strict):
    id: str
    label: str
    weight: int = Field(..., ge=1)
    passed: bool
    evidence: str = ""
    result_ids: list[str] = Field(default_factory=list)
    available_result_ids: list[str] = Field(default_factory=list)
    panel_id: str
    panel_label: str
    fix: str = ""


class ReportNextStep(_Strict):
    number: int = Field(..., ge=1)
    item_id: str
    text: str
    panel_id: str
    panel_label: str
    result_ids: list[str] = Field(default_factory=list)


class ReportScorecard(_Strict):
    schema_version: str = SCHEMA_VERSION
    stage: str = "S15"
    generated_at: str = ""
    items: list[ReportItem] = Field(default_factory=list)
    score: int = 0
    max_score: int = 100
    percent: float = 0.0
    outstanding: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_steps: list[ReportNextStep] = Field(default_factory=list)
    verdict: str
    status: ReportStatusName
    ready: bool = False


class ReportExportSpec(_Strict):
    kind: ReportKind
    label: str
    description: str
    filename: str
    media_type: str


class ReportStatus(_Strict):
    status: Literal["complete"] = "complete"
    stage: str = "S15"
    stage_status: str = "not_started"
    generated_at: str = ""
    scorecard: ReportScorecard
    exports: list[ReportExportSpec] = Field(default_factory=list)
    catalog_counts: dict[str, int] = Field(default_factory=dict)
    selection_count: int = 0
    scope: ReportScope = "all"
    mode: str = "print"
    privacy_note: str = (
        "Reports contain computed results and metadata only. Raw values are not embedded "
        "unless you explicitly exported them; no Gemini key is stored in a report."
    )


__all__ = [
    "EXPORT_KINDS",
    "ReportExportSpec",
    "ReportItem",
    "ReportKind",
    "ReportNextStep",
    "ReportScope",
    "ReportScorecard",
    "ReportStatus",
    "ReportStatusName",
]
