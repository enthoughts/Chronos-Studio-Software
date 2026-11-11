"""Shared report content: catalogue, readiness scorecard and manuscript prose."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from app.charts import completeness as C
from app.charts import exporters as E
from app.charts.theme import ThemeMode
from app.ts.report.prose import manuscript_sections
from app.ts.report.schemas import (
    ReportExportSpec,
    ReportScope,
    ReportScorecard,
    ReportStatus,
)
from app.ts.report.scorecard import build_scorecard, cache_catalog_ids
from app.ts.report.utils import analysis_settings, safe_json

MODES: tuple[ThemeMode, ...] = ("light", "dark", "print")

EXPORT_SPECS: tuple[ReportExportSpec, ...] = (
    ReportExportSpec(
        kind="html", label="One-file HTML", description="Self-contained report with embedded vector figures; works offline without Chromium.",
        filename="chronos-report.html", media_type="text/html; charset=utf-8",
    ),
    ReportExportSpec(
        kind="docx", label="Word manuscript", description="Numbered figures and tables with filled Methods–Forecast Accuracy prose.",
        filename="chronos-report.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ReportExportSpec(
        kind="pdf", label="Print PDF", description="Print-CSS PDF via local Chromium; browser Print-to-PDF from the HTML is the offline fallback.",
        filename="chronos-report.pdf", media_type="application/pdf",
    ),
    ReportExportSpec(
        kind="latex", label="LaTeX tables", description="Booktabs table snippets with captions and labels, ready for a thesis.",
        filename="chronos-tables.tex", media_type="application/x-tex; charset=utf-8",
    ),
    ReportExportSpec(
        kind="xlsx", label="Workbook", description="One sheet per catalogue table plus a README and scorecard sheet.",
        filename="chronos-tables.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    ReportExportSpec(
        kind="python", label="Reproducible Python", description="One script that replays the full staged analysis against the API.",
        filename="reproduce_analysis.py", media_type="text/x-python; charset=utf-8",
    ),
    ReportExportSpec(
        kind="r", label="Reproducible R", description="Optional equivalent using jsonlite and httr.",
        filename="reproduce_analysis.R", media_type="text/plain; charset=utf-8",
    ),
    ReportExportSpec(
        kind="chronos", label="Report archive", description="Data hash, settings, all results JSON and raw CSV; export only, not a resumable session.",
        filename="session.chronos", media_type="application/zip",
    ),
    ReportExportSpec(
        kind="figures", label="Publication figures", description="Vector SVG for every figure; 300-dpi PNG and PDF need optional local Chromium.",
        filename="figures.zip", media_type="application/zip",
    ),
    ReportExportSpec(
        kind="all", label="Everything", description="One ZIP containing every export above; errors.txt names any optional format that could not be built.",
        filename="chronos-report-bundle.zip", media_type="application/zip",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_mode(mode: str) -> ThemeMode:
    return mode if mode in MODES else "print"


def _public_catalog(session: Any, mode: ThemeMode, catalog: dict[str, Any] | None) -> dict[str, Any]:
    # A fresh session builds an empty catalogue cleanly. Once results exist, a
    # broken catalogue is a real error: it must never be silently replaced by an
    # empty document that still claims manuscript readiness.
    raw = catalog if catalog is not None else C.build_catalog(session, mode=mode)
    out = {
        "scorecard": dict(raw.get("scorecard") or {}),
        "plots": [C.public_item(x) for x in raw.get("plots") or []],
        "tables": [C.public_item(x) for x in raw.get("tables") or []],
    }
    if catalog is None:
        cache_catalog_ids(session, (str(x["id"]) for x in out["plots"] + out["tables"]))
    return out


def raw_table_csv(session: Any) -> bytes:
    """A raw CSV of the untouched table for reproducibility, when one was stored."""
    table = session.objects.get("raw_table")
    if table is not None and hasattr(table, "to_csv"):
        try:
            return table.to_csv(index=False).encode("utf-8")
        except Exception:
            pass
    series = session.objects.get("series_original")
    if series is not None and hasattr(series, "to_frame"):
        try:
            frame = series.to_frame(name="value").reset_index()
            return frame.to_csv(index=False).encode("utf-8")
        except Exception:
            pass
    return b""


@dataclass
class ReportBundle:
    session: Any
    mode: ThemeMode
    scope: ReportScope
    catalog: dict[str, Any]
    items: list[dict[str, Any]]
    plots: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    selection_ids: list[str]
    scorecard: ReportScorecard
    sections: list[dict[str, Any]]
    settings: dict[str, Any]
    data_csv: bytes
    generated_at: str
    figure_cache: dict[tuple[str, str, str], bytes] = field(default_factory=dict)

    @property
    def title(self) -> str:
        dataset = self.session.state.dataset
        return f"Chronos Studio report — {dataset.name}" if dataset else "Chronos Studio report"

    @property
    def data_hash(self) -> str:
        return str(self.settings.get("content_hash") or "")

    def figure_bytes(self, item: dict[str, Any], fmt: str) -> bytes:
        key = (str(item.get("id") or ""), fmt, self.mode)
        if key not in self.figure_cache:
            data, _ = E.download_bytes(item, fmt, mode=self.mode)
            self.figure_cache[key] = data
        return self.figure_cache[key]

    def results_json(self) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in self.session.all_results()]

    def session_export(self) -> dict[str, Any]:
        return self.session.export_state()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.settings.get("schema_version"),
            "generated_at": self.generated_at,
            "session_id": self.session.id,
            "stage": "S15",
            "stage_status": self.session.state.stages["S15"].status,
            "dataset": self.session.state.dataset.name if self.session.state.dataset else None,
            "data_hash": self.data_hash,
            "seed": self.session.seed,
            "mode": self.mode,
            "scope": self.scope,
            "selected_catalog_items": list(self.selection_ids),
            "catalog_counts": {"plots": len(self.catalog["plots"]), "tables": len(self.catalog["tables"])},
            "verdict": self.scorecard.verdict,
            "score": self.scorecard.score,
            "max_score": self.scorecard.max_score,
            "library_versions": dict(self.session.state.library_versions),
        }


def build_bundle(
    session: Any,
    *,
    mode: str = "print",
    scope: ReportScope = "all",
    catalog: dict[str, Any] | None = None,
) -> ReportBundle:
    theme = normalise_mode(mode)
    public = _public_catalog(session, theme, catalog)
    selection_ids = C.report_item_ids(session)
    all_items = public["plots"] + public["tables"]
    if scope == "selected":
        by_id = {str(x["id"]): x for x in all_items}
        items = [by_id[x] for x in selection_ids if x in by_id]
    else:
        items = list(all_items)
    scorecard = build_scorecard(session)
    sections = manuscript_sections(session, public, scorecard)
    settings = analysis_settings(session)
    return ReportBundle(
        session=session,
        mode=theme,
        scope=scope,
        catalog=public,
        items=items,
        plots=[x for x in items if x.get("kind") == "plot"],
        tables=[x for x in items if x.get("kind") == "table"],
        selection_ids=selection_ids,
        scorecard=scorecard,
        sections=sections,
        settings=settings,
        data_csv=raw_table_csv(session),
        generated_at=_utc_now(),
    )


def build_status_response(
    session: Any,
    *,
    mode: str = "print",
    scope: ReportScope = "all",
    catalog: dict[str, Any] | None = None,
) -> ReportStatus:
    bundle = build_bundle(session, mode=mode, scope=scope, catalog=catalog)
    return ReportStatus(
        stage_status=str(session.state.stages["S15"].status),
        generated_at=bundle.generated_at,
        scorecard=bundle.scorecard,
        exports=list(EXPORT_SPECS),
        catalog_counts={
            "plots": len(bundle.catalog["plots"]),
            "tables": len(bundle.catalog["tables"]),
            "items": len(bundle.items),
        },
        selection_count=len(bundle.selection_ids),
        scope=bundle.scope,
        mode=bundle.mode,
    )


def export_spec(kind: str) -> ReportExportSpec:
    for spec in EXPORT_SPECS:
        if spec.kind == kind:
            return spec
    raise KeyError(kind)


__all__ = [
    "EXPORT_SPECS", "ReportBundle", "build_bundle", "build_status_response",
    "export_spec", "normalise_mode", "raw_table_csv", "safe_json",
]
