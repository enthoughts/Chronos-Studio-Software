"""XLSX workbook: one sheet per catalogue table plus a README sheet."""

from __future__ import annotations

import io
import math
import re
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.charts.theme import (
    SPREADSHEET_FONT,
    SPREADSHEET_HEADER_FILL,
    SPREADSHEET_HEADER_FONT,
)
from app.core.errors import ChronosError
from app.ts.report.bundle import ReportBundle
from app.ts.report.utils import jsonable


def _cell_value(value: Any) -> Any:
    clean = jsonable(value)
    if clean is None or isinstance(clean, (bool, int)):
        return clean
    if isinstance(clean, float):
        return clean if math.isfinite(clean) else None
    if isinstance(clean, (list, dict)):
        from app.ts.report.utils import safe_json
        return safe_json(clean)[:1000]
    return str(clean)[:1000]


def _columns(item: dict[str, Any]) -> list[dict[str, Any]]:
    columns = [dict(x) for x in item.get("columns") or [] if isinstance(x, dict)]
    if columns:
        return columns
    keys: list[str] = []
    for row in item.get("rows") or []:
        if isinstance(row, dict):
            for key in row:
                if str(key) not in keys:
                    keys.append(str(key))
    return [{"key": k, "label": k.replace("_", " ").title()} for k in keys]


def _safe_sheet_name(value: str, used: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "-", value).strip("'")[:31] or "Table"
    name, i = base, 2
    while name in used:
        suffix = f"~{i}"
        name = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def _write_table(ws: Any, columns: list[dict[str, Any]], rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    header_fill = PatternFill("solid", fgColor=SPREADSHEET_HEADER_FILL)
    header_font = Font(name=SPREADSHEET_FONT, bold=True, color=SPREADSHEET_HEADER_FONT)
    ws.append([str(c.get("label") or c.get("key") or "Column") for c in columns] or ["Value"])
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    for row in rows:
        ws.append([_cell_value(row.get(str(c.get("key")))) for c in columns] if columns else [_cell_value(row)])
    ws.freeze_panes = "A2"
    for index, column in enumerate(columns or [{"key": "value", "label": "Value"}], start=1):
        width = min(42, max(11, len(str(column.get("label") or column.get("key") or "")) + 3))
        for row in rows[:100]:
            v = _cell_value(row.get(str(column.get("key")))) if columns else _cell_value(row)
            if v is not None:
                width = max(width, min(42, len(str(v)) + 2))
        ws.column_dimensions[get_column_letter(index)].width = width
    ws.append([])
    for key, label in [
        ("id", "Catalogue id"), ("title", "Title"), ("caption", "Caption"),
        ("alt_text", "Alt text"), ("stage", "Stage"), ("group", "Group"), ("source", "Source"),
    ]:
        ws.append([label, str(meta.get(key) or "")])
        ws.cell(ws.max_row, 1).font = Font(name=SPREADSHEET_FONT, bold=True)
        ws.cell(ws.max_row, 2).alignment = Alignment(wrap_text=True, vertical="top")


def render_xlsx(bundle: ReportBundle) -> bytes:
    try:
        wb = Workbook()
    except Exception as exc:  # pragma: no cover
        raise ChronosError(
            code="export.xlsx_missing", plain="XLSX export needs the openpyxl package.",
            fix="Install openpyxl, then try again.", technical=str(exc),
        ) from exc
    used: set[str] = set()
    readme = wb.active
    readme.title = _safe_sheet_name("README", used)
    readme.append(["Chronos Studio — reports workbook"])
    readme["A1"].font = Font(name=SPREADSHEET_FONT, bold=True, size=14)
    rows = [
        ("Verdict", bundle.scorecard.verdict),
        ("Score", f"{bundle.scorecard.score}/{bundle.scorecard.max_score}"),
        ("Generated", bundle.generated_at),
        ("Session", bundle.session.id),
        ("Dataset", bundle.session.state.dataset.name if bundle.session.state.dataset else ""),
        ("Data hash", bundle.data_hash),
        ("Seed", bundle.session.seed),
        ("Theme", bundle.mode),
        ("Scope", bundle.scope),
        ("Catalogue plots", len(bundle.plots)),
        ("Catalogue tables", len(bundle.tables)),
        ("Selected items", len(bundle.selection_ids)),
        ("Privacy", "No Gemini key is stored; values come from computed results and catalogue rows."),
        ("Sheets", "One sheet per catalogue table, plus README and Scorecard."),
    ]
    for key, value in rows:
        readme.append([key, value])
        readme.cell(readme.max_row, 1).font = Font(name=SPREADSHEET_FONT, bold=True)
        readme.cell(readme.max_row, 2).alignment = Alignment(wrap_text=True, vertical="top")
    readme.append([])
    readme.append(["What to do next"])
    readme.cell(readme.max_row, 1).font = Font(name=SPREADSHEET_FONT, bold=True)
    if bundle.scorecard.next_steps:
        for step in bundle.scorecard.next_steps:
            readme.append([step.number, step.text])
    else:
        readme.append([1, "No outstanding items."])
    readme.append([])
    readme.append(["Available exports"])
    readme.cell(readme.max_row, 1).font = Font(name=SPREADSHEET_FONT, bold=True)
    from app.ts.report.bundle import EXPORT_SPECS
    for spec in EXPORT_SPECS:
        readme.append([spec.label, spec.filename, spec.description])
    readme.column_dimensions["A"].width = 24
    readme.column_dimensions["B"].width = 82
    readme.column_dimensions["C"].width = 70
    readme.freeze_panes = "A2"

    score = wb.create_sheet(_safe_sheet_name("Scorecard", used))
    _write_table(
        score,
        [
            {"key": "check", "label": "Check"}, {"key": "weight", "label": "Weight"},
            {"key": "status", "label": "Status"}, {"key": "panel", "label": "Panel"},
            {"key": "results", "label": "Result links"}, {"key": "evidence", "label": "Evidence"},
        ],
        [{
            "check": x.label, "weight": x.weight, "status": "yes" if x.passed else "no",
            "panel": f"{x.panel_id} — {x.panel_label}",
            "results": ", ".join(x.available_result_ids or x.result_ids), "evidence": x.evidence,
        } for x in bundle.scorecard.items],
        {"id": "report.scorecard", "title": "Manuscript-readiness scorecard",
         "caption": bundle.scorecard.verdict, "stage": "S15", "group": "Readiness"},
    )
    for index, item in enumerate(bundle.tables, start=1):
        ws = wb.create_sheet(_safe_sheet_name(f"{index:02d} {item.get('id') or item.get('title')}", used))
        _write_table(ws, _columns(item), [dict(x) for x in item.get("rows") or [] if isinstance(x, dict)], item)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


__all__ = ["render_xlsx"]
