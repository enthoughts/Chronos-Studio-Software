"""DOCX manuscript with numbered figures/tables and filled template prose."""

from __future__ import annotations

import io
from typing import Any

from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Inches, Pt

from app.core.errors import ChronosError
from app.ts.report.bundle import ReportBundle
from app.ts.report.utils import cell_text

MAX_DOCX_ROWS = 35


def _columns(item: dict[str, Any]) -> list[dict[str, Any]]:
    columns = [dict(x) for x in item.get("columns") or [] if isinstance(x, dict)]
    if columns:
        return columns[:10]
    keys: list[str] = []
    for row in item.get("rows") or []:
        if isinstance(row, dict):
            for key in row:
                if str(key) not in keys:
                    keys.append(str(key))
    return [{"key": k, "label": k.replace("_", " ").title()} for k in keys[:10]]


def _caption(doc: Any, kind: str, number: int, item: dict[str, Any]) -> None:
    p = doc.add_paragraph()
    r = p.add_run(f"{kind} {number}. ")
    r.bold = True
    p.add_run(str(item.get("caption") or item.get("title") or ""))
    p.add_run(
        f"  [{item.get('id', '')} · {item.get('stage', '')} · {item.get('group', '')}]"
    ).italic = True
    p.style = doc.styles["Caption"] if "Caption" in doc.styles else p.style


def _add_scorecard(doc: Any, bundle: ReportBundle) -> None:
    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, ["Check", "Weight", "Status", "Panel", "Results", "Evidence"]):
        cell.text = text
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
    for x in bundle.scorecard.items:
        cells = table.add_row().cells
        values = [
            x.label, str(x.weight), "✓" if x.passed else "✗",
            f"{x.panel_id} — {x.panel_label}",
            ", ".join(x.available_result_ids or x.result_ids), x.evidence,
        ]
        for cell, value in zip(cells, values):
            cell.text = value
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8)


def render_docx(bundle: ReportBundle) -> bytes:
    try:
        doc = Document()
    except Exception as exc:  # pragma: no cover
        raise ChronosError(
            code="export.docx_missing",
            plain="DOCX export needs the python-docx package.",
            fix="Install python-docx, then try again.",
            technical=str(exc),
        ) from exc
    doc.core_properties.title = bundle.title
    doc.core_properties.subject = bundle.scorecard.verdict
    doc.core_properties.author = "Chronos Studio"
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.75)
    section.left_margin = section.right_margin = Inches(0.8)

    doc.add_heading(bundle.title, level=0)
    p = doc.add_paragraph()
    p.add_run(bundle.scorecard.verdict).bold = True
    p.add_run(f"  ·  {bundle.scorecard.score}/{bundle.scorecard.max_score} weighted points")
    doc.add_paragraph(
        f"Generated {bundle.generated_at} · seed {bundle.session.seed} · data hash {bundle.data_hash} · "
        f"{bundle.mode} theme · {bundle.scope} catalogue scope."
    )
    doc.add_paragraph(
        "Privacy & provenance: numbers are copied from stored results and the Part 16 catalogue. "
        "No raw values are sent to an AI service and no Gemini key is stored in this report."
    )
    doc.add_heading("Manuscript-readiness scorecard", level=1)
    _add_scorecard(doc, bundle)
    if bundle.scorecard.next_steps:
        doc.add_heading("What to do next", level=1)
        for x in bundle.scorecard.next_steps:
            doc.add_paragraph(f"{x.number}. {x.text}", style="List Number")

    doc.add_page_break()
    doc.add_heading("Manuscript template prose", level=1)
    doc.add_paragraph("Template prose — verify before submission; every statistic comes from a stored result.")
    for section_data in bundle.sections:
        doc.add_heading(str(section_data["title"]), level=2)
        for paragraph in section_data["paragraphs"]:
            doc.add_paragraph(str(paragraph))
        sources = ", ".join(section_data["result_ids"])
        note = doc.add_paragraph(f"Grounded in: {sources}. {section_data['note']}")
        note.runs[0].italic = True
        note.runs[0].font.size = Pt(9)

    doc.add_page_break()
    doc.add_heading("Figures", level=1)
    for number, item in enumerate(bundle.plots, start=1):
        try:
            png = bundle.figure_bytes(item, "png")
            doc.add_picture(io.BytesIO(png), width=Inches(6.4))
        except Exception as exc:
            doc.add_paragraph(f"Figure unavailable: {type(exc).__name__}. Use the HTML/SVG export instead.")
        _caption(doc, "Figure", number, item)
        if str(item.get("alt_text") or ""):
            alt = doc.add_paragraph(f"Alt text: {item['alt_text']}")
            alt.runs[0].italic = True
            alt.runs[0].font.size = Pt(8)

    doc.add_page_break()
    doc.add_heading("Tables", level=1)
    for number, item in enumerate(bundle.tables, start=1):
        columns = _columns(item)
        rows = [dict(x) for x in item.get("rows") or [] if isinstance(x, dict)][:MAX_DOCX_ROWS]
        table = doc.add_table(rows=1, cols=max(1, len(columns)))
        table.style = "Table Grid"
        if columns:
            for cell, column in zip(table.rows[0].cells, columns):
                cell.text = str(column.get("label") or column.get("key") or "")
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.bold = True
        for row in rows:
            cells = table.add_row().cells
            for cell, column in zip(cells, columns):
                cell.text = cell_text(row.get(str(column.get("key"))))
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8)
        _caption(doc, "Table", number, item)
        if len(item.get("rows") or []) > MAX_DOCX_ROWS:
            doc.add_paragraph(
                f"Showing first {MAX_DOCX_ROWS} of {len(item['rows'])} rows; the XLSX export has all rows."
            ).runs[0].font.size = Pt(8)
    footer = doc.add_paragraph(
        "Chronos Studio · numbered tables and figures · 300 dpi PNG · template prose filled from real results."
    )
    footer.runs[0].italic = True
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


__all__ = ["MAX_DOCX_ROWS", "render_docx"]
