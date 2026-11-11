"""LaTeX table snippets for a thesis."""

from __future__ import annotations

from typing import Any

from app.ts.report.bundle import ReportBundle
from app.ts.report.utils import cell_text, latex_text, safe_stem

MAX_LATEX_ROWS = 35
MAX_LATEX_COLUMNS = 8


def _columns(item: dict[str, Any]) -> list[dict[str, Any]]:
    columns = [dict(x) for x in item.get("columns") or [] if isinstance(x, dict)]
    if columns:
        return columns[:MAX_LATEX_COLUMNS]
    keys: list[str] = []
    for row in item.get("rows") or []:
        if isinstance(row, dict):
            for key in row:
                if str(key) not in keys:
                    keys.append(str(key))
    return [{"key": k, "label": k.replace("_", " ").title()} for k in keys[:MAX_LATEX_COLUMNS]]


def table_snippet(item: dict[str, Any], number: int) -> str:
    columns = _columns(item)
    rows = [dict(x) for x in item.get("rows") or [] if isinstance(x, dict)][:MAX_LATEX_ROWS]
    ncols = max(1, len(columns))
    spec = "@{}" + "l" * min(ncols, 4) + "r" * max(0, ncols - 4) + "@{}"
    label = safe_stem(item.get("id"), fallback=f"table{number}")
    caption = latex_text(item.get("caption") or item.get("title") or item.get("id"))
    out = [
        f"% {item.get('id', '')} · {item.get('stage', '')} · {item.get('group', '')}\n",
        "\\begin{longtable}{" + spec + "}\n",
        f"\\caption{{Table {number}. {caption}}}\\label{{tab:{label}}}\\\\\n",
        "\\toprule\n",
    ]
    if columns:
        out.append(" & ".join(latex_text(c.get("label") or c.get("key")) for c in columns) + " \\\\\n")
        out.append("\\midrule\\endfirsthead\n")
        out.append("\\toprule\n")
        out.append(" & ".join(latex_text(c.get("label") or c.get("key")) for c in columns) + " \\\\\n")
        out.append("\\midrule\\endhead\n")
    for row in rows:
        cells = [latex_text(row.get(str(c.get("key")))) for c in columns] if columns else []
        out.append(" & ".join(cells or [latex_text(cell_text(row))]) + " \\\\\n")
    total = len(item.get("rows") or [])
    if total > MAX_LATEX_ROWS:
        out.append(f"\\multicolumn{{{ncols}}}{{@{{}}l@{{}}}}{{\\textit{{First {MAX_LATEX_ROWS} of {total} rows; XLSX has all rows.}}}} \\\\\n")
    out.append("\\bottomrule\n\\end{longtable}\n\n")
    return "".join(out)


def render_latex(bundle: ReportBundle) -> bytes:
    head = (
        "% Chronos Studio · S15 LaTeX table snippets\n"
        "% Add to the preamble: \\usepackage{booktabs,longtable}\n"
        "% Captions are numbered by LaTeX; every value comes from a stored catalogue row.\n"
        f"% Report: {latex_text(bundle.title)}\n"
        f"% Verdict: {latex_text(bundle.scorecard.verdict)}\n"
        f"% Data hash: {latex_text(bundle.data_hash)}\\ · seed {latex_text(bundle.session.seed)}\n\n"
    )
    scorecard_item = {
        "id": "report.scorecard",
        "stage": "S15",
        "group": "Readiness",
        "title": "Manuscript-readiness scorecard",
        "caption": f"Weighted readiness scorecard: {bundle.scorecard.score}/{bundle.scorecard.max_score}.",
        "columns": [
            {"key": "check", "label": "Check"}, {"key": "weight", "label": "Weight"},
            {"key": "status", "label": "Status"}, {"key": "panel", "label": "Panel"},
            {"key": "results", "label": "Results"}, {"key": "evidence", "label": "Evidence"},
        ],
        "rows": [{
            "check": x.label, "weight": x.weight, "status": "yes" if x.passed else "no",
            "panel": f"{x.panel_id} — {x.panel_label}",
            "results": ", ".join(x.available_result_ids or x.result_ids), "evidence": x.evidence,
        } for x in bundle.scorecard.items],
    }
    parts = [head, table_snippet(scorecard_item, 1)]
    for i, item in enumerate(bundle.tables, start=2):
        parts.append(table_snippet(item, i))
    return "".join(parts).encode("utf-8")


__all__ = ["MAX_LATEX_COLUMNS", "MAX_LATEX_ROWS", "render_latex", "table_snippet"]
