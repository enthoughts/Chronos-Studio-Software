"""The 'everything' export: one ZIP with all Part 18 deliverables."""

from __future__ import annotations

import io
from zipfile import ZIP_DEFLATED, ZipFile

from app.ts.report.archive import render_archive
from app.ts.report.bundle import EXPORT_SPECS, ReportBundle
from app.ts.report.docx import render_docx
from app.ts.report.figures import render_figures
from app.ts.report.html import render_html
from app.ts.report.latex import render_latex
from app.ts.report.pdf import render_pdf
from app.ts.report.script import render_python, render_r
from app.ts.report.utils import safe_json
from app.ts.report.xlsx import render_xlsx

BUILDERS = {
    "html": ("chronos-report.html", render_html),
    "docx": ("chronos-report.docx", render_docx),
    "pdf": ("chronos-report.pdf", render_pdf),
    "latex": ("chronos-tables.tex", render_latex),
    "xlsx": ("chronos-tables.xlsx", render_xlsx),
    "python": ("reproduce_analysis.py", render_python),
    "r": ("reproduce_analysis.R", render_r),
    "chronos": ("session.chronos", render_archive),
    "figures": ("figures.zip", render_figures),
}

README = """Chronos Studio — Reports & Finalization bundle

chronos-report.html      Self-contained one-file HTML with embedded vector figures (default share)
chronos-report.docx      Numbered figures/tables, captions and filled template prose
chronos-report.pdf       Print-CSS PDF from the same HTML (requires local Chromium)
chronos-tables.tex       LaTeX booktabs table snippets
chronos-tables.xlsx      One sheet per table + README + scorecard
reproduce_analysis.py    One Python script that replays the entire staged analysis
reproduce_analysis.R     Optional R equivalent
session.chronos          Data hash, all settings, all results JSON, raw CSV (export only)
figures.zip              Vector SVG for every figure; PNG/PDF require local Chromium
errors.txt               Present only when an optional export could not be built
status.json              The readiness verdict, scorecard and export catalogue

Template prose is deterministic and grounded in stored results; verify before submission.
No Gemini API key is stored in the bundle.
"""


def render_all(bundle: ReportBundle) -> bytes:
    out = io.BytesIO()
    files: dict[str, bytes] = {}
    errors: dict[str, str] = {}
    for kind, (filename, builder) in BUILDERS.items():
        try:
            files[filename] = builder(bundle)
        except Exception as exc:
            errors[filename] = f"{type(exc).__name__}: {exc}"
    files["README.txt"] = README.encode("utf-8")
    files["status.json"] = safe_json({
        "generated_at": bundle.generated_at,
        "session_id": bundle.session.id,
        "verdict": bundle.scorecard.verdict,
        "score": bundle.scorecard.score,
        "max_score": bundle.scorecard.max_score,
        "stage_status": bundle.session.state.stages["S15"].status,
        "exports": [x.model_dump(mode="json") for x in EXPORT_SPECS],
        "catalog_counts": {"plots": len(bundle.plots), "tables": len(bundle.tables)},
        "errors": errors,
    }, indent=2).encode("utf-8")
    if errors:
        files["errors.txt"] = "".join(f"{k}: {v}\n" for k, v in errors.items()).encode("utf-8")
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return out.getvalue()


__all__ = ["BUILDERS", "README", "render_all"]
