"""The self-contained, one-file HTML report (§5.13)."""

from __future__ import annotations

import base64
from string import Template
from typing import Any

from app.charts.theme import ThemeMode, colours, with_alpha
from app.ts.report.bundle import ReportBundle
from app.ts.report.utils import html_text, safe_stem

# The report's print CSS is generated from the SAME single-source house theme as
# every Plotly figure. No colour literal lives in this module (§5.10, Part 16).
CSS_TEMPLATE = """
:root { --ink:$text; --soft:$muted; --rule:$rule; --paper:$paper; --accent:$accent; --ok:$ok; --warn:$warn; --bad:$bad; --wash:$wash; }
* { box-sizing:border-box; }
body { margin:0 auto; max-width:980px; padding:34px 28px 80px; font:15px/1.55 Georgia,'Times New Roman',serif; color:var(--ink); background:var(--paper); }
h1,h2,h3 { font-family:Arial,Helvetica,sans-serif; line-height:1.25; }
h1 { font-size:29px; margin:0 0 8px; } h2 { font-size:20px; margin:30px 0 10px; border-bottom:2px solid var(--accent); padding-bottom:5px; }
h3 { font-size:16px; margin:20px 0 6px; }
p { margin:8px 0 12px; } .meta,.eyebrow,.small { font:12px/1.5 Arial,Helvetica,sans-serif; color:var(--soft); }
.eyebrow { letter-spacing:.13em; text-transform:uppercase; }
.verdict { border-left:6px solid var(--accent); background:var(--wash); padding:14px 18px; margin:18px 0; font:17px/1.4 Arial,Helvetica,sans-serif; }
.verdict.ready { border-color:var(--ok); } .verdict.nearly { border-color:var(--warn); } .verdict.not_ready { border-color:var(--bad); }
.score { float:right; font-weight:700; }
.callout { border:1px solid var(--rule); background:var(--wash); padding:12px 16px; margin:14px 0; font:13px/1.5 Arial,Helvetica,sans-serif; }
table { border-collapse:collapse; width:100%; margin:10px 0 18px; font:12px/1.35 Arial,Helvetica,sans-serif; }
th,td { border:1px solid var(--rule); padding:5px 7px; text-align:left; vertical-align:top; }
th { background:$header; font-weight:700; } tr:nth-child(even) td { background:$even; }
.pass { color:var(--ok); font-weight:700; } .fail { color:var(--bad); font-weight:700; }
figure { margin:22px 0 30px; page-break-inside:avoid; }
figure img { display:block; width:100%; height:auto; border:1px solid var(--rule); background:var(--paper); }
figcaption { font:12px/1.45 Arial,Helvetica,sans-serif; color:var(--soft); margin-top:6px; }
.item { border-top:1px solid var(--rule); padding-top:8px; }
.item .tag { display:inline-block; border:1px solid var(--rule); border-radius:9px; padding:1px 7px; font:11px Arial,sans-serif; color:var(--soft); }
.footer { margin-top:40px; border-top:1px solid var(--rule); padding-top:10px; font:11px/1.5 Arial,Helvetica,sans-serif; color:var(--soft); }
@page { size:A4; margin:14mm; }
@media print { body { max-width:none; padding:0; font-size:10.5pt; } h2 { break-after:avoid; } figure,table { break-inside:avoid; } .page-break { break-before:page; } }
"""


def css_for(mode: ThemeMode) -> str:
    """Render CSS variables from the house theme, including dark/print variants."""
    ink = colours(mode)
    neutral = ink.get("neutral", ink["muted"])
    values = {
        "text": ink["text"],
        "muted": ink["muted"],
        "paper": ink["paper"],
        "accent": ink["series"],
        "ok": ink.get("ok", ink["series"]),
        "warn": ink.get("warn", ink["muted"]),
        "bad": ink.get("bad", ink["series"]),
        "rule": with_alpha(neutral, 0.22),
        "wash": with_alpha(neutral, 0.06),
        "header": with_alpha(neutral, 0.12),
        "even": with_alpha(neutral, 0.03),
    }
    return Template(CSS_TEMPLATE).substitute(values)



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
    return [{"key": k, "label": k.replace("_", " ").title(), "dtype": "string"} for k in keys]


def html_table(item: dict[str, Any], *, max_rows: int = 60, number: int | None = None) -> str:
    columns = _columns(item)[:12]
    rows = [dict(x) for x in (item.get("rows") or []) if isinstance(x, dict)][:max_rows]
    caption = f"Table {number}. " if number else ""
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{html_text(c.get('label') or c.get('key'))}</th>" for c in columns)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        out.extend(f"<td>{html_text(row.get(str(c.get('key'))))}</td>" for c in columns)
        out.append("</tr>")
    out.append("</tbody></table>")
    if len(item.get("rows") or []) > max_rows:
        out.append(f"<p class='small'>Showing first {max_rows} of {len(item['rows'])} rows; the XLSX export contains all rows.</p>")
    out.append(f"<figcaption>{html_text(caption + str(item.get('caption') or ''))}</figcaption>")
    return "".join(out)


def scorecard_table(bundle: ReportBundle) -> str:
    out = [
        "<table><thead><tr><th>Check</th><th>Weight</th><th>Status</th><th>Panel</th>",
        "<th>Result links</th><th>Evidence</th></tr></thead><tbody>",
    ]
    for x in bundle.scorecard.items:
        out.append(
            f"<tr><td>{html_text(x.label)}</td><td>{x.weight}</td>"
            f"<td class='{'pass' if x.passed else 'fail'}'>{'✓' if x.passed else '✗'}</td>"
            f"<td>{html_text(x.panel_id)} — {html_text(x.panel_label)}</td>"
            f"<td>{html_text(', '.join(x.available_result_ids or x.result_ids))}</td>"
            f"<td>{html_text(x.evidence)}</td></tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def embedded_figure_html(bundle: ReportBundle, item: dict[str, Any], number: int) -> str:
    caption = str(item.get("caption") or "")
    alt = str(item.get("alt_text") or caption)
    try:
        svg = bundle.figure_bytes(item, "svg")
        uri = "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")
        image = f"<img src='{uri}' alt='{html_text(alt)}' />"
    except Exception as exc:
        image = (
            "<div class='callout'>Static figure unavailable: "
            f"{html_text(type(exc).__name__)}. Download the XLSX/data-behind export instead.</div>"
        )
    return (
        f"<figure class='item' id='{html_text(safe_stem(item.get('id')))}'>{image}"
        f"<figcaption><strong>Figure {number}.</strong> {html_text(caption)} "
        f"<span class='small'>({html_text(item.get('id'))} · {html_text(item.get('stage'))} · "
        f"{html_text(item.get('group'))})</span></figcaption></figure>"
    )


def render_html(bundle: ReportBundle) -> bytes:
    """Return one UTF-8 HTML file with inline CSS and data-URI figures."""
    s = bundle.scorecard
    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html_text(bundle.title)}</title><style>{css_for(bundle.mode)}</style></head><body>",
        "<div class='eyebrow'>Chronos Studio · S15 Reports &amp; Finalization</div>",
        f"<h1>{html_text(bundle.title)}</h1>",
        f"<p class='meta'>Generated {html_text(bundle.generated_at)} · seed {html_text(bundle.session.seed)} "
        f"· data hash <code>{html_text(bundle.data_hash)}</code> · {html_text(bundle.mode)} theme · "
        f"{html_text(bundle.scope)} catalogue scope</p>",
        f"<div class='verdict {html_text(s.status)}'><span class='score'>{s.score}/{s.max_score}</span>"
        f"{html_text(s.verdict)}</div>",
        "<div class='callout'><strong>Privacy &amp; provenance.</strong> Numbers are copied from stored "
        "results and the Part 16 catalogue. No raw values are sent to an AI service and no Gemini key "
        "is stored in this report. Template prose is deterministic; verify it before submission.</div>",
        f"<h2>Manuscript-readiness scorecard</h2>{scorecard_table(bundle)}",
    ]
    if s.next_steps:
        parts.append("<h2>What to do next</h2><ol>")
        parts.extend(
            f"<li>{html_text(x.text)} <span class='small'>"
            f"[{html_text(', '.join(x.result_ids))}]</span></li>" for x in s.next_steps
        )
        parts.append("</ol>")
    parts.append("<h2 class='page-break'>Manuscript template prose</h2>")
    for section in bundle.sections:
        parts.append(f"<h3>{html_text(section['title'])}</h3>")
        parts.extend(f"<p>{html_text(p)}</p>" for p in section["paragraphs"])
        parts.append(
            f"<p class='small'>Grounded in: {html_text(', '.join(section['result_ids']))}. "
            f"{html_text(section['note'])}</p>"
        )
    parts.append("<h2 class='page-break'>Figures</h2>")
    for i, item in enumerate(bundle.plots, start=1):
        parts.append(embedded_figure_html(bundle, item, i))
    parts.append("<h2 class='page-break'>Tables</h2>")
    for i, item in enumerate(bundle.tables, start=1):
        parts.append(f"<figure class='item' id='{html_text(safe_stem(item.get('id')))}'>")
        parts.append(html_table(item, number=i))
        parts.append("</figure>")
    libs = ", ".join(f"{k} {v}" for k, v in sorted(bundle.session.state.library_versions.items())[:18])
    parts.append(
        "<div class='footer'>Chronos Studio · one-file HTML · embedded vector figures at "
        f"{html_text(bundle.mode)} house theme · 300 dpi static downloads available separately. "
        f"Library versions: {html_text(libs)}.</div></body></html>"
    )
    return "".join(parts).encode("utf-8")


__all__ = ["CSS_TEMPLATE", "css_for", "embedded_figure_html", "html_table", "render_html", "scorecard_table"]
