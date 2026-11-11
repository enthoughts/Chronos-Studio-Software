"""
charts/exporters.py — Part 16's per-item static and data exports (§5.10, §5.15).

One catalogue item can be downloaded as:

* PNG at **300 dpi** (1200 CSS px × 3.125 = 3750 raster px),
* SVG (vector, print-safe),
* PDF (vector page),
* XLSX (the numbers behind the figure/table, plus caption and provenance),
* CSV (the same numbers in a universally readable form).

All static rendering goes through the same house theme as the interactive
figures. PNG/PDF use a lazily started local Kaleido/Chromium server with the
local Plotly UMD bundle; no network access or cloud export is involved. SVG
falls back to a pure-Python vector renderer when that optional browser stack
is absent, so the one-file HTML report keeps its figures in the standalone
build.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import threading
import warnings
from pathlib import Path
from typing import Any, Literal

import plotly

from app.charts.svg_fallback import render_svg
from app.charts.theme import (
    SPREADSHEET_FONT,
    SPREADSHEET_HEADER_FILL,
    SPREADSHEET_HEADER_FONT,
    ThemeMode,
    restyled,
)
from app.core.browser import find_chromium
from app.core.errors import ChronosError

ExportFormat = Literal["png", "svg", "pdf", "xlsx", "csv"]
STATIC_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")
DATA_FORMATS: tuple[str, ...] = ("xlsx", "csv")
FORMATS: tuple[str, ...] = (*STATIC_FORMATS, *DATA_FORMATS)

#: Browser CSS is 96 dpi, so 300 dpi means a 3.125× raster scale.
DPI = 300
CSS_WIDTH = 1200
CSS_HEIGHT = 800
PNG_SCALE = DPI / 96.0

_LOCK = threading.Lock()
_STARTED = False
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"

MEDIA_TYPES: dict[str, str] = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv; charset=utf-8",
}


def _chrome_path() -> str:
    """The locally installed Chromium used by Kaleido, with a clear refusal."""
    found = find_chromium()
    if found:
        return found
    raise ChronosError(
        plain="Static PNG and PDF export needs a local Chromium, but none was found.",
        fix="Install Chromium or set CHRONOS_CHROME to its executable path, then try again. "
            "XLSX and CSV data downloads still work without it.",
        technical="Kaleido 1.3 requires a Chrome/Chromium binary; the checked paths are "
                  "CHRONOS_CHROME, CHROME_PATH, the Playwright cache and system browser paths.",
        code="catalog.chromium_missing",
    )


def _ensure_server() -> Any:
    """Start the process-wide Kaleido server once, with local assets only."""
    global _STARTED
    with _LOCK:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import kaleido

        if _STARTED and kaleido._global_server.is_running():
            return kaleido
        if not kaleido._global_server.is_running():
            chrome = _chrome_path()
            if not PLOTLY_JS.is_file():
                raise ChronosError(
                    plain="The local Plotly chart library is missing, so static exports cannot be built.",
                    fix="Reinstall the plotly package, then try the export again.",
                    code="catalog.plotlyjs_missing",
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kaleido.start_sync_server(
                    path=chrome,
                    plotlyjs=PLOTLY_JS,
                    mathjax=False,
                    silence_warnings=True,
                )
        _STARTED = True
        return kaleido


def _figure_dict(figure_json: dict[str, Any], mode: ThemeMode, title: str) -> dict[str, Any]:
    """Restyle and size a figure for a deterministic static page."""
    fig = restyled(figure_json, mode)
    layout = fig.setdefault("layout", {})
    layout["width"] = CSS_WIDTH
    layout["height"] = CSS_HEIGHT
    layout["autosize"] = False
    layout.pop("template", None)
    if title and not (layout.get("title") or {}).get("text"):
        layout["title"] = {"text": title, "x": 0.0, "xanchor": "left"}
    return fig


def static_image_bytes(
    figure_json: dict[str, Any],
    fmt: str,
    *,
    mode: ThemeMode = "light",
    title: str = "",
) -> bytes:
    """Render one figure to PNG/SVG/PDF, with an offline SVG fallback."""
    if fmt not in STATIC_FORMATS:
        raise ChronosError(
            plain=f"'{fmt}' is not a static image format.",
            fix="Choose PNG, SVG or PDF.",
            code="catalog.unsupported_format",
        )
    fig = _figure_dict(figure_json, mode, title)
    if fmt == "svg":
        # SVG is also the image format embedded in the one-file HTML report.
        # Missing Kaleido, a missing browser or a failed local server must not
        # turn every report figure into an error callout.
        try:
            kaleido = _ensure_server()
        except Exception:
            return render_svg(
                fig, mode=mode, title=title,
                width=CSS_WIDTH, height=CSS_HEIGHT,
            )
    else:
        try:
            kaleido = _ensure_server()
        except ImportError as exc:
            raise ChronosError(
                plain="This static export needs Kaleido and a local Chromium.",
                fix="Use SVG, HTML, XLSX or CSV now; install Kaleido and a local "
                    "Chromium to enable 300-dpi PNG and vector PDF downloads.",
                technical=str(exc),
                code="catalog.kaleido_missing",
            ) from exc
    opts: dict[str, Any] = {
        "format": fmt,
        "width": CSS_WIDTH,
        "height": CSS_HEIGHT,
        "scale": PNG_SCALE if fmt == "png" else 1,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = kaleido.write_fig_from_object_sync(
                {"fig": fig, "opts": opts},
                _write=False,
                cancel_on_error=True,
            )
    except Exception as exc:  # pragma: no cover - surfaced as a friendly API error
        raise ChronosError(
            plain="The static image could not be rendered from this figure.",
            fix="Try SVG or PDF, or download the numbers as XLSX/CSV instead.",
            technical=f"{type(exc).__name__}: {exc}",
            code="catalog.render_failed",
        ) from exc
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ChronosError(
            plain="The static image renderer returned an empty file.",
            fix="Try another format, or download the numbers as XLSX/CSV.",
            code="catalog.empty_render",
        )
    return bytes(data)


def _cell_value(value: Any) -> Any:
    """A scalar that openpyxl/csv can write without losing information."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)[:1000]
    if isinstance(value, str):
        return value
    return str(value)


def _table_parts(item: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    columns = list(item.get("columns") or [])
    rows = [dict(r) if isinstance(r, dict) else {"value": r} for r in (item.get("rows") or [])]
    if not columns:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        columns = [{"key": k, "label": k.replace("_", " ").title(),
                    "dtype": "string", "align": "left"} for k in keys]
    return columns, rows


def csv_bytes(item: dict[str, Any]) -> bytes:
    columns, rows = _table_parts(item)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([str(c.get("label") or c.get("key") or "Column") for c in columns])
    for row in rows:
        writer.writerow([_cell_value(row.get(str(c.get("key")))) for c in columns])
    return b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")


def xlsx_bytes(item: dict[str, Any]) -> bytes:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise ChronosError(
            plain="XLSX export needs the openpyxl package, which is not installed.",
            fix="Install openpyxl, or use the CSV data download instead.",
            code="catalog.openpyxl_missing",
        ) from exc
    columns, rows = _table_parts(item)
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    header_fill = PatternFill("solid", fgColor=SPREADSHEET_HEADER_FILL)
    header_font = Font(name=SPREADSHEET_FONT, bold=True, color=SPREADSHEET_HEADER_FONT)
    ws.append([str(c.get("label") or c.get("key") or "Column") for c in columns])
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    for row in rows:
        ws.append([_cell_value(row.get(str(c.get("key")))) for c in columns])
    ws.freeze_panes = "A2"
    for i, col in enumerate(columns, start=1):
        width = min(42, max(11, len(str(col.get("label") or col.get("key") or "")) + 3))
        for row in rows[:80]:
            v = _cell_value(row.get(str(col.get("key"))))
            if v is not None:
                width = max(width, min(42, len(str(v)) + 2))
        ws.column_dimensions[get_column_letter(i)].width = width
    meta = wb.create_sheet("About this item")
    meta.append(["Catalogue item", str(item.get("id") or "")])
    meta.append(["Title", str(item.get("title") or "")])
    meta.append(["Caption", str(item.get("caption") or "")])
    meta.append(["Alt text", str(item.get("alt_text") or "")])
    meta.append(["Group", str(item.get("group") or "")])
    meta.append(["Stage", str(item.get("stage") or "")])
    meta.append(["Source", str(item.get("source") or "")])
    meta.append(["Rows", len(rows)])
    meta.append(["Export DPI", DPI])
    meta.append(["Formats", ", ".join(FORMATS)])
    for row in meta.iter_rows():
        row[0].font = Font(name=SPREADSHEET_FONT, bold=True)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    meta.column_dimensions["A"].width = 20
    meta.column_dimensions["B"].width = 92
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def _table_figure_json(item: dict[str, Any], mode: ThemeMode) -> dict[str, Any]:
    if isinstance(item.get("figure_json"), dict) and item["figure_json"]:
        return item["figure_json"]
    from app.charts import primitives as P

    columns, rows = _table_parts(item)
    chart = P.table_figure(
        columns, rows, title=str(item.get("title") or "Table"),
        subtitle=str(item.get("caption") or "")[:120], mode=mode,
    )
    return chart.figure_json


def download_bytes(
    item: dict[str, Any],
    fmt: str,
    *,
    mode: ThemeMode = "light",
) -> tuple[bytes, str]:
    """Return `(bytes, media_type)` for any catalogue item and any Part 16 format."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ChronosError(
            plain=f"'{fmt}' is not a catalogue export format.",
            fix="Choose PNG, SVG, PDF, XLSX or CSV.",
            code="catalog.unsupported_format",
        )
    if fmt in STATIC_FORMATS:
        figure_json = (
            item.get("figure_json") if item.get("kind") == "plot"
            else _table_figure_json(item, mode)
        )
        return static_image_bytes(
            figure_json, fmt, mode=mode, title=str(item.get("title") or "")
        ), MEDIA_TYPES[fmt]
    if fmt == "csv":
        return csv_bytes(item), MEDIA_TYPES[fmt]
    return xlsx_bytes(item), MEDIA_TYPES[fmt]


def filename(item: dict[str, Any], fmt: str) -> str:
    """A safe, stable download name that keeps the catalogue id visible."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(item.get("id") or item.get("title") or "catalog-item"))
    stem = stem.strip("-._") or "catalog-item"
    return f"{stem}.{fmt}"


def media_type(fmt: str) -> str:
    return MEDIA_TYPES.get(fmt.lower(), "application/octet-stream")


__all__ = [
    "FORMATS", "STATIC_FORMATS", "DATA_FORMATS", "DPI", "PNG_SCALE",
    "MEDIA_TYPES", "static_image_bytes", "csv_bytes", "xlsx_bytes",
    "download_bytes", "filename", "media_type",
]
