"""Publication-resolution PNG/SVG/PDF exports for every catalogue figure."""

from __future__ import annotations

import io
from zipfile import ZIP_DEFLATED, ZipFile

from app.charts import exporters as E
from app.ts.report.bundle import ReportBundle
from app.ts.report.utils import safe_json, safe_stem

STATIC_FORMATS = ("png", "svg", "pdf")


def render_figures(bundle: ReportBundle) -> bytes:
    out = io.BytesIO()
    manifest: list[dict[str, object]] = []
    errors: list[str] = []
    readme = (
        "Chronos Studio publication figures\n"
        f"Report: {bundle.title}\nGenerated: {bundle.generated_at}\n"
        f"House theme: {bundle.mode} · PNG scale: {E.PNG_SCALE} (300 dpi) · "
        f"CSS page: {E.CSS_WIDTH}×{E.CSS_HEIGHT}\n"
        "Every catalogue figure is supplied as PNG (3750 px wide), vector SVG and vector PDF.\n"
        "Captions and alt text are in captions.json.\n"
    )
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme)
        for item in bundle.plots:
            stem = safe_stem(item.get("id"), fallback="figure")
            entry: dict[str, object] = {
                "id": item.get("id"), "title": item.get("title"), "stem": stem,
                "caption": item.get("caption"), "alt_text": item.get("alt_text"),
                "stage": item.get("stage"), "group": item.get("group"), "formats": [],
            }
            for fmt in STATIC_FORMATS:
                try:
                    data = bundle.figure_bytes(item, fmt)
                    name = f"figures/{stem}.{fmt}"
                    zf.writestr(name, data)
                    (entry["formats"] if isinstance(entry["formats"], list) else []).append(
                        {"format": fmt, "file": name, "bytes": len(data),
                         "media_type": E.media_type(fmt)}
                    )
                except Exception as exc:
                    errors.append(f"{item.get('id')} {fmt}: {type(exc).__name__}: {exc}")
            manifest.append(entry)
        zf.writestr("captions.json", safe_json(manifest, indent=2))
        zf.writestr("dpi.txt", f"{E.DPI}\n")
        if errors:
            zf.writestr("errors.txt", "\n".join(errors) + "\n")
        zf.writestr("manifest.json", safe_json({
            "schema_version": bundle.settings.get("schema_version"),
            "generated_at": bundle.generated_at,
            "session_id": bundle.session.id,
            "data_hash": bundle.data_hash,
            "mode": bundle.mode,
            "scope": bundle.scope,
            "figures": len(bundle.plots),
            "formats": list(STATIC_FORMATS),
            "dpi": E.DPI,
            "errors": errors,
        }, indent=2))
    return out.getvalue()


__all__ = ["STATIC_FORMATS", "render_figures"]
