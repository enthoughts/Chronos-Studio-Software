"""
api/catalog.py — Part 16's completeness catalogue, export and report-insert endpoints.

This is deliberately **not** Part 18's whole-report / archive router: the
Part 18 export router consumes this catalogue. Part 16 owns the
per-item scorecard, the data-behind downloads, the 300 dpi/static/vector exports and
the one-click "insert into report" basket that Part 18 will consume later.

No new pipeline stage is introduced. The catalogue reads stored results and stored
objects, attributes every item to an existing S0–S15 stage, and leaves the session's
pipeline state alone.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel, Field

from app.api.deps import session_dep
from app.charts import completeness as C
from app.charts import exporters as E
from app.charts.theme import ThemeMode
from app.core.errors import ChronosError
from app.core.session import Session

router = APIRouter(tags=["catalog"])


class CatalogItemNotFound(ChronosError):
    code = "catalog.item_not_found"
    http_status = 404
    plain = "That plot or table is not in this session's catalogue."
    fix = "Refresh the catalogue and choose an item that has been computed."


class ReportAction(BaseModel):
    """One-click report insertion, removal, bulk selection or clear."""
    action: Literal["add", "remove", "set", "clear"] = "add"
    item_id: str | None = Field(default=None, description="Item to add or remove.")
    item_ids: list[str] | None = Field(default=None, description="Items for a bulk set.")


def _catalog(session: Session, mode: ThemeMode) -> dict[str, Any]:
    return C.build_catalog(session, mode=mode)


def _report_payload(session: Session, catalog: dict[str, Any]) -> dict[str, Any]:
    ids: list[str] = []
    items: list[dict[str, Any]] = []
    for item_id in C.report_item_ids(session):
        item = C.find_item(catalog, item_id)
        if item is not None:
            ids.append(item_id)
            items.append(C.public_item(item))
    return {"item_ids": ids, "items": items, "count": len(items)}


@router.get("/analysis/catalog")
def read_catalog(
    mode: ThemeMode = Query("light", description="House theme: light, dark or print."),
    session: Session = Depends(session_dep),
) -> dict[str, Any]:
    """The §5.10 scorecard: every computed plot and table, with captions and provenance."""
    catalog = _catalog(session, mode)
    return {
        "status": "complete",
        "stage": "S15",
        "mode": mode,
        "scorecard": catalog["scorecard"],
        "groups": list(C.GROUPS),
        "plots": [C.public_item(x) for x in catalog["plots"]],
        "tables": [C.public_item(x) for x in catalog["tables"]],
        "report": _report_payload(session, catalog),
        "exports": {
            "formats": list(E.FORMATS),
            "dpi": E.DPI,
            "scale": E.PNG_SCALE,
            "endpoint": "/api/v1/analysis/catalog/{item_id}/download",
        },
    }


@router.get("/analysis/catalog/report")
def read_report_selection(
    mode: ThemeMode = Query("light"),
    session: Session = Depends(session_dep),
) -> dict[str, Any]:
    """Items the user has inserted into the future write-up."""
    catalog = _catalog(session, mode)
    return {"status": "complete", **_report_payload(session, catalog)}


@router.post("/analysis/catalog/report")
def update_report_selection(
    body: ReportAction,
    mode: ThemeMode = Query("light"),
    session: Session = Depends(session_dep),
) -> dict[str, Any]:
    """Add/remove/replace/clear report-insert selections, refusing unknown ids."""
    catalog = _catalog(session, mode)
    current = C.report_item_ids(session)
    wanted: list[str] = []
    if body.action == "clear":
        wanted = []
    elif body.action == "set":
        wanted = list(body.item_ids or ([body.item_id] if body.item_id else []))
    elif body.action == "add":
        wanted = current + list(body.item_ids or ([body.item_id] if body.item_id else []))
    elif body.action == "remove":
        drop = set(body.item_ids or ([body.item_id] if body.item_id else []))
        wanted = [x for x in current if x not in drop]
    for item_id in wanted:
        if C.find_item(catalog, item_id) is None:
            raise CatalogItemNotFound(
                context={"item_id": item_id},
                technical=f"No plot or table with id '{item_id}' exists in this session.",
            )
    C.set_report_item_ids(session, wanted)
    return {"status": "complete", "action": body.action, **_report_payload(session, catalog)}


@router.get("/analysis/catalog/{item_id}/download")
def download_catalog_item(
    item_id: str,
    format: str = Query("png", description="png, svg, pdf, xlsx or csv."),
    mode: ThemeMode = Query("light", description="House theme for static exports."),
    session: Session = Depends(session_dep),
) -> Response:
    """One catalogue item as a 300 dpi PNG, SVG, PDF, XLSX or CSV download."""
    catalog = _catalog(session, mode)
    item = C.find_item(catalog, item_id)
    if item is None:
        raise CatalogItemNotFound(context={"item_id": item_id})
    data, media = E.download_bytes(item, format, mode=mode)
    safe_name = E.filename(item, format.lower())
    return Response(
        content=data,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "X-Catalog-Item": item_id,
            "X-Catalog-Dpi": str(E.DPI),
            "X-Catalog-Mode": mode,
            "Cache-Control": "no-store",
        },
    )


__all__ = ["router"]
