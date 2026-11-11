"""
api/datasets.py — S0 endpoints: upload, examples, mapping, preview.

The HTTP layer stays thin: it validates the transport, calls `ts/io/ingest.py`,
and stores the resulting series on the session's server-side objects. All
parsing logic lives in `app/ts/io/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import session_dep
from app.core.config import ASSETS_DIR, optional_features
from app.core.errors import ChronosError, ColumnMappingError, UnsupportedFormatError
from app.core.schemas import DataClass, ParseIssue, Stage
from app.core.session import Session
from app.ts.io.ingest import (
    ColumnMapping,
    IngestResult,
    available_calendars,
    ingest_table,
    ingest_text,
    suggest_mapping,
)
from app.ts.io.readers import read_any, supported_extensions
from app.ts.io.validate import profile_columns

router = APIRouter(tags=["data"])

EXAMPLES_DIR = Path(ASSETS_DIR) / "examples"


def _event_times(result: IngestResult) -> pd.DatetimeIndex:
    """
    The raw event timestamps from an event-mode ingest.

    `raw` holds the original table, so the dates are read back from the column the mapping chose —
    including duplicates, which is the point: two events on one day are two events, and a count series
    that collapsed them would leave a Hawkes model with nothing to model.
    """
    raw = result.raw
    df = getattr(raw, "df", None)
    col = result.mapping.time_column if result.mapping is not None else None
    if df is None or col is None or col not in df.columns:
        return pd.DatetimeIndex([])
    parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=bool(result.mapping.dayfirst))
    return pd.DatetimeIndex(parsed.dropna().sort_values())


def _mapping_from_payload(raw: dict[str, Any] | None) -> ColumnMapping | None:
    if not raw:
        return None
    allowed = {f for f in ColumnMapping.__dataclass_fields__}  # type: ignore[attr-defined]
    return ColumnMapping(**{k: v for k, v in raw.items() if k in allowed})


def _store(session: Session, result: IngestResult) -> None:
    """Persist the dataset + series and mark S0 complete."""
    session.state.dataset = result.dataset
    session.objects["series"] = result.series
    # The untouched series is what undo replays from, so it is captured once at load
    # and never overwritten by preprocessing.
    session.objects["series_original"] = result.series.copy()
    session.objects["exogenous"] = result.exogenous
    # Event-timestamp input keeps its raw timestamps: the point-process models need the intervals,
    # and a count series cannot recover them.
    if result.dataset.data_class is DataClass.EVENT_TIMESTAMP:
        session.objects["event_times"] = _event_times(result)
    session.objects["raw_table"] = result.raw
    session.objects["mapping"] = result.mapping
    session.objects["profiles"] = result.profiles
    session.state.series_version = 1
    session.mark_stage(Stage.S0_INGEST, "complete",
                       f"{result.dataset.n_observations} observations loaded")


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@router.get("/data/formats")
def formats() -> dict[str, Any]:
    return {
        "extensions": list(supported_extensions()),
        "max_upload_mb": 64,
        "calendars": available_calendars(),
        "optional_readers": optional_features(),
    }


@router.get("/data/examples")
def list_examples() -> dict[str, Any]:
    """The bundled teaching datasets, each with what it is meant to teach."""
    manifest_path = EXAMPLES_DIR / "manifest.json"
    if not manifest_path.exists():
        return {"examples": []}
    manifest = json.loads(manifest_path.read_text())
    for item in manifest:
        item["available"] = (EXAMPLES_DIR / item["file"]).exists()
    return {"examples": manifest}


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
class IngestResponse(BaseModel):
    dataset: dict[str, Any]
    mapping: dict[str, Any]
    cross_sell: list[str] = Field(
        default_factory=list,
        description="Other time/value columns found in the file, offered in one click.",
    )
    profiles: list[dict[str, Any]] = Field(default_factory=list)
    candidate_frequencies: list[dict[str, Any]] = Field(default_factory=list)
    next: str = "S1"


@router.post("/data/upload", response_model=IngestResponse, status_code=201)
async def upload(
    file: UploadFile = File(...),
    mapping_json: str | None = Form(None),
    sheet_name: str | None = Form(None),
    session: Session = Depends(session_dep),
) -> IngestResponse:
    """Accept a file and run S0. `mapping_json` lets the user override our guesses."""
    data = await file.read()
    mapping = _mapping_from_payload(json.loads(mapping_json) if mapping_json else None)
    result = ingest_table(
        read_any(data, filename=file.filename or "upload.csv", sheet_name=sheet_name),
        mapping=mapping,
        name=file.filename or "upload",
    )
    _store(session, result)
    return _to_response(result)


class PasteRequest(BaseModel):
    text: str
    mapping: dict[str, Any] | None = None


@router.post("/data/paste", response_model=IngestResponse, status_code=201)
def paste(body: PasteRequest, session: Session = Depends(session_dep)) -> IngestResponse:
    """Clipboard paste — the fastest path for a table copied out of Excel."""
    result = ingest_text(body.text, mapping=_mapping_from_payload(body.mapping))
    _store(session, result)
    return _to_response(result)


class ExampleRequest(BaseModel):
    example_id: str
    mapping: dict[str, Any] | None = None


@router.post("/data/example", response_model=IngestResponse, status_code=201)
def load_example(body: ExampleRequest, session: Session = Depends(session_dep)) -> IngestResponse:
    manifest_path = EXAMPLES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise ChronosError(
            plain="The example datasets are missing from this installation.",
            fix="Run `python scripts/gen_examples.py` in the backend folder to build them.",
        )
    manifest = {m["id"]: m for m in json.loads(manifest_path.read_text())}
    item = manifest.get(body.example_id)
    if item is None:
        raise ChronosError(
            code="data.example_unknown",
            plain=f"There is no example dataset called '{body.example_id}'.",
            fix="Pick one from the Example data list.",
        )
    path = EXAMPLES_DIR / item["file"]
    result = ingest_table(
        read_any(path.read_bytes(), filename=item["file"]),
        mapping=_mapping_from_payload(body.mapping),
        name=item["title"],
        source="example",
    )
    # Sanity-check our detection against the manifest's expectation, and say so if
    # they disagree — that is a bug worth surfacing, not hiding.
    expected = item.get("data_class")
    if expected and result.dataset.data_class.value != expected:
        result.dataset.quality.issues.append(
            ParseIssue(
                code="data.class_mismatch", severity="warning",
                plain=(
                    f"This dataset is documented as '{expected}' but was detected as "
                    f"'{result.dataset.data_class.value}'. You can override the data class "
                    "in Column mapping."
                ),
                fix="Report this if the detection looks wrong.",
            )
        )
    _store(session, result)
    resp = _to_response(result)
    resp.profiles = [p.model_dump(mode="json") for p in result.profiles]
    return resp


# --------------------------------------------------------------------------- #
# Mapping assistant (before upload, for the UI's column pickers)
# --------------------------------------------------------------------------- #
@router.post("/data/inspect")
async def inspect(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(None),
) -> dict[str, Any]:
    """
    Load the file, propose a mapping, and return previews — without committing.
    The UI uses this to show the column picker with our best guess preselected.
    """
    data = await file.read()
    table = read_any(data, filename=file.filename or "upload.csv", sheet_name=sheet_name)
    mapping, issues, notes = suggest_mapping(table.df)
    return {
        "columns": [
            {
                "name": str(c),
                "samples": [str(x) for x in table.df[c].dropna().head(5).tolist()],
            }
            for c in table.df.columns
        ],
        "suggested_mapping": mapping.__dict__,
        "issues": [i.model_dump(mode="json") for i in issues],
        "notes": notes,
        "n_rows": int(len(table.df)),
        "format": table.format,
        "sheet_name": table.sheet_name,
        "loader_notes": table.notes,
        "profiles": [p.model_dump(mode="json") for p in profile_columns(table.df)],
    }


class RemapRequest(BaseModel):
    mapping: dict[str, Any]


@router.post("/data/remap", response_model=IngestResponse)
def remap(body: RemapRequest, session: Session = Depends(session_dep)) -> IngestResponse:
    """Re-run S0 with a different column mapping — no re-upload needed."""
    raw = session.objects.get("raw_table")
    if raw is None:
        raise ColumnMappingError(
            plain="There is no uploaded table in this session to re-map.",
            fix="Upload the file again.",
        )
    result = ingest_table(raw, mapping=_mapping_from_payload(body.mapping), name=session.state.dataset.name
                          if session.state.dataset else raw.source, source="upload")
    _store(session, result)
    return _to_response(result)


# --------------------------------------------------------------------------- #
# Current dataset
# --------------------------------------------------------------------------- #
@router.get("/data/current")
def current_dataset(session: Session = Depends(session_dep)) -> dict[str, Any]:
    ds = session.state.dataset
    if ds is None:
        return {"dataset": None, "message": "No data has been loaded yet."}
    series: pd.Series | None = session.objects.get("series")
    return {
        "dataset": ds.model_dump(mode="json"),
        "quality": ds.quality.model_dump(mode="json"),
        "mapping": session.objects.get("mapping").__dict__
        if session.objects.get("mapping") else None,
        "series_head": (
            [{"time": str(t), "value": (None if pd.isna(v) else float(v))}
             for t, v in series.head(10).items()]
            if series is not None else []
        ),
        "n_missing": int(series.isna().sum()) if series is not None else 0,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_response(result: IngestResult) -> IngestResponse:
    df = result.raw.df
    ds = result.dataset
    others = [
        str(c) for c in df.columns
        if str(c) not in {ds.target_column, (ds.quality.time_column or "")}
        and str(c) not in ds.exogenous_columns
    ][:5]
    return IngestResponse(
        dataset=ds.model_dump(mode="json"),
        mapping=result.mapping.__dict__,
        cross_sell=others,
        profiles=[p.model_dump(mode="json") for p in result.profiles],
        candidate_frequencies=[
            f.model_dump(mode="json") for f in result.candidate_frequencies
        ],
    )


__all__ = ["router"]
