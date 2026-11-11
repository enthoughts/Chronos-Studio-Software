"""Small, dependency-light helpers shared by the report builders."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from app.core.config import ASSETS_DIR, optional_features
from app.core.schemas import SCHEMA_VERSION, STAGE_LABELS, Stage

FALLBACK_TEMPLATE = "the app has not computed that yet — open the X panel"
BANNED_SETTING_WORDS = (
    "key", "secret", "token", "password", "credential", "raw_table", "series_original",
    "dataframe", "frame", "fit", "wrapper",
)


def stage_label(stage: str | Stage) -> str:
    try:
        s = stage if isinstance(stage, Stage) else Stage(str(stage))
        return STAGE_LABELS[s]
    except (ValueError, KeyError):
        return "relevant"


def fallback_for(stage: str | Stage) -> str:
    return FALLBACK_TEMPLATE.replace("X", stage_label(stage))


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        try:
            out = value.model_dump(mode="json")
            return out if isinstance(out, dict) else {}
        except Exception:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return asdict(value)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            out = value.to_dict()
            return out if isinstance(out, dict) else {}
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {}


def jsonable(value: Any, *, depth: int = 0) -> Any:
    """Strict-finite JSON conversion, including Pydantic/dataclass/numpy objects."""
    if depth > 6:
        return str(value)[:300]
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return jsonable(value.item(), depth=depth + 1)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="json"), depth=depth + 1)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return jsonable(asdict(value), depth=depth + 1)
        except Exception:
            return str(value)[:300]
    if isinstance(value, dict):
        return {str(k): jsonable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v, depth=depth + 1) for v in value]
    if hasattr(value, "to_numpy"):
        try:
            return jsonable(value.to_numpy().tolist(), depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist(), depth=depth + 1)
        except Exception:
            pass
    return str(value)[:300]


def safe_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        jsonable(value), ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"),
        allow_nan=False,
    )


def cell_text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return str(value) if math.isfinite(value) else "—"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)[:300]
    return str(value)


def html_text(value: Any) -> str:
    return html.escape(cell_text(value), quote=True)


def latex_text(value: Any) -> str:
    text = cell_text(value)
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
        "&": r"\&", "#": r"\#", "%": r"\%", "_": r"\_", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def safe_stem(value: Any, fallback: str = "chronos-item") -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-._")
    return (stem or fallback)[:100]


def content_hash(values: list[float]) -> str:
    data = json.dumps([float(v) for v in values], separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def example_id_for(dataset_name: str) -> str | None:
    path = ASSETS_DIR / "examples" / "manifest.json"
    if not path.is_file():
        return None
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(rows, dict):
        rows = [{"id": key, **value} if isinstance(value, dict) else {"id": key}
                for key, value in rows.items()]
    name = str(dataset_name or "").strip().lower()
    for row in rows:
        choices = {
            str(row.get("id", "")).lower(),
            str(row.get("title", "")).lower(),
            str(row.get("file", "")).replace(".csv", "").lower(),
        }
        if name and (name in choices or any(name in x for x in choices if x)):
            return str(row.get("id") or "") or None
    return None


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if any(word in lower for word in BANNED_SETTING_WORDS):
                continue
            out[str(key)] = _scrub(item)
        return out
    if isinstance(value, list):
        return [_scrub(x) for x in value]
    return value


def analysis_settings(session: Any) -> dict[str, Any]:
    """The replayable, non-secret settings a reproducible script needs."""
    dataset = session.state.dataset
    forecast = as_dict(session.objects.get("forecast"))
    bundle = forecast.get("bundle")
    horizon = getattr(bundle, "horizon", None) or as_dict(bundle).get("horizon") or 6
    evaluation = as_dict(session.objects.get("evaluation"))
    report = as_dict(evaluation.get("report"))
    inputs = as_dict(report.get("inputs"))
    ev_horizon = inputs.get("horizon") or inputs.get("forecast_horizon") or horizon
    folds = inputs.get("folds") or 4
    n_paths = inputs.get("n_paths") or 2000
    reps = inputs.get("rqr_reps") or inputs.get("reps") or 5
    decomp = as_dict(session.objects.get("decomposition"))
    method = decomp.get("method") or "stl"
    mapping = jsonable(as_dict(session.objects.get("mapping")))
    stage_calls = [
        {"path": "profile", "payload": {}},
        {"path": "advice", "payload": {}},
        {"path": "decompose", "payload": {"method": method}},
        {"path": "tests", "payload": {}},
        {"path": "correlate", "payload": None, "method": "GET"},
        {"path": "next", "payload": None, "method": "GET"},
        {"path": "gate", "payload": {}},
        {"path": "suggest", "payload": {}},
        {"path": "fit", "payload": {"model_ids": list(as_dict(session.objects.get("fits")).keys())[:8]}},
        {"path": "select", "payload": {"horizon": ev_horizon, "folds": folds}},
        {"path": "diagnose", "payload": {}},
        {"path": "forecast", "payload": {"horizon": horizon}},
        {"path": "evaluate", "payload": {
            "horizon": ev_horizon, "folds": folds, "n_paths": n_paths, "reps": reps,
        }},
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": session.seed,
        "library_versions": dict(session.state.library_versions),
        "dataset": jsonable(dataset.model_dump(mode="json")) if dataset else None,
        "content_hash": dataset.content_hash if dataset else "",
        "example_id": example_id_for(dataset.name) if dataset else None,
        "source": dataset.source if dataset else "upload",
        "mapping": mapping,
        "preprocessing": jsonable([p.model_dump(mode="json") for p in session.state.preprocessing]),
        "series_version": session.state.series_version,
        "stage_status": {k: v.status for k, v in session.state.stages.items()},
        "stage_calls": stage_calls,
        "catalog_report": list(session.objects.get("catalog_report") or []),
        "optional_features": optional_features(),
    }


def settings_payload(session: Any) -> dict[str, Any]:
    return _scrub(jsonable(analysis_settings(session)))


__all__ = [
    "FALLBACK_TEMPLATE", "analysis_settings", "as_dict", "cell_text", "content_hash",
    "example_id_for", "fallback_for", "html_text", "jsonable", "latex_text",
    "safe_json", "safe_stem", "settings_payload", "stage_label",
]
