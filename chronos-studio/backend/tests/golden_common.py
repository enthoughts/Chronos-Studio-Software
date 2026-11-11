"""
golden_common.py — shared case definitions for the Part 19 golden-file tests.

Kept outside the test module so `scripts/generate_golden.py` can regenerate the
stored reference with exactly the same code the test uses. A golden test must
never compute its reference with different logic than the generator did.

One case per model *family* in the registry (§8 Part 19: "golden-file fixtures
per model family"). Each case loads a bundled seed dataset through the real
ingestion path (`read_any` → `ingest_table`) and fits through `fit_one`, so the
fixture exercises the same public surface the application uses.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from app.core.schemas import DataClass
from app.ts.io.ingest import ingest_table
from app.ts.io.readers import read_any
from app.ts.models.fit import fit_one

warnings.simplefilter("ignore")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = BACKEND_ROOT / "app" / "assets" / "examples"
FIXTURES = BACKEND_ROOT / "tests" / "fixtures"
GOLDEN_MODELS_PATH = FIXTURES / "golden_models.json"
REFERENCE_PATH = FIXTURES / "airpassengers_reference.json"

MANIFEST = {m["id"]: m for m in json.loads((EXAMPLES / "manifest.json").read_text())}

# (family, model_id, bundled example) — one per implemented family.
CASES: tuple[tuple[str, str, str], ...] = (
    ("baseline", "naive", "airpassengers"),
    ("smoothing", "ets", "airpassengers"),
    ("arma", "arima", "airpassengers"),
    ("structural", "ucm", "airpassengers"),
    ("count", "poisson", "outbreak_counts"),
    ("intermittent", "croston", "spare_parts"),
    ("volatility", "garch", "volatile_returns"),
    ("multivariate", "var", "traffic_spend"),
    ("exogenous", "arimax", "traffic_spend"),
    ("nonlinear", "setar", "lynx"),
    ("ml", "random_forest", "airpassengers"),
    ("point_process", "renewal", "machine_service"),
    ("combination", "combination", "airpassengers"),
)


def load_example(example_id: str):
    """Load a seed dataset exactly as the API does, including event timestamps."""
    item = MANIFEST[example_id]
    raw = read_any((EXAMPLES / item["file"]).read_bytes(), filename=item["file"])
    res = ingest_table(raw, name=item["title"], source="example")
    # Event-timestamp inputs need the raw parsed timestamps (see datasets.py).
    col = res.mapping.time_column
    frame = getattr(res.raw, "df", None)
    event_times = pd.DatetimeIndex([])
    if frame is not None and col and col in frame.columns:
        parsed = pd.to_datetime(frame[col], errors="coerce",
                                dayfirst=bool(res.mapping.dayfirst))
        event_times = pd.DatetimeIndex(parsed.dropna().sort_values())
    return res, event_times


def compute_case(family: str, model_id: str, example_id: str) -> dict:
    """Fit one model and return the stable, comparable slice of its summary."""
    res, event_times = load_example(example_id)
    ds = res.dataset
    dc = ds.data_class if isinstance(ds.data_class, DataClass) else DataClass(ds.data_class)
    try:
        outcome = fit_one(
            model_id, res.series,
            period=ds.frequency.period,
            data_class=dc,
            exog=res.exogenous,
            event_times=event_times,
            frequency_code=ds.frequency.code,
        )
        s = outcome.summary
        return {
            "family": family, "model_id": model_id, "example_id": example_id,
            "failed": bool(s.failed), "converged": bool(s.converged),
            "n_obs_used": int(s.n_obs_used), "n_params": int(s.n_params),
            "aic": s.aic, "aicc": s.aicc, "bic": s.bic, "loglik": s.loglik,
            "ic_comparable": bool(s.ic_comparable),
            "failure_reason": str(s.failure_reason)[:200],
        }
    except Exception as exc:  # a fit that raises is still a golden outcome
        return {
            "family": family, "model_id": model_id, "example_id": example_id,
            "failed": True, "converged": False, "n_obs_used": 0, "n_params": 0,
            "aic": None, "aicc": None, "bic": None, "loglik": None,
            "ic_comparable": False,
            "failure_reason": f"{type(exc).__name__}: {exc}"[:200],
        }


def library_versions() -> dict[str, str]:
    import importlib.metadata as md
    out = {}
    for pkg in ("statsmodels", "scikit-learn", "scipy", "numpy", "pandas"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def build_fixture() -> dict:
    return {
        "source": (
            "Per-family golden fits on bundled seed data. Numeric fields are "
            "information-criterion values produced by statsmodels / scikit-learn "
            "through app.ts.models.fit.fit_one. Regenerate with "
            "PYTHONPATH=. python scripts/generate_golden.py (see CONTRIBUTING.md)."
        ),
        "libraries": library_versions(),
        "cases": [compute_case(*c) for c in CASES],
    }
