"""
api/eda.py — S1 endpoints: run and read the exploratory analysis.

`POST /analysis/profile` runs S1 and stores the results on the session.
`GET  /analysis/profile` reads them back (so a page reload costs nothing).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.deps import session_dep
from app.core.errors import ChronosError
from app.core.schemas import AnalysisResponse, DataClass, STAGE_LABELS, Stage
from app.core.session import Session
from app.ts.eda.profile import EdaOutput, run_eda

router = APIRouter(tags=["analysis"])


class ProfileRequest(BaseModel):
    window: int | None = None
    audience: str = "beginner"


@router.post("/analysis/profile", response_model=AnalysisResponse)
def run_profile(
    body: ProfileRequest | None = None,
    session: Session = Depends(session_dep),
) -> AnalysisResponse:
    """Run the S1 exploratory battery. Safe to call repeatedly (idempotent)."""
    t0 = time.perf_counter()
    ds = session.state.dataset
    series: pd.Series | None = session.objects.get("series")
    if ds is None or series is None:
        raise ChronosError(
            plain="There is no data loaded in this session yet.",
            fix="Load a file or an example dataset in Step 0 first.",
        )
    session.require_stage(Stage.S1_PROFILE)

    out: EdaOutput = run_eda(
        series,
        name=ds.name,
        frequency_label=ds.frequency.label,
        freq_code=ds.frequency.code,
        period=ds.frequency.period,
        data_class=DataClass(ds.data_class),
        window=(body.window if body else None),
        audience=(body.audience if body else "beginner"),
    )

    # Replace any previous S1 results so re-running never doubles the panel.
    session.results[Stage.S1_PROFILE] = {}
    for result in out.results:
        session.add_result(result)

    session.objects["eda_facts"] = out.facts
    # The flagged points themselves, not just the count: the exogenous models build their
    # intervention dummies from exactly these, and a second detection pass inside the model would be
    # a second answer to the same question.
    session.objects["eda_outliers"] = [
        {"time": str(o.time), "kind": o.type}
        for o in out.outliers
    ]
    # Which series version these facts describe. Advice computed from facts that belong
    # to a different version would be advice about data the user no longer has.
    session.objects["eda_series_version"] = session.state.series_version
    session.mark_stage(
        Stage.S1_PROFILE, "complete",
        f"{len(out.results)} results across 1 series",
    )
    for cav in out.caveats:
        if cav not in session.state.caveats:
            session.state.caveats.append(cav)

    return AnalysisResponse(
        stage=Stage.S1_PROFILE,
        advisor=None,     # S1 makes no recommendations; S2 does.
        results=out.results,
        pipeline=session.state.progress(),
        caveats=out.caveats,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/analysis/profile/results")
def read_profile(session: Session = Depends(session_dep)) -> dict[str, Any]:
    results = session.results_for(Stage.S1_PROFILE)
    return {
        "stage": Stage.S1_PROFILE.value,
        "label": STAGE_LABELS[Stage.S1_PROFILE],
        "status": session.state.stages[Stage.S1_PROFILE.value].status,
        "results": [r.model_dump(mode="json") for r in results],
        "facts": session.objects.get("eda_facts", {}),
        "caveats": session.state.caveats,
    }


@router.get("/analysis/facts")
def read_facts(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    The flat fact dictionary produced so far, per stage.

    Exposed deliberately: the advisor rules (Part 3) and the AI digest (Part 17)
    both read from here, so a reviewer can see exactly what was computed.
    """
    return {
        "eda": session.objects.get("eda_facts", {}),
        "dataset": (
            {
                "n": session.state.dataset.n_observations,
                "frequency": session.state.dataset.frequency.code,
                "period": session.state.dataset.frequency.period,
                "data_class": session.state.dataset.data_class.value
                if hasattr(session.state.dataset.data_class, "value")
                else str(session.state.dataset.data_class),
                "content_hash": session.state.dataset.content_hash,
            } if session.state.dataset else {}
        ),
        "series_version": session.state.series_version,
        "seed": session.state.seed,
        # Every caveat any stage has attached, in order. The report and the AI digest both read
        # this list, so it has to be visible here — an endpoint called "facts" that omits the
        # travelling caveats forces every consumer to reach into the session instead.
        "caveats": list(session.state.caveats),
    }


@router.get("/analysis/series")
def read_series(
    limit: int = Query(2000, ge=10, le=20000),
    session: Session = Depends(session_dep),
) -> dict[str, Any]:
    """
    The series as points, for client-side plotting and for "download the data
    behind this plot". Large series are decimated for *display* only, with an
    explicit flag so nothing pretends the points are the full data (§5.1).
    """
    series: pd.Series | None = session.objects.get("series")
    if series is None:
        return {"points": [], "n": 0, "decimated": False, "message": "No data loaded."}
    clean = series.dropna()
    total = len(clean)
    decimated = total > limit
    view = clean.iloc[:: max(1, int(np.ceil(total / limit)))] if decimated else clean
    return {
        "points": [
            {"time": str(t), "value": float(v)} for t, v in view.items()
        ],
        "n": int(total),
        "n_displayed": int(len(view)),
        "decimated": bool(decimated),
        "note": (
            f"Showing {len(view)} of {total} points for display speed. "
            "All statistics use the full series."
        ) if decimated else "",
        "missing": int(series.isna().sum()),
    }

__all__ = ["router"]
