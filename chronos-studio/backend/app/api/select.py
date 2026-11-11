"""
api/select.py — S11 (select) and S12 (diagnose) (§5.9, Part 13).

  POST /analysis/select           run rolling-origin CV, rank, and build the selection report
  GET  /analysis/selection        the stored selection report
  POST /analysis/diagnose         the residual battery + adequacy verdict for one fit
  GET  /analysis/diagnostics      every stored diagnostics block (what S12 renders)

Three deliberate API decisions:

**1. Selection is a POST, not a GET.** It refits every model on every fold — it is expensive and it is
an action. A GET that takes forty seconds is a GET somebody's browser retries.

**2. The stored report keeps the objects it needs.** The CV results and the per-model errors live in
`session.objects` so `/analysis/selection` can re-render the table without re-running the folds, and
`/analysis/diagnose` can read the same fits the report ranked.

**3. Diagnose is per-fit and defaults to all of them, because a verdict on one model with nothing to
compare it against is a number without a context.** The panel shows the cheap rows (baseline, white
noise) beside the winner for exactly this reason.

Stage gating does the rest: S11 requires S10 complete, S12 requires S11. A selection report computed
before anything was fitted is the one thing this endpoint must never produce.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import session_dep
from app.core.schemas import AnalysisResponse, Confidence, RuleOut, Stage
from app.core.session import Session
from app.ts.diagnose import diagnose_model, judge, run_battery
from app.ts.select import (
    CVResult,
    CVScheme,
    FitIdentity,
    build_selection_report,
    cross_validate,
    rank_by_cv,
)
from app.ts.select.ic import ICKind

router = APIRouter()

SELECTION = "selection"
DIAGNOSTICS = "diagnostics"

#: The stages that can be cross-validated. ML rows forecast recursively and are handled the same way;
#: the combination row is excluded because it is *built from* held-out forecasts and would be scored on
#: the same data its weights came from.
CV_EXCLUDED = {"combination"}


class SelectionRequest(BaseModel):
    horizon: int = Field(12, ge=1, le=120, description="Forecast horizon; also the fold length.")
    folds: int = Field(4, ge=2, le=10)
    scheme: str = Field("expanding", description="expanding or sliding")
    metric: str = Field("mase", description="The primary criterion. mase, mae, rmse or smape.")
    models: list[str] = Field(
        default_factory=list,
        description="Which stored fits to cross-validate. Empty = every successful non-combination fit.",
    )
    include_baselines: bool = True


class DiagnoseRequest(BaseModel):
    models: list[str] = Field(
        default_factory=list, description="Empty = every stored fit, including the baselines."
    )
    period: int | None = None


# --------------------------------------------------------------------------- #
# Helpers shared with api/models.py's fit store
# --------------------------------------------------------------------------- #
def _fits(session: Session) -> dict[str, Any] | None:
    stored = session.objects.get("fits")
    return stored if stored else None


def _identities(outcomes: list[Any], *, period: int | None) -> list[FitIdentity]:
    """
    Build the comparability identities from the stored fits.

    Every fact here comes from the fit that was actually run — `n_eff`, the differencing the model
    declares in `fitted_on`, and whether it produced a likelihood at all. Nothing is re-derived, which
    is what makes the guard trustworthy: it compares what happened, not what should have.
    """
    from app.ts.select.ic import information_criteria

    out: list[FitIdentity] = []
    for o in outcomes:
        s = o.summary
        if s.failed:
            continue
        fitted_on = (s.fitted_on or "levels").lower()
        d = 1 if fitted_on.startswith("diff") else 0
        stack: tuple[str, ...] = ()
        if "log" in fitted_on:
            stack = ("log",)
        # The likelihood class decides whether this row may be *sorted* against the others at all.
        if not s.ic_comparable or s.loglik is None:
            model_class = "none"
        elif s.family.value in {"volatility", "count", "intermittent", "serial_count"}:
            model_class = "quasi"
        elif s.model_id in {"arfima", "croston", "tsb", "adida"}:
            model_class = "ima"
        else:
            model_class = "arma"
        ic = information_criteria(s.loglik, n=s.n_obs_used or 0, k=max(s.n_params, 1)) if s.loglik is not None else None
        out.append(
            FitIdentity(
                model_id=s.model_id,
                model_class=model_class,
                n_obs=s.n_obs_used or 0,
                d=d,
                D=0,
                transform_stack=stack,
                loglik=s.loglik,
                k=s.n_params,
                ic=ic,
            )
        )
    return out


def _factory(model_id: str, *, period: int | None, d: int, D: int, data_class: Any,
             exog: pd.DataFrame | None, future_exog: pd.DataFrame | None,
             interventions: list[dict] | None, frequency_code: str | None):
    """A fresh wrapper per (model, fold). The CV engine calls this once per fold and never caches."""
    from app.ts.models.fit import build_multi_frame, build_wrapper, days_per_period, MULTIVARIATE_IDS

    def make(mid: str):
        wrapper = build_wrapper(
            mid, period=period, d_hint=d, D_hint=D,
            exog=exog, future_exog=future_exog, interventions=interventions,
        )
        if mid in MULTIVARIATE_IDS:
            # The multivariate rows need the companion column; without it they refuse with their own
            # measured reason rather than crashing the fold.
            wrapper._multi = None  # type: ignore[attr-defined]
        wrapper.period_days = days_per_period(frequency_code, period)
        return wrapper

    return make


def _period(session: Session) -> int | None:
    from app.api.models import _period as _p

    return _p(session)


def _d_hint(session: Session) -> tuple[int, int, str]:
    from app.api.models import _d_hint as _d

    return _d(session)


# --------------------------------------------------------------------------- #
# S11
# --------------------------------------------------------------------------- #
@router.post("/analysis/select", response_model=AnalysisResponse)
def run_selection(body: SelectionRequest, session: Session = Depends(session_dep)) -> AnalysisResponse:
    """
    Cross-validate the stored fits and rank them.

    Refuses when nothing has been fitted: the answer to "which model is best" with no models is not a
    ranking, and the stage gate would let it through otherwise.
    """
    t0 = time.perf_counter()
    session.require_stage(Stage.S11_SELECT)
    stored = _fits(session)
    if not stored:
        from app.core.errors import StageOrderError

        raise StageOrderError(
            plain="No models have been fitted, so there is nothing to select between.",
            fix="Run Step 10 first — selection compares fits, it does not create them.",
        )

    outcomes: list[Any] = stored["outcomes"]
    ok = [o for o in outcomes if not o.summary.failed]
    by_id = {o.summary.model_id: o for o in ok}
    wanted = list(body.models) or [
        o.summary.model_id
        for o in ok
        if body.include_baselines or not o.summary.is_baseline
    ]
    wanted = [m for m in dict.fromkeys(wanted) if m in by_id and m not in CV_EXCLUDED]
    if len(wanted) < 2:
        from app.core.errors import StageOrderError

        raise StageOrderError(
            plain=(
                "Cross-validation needs at least two fitted models to compare and "
                f"{len(wanted)} {'is' if len(wanted) == 1 else 'are'} available."
            ),
            fix="Fit a second model in Step 10 — a single model's CV score is a number with nothing "
            "to be better or worse than.",
            context={"available": sorted(by_id), "requested": wanted},
        )

    scheme = CVScheme(body.scheme) if body.scheme in {s.value for s in CVScheme} else CVScheme.EXPANDING
    series = _series(session)
    d, D, _ = _d_hint(session)
    period = _period(session)
    data_class = stored.get("data_class")
    from app.core.schemas import DataClass

    exog, future_exog = _exog(session)

    from app.ts.models.fit import DAYS_PER_PERIOD

    freq = _frequency_code(session)
    try:
        cv_results = cross_validate(
            wanted,
            series,
            factory=_factory(
                "", period=period, d=d, D=D, data_class=data_class, exog=exog,
                future_exog=future_exog, interventions=_interventions(session), frequency_code=freq,
            ),
            horizon=body.horizon,
            period=period,
            folds=body.folds,
            scheme=scheme,
        )
    except ValueError as exc:
        from app.core.errors import StageOrderError

        raise StageOrderError(
            plain=str(exc),
            fix="Lower the horizon or the fold count, or use a longer series.",
            context={"horizon": body.horizon, "folds": body.folds, "n": int(len(series))},
        ) from exc

    # The paired error series come straight from the CV results, in fold order. The first version of
    # this block reconstructed them from per-fold `bias` and then passed `errors_by_model=None` anyway
    # — a test that needs errors from the same origins cannot be built from fold means, and silently
    # omitting it made the panel claim the tests were unavailable.
    errors_by_model: dict[str, np.ndarray] = {}
    actual_pooled: list[float] = []
    for r in cv_results:
        if r.failed or not r.errors:
            continue
        errors_by_model[r.model_id] = np.asarray(r.errors, dtype=float)
        if not actual_pooled:
            actual_pooled = list(r.actuals)
    if actual_pooled:
        errors_by_model["__actual__"] = np.asarray(actual_pooled, dtype=float)

    identities = _identities(ok, period=period)
    ic_kind = ICKind(body.metric) if body.metric in {k.value for k in ICKind} else ICKind.AICC
    report = build_selection_report(
        cv_results,
        identities=identities,
        errors_by_model=errors_by_model or None,
        horizon=body.horizon,
        metric="mase" if ic_kind is not ICKind.AICC else body.metric,
        ic_kind=ICKind.AICC,
    )

    session.objects[SELECTION] = {
        "version": session.state.series_version,
        "at": time.time(),
        "report": report,
        "cv": {r.model_id: r for r in cv_results},
        "request": body.model_dump(),
        "wanted": wanted,
        "period": period,
    }

    result = _selection_table(report)
    session.add_result(result)
    winner = report.winner_name or report.winner or "none"
    session.mark_stage(
        Stage.S11_SELECT, "complete",
        f"Ranked {len(report.shortlist)} model(s) on rolling-origin {report.primary_criterion}; "
        f"{winner} first",
    )

    return AnalysisResponse(
        stage=Stage.S11_SELECT,
        advisor=_selection_advisor(report),
        results=[result],
        pipeline=session.state.progress(),
        caveats=[report.caveat] + [f.detail for f in []],
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/analysis/selection")
def get_selection(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """The stored selection report, re-rendered. Never re-runs the folds."""
    stored = session.objects.get(SELECTION)
    if not stored:
        return {
            "stage": Stage.S11_SELECT.value,
            "status": "not_started",
            "message": "Selection has not been run yet.",
            "report": None,
        }
    report = stored["report"]
    cv: dict[str, CVResult] = stored["cv"]
    return {
        "stage": Stage.S11_SELECT.value,
        "status": "complete",
        "stale": stored["version"] != session.state.series_version,
        "computed_for_version": stored["version"],
        "current_version": session.state.series_version,
        "report": report.as_dict(),
        "cv": [cv[m].as_dict() for m in cv],
        "request": stored["request"],
    }


def _selection_table(report: Any):
    """The shortlist as a TableResult, so the generic result renderer has something to draw."""
    from app.core.schemas import TableColumn, TableResult

    rows = []
    for e in report.shortlist:
        rows.append(
            {
                "rank": e.rank,
                "model": e.display_name,
                "cv_mase": None if e.cv_mase is None else round(e.cv_mase, 3),
                "cv_mae": None if e.cv_mae is None else round(e.cv_mae, 2),
                "spread": None if e.cv_spread is None else round(e.cv_spread, 3),
                "worst_fold": None if e.worst_fold_mase is None else round(e.worst_fold_mase, 3),
                "aicc": None if e.aicc is None else round(e.aicc, 2),
                "delta_aicc": None if e.delta_aicc is None else round(e.delta_aicc, 2),
                "weight": None if e.akaike_weight is None else round(e.akaike_weight, 3),
                "support": e.support,
            }
        )
    return TableResult(
        id="select.shortlist",
        stage=Stage.S11_SELECT,
        title="Ranked shortlist — rolling-origin accuracy first",
        plain=(
            "The models ordered by how well they forecast data they never saw, worst-ranked last."
        ),
        why=(
            "This is the table that answers 'which model should I use'. It is ordered by held-out "
            "accuracy, not by in-sample fit, because an information criterion rewards a model for "
            "explaining the past and says nothing about the future."
        ),
        method=(
            "Rolling-origin cross-validation: each model is refitted on every fold's training window "
            "and scored on the following h observations, h being the forecast horizon. The MASE "
            "denominator is computed on the training window only, so the test data never enters the "
            "scale. AICc is reported for reference and is blank where the comparability guard refused "
            "the comparison."
        ),
        reproducible_snippet=(
            "from app.ts.select import cross_validate, rank_by_cv, build_selection_report\n"
            "cv = cross_validate(ids, y, factory=make, horizon=12, period=12, folds=4)\n"
            "report = build_selection_report(cv, identities=identities, horizon=12)\n"
            "print(report.why)"
        ),
        columns=[
            TableColumn(key="rank", label="#"),
            TableColumn(key="model", label="Model"),
            TableColumn(key="cv_mase", label="CV MASE", numeric=True),
            TableColumn(key="cv_mae", label="CV MAE", numeric=True),
            TableColumn(key="spread", label="fold spread", numeric=True),
            TableColumn(key="worst_fold", label="worst fold", numeric=True),
            TableColumn(key="aicc", label="AICc", numeric=True),
            TableColumn(key="delta_aicc", label="ΔAICc", numeric=True),
            TableColumn(key="weight", label="Akaike w", numeric=True),
            TableColumn(key="support", label="support"),
        ],
        rows=rows,
        footnote=report.why,
        caption=(
            "Ranked on held-out forecast accuracy at the requested horizon. The AICc columns are "
            "secondary and are blank where the comparability guard refused the comparison."
        ),
    )


def _selection_advisor(report: Any):
    from app.core.schemas import AdvisorReport

    recs = []
    if report.shortlist:
        top = report.shortlist[0]
        recs.append(
            RuleOut(
                id="select.winner",
                stage=Stage.S11_SELECT,
                recommend=f"Use {top.display_name}"
                + (" (or its runner-up — they are not separable)" if report.indistinguishable else ""),
                confidence=Confidence.MEDIUM if report.indistinguishable else Confidence.HIGH,
                because=_trim(report.why, 300),
                provenance=["rolling_origin_cv", "info_criteria", "pairwise_tests"],
                alternative=(
                    f"{report.shortlist[1].display_name} scores "
                    f"{report.shortlist[1].cv_mase:.3f} against {top.cv_mase:.3f} — the choice is not "
                    "established by this data."
                    if report.indistinguishable and len(report.shortlist) > 1
                    else ""
                ),
                risk_if_ignored=(
                    "Choosing on in-sample fit instead throws away the only evidence about "
                    "forecasting that this app has."
                ),
                how_to="Step 11 → read the shortlist → Step 12 for that model's diagnostics",
            )
        )
    return AdvisorReport(
        stage=Stage.S11_SELECT,
        headline=(
            f"{report.winner_name or 'No model'} ranks first on {report.primary_criterion}"
            + (", but not separably so" if report.indistinguishable else "")
            + "."
        ),
        recommendations=recs,
        facts_used={
            "n_folds": report.n_folds,
            "horizon": report.horizon,
            "scheme": report.scheme,
            "ic_comparable": report.ic_comparable,
        },
    )


def _trim(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# S12
# --------------------------------------------------------------------------- #
@router.post("/analysis/diagnose", response_model=AnalysisResponse)
def run_diagnose(body: DiagnoseRequest, session: Session = Depends(session_dep)) -> AnalysisResponse:
    """
    Run the battery and the verdict for the requested fits.

    Every model gets a verdict, including the baselines — a white-noise verdict of ADEQUATE beside a
    model's INADEQUATE is the comparison that makes the verdict mean something.
    """
    t0 = time.perf_counter()
    session.require_stage(Stage.S12_DIAGNOSE)
    stored = _fits(session)
    if not stored:
        from app.core.errors import StageOrderError

        raise StageOrderError(
            plain="No models have been fitted, so there are no residuals to diagnose.",
            fix="Run Step 10 first.",
        )
    outcomes: list[Any] = stored["outcomes"]
    ok = [o for o in outcomes if not o.summary.failed]
    by_id = {o.summary.model_id: o for o in ok}
    wanted = list(body.models) or [o.summary.model_id for o in ok]
    missing = [m for m in wanted if m not in by_id]
    if missing:
        from app.core.errors import StageOrderError

        raise StageOrderError(
            plain=f"{', '.join(missing)} is not a successful fit in this session.",
            fix="Diagnose a model that fitted; the failed ones already carry their reason.",
            context={"requested": wanted, "available": sorted(by_id)},
        )

    period = body.period if body.period is not None else stored.get("period")
    blocks: dict[str, Any] = {}
    for mid in dict.fromkeys(wanted):
        o = by_id[mid]
        from app.core.schemas import DataClass

        is_count = o.summary.family.value in {"count", "intermittent", "serial_count", "point_process"}
        is_vol = o.summary.family.value == "volatility"
        blocks[mid] = diagnose_model(o, period=period, is_count_model=is_count, is_volatility_model=is_vol)

    session.objects[DIAGNOSTICS] = {
        "version": session.state.series_version,
        "at": time.time(),
        "blocks": blocks,
        "period": period,
    }

    result = _diagnostics_table(blocks)
    session.add_result(result)
    n_ok = sum(1 for b in blocks.values() if b.get("verdict", {}) and b["verdict"].get("adequacy") == "adequate")
    n_inad = sum(
        1 for b in blocks.values() if b.get("verdict", {}) and b["verdict"].get("adequacy") == "inadequate"
    )
    session.mark_stage(
        Stage.S12_DIAGNOSE, "complete",
        f"Diagnosed {len(blocks)} model(s): {n_ok} adequate, {n_inad} inadequate",
    )

    return AnalysisResponse(
        stage=Stage.S12_DIAGNOSE,
        advisor=_diagnostics_advisor(blocks),
        results=[result],
        pipeline=session.state.progress(),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/analysis/diagnostics")
def get_diagnostics(session: Session = Depends(session_dep)) -> dict[str, Any]:
    stored = session.objects.get(DIAGNOSTICS)
    if not stored:
        return {
            "stage": Stage.S12_DIAGNOSE.value,
            "status": "not_started",
            "message": "Diagnostics have not been run yet.",
            "blocks": {},
        }
    return {
        "stage": Stage.S12_DIAGNOSE.value,
        "status": "complete",
        "stale": stored["version"] != session.state.series_version,
        "computed_for_version": stored["version"],
        "current_version": session.state.series_version,
        "period": stored["period"],
        "blocks": stored["blocks"],
    }


def _diagnostics_table(blocks: dict[str, Any]):
    from app.core.schemas import TableColumn, TableResult

    rows = []
    for mid, b in blocks.items():
        if b.get("status") != "ok":
            rows.append(
                {
                    "model": b.get("display_name") or mid,
                    "adequacy": "not applicable",
                    "lb": None, "jb": None, "arch": None,
                    "reason": _trim(b.get("message", ""), 90),
                }
            )
            continue
        bat = b["battery"]
        verdict = b["verdict"]
        lb = (bat.get("ljung_box") or {}).get("results") or {}
        # The smallest p across the lag grid is the one to read: a model that fails at the seasonal lag
        # and passes at 10 would otherwise look clean.
        lb_p = min((v.get("p") for v in lb.values() if v.get("p") is not None), default=None)
        rows.append(
            {
                "model": b.get("display_name") or mid,
                "adequacy": verdict.get("adequacy", "").replace("_", " "),
                "lb": None if lb_p is None else round(lb_p, 4),
                "jb": (bat.get("normality") or {}).get("jarque_bera", {}).get("p"),
                "arch": (bat.get("arch_lm") or {}).get("p"),
                "reason": _trim(
                    (verdict.get("reasons") or [{}])[0].get("detail", "") if verdict.get("reasons") else verdict.get("headline", ""),
                    90,
                ),
            }
        )
    return TableResult(
        id="diagnose.verdicts",
        stage=Stage.S12_DIAGNOSE,
        title="Adequacy verdicts",
        plain=(
            "Whether each model's leftovers — the parts it could not explain — look like noise, or "
            "like a pattern it missed."
        ),
        why=(
            "A model can fit the sample well and still be wrong: if its residuals still contain a "
            "pattern, the pattern is in the data and not in the model. The intervals are then too "
            "narrow even when the point forecast looks fine."
        ),
        method=(
            "Ljung–Box at 5, 10, s and 2s lags; Box–Pierce; Jarque–Bera, Shapiro–Wilk and "
            "Anderson–Darling; ARCH-LM; a runs test and a sign test; CUSUM of the standardised "
            "residuals; and a MAD-scaled outlier scan. The verdict names the failing test and its "
            "remedy, and treats a lone rejection among many as a multiple-testing artefact rather "
            "than a revision."
        ),
        reproducible_snippet=(
            "from app.ts.diagnose import run_battery, judge\n"
            "battery = run_battery(model.residuals(), period=12)\n"
            "verdict = judge(battery, period=12)\n"
            "print(verdict.adequacy, verdict.headline)"
        ),
        columns=[
            TableColumn(key="model", label="Model"),
            TableColumn(key="adequacy", label="Verdict"),
            TableColumn(key="lb", label="Ljung–Box p", dtype="number", format="p", align="right"),
            TableColumn(key="jb", label="Jarque–Bera p", dtype="number", format="p", align="right"),
            TableColumn(key="arch", label="ARCH-LM p", dtype="number", format="p", align="right"),
            TableColumn(key="reason", label="what the verdict rests on"),
        ],
        rows=rows,
        footnote=(
            "Ljung–Box p is the smallest across the lag grid (5, 10, s, 2s), so a seasonal failure "
            "cannot hide behind a clean lag-10 result. Non-normal residuals and ARCH are recorded as "
            "caveats, not failures — neither invalidates the point forecast."
        ),
    )


def _diagnostics_advisor(blocks: dict[str, Any]):
    from app.core.schemas import AdvisorReport

    bad = [
        (mid, b)
        for mid, b in blocks.items()
        if (b.get("verdict") or {}).get("adequacy") == "inadequate"
    ]
    recs = []
    for mid, b in bad[:3]:
        verdict = b["verdict"]
        findings = verdict.get("reasons") or []
        recs.append(
            RuleOut(
                id=f"diagnose.{mid}",
                stage=Stage.S12_DIAGNOSE,
                recommend=f"Revise {b.get('display_name') or mid}",
                confidence=Confidence.HIGH,
                because=_trim(verdict.get("headline", ""), 260),
                provenance=["residual_battery", "adequacy_verdict"],
                alternative=findings[0].get("remedy", "") if findings else "",
                risk_if_ignored=(
                    "An inadequate fit's intervals are not trustworthy even when its point forecast "
                    "looks fine."
                ),
                how_to="Step 12 → read the named failing test → apply its remedy → refit in Step 10",
            )
        )
    return AdvisorReport(
        stage=Stage.S12_DIAGNOSE,
        headline=(
            f"{len(bad)} model(s) of {len(blocks)} failed the adequacy battery in a way that "
            "corroborates across independent checks."
            if bad
            else f"All {len(blocks)} diagnosed model(s) are adequate or adequate with caveats."
        ),
        recommendations=recs,
        facts_used={"n_diagnosed": len(blocks), "n_inadequate": len(bad)},
    )


# --------------------------------------------------------------------------- #
# The series helpers, imported late to avoid a circular import with api/models.py
# --------------------------------------------------------------------------- #
def _series(session: Session):
    from app.api.models import _series as _s

    return _s(session)


def _exog(session: Session):
    from app.api.models import _exog as _e

    return _e(session)


def _interventions(session: Session):
    from app.api.models import _interventions as _i

    return _i(session)


def _frequency_code(session: Session):
    from app.api.models import _frequency_code as _f

    return _f(session)


__all__ = ["router"]
