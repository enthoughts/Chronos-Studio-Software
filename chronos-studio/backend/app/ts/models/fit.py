"""
ts/models/fit.py — fit a set of candidates, with a fallback that the user can see (§5.8).

The requirement in the spec is blunt: "a fit-failure path for every single entry: catch, log the
reason, fall back to the next candidate, and never show an empty panel". This module is that path.

Two design points:

* **A failure is a row in the table, not a stack trace.** Every candidate produces a `FittedSummary`
  whether it fitted or not, so the comparison table can show "SARIMA — did not converge, because …"
  rather than silently shrinking.
* **There is always something to show.** If every non-baseline model fails, the baselines are still
  fitted and the panel says so explicitly, with the reason each real model was lost. An empty panel
  reads as a broken app; a panel of baselines plus four named failures reads as an honest one.

Fitted objects stay in `session.objects["fits"]` — they are large and not JSON, and shipping them on
the wire would be both slow and a reproducibility hazard (the report must cite the fit that produced
it, not a re-fit of the same code).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from app.core.schemas import DataClass, FitSummary, ModelFamily
from app.ts.models.arma import IMPLEMENTED_ARMA
from app.ts.models.base import FittedSummary, ModelWrapper
from app.ts.models.combination import METHODS as COMBINATION_METHODS
from app.ts.models.combination import CombinationModel, ComponentForecast
from app.ts.select import fold_origins
from app.ts.models.baselines import BASELINE_CLASSES
from app.ts.models.counts import IMPLEMENTED_COUNT
from app.ts.models.exog import IMPLEMENTED_EXOG
from app.ts.models.ingarch import IMPLEMENTED_SERIAL_COUNT
from app.ts.models.pointprocess import IMPLEMENTED_POINT
from app.ts.models.volatility import IMPLEMENTED_VOLATILITY
from app.ts.models.intermittent import IMPLEMENTED_INTERMITTENT
from app.ts.models.ml import IMPLEMENTED_ML
from app.ts.models.multivariate import IMPLEMENTED_MULTIVARIATE
from app.ts.models.regimes import IMPLEMENTED_NONLINEAR
from app.ts.models.ucm import IMPLEMENTED_STRUCTURAL
from app.ts.models.registry import REGISTRY, get_spec, is_implemented
from app.ts.models.smoothing import IMPLEMENTED_SMOOTHING

#: The one row that is fitted *from the other fits* rather than from the series. It is handled at the
#: end of `fit_many`, after everything else exists, because it has nothing to combine before then.
COMBINATION_ID = "combination"

#: Every model this build can actually fit, by id.
IMPLEMENTED: dict[str, type[ModelWrapper]] = {
    **BASELINE_CLASSES,
    **IMPLEMENTED_SMOOTHING,
    **IMPLEMENTED_ARMA,
    **IMPLEMENTED_COUNT,
    **IMPLEMENTED_INTERMITTENT,
    **IMPLEMENTED_STRUCTURAL,
    **IMPLEMENTED_EXOG,
    **IMPLEMENTED_NONLINEAR,
    **IMPLEMENTED_SERIAL_COUNT,
    **IMPLEMENTED_POINT,
    **IMPLEMENTED_VOLATILITY,
    **IMPLEMENTED_MULTIVARIATE,
    **IMPLEMENTED_ML,
    # Registered here so it counts as implemented and can be ticked, but its `_fit` refuses unless it
    # has been given the component forecasts — which only `fit_many` can do, at the end.
    COMBINATION_ID: CombinationModel,
}

#: Models whose predictive distribution is a PMF, not a density. Read by the evaluation battery in
#: Part 15 so the metric policy can switch to CRPS/PIT without asking each wrapper.
DISCRETE_PREDICTIVE: frozenset[str] = frozenset(
    {"poisson", "neg_binomial", "zero_inflated", "ingarch", "inar", "renewal", "hawkes"}
)

#: Models that take the exogenous frame and the intervention list, i.e. the ones whose *inputs* are
#: not just the series. Kept as a set rather than a family check because it is a property of the
#: wrapper's constructor, and the next exogenous model should have to say so explicitly.
EXOGENOUS_IDS = frozenset(IMPLEMENTED_EXOG)

#: The multivariate family takes the *target and its companions* as one frame, so the fit step has to
#: hand it more than `y`. Kept as a set for the same reason as `EXOGENOUS_IDS`: it is a property of the
#: wrapper's constructor, and the next multivariate row should have to declare itself.
MULTIVARIATE_IDS = frozenset(IMPLEMENTED_MULTIVARIATE)

#: The ML family takes the driver frame as lagged *features* rather than as a regression design.
ML_IDS = frozenset(IMPLEMENTED_ML)

#: Families whose evaluation must ban MAPE (§5.8's count/volatility rule).
MAPE_BANNED = {DataClass.COUNT, DataClass.INTERMITTENT}


@dataclass
class FitOutcome:
    model_id: str
    wrapper: ModelWrapper
    summary: FittedSummary
    seconds: float
    diagnostics: dict[str, object] = field(default_factory=dict)


#: Days in one recording period, by calendar code. Weekly is 7; monthly is the average Gregorian month.
DAYS_PER_PERIOD: dict[str, float] = {
    "D": 1.0, "W": 7.0, "MS": 365.25 / 12, "QS": 365.25 / 4, "YS": 365.25, "H": 1.0 / 24,
}


def days_per_period(frequency_code: str | None, period: int | None = None) -> float | None:
    """
    Length of one recording period in days.

    Taken from the calendar code, never from the seasonal period: a weekly series has `period=52` for
    seasonality and still lasts 7 days per row, and dividing the year by 52 only matches by coincidence
    on a 365-day year. Falls back to the seasonal period only when no code is available.
    """
    if frequency_code and frequency_code in DAYS_PER_PERIOD:
        return DAYS_PER_PERIOD[frequency_code]
    if period and period >= 2 and period <= 366:
        return 365.25 / period
    return None


def build_wrapper(
    model_id: str,
    *,
    period: int | None = None,
    d_hint: int = 0,
    D_hint: int = 0,
    exog: pd.DataFrame | None = None,
    future_exog: pd.DataFrame | None = None,
    interventions: list[dict] | None = None,
) -> ModelWrapper:
    """Construct a wrapper for an id, with the shared context every fit needs."""
    cls = IMPLEMENTED.get(model_id)
    if cls is None:
        raise KeyError(f"'{model_id}' cannot be fitted in this build.")
    # `period` is on the base class; the ARMA family additionally takes the differencing hints.
    kwargs: dict[str, object] = {"period": period}
    if model_id in {"arima", "sarima", "arfima"}:
        kwargs["d_hint"] = d_hint
    if model_id == "sarima":
        kwargs["D_hint"] = D_hint
    if model_id in EXOGENOUS_IDS:
        kwargs.update(exog=exog, future_exog=future_exog, interventions=interventions)
    if model_id in ML_IDS:
        # Lag-1 drivers only; the ML wrapper builds the design itself so it can guarantee no leakage.
        kwargs.update(exog=exog)
    return cls(**kwargs)  # type: ignore[arg-type]


def build_multi_frame(y: pd.Series, exog: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Assemble the multivariate frame: the target first, then every mapped numeric column.

    The target goes first because the VAR's forecast, its reported equation and its IRF are all read
    against column 0. `None` when there is nothing to add — the model then refuses with its own
    measured reason ("needs at least two series and it was given 1") instead of failing obscurely.
    """
    if exog is None or getattr(exog, "shape", (0, 0))[1] == 0:
        return None
    frame = pd.DataFrame(exog).copy()
    frame.insert(0, "__target__", pd.Series(y).reindex(frame.index).astype(float))
    frame = frame.astype(float)
    frame.columns = ["the target"] + [str(c) for c in frame.columns[1:]]
    return frame


def fit_one(
    model_id: str,
    y: pd.Series,
    *,
    period: int | None,
    d_hint: int = 0,
    D_hint: int = 0,
    data_class: DataClass = DataClass.CONTINUOUS,
    exog: pd.DataFrame | None = None,
    future_exog: pd.DataFrame | None = None,
    interventions: list[dict] | None = None,
    event_times: pd.DatetimeIndex | None = None,
    frequency_code: str | None = None,
) -> FitOutcome:
    """Fit one model, always returning an outcome — the failure lives inside the summary."""
    t0 = time.perf_counter()
    try:
        wrapper = build_wrapper(
            model_id, period=period, d_hint=d_hint, D_hint=D_hint,
            exog=exog, future_exog=future_exog, interventions=interventions,
        )
        if model_id in MULTIVARIATE_IDS:
            # `multi` is set after construction because `build_wrapper` keeps one signature for all
            # twenty-odd constructors; the frame is a fit-time input, not a configuration choice.
            wrapper._multi = build_multi_frame(y, exog)  # type: ignore[attr-defined]
        if wrapper is not None:
            wrapper.period_days = days_per_period(frequency_code, period)
    except KeyError as exc:
        # An unimplemented or unknown id is not a modelling failure; it should never reach here, but
        # if it does the caller gets a visible row rather than a 500 at the API boundary.
        from app.ts.models.base import FittedSummary as _FS

        family = ModelFamily.BASELINE
        if model_id in REGISTRY:
            family = REGISTRY[model_id].family
        summary = _FS(
            model_id=model_id,
            display_name=REGISTRY[model_id].display_name if model_id in REGISTRY else model_id,
            family=family,
            fit_id=f"fit-{model_id}",
            failed=True,
            failure_reason=str(exc).strip("'\""),
            converged=False,
        )
        return FitOutcome(model_id, None, summary, time.perf_counter() - t0)  # type: ignore[arg-type]

    # A point-process model is fitted to the event *timestamps*, not to the counts: the intervals are
    # the data, and handing it the count series would silently model the wrong thing (and fail on a
    # DatetimeIndex check that would look like a bug rather than a mismatch).
    y_fit = y
    if model_id in {"renewal", "hawkes"} and event_times is not None and len(event_times) >= 3:
        y_fit = pd.Series(np.ones(len(event_times), dtype=float), index=pd.DatetimeIndex(event_times))
    wrapper.fit(y_fit)
    summary = wrapper.summary  # type: ignore[assignment]
    if not wrapper.failed:
        wrapper.compute_diagnostics(y, ban_mape=data_class in MAPE_BANNED)
    return FitOutcome(
        model_id=model_id,
        wrapper=wrapper,
        summary=summary,
        seconds=time.perf_counter() - t0,
        diagnostics=wrapper.diagnostics.as_dict() if not wrapper.failed else {},
    )


def fit_many(
    model_ids: list[str],
    y: pd.Series,
    *,
    period: int | None,
    d_hint: int = 0,
    D_hint: int = 0,
    data_class: DataClass = DataClass.CONTINUOUS,
    ensure_baselines: bool = True,
    event_times: pd.DatetimeIndex | None = None,
    frequency_code: str | None = None,
    exog: pd.DataFrame | None = None,
    future_exog: pd.DataFrame | None = None,
    interventions: list[dict] | None = None,
    combination_method: str = "mean",
    combination_horizon: int | None = None,
) -> list[FitOutcome]:
    """
    Fit every requested model, and — always — the baselines.

    The baselines are not optional even when the caller forgot them: a comparison table without the
    naive benchmark cannot answer "was this worth it?", which is the only question the table exists
    to answer.
    """
    ids = list(dict.fromkeys(model_ids))
    if ensure_baselines:
        from app.ts.models.registry import BASELINE_IDS

        for b in BASELINE_IDS:
            if b not in ids:
                ids.insert(0, b)
        # Keep the ladder in registry order.
        ids = [b for b in BASELINE_IDS if b in ids] + [i for i in ids if i not in BASELINE_IDS]
    wants_combination = COMBINATION_ID in ids
    plain_ids = [i for i in ids if i != COMBINATION_ID]
    outcomes = [
        fit_one(
            mid, y, period=period, d_hint=d_hint, D_hint=D_hint, data_class=data_class,
            exog=exog, future_exog=future_exog, interventions=interventions,
            event_times=event_times, frequency_code=frequency_code,
        )
        for mid in plain_ids
    ]
    if wants_combination:
        outcomes.append(
            _fit_combination(
                outcomes, y, period=period, method=combination_method,
                horizon=combination_horizon, d_hint=d_hint, D_hint=D_hint, data_class=data_class,
                exog=exog, future_exog=future_exog, interventions=interventions,
                frequency_code=frequency_code,
            )
        )
    return outcomes


def _fit_combination(
    outcomes: list[FitOutcome],
    y: pd.Series,
    *,
    period: int | None,
    method: str = "mean",
    horizon: int | None = None,
    d_hint: int = 0,
    D_hint: int = 0,
    data_class: DataClass = DataClass.CONTINUOUS,
    exog: pd.DataFrame | None = None,
    future_exog: pd.DataFrame | None = None,
    interventions: list[dict] | None = None,
    frequency_code: str | None = None,
    folds: int = 3,
) -> FitOutcome:
    """
    Build the combination from the other rows' **held-out** forecasts.

    The rule this function exists to enforce: the weights must be estimated where the components were
    *forecasting*, not where they were fitted. So the components are re-fitted on rolling origins, the
    errors at those origins form the regression, and only then is the combination fitted. A
    combination whose weights came from in-sample fits beats its own components for a reason that
    disappears on new data — the single most common way this row is implemented wrongly.

    A component that cannot be cross-validated (too few observations, a model without a residual path)
    is dropped, and the card names it. Dropping silently would make "the combination of five models"
    a claim about a set the reader cannot see.
    """
    t0 = time.perf_counter()
    excluded: list[dict[str, str]] = []
    usable = [
        o
        for o in outcomes
        if not o.summary.failed and o.wrapper is not None and not getattr(o.summary, "is_baseline", False)
    ]
    # Components are the *real* models, not the benchmark ladder. Averaging mean + naive + seasonal
    # naive + drift + white noise with one fitted model produces a forecast that is mostly a random
    # walk wearing a model's name — the first run of this scored MASE 2.10 against a component that
    # scored 0.25. One benchmark is allowed in when there is only a single real model, because
    # "does adding the naive forecast to the average help?" is a legitimate question; five are not.
    if len(usable) >= 2:
        components = usable
    else:
        naive = [o for o in outcomes if not o.summary.failed and o.summary.model_id == "naive"]
        components = usable + naive
        if components:
            excluded.append(
                {
                    "model_id": "benchmarks",
                    "why": (
                        "the other benchmarks (mean, white noise, seasonal naive, drift) were not "
                        "used as components: averaging four near-identical benchmarks with a single "
                        "real model produces a random walk, not a combination"
                    ),
                }
            )
    if len(components) < 2:
        return _failed_combination(
            t0,
            "A combination needs at least two successfully fitted models and there "
            f"{'is' if len(components) == 1 else 'are'} {len(components)}.",
            "Fit at least two models in Step 10, then tick the combination again.",
            reason_kind="length",
        )

    horizon = int(horizon or 12)
    values = pd.Series(y).astype(float)
    arr = values.to_numpy()
    try:
        # Many overlapping origins, not a handful of CV folds: the weights are a regression, and a
        # 3-point regression on 3 CV folds is fitting noise. One origin per period forward, capped so
        # the refit cost stays bounded.
        origins = fold_origins(
            len(arr), horizon=horizon, folds=max(6, min(30, len(arr) // 5)), step=1
        )
    except ValueError as exc:
        return _failed_combination(
            t0,
            f"The combination could not be given held-out forecasts: {exc}",
            "Use a shorter horizon for the combination, or fit it on a longer series.",
            reason_kind="length",
        )

    # (origins × components) matrix of held-out forecasts, plus the realisation at each origin.
    columns: dict[str, list[float]] = {}
    realised: list[float] = []
    for fold in origins:
        realised.append(float(arr[fold.test_start]))  # first step of the test block
    for outcome in components:
        mid = outcome.summary.model_id
        path: list[float] = []
        for fold in origins:
            try:
                wrapper = build_wrapper(
                    mid, period=period, d_hint=d_hint, D_hint=D_hint,
                    exog=exog, future_exog=future_exog, interventions=interventions,
                )
                if mid in MULTIVARIATE_IDS:
                    wrapper._multi = build_multi_frame(values.iloc[: fold.train_end + 1], exog)  # type: ignore[attr-defined]
                wrapper.period_days = days_per_period(frequency_code, period)
                wrapper.fit(values.iloc[fold.train_start : fold.train_end + 1])
                if wrapper.failed:
                    raise RuntimeError(wrapper.summary.failure_reason or "the fit failed")
                fc = wrapper.predict(horizon=fold.n_test)
                path.append(float(pd.Series(fc["mean"]).iloc[0]))
            except Exception as exc:  # noqa: BLE001
                excluded.append({"model_id": mid, "why": f"{type(exc).__name__}: {exc}"[:160]})
                path = []
                break
        if path:
            columns[mid] = path

    if len(columns) < 2:
        return _failed_combination(
            t0,
            "Fewer than two models could produce a held-out forecast, so there is nothing to combine. "
            + "; ".join(f"{e['model_id']}: {e['why']}" for e in excluded[:3]),
            "Choose models that can be re-fitted on a shorter window — or a shorter combination horizon.",
            reason_kind="length",
        )

    names = list(columns)
    matrix = np.column_stack([columns[n] for n in names])
    aicc = []
    by_id = {o.summary.model_id: o for o in outcomes}
    for n in names:
        s0 = by_id[n].summary
        aicc.append(s0.aicc if s0.ic_comparable else None)

    wrapper = CombinationModel(
        method=method, components=names, period=period,  # type: ignore[arg-type]
    )
    wrapper.set_inputs(
        matrix,
        np.asarray(realised, dtype=float),
        aicc=aicc,
        origins=origins,
        full_series=arr,
        component_forecasts=_component_paths(names, by_id, by_id, horizon=horizon, period=period,
                                             d_hint=d_hint, D_hint=D_hint, values=values,
                                             exog=exog, future_exog=future_exog,
                                             interventions=interventions,
                                             frequency_code=frequency_code),
    )
    wrapper.fit(values)
    summary = wrapper.summary  # type: ignore[assignment]
    if not wrapper.failed:
        wrapper.compute_diagnostics(values, ban_mape=data_class in MAPE_BANNED)
        if excluded:
            summary.caveats.append(
                f"{len(excluded)} model(s) could not be cross-validated and are not in the "
                "combination: " + "; ".join(f"{e['model_id']} ({e['why']})" for e in excluded[:4]) + "."
            )
    return FitOutcome(
        model_id=COMBINATION_ID,
        wrapper=wrapper,
        summary=summary,
        seconds=time.perf_counter() - t0,
        diagnostics=wrapper.diagnostics.as_dict() if not wrapper.failed else {},
    )


def _component_paths(
    names: Sequence[str],
    by_id: dict[str, FitOutcome],
    _unused: dict[str, FitOutcome],
    *,
    horizon: int,
    period: int | None,
    d_hint: int,
    D_hint: int,
    values: pd.Series,
    exog: pd.DataFrame | None,
    future_exog: pd.DataFrame | None,
    interventions: list[dict] | None,
    frequency_code: str | None,
) -> list[ComponentForecast]:
    """
    Each component's *future* path, refitted on the whole series.

    These are the paths the combination's forecast and envelope are built from. They are separate from
    the held-out matrix on purpose: the matrix estimates the weights, the paths produce the answer, and
    confusing the two is how a combination ends up forecasting from a model fitted to a third of the
    data.
    """
    out: list[ComponentForecast] = []
    for name in names:
        outcome = by_id.get(name)
        if outcome is None or outcome.wrapper is None:
            continue
        try:
            fc = outcome.wrapper.predict(horizon=horizon)
            out.append(
                ComponentForecast(
                    model_id=name,
                    display_name=outcome.summary.display_name,
                    mean=[float(v) for v in pd.Series(fc["mean"]).to_numpy()],
                    lower=[float(v) for v in pd.Series(fc["lower"]).to_numpy()],
                    upper=[float(v) for v in pd.Series(fc["upper"]).to_numpy()],
                )
            )
        except Exception:  # noqa: BLE001 - a component without a forecast is simply not in the envelope
            continue
    return out


def _failed_combination(t0: float, why: str, fix: str, *, reason_kind: str = "modelling") -> FitOutcome:
    """A combination that could not be built is still a row, with the reason on it."""
    summary = FittedSummary(
        model_id=COMBINATION_ID,
        display_name=REGISTRY[COMBINATION_ID].display_name if COMBINATION_ID in REGISTRY else "Forecast combination",
        family=ModelFamily.COMBINATION,
        fit_id=f"fit-{COMBINATION_ID}",
        failed=True,
        converged=False,
        failure_reason=why,
        is_baseline=False,
    )
    return FitOutcome(COMBINATION_ID, None, summary, time.perf_counter() - t0)  # type: ignore[arg-type]


def to_schema(outcome: FitOutcome) -> FitSummary:
    """Convert the internal summary to the wire schema."""
    s: FittedSummary = outcome.summary
    return FitSummary(
        model_id=s.model_id,
        display_name=s.display_name,
        fit_id=s.fit_id,
        converged=s.converged,
        n_params=s.n_params,
        loglik=s.loglik,
        aic=s.aic,
        aicc=s.aicc,
        bic=s.bic,
        n_effective=s.n_obs_used,
        data_hash="",
        transform_stack=[],
        converged_warnings=s.converged_warnings,
        failed=s.failed,
        failure_reason=s.failure_reason,
        plain_one_liner=s.plain_one_liner,
        family=s.family,
        params=s.params,
        fitted_on=s.fitted_on,
        roots_ok=s.roots_ok,
        roots_note=s.roots_note,
        n_obs_used=s.n_obs_used,
        caveats=s.caveats,
        is_baseline=s.is_baseline,
    )


def summarise_failures(outcomes: list[FitOutcome]) -> str:
    """A sentence naming what was lost and why — used when the panel would otherwise look empty."""
    bad = [o for o in outcomes if o.summary.failed]
    if not bad:
        return ""
    names = "; ".join(f"{o.summary.display_name} ({o.summary.failure_reason[:90]})" for o in bad[:4])
    more = f" and {len(bad) - 4} more" if len(bad) > 4 else ""
    return (
        f"{len(bad)} of {len(outcomes)} fits failed: {names}{more}. The baselines above are still "
        "valid benchmarks — a series that no model can beat the naive forecast on is itself a "
        "finding."
    )


def best_by_aicc(outcomes: list[FitOutcome]) -> list[FitOutcome]:
    """
    Rank the successful fits for display: comparable ICs first, then the rest, baselines last.

    Models with `ic_comparable=False` (Croston, Theta, ARFIMA, the hurdle) are *not* ranked against
    the others — they are appended with their own note, because sorting a number that means something
    next to a number that does not is worse than leaving them out of the ranking.
    """
    ok = [o for o in outcomes if not o.summary.failed]
    comparable = [o for o in ok if o.summary.ic_comparable and o.summary.aicc is not None]
    # No IC: Theta, ARFIMA, Croston, the hurdle. Ranked by MASE where there is one, because a
    # forecaster reading this table is comparing *forecasts*. Ordering them alphabetically would put
    # Theta above a much better fit for no reason a user could see.
    unranked = [o for o in ok if not (o.summary.ic_comparable and o.summary.aicc is not None) and not o.summary.is_baseline]
    baselines = [o for o in ok if o.summary.is_baseline]
    comparable.sort(key=lambda o: (o.summary.aicc or np.inf))
    # Rows without an IC are ordered by their *measured* out-of-sample MASE where they have one, then
    # by parameter count. Alphabetical order would put k-NN above a Gaussian process that beats it by
    # half, for no reason a reader could see.
    unranked.sort(
        key=lambda o: (
            o.diagnostics.get("mase") is None,
            o.diagnostics.get("mase") if o.diagnostics.get("mase") is not None else np.inf,
            o.summary.n_params,
            o.summary.model_id,
        )
    )
    baselines.sort(key=lambda o: (o.summary.n_params, o.summary.model_id))
    return comparable + unranked + baselines


__all__ = [
    "EXOGENOUS_IDS",
    "IMPLEMENTED",
    "FitOutcome",
    "MAPE_BANNED",
    "best_by_aicc",
    "build_wrapper",
    "fit_many",
    "fit_one",
    "summarise_failures",
    "to_schema",
]
