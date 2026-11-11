"""
api/decompose.py — S4: take the series apart (§5.4).

  POST /analysis/decompose          run one method, store the result
  GET  /analysis/decompose          read back whatever is stored
  POST /analysis/decompose/compare  run several methods and show their trends together

Design notes:

* **The method menu is data, not prose.** `GET /analysis/decompose/methods` returns what can run
  on this series and why a method cannot — so the UI can grey a button out with a reason instead
  of offering a choice that will fail.
* **A blocked method is a refusal with a reason**, not a silent substitution: multiplicative
  decomposition on non-positive data raises a `ChronosError` that names the offending values.
* **Every number shown travels with its own sentence.** The scorecard, the three component
  findings and the recommendation all read from the same computed facts, so the chart, the table
  and the advice cannot disagree.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.advisor.rules_decompose import DecompContext, rank
from app.api.deps import session_dep
from app.core.errors import ChronosError
from app.core.schemas import (
    AnalysisResponse,
    AdvisorReport,
    Confidence,
    FigureResult,
    NarrativeResult,
    RuleOut,
    Stage,
    TableColumn,
    TableResult,
    StatResult,
)
from app.core.session import Session
from app.ts.decompose import figures as F
from app.ts.decompose.components import scorecard
from app.ts.decompose.methods import Decomposition, Method, Model, check_model, decompose

router = APIRouter(tags=["analysis"])

CURRENT = "series"
STORE = "decomposition"


def _series(session: Session) -> pd.Series:
    series: pd.Series | None = session.objects.get(CURRENT)
    if session.state.dataset is None or series is None:
        raise ChronosError(
            plain="There is no data loaded in this session yet.",
            fix="Load a file or an example dataset in Step 0 first.",
        )
    return series


def _period(session: Session) -> int | None:
    ds = session.state.dataset
    return ds.frequency.period if ds is not None else None


def _require_decomposable(session: Session) -> None:
    """S4 is not gated on S2/S3: a user may look at components before deciding anything."""
    if session.state.stages[Stage.S1_PROFILE.value].status not in {"complete", "stale"}:
        raise ChronosError(
            plain="The series has not been examined yet, so its components would have no baseline.",
            fix="Run Step 1 (Explore) first — it detects the period the decomposition needs.",
            code="decompose.no_profile",
        )


def _plain_error(exc: Exception, method: str) -> ChronosError:
    """Turn a ValueError from the statistics layer into a plain sentence plus a fix."""
    msg = str(exc)
    fix = "Try a shorter window, a different method, or add observations."
    if "positive" in msg or "multiplicative" in msg.lower():
        fix = "Use additive decomposition, or transform the series first (Step 2 → log)."
    elif method == "CF" and ("period" in msg.lower() or "observations" in msg.lower()):
        # The band-pass filter's "period" is a *frequency band*, not a calendar season. Sending the
        # seasonal advice below would tell the user to fix a calendar that has nothing to do with
        # the filter's sample-length requirement.
        fix = (
            "The band-pass filter needs about three times the longest period in the band as "
            "observations. Use a longer series, shorten the band's upper bound, or choose "
            "Hodrick–Prescott or Unobserved components instead."
        )
    elif method in {"STL", "CLASSICAL", "MSTL"} and ("period" in msg.lower() or "cycles" in msg.lower()):
        # "Choose a longer seasonal period" was the wrong advice for a 100-minute series whose
        # nominal daily period is 1440, or for a yearly series with no seasonal calendar at all.
        # The honest fix is to name the methods that do not need a season, or to aggregate.
        fix = (
            "This series has no usable seasonal period at its current length. Choose "
            "Hodrick–Prescott, Unobserved components or Band-pass (cycle) — none of them needs "
            "one — or aggregate to a coarser calendar (Step 2). Only supply a period explicitly "
            "if you know the series really repeats on it."
        )
    return ChronosError(
        plain=f"{method} could not be run on this series: {msg}",
        fix=fix,
        code="decompose.failed",
    )


# --------------------------------------------------------------------------- #
# Methods available for this series
# --------------------------------------------------------------------------- #
def _reproducible_snippet(dec: Decomposition, body: "DecomposeRequest") -> str:
    """
    Code that actually reproduces *this* decomposition.

    The first version hard-coded `STL(s, period=12).fit()` on every result block, including HP,
    band-pass, classical and state-space runs. A snippet labelled "reproducible" that reconstructs
    a different decomposition is worse than no snippet: a reader who runs it gets different numbers
    and concludes the app's figures are wrong. The method and its options come from the same
    `dec`/`body` pair that produced the result.
    """
    p = dec.period or body.period or 12
    yy = "np.log(y)" if body.model == "multiplicative" else "y"
    head = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
        "y = s.interpolate().dropna()\n"
        f"yy = {yy}\n"
    )
    if body.method == "stl":
        return head + (
            "from statsmodels.tsa.seasonal import STL\n"
            f"fit = STL(yy, period={p}, robust={body.robust}).fit()\n"
            "trend, seasonal, remainder = fit.trend, fit.seasonal, fit.resid\n"
            "print(trend.std(), seasonal.std(), remainder.std())\n"
        )
    if body.method == "classical":
        return head + (
            "from statsmodels.tsa.seasonal import seasonal_decompose\n"
            f"fit = seasonal_decompose(yy, model='{body.model}', period={p}, "
            "extrapolate_trend=0)\n"
            "print(fit.trend.std(), fit.seasonal.std(), fit.resid.std())\n"
        )
    if body.method == "mstl":
        periods = body.periods or [p]
        return head + (
            "from statsmodels.tsa.seasonal import MSTL\n"
            f"fit = MSTL(yy, periods={list(map(int, periods))}).fit()\n"
            "print(fit.trend.std(), fit.seasonal.std(), fit.resid.std())\n"
        )
    if body.method == "ucm":
        seasonal = (
            f", seasonal={p}, stochastic_seasonal=True" if dec.period else ""
        )
        return head + (
            "from statsmodels.tsa.statespace.structural import UnobservedComponents\n"
            f"mod = UnobservedComponents(yy, level=True, trend=True, irregular=True{seasonal})\n"
            "res = mod.fit(disp=False, maxiter=200)\n"
            "print(res.smoothed_state)\n"
        )
    if body.method == "hp":
        lam = body.lamb if body.lamb is not None else (1600 if len(dec.observed) >= 100 else 100)
        return head + (
            "from statsmodels.tsa.filters.hp_filter import hpfilter\n"
            f"cycle, trend = hpfilter(yy, lamb={lam})\n"
            "print(trend.std(), cycle.std())\n"
        )
    # cf
    return head + (
        "from statsmodels.tsa.filters.cf_filter import cffilter\n"
        f"cycle, trend = cffilter(yy, low={body.low}, high={body.high}, drift=True)\n"
        "print(trend.std(), cycle.std())\n"
    )


@router.get("/analysis/decompose/methods")
def methods(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    What can run here, and what cannot — with the reason.

    The UI uses this to grey out a button *and say why*, which is the difference between a
    disabled control and a dead end.
    """
    series = _series(session)
    n = len(series.dropna())
    period = _period(session)
    out: list[dict[str, Any]] = []

    def entry(method: str, label: str, needs: str, ok: bool, why: str = "") -> None:
        out.append({"method": method, "label": label, "needs": needs,
                    "available": ok, "why_not": why})

    seasonal_ok = bool(period and period >= 2 and n >= 2 * (period or 0) + 2)
    why_seasonal = "" if seasonal_ok else (
        f"Needs two full cycles: {2 * (period or 0) + 2} observations for a period of {period}. "
        f"This series has {n}."
        if period else "This series has no seasonal calendar, so there is nothing to decompose."
    )
    entry("stl", "STL (recommended)", "a seasonal period, 2+ cycles", seasonal_ok, why_seasonal)
    entry("classical", "Classical", "a seasonal period, 2+ cycles", seasonal_ok, why_seasonal)
    entry("mstl", "MSTL (multiple seasons)", "2+ cycles of the longest period", seasonal_ok, why_seasonal)
    entry("ucm", "Unobserved components", "about 20 observations", n >= 20,
          "" if n >= 20 else f"Needs about 20 observations; this series has {n}.")
    entry("hp", "Hodrick–Prescott", "10 observations, no seasonal term needed", n >= 10,
          "" if n >= 10 else f"Needs 10 observations; this series has {n}.")
    entry("cf", "Band-pass (cycle)", "about 3× the longest period", n >= 96,
          "" if n >= 96 else f"Needs about 96 observations for a 32-period band; this has {n}.")

    return {
        "n": n,
        "period": period,
        "methods": out,
        "models": [
            {"model": "additive", "available": True, "why_not": ""},
            {
                "model": "multiplicative",
                "available": bool((series.dropna() > 0).all()),
                "why_not": "" if bool((series.dropna() > 0).all()) else (
                    f"{int((series.dropna() <= 0).sum())} values are zero or negative, and a "
                    "multiplicative decomposition is undefined there."
                ),
            },
        ],
    }


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
class DecomposeRequest(BaseModel):
    method: Literal["stl", "classical", "mstl", "ucm", "hp", "cf"] = "stl"
    model: Literal["additive", "multiplicative"] = "additive"
    period: int | None = Field(None, ge=2, le=365)
    periods: list[int] | None = None
    robust: bool = True
    seasonal_window: int | None = None
    trend_window: int | None = None
    lamb: float | None = None
    low: int = 6
    high: int = 32


def _ctx(dec: Decomposition, sc: dict[str, Any], session: Session) -> DecompContext:
    findings = {f.name: f for f in sc["findings"]}
    trend, seas = findings["Trend"], findings["Seasonality"]
    rem, cyc = findings["Remainder"], findings["Cycle"]
    series = _series(session)
    n = len(series.dropna())

    def pick(detail: str, marker: str) -> float | None:
        """Pull a number back out of the detail string the finding already computed."""
        for part in detail.split(";"):
            if marker in part:
                for token in part.replace("(", " ").replace(")", " ").split():
                    try:
                        return float(token)
                    except ValueError:
                        continue
        return None

    ds = session.state.dataset
    return DecompContext(
        series_name=ds.name if ds else "your series",
        n=n,
        period=_period(session),
        method=dec.method,
        model=dec.model,
        trend_strength=dec.trend_strength,
        seasonal_strength=dec.seasonal_strength,
        trend_share=dec.shares.get("trend"),
        seasonal_share=dec.shares.get("seasonal"),
        remainder_share=dec.shares.get("remainder"),
        trend_p=trend.p_value,
        trend_slope=pick(trend.detail, "Sen's") if trend.detail else None,
        seasonal_kw_p=seas.p_value,
        seasonal_amplitude_pct=pick(seas.detail, "%"),
        seasonal_growing=any("swing is" in c for c in seas.caveats),
        remainder_lb_p=rem.p_value,
        remainder_arch_p=pick(rem.detail, "ARCH-LM"),
        remainder_jb_p=pick(rem.detail, "Jarque"),
        cycle_p=cyc.p_value,
        cycle_period=(cyc.table[0]["period"] if cyc.table else None),
        cycle_on_calendar=bool(
            cyc.table and _period(session) and cyc.table[0].get("period")
            and abs(float(cyc.table[0]["period"]) - float(_period(session) or 0))
            <= max(1.0, 0.15 * float(_period(session) or 1))
        ),
        nested_seasonality=(dec.method in {"classical"} and (dec.seasonal_strength or 0) >= 0.64),
        multiplicative_blocked=(dec.model == "multiplicative"
                                and bool((series.dropna() <= 0).any())),
        n_nonpositive=int((series.dropna() <= 0).sum()),
        methods_compared=len(session.objects.get("decomp_methods_ran") or []),
        trend_spread=session.objects.get("decomp_trend_spread"),
    )


@router.post("/analysis/decompose", response_model=AnalysisResponse)
def run_decompose(
    body: DecomposeRequest | None = None,
    session: Session = Depends(session_dep),
) -> AnalysisResponse:
    """Run one decomposition. Idempotent: re-running replaces the stored result."""
    t0 = time.perf_counter()
    series = _series(session)
    _require_decomposable(session)
    body = body or DecomposeRequest()
    period = body.period or _period(session)
    ds = session.state.dataset

    # The guardrail: refuse, do not switch.
    try:
        check_model(body.model, series)
    except ValueError as exc:
        raise ChronosError(
            plain=str(exc),
            fix="Pick 'additive', or transform the series in Step 2 (a log needs positive data).",
            code="decompose.multiplicative_undefined",
        ) from exc

    # Only the keyword arguments each method accepts; passing all of them makes every method
    # fail on the ones it does not take.
    extra: dict[str, Any] = {}
    if body.method == "stl":
        extra.update(robust=body.robust, seasonal_window=body.seasonal_window,
                     trend_window=body.trend_window)
    elif body.method == "mstl":
        extra.update(periods=body.periods, robust=body.robust)
    elif body.method == "hp":
        extra.update(lamb=body.lamb)
    elif body.method == "cf":
        extra.update(low=body.low, high=body.high)
    try:
        dec = decompose(series, body.method, period=period, model=body.model, **extra)
    except ValueError as exc:
        raise _plain_error(exc, body.method.upper()) from exc

    sc = scorecard(dec, series, period)
    name = ds.name if ds else "your series"

    session.results[Stage.S4_DECOMPOSE] = {}
    session.add_result(NarrativeResult(
        id="narrative.decomp_scorecard",
        stage=Stage.S4_DECOMPOSE,
        title=f"What this series is made of ({dec.label})",
        plain=sc["plain"],
        why=(
            "The split decides which terms the model needs: a strong trend needs a trend term, a "
            "strong seasonal pattern needs a seasonal one, and a remainder that is not white means "
            "the model still has work to do."
        ),
        method=f"{dec.label} via app/ts/decompose/methods.py; shares from component variances.",
        paragraphs=[sc["plain"]] + [f.plain for f in sc["findings"]],
        caveats=[c for f in sc["findings"] for c in f.caveats],
        reproducible_snippet=(
            "import pandas as pd\n"
            "from statsmodels.tsa.seasonal import STL\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            f"fit = STL(s, period={period or 12}, robust=True).fit()\n"
            "print(fit.trend.head(), fit.seasonal.head(), fit.resid.head())\n"
        ),
    ))
    session.add_result(TableResult(
        id="table.decomp_scorecard",
        stage=Stage.S4_DECOMPOSE,
        title="Decomposition scorecard",
        plain=(
            "How much of the series' variation each component carries. The three shares sum to "
            "100%."
        ),
        why="This is the ranking that decides which model terms are needed.",
        method=f"Variance shares of the {dec.label} components; F_T and F_S as in Hyndman §3.4.",
        columns=[
            TableColumn(key="component", label="Component"),
            TableColumn(key="share", label="Share of variation", dtype="number", align="right",
                        format=".1%"),
            TableColumn(key="reading", label="Strength"),
        ],
        rows=sc["rows"],
        caption=f"Method: {dec.label}.",
        reproducible_snippet=(
            "import numpy as np\n"
            "# the shares are the variance of each component, all three summing to the total\n"
            "print(np.var(trend), np.var(seasonal), np.var(remainder))\n"
        ),
    ))

    for f in sc["findings"]:
        if f.statistic is None:
            # A finding without a statistic is stated as text, never dressed up as a number.
            session.add_result(NarrativeResult(
                id=f"narrative.{f.id.replace('.', '_')}",
                stage=Stage.S4_DECOMPOSE,
                title=f.name,
                plain=f.plain,
                why="Reported because a component that cannot be quantified is a caveat, not a result.",
                method=dec.label,
                paragraphs=[f.plain],
                caveats=f.caveats,
                reproducible_snippet="print('not computable on this series')\n",
            ))
            continue
        session.add_result(StatResult(
            id=f.id.replace("stat.", "stat.decomp."),
            stage=Stage.S4_DECOMPOSE,
            title=f.name,
            plain=f.plain,
            why=(
                "Each component gets a number as well as a picture: a plot alone cannot tell "
                "'strong' from 'looks strong'."
            ),
            technical=f.detail,
            method=dec.label,
            value=float(f.statistic),
            p_value=f.p_value,
            statistic_name=f.name,
            interpretation=f.plain,
            band=f.band if f.band in {"none", "weak", "moderate", "strong"} else "none",
            caveats=f.caveats,
            reproducible_snippet=_reproducible_snippet(dec, body),
        ))
        if f.table:
            session.add_result(TableResult(
                id=f"table.{f.id.split('.')[-1]}",
                stage=Stage.S4_DECOMPOSE,
                title=f"{f.name} by position",
                plain=f"The {f.name.lower()} effect at each position of the cycle.",
                why="The shape, not just its strength: this is what the model will reproduce.",
                method=dec.label,
                columns=[
                    TableColumn(key="position", label="Position", dtype="integer", align="right"),
                    TableColumn(key="effect", label="Effect", dtype="number", align="right"),
                ],
                rows=f.table,
                reproducible_snippet="print(seasonal.shape)\n",
            ))

    grid = F.component_grid(dec, name)
    session.add_result(FigureResult(
        id="fig.decomposition",
        stage=Stage.S4_DECOMPOSE,
        title=f"Decomposition ({dec.label})",
        plain=(
            "The series split into the three parts that add back up to it: trend, seasonal "
            "pattern and what is left over."
        ),
        why="Seeing the parts separately is what makes 'the series is trending' checkable.",
        method=dec.label,
        figure_json=grid.figure_json,
        alt_text=grid.alt_text,
        caveats=dec.caveats,
        reproducible_snippet=(
            "import pandas as pd\n"
            "from statsmodels.tsa.seasonal import STL\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            f"STL(s, period={period or 12}, robust=True).fit().plot()\n"
        ),
    ))
    if dec.seasonal_strength is not None and (dec.seasonal_strength or 0) > 0.05:
        seas_finding = next(f for f in sc["findings"] if f.name == "Seasonality")
        bars = F.seasonal_index_bars(seas_finding.table, period or 12, name)
        if bars is not None:
            session.add_result(FigureResult(
                id="fig.seasonal_index",
                stage=Stage.S4_DECOMPOSE,
                title="Seasonal shape",
                plain="The average deviation at each position in the cycle — the seasonal 'index'.",
                why="The peak and trough positions are what a reader checks the model against.",
                method=dec.label,
                figure_json=bars.figure_json,
                alt_text=bars.alt_text,
                reproducible_snippet="print(seasonal.groupby(seasonal.index.month).mean())\n",
            ))
    cyc = next(f for f in sc["findings"] if f.name == "Cycle")
    perio = F.periodogram_chart(series, period, cyc.table)
    session.add_result(FigureResult(
        id="fig.periodogram",
        stage=Stage.S4_DECOMPOSE,
        title="Periodogram",
        plain=(
            "How much of the variation sits at each periodicity. A tall spike at the calendar "
            "period is seasonality; a spike anywhere else is a cycle."
        ),
        why="It is the only honest way to answer 'is there a cycle, and how long is it?'",
        method="FFT periodogram of the detrended series; peak significance by Fisher's g.",
        figure_json=perio.figure_json,
        alt_text=perio.alt_text,
        caveats=cyc.caveats,
        reproducible_snippet=(
            "import numpy as np, pandas as pd\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "p = np.abs(np.fft.rfft(s - s.mean())) ** 2\n"
            "print(np.argsort(p)[-5:])\n"
        ),
    ))

    # ---- advice -----------------------------------------------------------
    ctx = _ctx(dec, sc, session)
    hits = rank(ctx)
    recs = [RuleOut(id=r.id, stage=Stage.S4_DECOMPOSE, recommend=r.recommend,
                    confidence=r.confidence, because=r.because(ctx), provenance=r.provenance,
                    alternative=r.alternative, risk_if_ignored=r.risk_if_ignored, how_to=r.how_to)
            for r in hits if r.kind != "blocker"]
    blockers = [RuleOut(id=r.id, stage=Stage.S4_DECOMPOSE, recommend=r.recommend,
                        confidence=r.confidence, because=r.because(ctx), provenance=r.provenance,
                        alternative=r.alternative, risk_if_ignored=r.risk_if_ignored, how_to=r.how_to)
                for r in hits if r.kind == "blocker"]
    report = AdvisorReport(
        stage=Stage.S4_DECOMPOSE,
        headline=sc.get("headline") or sc["plain"],
        recommendations=recs,
        blockers=blockers,
        facts_used={
            "trend_strength": round(float(dec.trend_strength), 4) if dec.trend_strength else 0.0,
            "seasonal_strength": round(float(dec.seasonal_strength), 4) if dec.seasonal_strength else 0.0,
            "trend_share": round(float(dec.shares.get("trend", 0.0)), 4),
            "seasonal_share": round(float(dec.shares.get("seasonal", 0.0)), 4),
            "remainder_share": round(float(dec.shares.get("remainder", 0.0)), 4),
            "n": ctx.n,
        },
        confidence=Confidence.HIGH if blockers == [] else Confidence.MEDIUM,
        next_stage_ready=not blockers,
        narrative=sc["plain"],
    )
    session.advice[Stage.S4_DECOMPOSE] = report
    session.objects[STORE] = {
        "method": dec.method,
        "model": dec.model,
        "label": dec.label,
        "period": period,
        "scorecard": {"rows": sc["rows"], "plain": sc["plain"]},
        "findings": [
            {"id": f.id, "name": f.name, "statistic": f.statistic, "p_value": f.p_value,
             "band": f.band, "plain": f.plain, "detail": f.detail, "caveats": f.caveats,
             "table": f.table}
            for f in sc["findings"]
        ],
        "shares": dec.shares,
        "caveats": dec.caveats + [c for f in sc["findings"] for c in f.caveats],
        "n": ctx.n,
    }
    session.mark_stage(Stage.S4_DECOMPOSE, "complete",
                       f"{dec.label}: {sc['plain'][:80]}")

    return AnalysisResponse(
        stage=Stage.S4_DECOMPOSE,
        advisor=report,
        results=list(session.results_for(Stage.S4_DECOMPOSE)),
        pipeline=session.state.progress(),
        caveats=dec.caveats,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
class CompareRequest(BaseModel):
    methods: list[str] = Field(default_factory=lambda: ["stl", "classical", "ucm"])
    model: Literal["additive", "multiplicative"] = "additive"


@router.post("/analysis/decompose/compare")
def compare_methods(
    body: CompareRequest | None = None,
    session: Session = Depends(session_dep),
) -> dict[str, Any]:
    """
    Run several methods and overlay their trends.

    The point is disagreement: where two methods' trends separate, the trend is a choice. That
    fact is more useful than any single decomposition, and it is the one a textbook picture hides.
    """
    series = _series(session)
    _require_decomposable(session)
    body = body or CompareRequest()
    period = _period(session)
    ds = session.state.dataset

    ran: list[Decomposition] = []
    skipped: list[dict[str, str]] = []
    for method in body.methods:
        try:
            ran.append(decompose(series, method, period=period, model=body.model))  # type: ignore[arg-type]
        except (ValueError, TypeError) as exc:
            skipped.append({"method": method, "why": str(exc)[:200]})
    if not ran:
        raise ChronosError(
            plain="None of the chosen methods could run on this series.",
            fix="Pick STL with the period detected in Step 0, or add more observations.",
            code="decompose.none_ran",
            context={"skipped": skipped},
        )

    fig = F.side_by_side(ran, ds.name if ds else "your series")
    rows = []
    for dec in ran:
        rows.append({
            "method": dec.label,
            "trend_share": round(float(dec.shares.get("trend", float("nan"))), 4),
            "seasonal_share": round(float(dec.shares.get("seasonal", float("nan"))), 4),
            "remainder_share": round(float(dec.shares.get("remainder", float("nan"))), 4),
            "F_T": round(float(dec.trend_strength), 4)
            if dec.trend_strength is not None and np.isfinite(dec.trend_strength) else None,
            "F_S": round(float(dec.seasonal_strength), 4)
            if dec.seasonal_strength is not None and np.isfinite(dec.seasonal_strength) else None,
        })

    # F_T / F_S need a remainder to compare against; the HP and band-pass filters have none, so
    # those cells are reported as not computed rather than as a zero that looks like a finding.
    notes = [
        f"{d.label} has no separate remainder, so F_T/F_S are not defined for it."
        for d in ran if d.trend_strength is None
    ]

    # How far apart are the trends, relative to the series' own spread?
    spread = None
    if len(ran) >= 2:
        matrix = np.vstack([np.asarray(d.trend, dtype=float) for d in ran])
        spread = float(np.nanmax(np.nanstd(matrix, axis=0)))

    session.add_result(FigureResult(
        id="fig.decomp_methods",
        stage=Stage.S4_DECOMPOSE,
        title="Trend by method",
        plain=(
            "The same series decomposed several ways, with only the trends drawn. Where the lines "
            "separate, the trend is a consequence of the method."
        ),
        why="This is the cheapest guard against quoting one method's trend as the series' trend.",
        method=", ".join(d.label for d in ran),
        figure_json=fig.figure_json,
        alt_text=fig.alt_text,
        caveats=notes + (
            [f"{s['method']} could not run: {s['why']}" for s in skipped]
            + ([f"The methods' trends differ by up to {spread:.4g} at a single date — comparable "
                "to the series' own variation, so treat the exact trend path with caution."]
               if spread else [])
        ),
        reproducible_snippet=(
            "import pandas as pd\n"
            "from statsmodels.tsa.seasonal import STL, seasonal_decompose\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            f"print(STL(s, period={period or 12}).fit().trend.tail())\n"
            f"print(seasonal_decompose(s, period={period or 12}).trend.tail())\n"
        ),
    ))

    session.objects["decomp_methods_ran"] = [d.label for d in ran]
    session.objects["decomp_trend_spread"] = float(spread) if spread is not None else None
    return {
        "stage": Stage.S4_DECOMPOSE.value,
        "ran": [d.label for d in ran],
        "skipped": skipped,
        "rows": rows,
        "trend_spread": round(spread, 6) if spread is not None else None,
        "notes": notes,
        "figure": fig.figure_json,
        "alt_text": fig.alt_text,
    }


# --------------------------------------------------------------------------- #
# Read back
# --------------------------------------------------------------------------- #
@router.get("/analysis/decompose")
def read_decompose(session: Session = Depends(session_dep)) -> dict[str, Any]:
    stored = session.objects.get(STORE)
    if not stored:
        return {
            "stage": Stage.S4_DECOMPOSE.value,
            "status": "not_started",
            "scorecard": None,
            "findings": [],
            "results": [],
        }
    report = session.advice.get(Stage.S4_DECOMPOSE)
    return {
        "stage": Stage.S4_DECOMPOSE.value,
        "status": session.state.stages[Stage.S4_DECOMPOSE.value].status,
        **stored,
        "results": [r.model_dump(mode="json") for r in session.results_for(Stage.S4_DECOMPOSE)],
        "recommendations": [r.model_dump(mode="json") for r in report.recommendations] if report else [],
        "blockers": [r.model_dump(mode="json") for r in report.blockers] if report else [],
        "headline": report.headline if report else "",
    }


__all__ = ["router"]
