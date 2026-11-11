"""
api/correlate.py — S6 (ACF / PACF) and S7 (the "what next" advisor) (§5.6).

  GET  /analysis/correlate          the correlogram battery for the current series
  POST /analysis/correlate/ccf      cross-correlation against a column, pre-whitened
  GET  /analysis/next               the ranked action list, with previews
  POST /analysis/next/preview       "what would happen if I did this?" — on a copy
  POST /analysis/next/apply         commit a recommended action (delegates to the S3 machinery)

Why S6 and S7 share a router: S7's rules exist to interpret S6's output, and splitting them across
two modules would mean passing the whole correlogram through the session between two HTTP calls
that a user never asked to make. The two `GET`s stay separate so the ACF screen renders without
waiting for the preview battery, which is the only expensive part.

Everything the advisor says here is *ranked against doing nothing*. The preview runs on copies;
`/apply` is the only endpoint that touches the series, and it goes through the same versioned,
reversible path as every other preprocessing operation.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.advisor.rules_next import NextContext, headline, rank
from app.api.deps import session_dep
from app.core.errors import ChronosError
from app.core.formatting import p_phrase
from app.core.schemas import (
    AdvisorReport,
    AnalysisResponse,
    Confidence,
    FigureResult,
    NarrativeResult,
    RuleOut,
    Stage,
    StatResult,
    TableColumn,
    TableResult,
)
from app.core.session import Session
from app.ts.correlate._acf import (
    Pattern,
    acf,
    box_pierce,
    default_max_lag,
    ljung_box,
    order_hints,
    pacf,
    read_pattern,
    variance_ratio,
)
from app.ts.correlate._ccf import ccf, spurious_warning
from app.ts.correlate._figures import acf_comparison_grid, acf_pacf_grid, ccf_chart
from app.ts.correlate._preview import preview_action

router = APIRouter(tags=["analysis"])

CORRELOGRAM = "correlogram"
NEXT_ADVICE = "next_advice"


# --------------------------------------------------------------------------- #
# Session plumbing
# --------------------------------------------------------------------------- #
def _require_series(session: Session) -> pd.Series:
    series: pd.Series | None = session.objects.get("series")
    if session.state.dataset is None or series is None:
        raise ChronosError(
            plain="There is no data loaded in this session yet.",
            fix="Load a file or an example dataset in Step 0 first.",
            code="correlate.no_data",
        )
    if len(series.dropna()) < 20:
        raise ChronosError(
            plain=f"Autocorrelation needs about 20 observations; this series has "
                  f"{int(series.notna().sum())}.",
            fix="Load a longer series, or aggregate the existing one to a coarser frequency.",
            code="correlate.too_short",
        )
    return series


def _period(session: Session) -> int | None:
    ds = session.state.dataset
    p = ds.frequency.period if ds is not None else None
    return p if p and p >= 2 else None


def _require_s5(session: Session) -> dict[str, Any]:
    """
    S7 interprets S5, so it must not invent a stationarity verdict.

    Returning a confident "difference it" on the basis of the ACF alone is precisely the mistake
    the two-stage design exists to prevent, so a missing S5 is an error with a fix, not a default.
    """
    # Goes through `_verdict_dict`, not `session.objects` directly: the raw stored dict has no
    # `seasonal_differencing_needed` or test count, and reading it directly is how S7 silently lost
    # the seasonal half of the verdict while still looking like it had one.
    rec = _verdict_dict(session)
    if rec is None:
        raise ChronosError(
            plain="The 'what next' advice needs the stationarity verdict from Step 5.",
            fix="Run Step 5 (Stationarity) first — the ACF alone cannot tell a unit root from a "
                "trend, and the advice must not guess.",
            code="next.missing_stationarity",
        )
    return rec


def _verdict_stamp(session: Session) -> str:
    """A value that changes whenever S5's verdict changes — the correlogram cache's second key."""
    rec = _verdict_dict(session)
    if rec is None:
        return "none"
    return f"{rec['verdict']}|{rec['differencing_needed']}|{rec.get('seasonal_differencing_needed')}"


def _verdict_dict(session: Session) -> dict[str, Any] | None:
    """
    S5 stores its verdict as a plain dict (`rec.as_dict()`), so accept either shape here.

    Normalising in one place keeps every S6/S7 consumer honest: there is exactly one way this code
    reads a verdict, so a change to S5's storage cannot silently leave one caller reading an
    attribute that no longer exists.
    """
    rec = session.objects.get("stat_verdict")
    if rec is None:
        return None
    if isinstance(rec, dict):
        out = dict(rec)
    else:                                        # pragma: no cover - older in-memory shape
        out = {
            "verdict": rec.verdict,
            "confidence": rec.confidence,
            "differencing_needed": rec.differencing_needed,
            "plain": getattr(rec, "plain", ""),
            "seasonal_differencing_needed": getattr(rec, "seasonal_differencing_needed", False),
            "n_unit_root_tests": len(getattr(rec, "unit_root_tests", []) or []),
        }
    out.setdefault("verdict", "inconclusive")
    out.setdefault("confidence", "low")
    out.setdefault("differencing_needed", False)
    out.setdefault("plain", "")

    # `Reconciliation.as_dict()` does not carry the seasonal-differencing flag or a test count, so
    # they are derived from the stored test rows rather than guessed.
    #
    # The direction is easy to get backwards and matters enormously: **OCSB's null hypothesis IS
    # the seasonal unit root**, so `rejects_null is False` means the seasonal pattern needs a
    # seasonal difference. Reading that flag as "the test found something bad" inverts the advice
    # and would tell a user not to difference a series that needs it. Only OCSB decides this;
    # Canova–Hansen measures seasonal stability and is not the same question.
    st = session.objects.get("stat_battery") or {}
    seasonal_rows = st.get("seasonal") or []
    ocsb = next((r for r in seasonal_rows if "ocsb" in str(r.get("id", ""))), None)
    out.setdefault(
        "seasonal_differencing_needed",
        bool(ocsb is not None and ocsb.get("rejects_null") is False),
    )
    out.setdefault(
        "n_unit_root_tests",
        len(st.get("unit_root") or []) + len(seasonal_rows),
    )
    return out


# --------------------------------------------------------------------------- #
# The correlogram
# --------------------------------------------------------------------------- #
class CCFRequest(BaseModel):
    column: str = Field(..., description="Name of the column to correlate the series against")
    max_lag: int | None = None
    prewhiten: bool = True


def _build_correlogram(session: Session) -> dict[str, Any]:
    series = _require_series(session)
    period = _period(session)
    max_lag = default_max_lag(len(series.dropna()), period)

    # A constant series has zero variance, so `acf` raises. Left unhandled that became a 500 with
    # a Python traceback on Step 6 — the exact opposite of the plain-English refusal the spec asks
    # for. Say what is true about the data and where to fix it instead.
    try:
        a = acf(series, max_lag=max_lag, period=period)
        p = pacf(series, max_lag=max_lag, period=period)
    except ValueError as exc:
        msg = str(exc)
        if "constant" in msg.lower():
            raise ChronosError(
                plain=(
                    "Every value in this series is the same, so there is no variation for an "
                    "autocorrelation to measure. ACF, PACF and the Ljung–Box test are undefined "
                    "for a constant series."
                ),
                fix=(
                    "Check the column mapping in Step 0 — a constant column is often a label or a "
                    "flag accidentally chosen as the values. Otherwise load a series that varies; "
                    "a mean (white-noise) model is the only forecast it can support."
                ),
                code="correlate.constant_series",
            ) from exc
        raise ChronosError(
            plain=f"The correlogram could not be computed: {msg}",
            fix="Check that at least 8 observations are present, then re-run Step 6.",
            code="correlate.failed",
        ) from exc

    lb_lags = [10] if max_lag >= 10 else [max_lag]
    if period and 2 * period <= max_lag:
        lb_lags += [period, 2 * period]
    elif period and period <= max_lag:
        lb_lags += [period]
    lb_lags = sorted(set(lb_lags))
    lb = ljung_box(a, lb_lags)
    bp = box_pierce(a, lb_lags)

    lb10 = lb.at(lb_lags[0]) or {"p_value": None, "statistic": None}
    seasonal_lb = lb.at(period) if period else None

    pattern = read_pattern(a, p, period=period, ljung_box_p=lb10["p_value"])

    try:
        vr = variance_ratio(series, q=2)
    except ValueError:
        vr = None

    # The stationarity verdict, when S5 has run. The pattern's hints are cross-checked against it.
    rec = _verdict_dict(session)
    verdict = None
    if rec is not None:
        verdict = {
            "verdict": rec["verdict"],
            "confidence": rec["confidence"],
            "differencing_needed": bool(rec["differencing_needed"]),
            "plain": rec["plain"],
        }
        if rec["differencing_needed"] and pattern.kind == "white_noise":
            pattern.caveats.append(
                "This correlogram looks white-noise-like, but Step 5 found a unit root. Check "
                "whether Step 5 ran on a different version of the series — that disagreement is "
                "more informative than either result on its own."
            )
        if (not rec["differencing_needed"]) and pattern.kind == "nonstationary_decay":
            pattern.caveats.append(
                f"Step 5 says {rec['verdict'].replace('_', ' ')}, so the slow decay is not a unit "
                "root — a trend, a break or long memory produces the same shape. Do not difference "
                "on the strength of the ACF alone."
            )

    hints = order_hints(
        a, p,
        pattern=pattern,
        period=period,
        differencing_needed=bool(rec["differencing_needed"]) if rec is not None else False,
        seasonal_differencing_needed=bool(rec.get("seasonal_differencing_needed", False))
        if rec is not None else False,
        variance_ratio=vr,
    )

    seasonal_acf = None
    if period and period <= a.max_lag:
        seasonal_acf = {
            "period": period,
            "r": float(a.at(period) or 0.0),
            "significant": a.significant(period),
            "band": float(a.band_bartlett[period - 1]),
        }

    return {
        "series_name": session.state.dataset.name if session.state.dataset else "your series",
        "n": a.n,
        "period": period,
        "max_lag": a.max_lag,
        "acf": a,
        "pacf": p,
        "ljung_box": lb,
        "box_pierce": bp,
        "ljung_box_early": lb10,
        "ljung_box_seasonal": seasonal_lb,
        "pattern": pattern,
        "variance_ratio": vr,
        "order_hints": hints,
        "seasonal_acf": seasonal_acf,
        "verdict": verdict,
    }


def _acf_results(c: dict[str, Any]) -> list[Any]:
    """Figures, tables and stats for the ACF screen — every one explaining itself."""
    a, p, pattern = c["acf"], c["pacf"], c["pattern"]
    period = c["period"]
    name = c["series_name"]
    out: list[Any] = []

    # ---- 1. the badges, as a table so they are readable and citable ----
    out.append(TableResult(
        id="table.correlogram_reading",
        stage=Stage.S6_CORRELATE,
        title="What the correlogram looks like",
        plain=(
            f"Automatically read from the shape of the ACF and PACF: {pattern.label}. "
            "This is a description of the picture, and Step 5's tests are what decide whether the "
            "series is stationary."
        ),
        why=(
            "Reading a correlogram by eye is where most ARIMA orders come from, and it is also "
            "where most of them go wrong. Naming the pattern explicitly makes the guess checkable."
        ),
        method=(
            f"ACF with n-denominator to lag {a.max_lag}; PACF by Durbin–Levinson; significance "
            "against Bartlett bands (1 + 2·Σr²)/n at 95%."
        ),
        columns=[
            TableColumn(key="feature", label="Feature"),
            TableColumn(key="value", label="What was found"),
            TableColumn(key="reading", label="What it suggests"),
        ],
        rows=[
            {"feature": "Pattern", "value": pattern.label, "reading": pattern.suggests or "—"},
            {"feature": "Confidence", "value": pattern.confidence,
             "reading": "how strongly the shape supports the label"},
            {"feature": "r at lag 1", "value": f"{a.values[0]:+.3f}",
             "reading": "persistence at the shortest lag"},
            {"feature": "Lags outside the band",
             "value": f"{len(a.significant_lags())} of {a.max_lag}",
             "reading": "; ".join(str(k) for k in a.significant_lags()[:10]) or "none"},
            *(
                [{"feature": f"r at lag {period}",
                  "value": f"{(c['seasonal_acf'] or {}).get('r', 0):+.3f}",
                  "reading": "seasonal period detected in Step 0"}]
                if c.get("seasonal_acf") else []
            ),
        ],
        caption="Every number in this table is on the chart behind it.",
        footnote="At 5% significance, roughly one lag in twenty is expected to clear the band by "
                 "chance. Isolated lags are not a pattern.",
        caveats=list(pattern.caveats),
        reproducible_snippet=(
            "import pandas as pd\n"
            "from statsmodels.tsa.stattools import acf, pacf\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            f"print(acf(s, nlags={a.max_lag}, fft=False))\n"
            f"print(pacf(s, nlags={a.max_lag}, method='ywm'))\n"
        ),
    ))

    # ---- 2. the grid figure ----
    grid = acf_pacf_grid(a, p, name=name)
    out.append(FigureResult(
        id="fig.acf_pacf",
        stage=Stage.S6_CORRELATE,
        title="ACF and PACF",
        plain=(
            "The left panel shows how strongly each observation is correlated with the one k steps "
            "back. The right panel shows the part of that correlation not explained by the steps in "
            "between — which is what tells AR order from MA order."
        ),
        why="These two pictures are the standard input to choosing p and q, and the difference "
            "between them is the single most misunderstood thing in the box-Jenkins method.",
        method="Sample ACF (n denominator) and PACF (Durbin–Levinson), ±1.96/√n and Bartlett bands.",
        figure_json=grid.figure_json,
        alt_text=grid.alt_text,
        caption=f"{name}, lags 1–{a.max_lag}.",
        caveats=list(pattern.caveats),
        reproducible_snippet=(
            "import pandas as pd\nimport matplotlib.pyplot as plt\n"
            "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            f"plot_acf(s, lags={a.max_lag}); plot_pacf(s, lags={min(a.max_lag, 20)}); plt.show()\n"
        ),
    ))

    # ---- 3. the raw vs differenced comparison (the teaching moment) ----
    frames = [("raw", a, p)]
    try:
        d1 = c["_series"].diff().dropna()
        if len(d1) >= 20:
            frames.append(("after 1 difference",
                           acf(d1, max_lag=a.max_lag, period=period),
                           pacf(d1, max_lag=a.max_lag, period=period)))
    except Exception:                                    # pragma: no cover - defensive
        pass
    if period and period >= 2:
        try:
            ds = c["_series"].diff(period).dropna()
            if len(ds) >= 20:
                frames.append((f"after a seasonal difference ({period})",
                               acf(ds, max_lag=a.max_lag, period=period),
                               pacf(ds, max_lag=a.max_lag, period=period)))
        except Exception:                                # pragma: no cover - defensive
            pass

    if len(frames) > 1:
        cmp_fig = acf_comparison_grid(frames, name=name)
        out.append(FigureResult(
            id="fig.acf_comparison",
            stage=Stage.S6_CORRELATE,
            title="The same correlogram after each transformation",
            plain=(
                "Each row is a version of the series. Reading down the left column shows whether a "
                "transformation actually removed the dependence or only moved it."
            ),
            why=(
                "This is the picture that makes over-differencing visible: a lag-1 bar swinging "
                "strongly negative is the signature of a difference that was not needed."
            ),
            method="ACF and PACF recomputed on each derived series independently.",
            figure_json=cmp_fig.figure_json,
            alt_text=cmp_fig.alt_text,
            caption="Nothing is applied by looking at this chart — the preview in Step 7 is where "
                    "a change is chosen.",
            caveats=[
                "Each difference removes one observation from the ends, so the panels are not "
                "computed from identical samples.",
            ],
            reproducible_snippet=(
                "import pandas as pd\nfrom statsmodels.graphics.tsaplots import plot_acf\n"
                "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
                "plot_acf(s); plot_acf(s.diff().dropna());\n"
                + (f"plot_acf(s.diff({period}).dropna())\n" if period else "")
            ),
        ))

    # ---- 4. portmanteau ----
    lb, bp = c["ljung_box"], c["box_pierce"]
    lb_rows = []
    for row in lb.rows():
        bp_row = next((r for r in bp.rows() if r["lag"] == row["lag"]), None)
        lb_rows.append({
            **row,
            "box_pierce": bp_row["statistic"] if bp_row else None,
            "box_pierce_p": bp_row["p_value"] if bp_row else None,
            "reading": (
                "autocorrelation remains — there is structure a white-noise model would miss"
                if row["rejects"] else
                "no evidence of remaining autocorrelation at these lags"
            ),
        })
    out.append(TableResult(
        id="table.portmanteau",
        stage=Stage.S6_CORRELATE,
        title="Ljung–Box (and Box–Pierce) at several lags",
        plain=(
            "One test covering all the lags up to h at once: it asks whether *any* of them is "
            "non-zero. A single Ljung–Box at lag 10 cannot see a seasonal spike at lag 12, which is "
            "why several lags are reported."
        ),
        why="The per-lag ACF bars answer 'is this lag significant'; the portmanteau answers the "
            "question a model actually needs: is there anything left at all?",
        method="Ljung–Box Q = n(n+2)·Σ r_k²/(n−k), χ² with h degrees of freedom; Box–Pierce "
               "reported for comparison because it is known to be badly sized on short samples.",
        columns=[
            TableColumn(key="lag", label="Up to lag", dtype="integer", align="right"),
            TableColumn(key="statistic", label="Ljung–Box Q", dtype="number", align="right"),
            TableColumn(key="df", label="df", dtype="integer", align="right"),
            TableColumn(key="p_value", label="p-value", dtype="number", format="p", align="right"),
            TableColumn(key="box_pierce", label="Box–Pierce Q", dtype="number", align="right"),
            TableColumn(key="reading", label="What it says"),
        ],
        rows=lb_rows,
        caption="Rejecting means autocorrelation remains; failing to reject means none was detected.",
        footnote="Ljung–Box loses power against a single sharp seasonal spike and has inflated "
                 "size when the model's parameters were estimated from the same data.",
        caveats=[
            "Failing to reject is not proof of independence — with a short series the test has "
            "little power against anything but strong dependence.",
        ],
        reproducible_snippet=(
            "import pandas as pd\nfrom statsmodels.stats.diagnostic import acorr_ljungbox\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            f"print(acorr_ljungbox(s, lags={lb.lags}, return_df=True))\n"
        ),
    ))

    # ---- 5. variance ratio ----
    vr = c["variance_ratio"]
    if vr is not None:
        out.append(StatResult(
            id="stat.variance_ratio",
            stage=Stage.S6_CORRELATE,
            title="Variance ratio (random-walk check)",
            plain=(
                f"The variance of 2-step changes is {vr['ratio']:.3f} times twice the variance of "
                f"1-step changes. For a random walk that ratio is exactly 1; {vr['reading']}."
            ),
            why=(
                "This is the quantified version of 'the variance went up after differencing' — the "
                "symptom of over-differencing that is otherwise noticed only by eye."
            ),
            method="Lo–MacKinlay variance ratio with a heteroskedasticity-robust variance.",
            value=float(vr["ratio"]),
            p_value=float(vr["p_value"]),
            statistic_name="VR(2)",
            ci_low=None,
            ci_high=None,
            interpretation=vr["reading"],
            band=("none" if 0.8 <= vr["ratio"] <= 1.25 else "moderate"),
            caveats=["The test assumes the series' variance exists; on a very persistent series it "
                     "has low power."],
            citations=["Lo & MacKinlay (1988), 'Stock market prices do not follow random walks'"],
            reproducible_snippet=(
                "import numpy as np, pandas as pd\n"
                "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
                "d1, d2 = np.diff(s), s.diff(2).dropna()\n"
                "print(d2.var(ddof=1) / (2 * d1.var(ddof=1)))\n"
            ),
        ))

    # ---- 6. order hints ----
    hints = c["order_hints"]
    if hints:
        out.append(TableResult(
            id="table.order_hints",
            stage=Stage.S6_CORRELATE,
            title="Candidate model orders (starting points, not decisions)",
            plain=(
                "The smallest models consistent with the shape of the correlogram. Each one lists "
                "what supports it and what would have to be checked — a hint without its "
                "counter-evidence is just a guess."
            ),
            why="A starting grid of two or three orders is compared by information criterion and out "
                "of sample; the correlogram narrows the grid but cannot choose from it.",
            method="Cut-offs read from the ACF (q, Q) and PACF (p, P); d and D come from Step 5's "
                   "unit-root tests, not from the correlogram.",
            columns=[
                TableColumn(key="notation", label="Order"),
                TableColumn(key="plain", label="Why this order"),
                TableColumn(key="support", label="Supporting features"),
                TableColumn(key="against", label="What to check"),
            ],
            rows=[
                {
                    "notation": h.label,
                    "plain": h.plain,
                    "support": " · ".join(h.support),
                    "against": " · ".join(h.against),
                }
                for h in hints
            ],
            caption="d and D are taken from Step 5. If you disagree with them, change Step 5's "
                    "inputs, not this table.",
            footnote="Over-differencing is far more expensive than under-differencing: it cannot be "
                     "undone by adding AR terms.",
            caveats=[
                "An order read off a correlogram is a hypothesis. Only an out-of-sample comparison "
                "can tell you whether it was right.",
            ],
            reproducible_snippet=(
                "import pandas as pd\n"
                "from pmdarima import auto_arima\n"
                "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
                "print(auto_arima(s, seasonal=True, stepwise=True).summary())\n"
            ),
        ))
    return out


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/analysis/correlate")
def get_correlate(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    The full correlogram battery. Cached in the session so the S6 screen can re-render freely.
    """
    t0 = time.perf_counter()
    store = session.objects.get(CORRELOGRAM)
    # The cache key covers the series *and* the S5 verdict. Keying only on the series version meant
    # that running S6 before S5 cached a payload with no verdict, and S7 then read that stale copy
    # and lost the seasonal-differencing flag and the test count — the advice silently degraded to
    # "the ACF says so", which is exactly what the two-stage design exists to prevent.
    key = (session.state.series_version, _verdict_stamp(session))
    if store and store.get("key") == key:
        payload = store["payload"]
        payload["cached"] = True
        return payload
    version = session.state.series_version

    c = _build_correlogram(session)
    c["_series"] = session.objects.get("series")
    results = _acf_results(c)
    session.mark_stage(Stage.S6_CORRELATE, "complete",
                       f"{len(results)} result blocks from {c['acf'].max_lag} lags")

    payload = {
        "stage": Stage.S6_CORRELATE.value,
        "status": "complete",
        "cached": False,
        "series_name": c["series_name"],
        "n": c["n"],
        "period": c["period"],
        "max_lag": c["max_lag"],
        "band_white": c["acf"].band_white,
        "band_bartlett": c["acf"].band_bartlett,
        "acf": c["acf"].rows(),
        "pacf": c["pacf"].rows(),
        "pattern": c["pattern"].to_dict(),
        "ljung_box": c["ljung_box"].rows(),
        "box_pierce": c["box_pierce"].rows(),
        "seasonal_acf": c["seasonal_acf"],
        "variance_ratio": c["variance_ratio"],
        "order_hints": [h.to_dict() for h in c["order_hints"]],
        "stationarity": c["verdict"],
        "results": [r.model_dump(mode="json") for r in results],
        "duration_ms": int((time.perf_counter() - t0) * 1000),
    }
    session.objects[CORRELOGRAM] = {"key": key, "version": version, "payload": payload}
    return payload


@router.post("/analysis/correlate/ccf")
def post_ccf(body: CCFRequest, session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    Cross-correlation between the series and another column, raw and pre-whitened.

    Both are returned. The pre-whitened one is the one to read; the raw one is there so the
    difference between them is visible rather than asserted.
    """
    series = _require_series(session)
    ds = session.state.dataset
    columns: dict[str, pd.Series] = session.objects.get("exog_columns") or {}
    target = columns.get(body.column)
    if target is None:
        available = sorted(columns) or []
        raise ChronosError(
            plain=f"'{body.column}' is not an extra column in this session.",
            fix=(
                "Columns available to correlate against: " + ", ".join(available) + "."
                if available else
                "This dataset has only one column, so there is nothing to correlate against. Upload "
                "a file with a second column to use cross-correlation."
            ),
            code="correlate.no_column",
        )

    max_lag = body.max_lag or min(24, max(4, len(series.dropna()) // 6))
    pw = ccf(target, series, max_lag=max_lag, x_name=body.column,
             y_name=ds.name if ds else "your series", prewhiten=True)
    raw = ccf(target, series, max_lag=max_lag, x_name=body.column,
              y_name=ds.name if ds else "your series", prewhiten=False)
    fig = ccf_chart(pw)
    warn = spurious_warning(pw, raw)

    return {
        "stage": Stage.S6_CORRELATE.value,
        "column": body.column,
        "n": pw.n,
        "band": pw.band,
        "peak": pw.peak(),
        "raw_peak": raw.peak(),
        "prewhitened": True,
        "filter_ar": pw.filter_ar,
        "rows": pw.rows(),
        "raw_rows": raw.rows(),
        "warning": warn,
        "figure": fig.figure_json,
        "alt_text": fig.alt_text,
        "caveats": [
            "Cross-correlation is not causation, and with two autocorrelated series a large CCF "
            "appears at some lag even when neither drives the other.",
            "Pre-whitening removes the shared memory; if the two series are related only through a "
            "common trend, the pre-whitened CCF will be small, which is the correct answer.",
        ],
    }


@router.get("/analysis/next")
def get_next(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    The ranked "what next" list, each item carrying its preview.

    The previews run here rather than only on demand because the ranking depends on them: an
    action whose preview says "this makes it worse" is demoted, and a list that demoted without
    saying why would be worse than no ranking at all.
    """
    t0 = time.perf_counter()
    series = _require_series(session)
    rec = _require_s5(session)
    period = _period(session)

    # S6 — recompute if the cached version is stale.
    store = session.objects.get(CORRELOGRAM)
    if not store or store.get("key") != (session.state.series_version, _verdict_stamp(session)):
        # Rebuild rather than trusting a payload computed before S5 ran.
        get_correlate(session=session)
        store = session.objects.get(CORRELOGRAM)
    if not store:                                  # pragma: no cover - defensive
        raise ChronosError(
            plain="The correlogram could not be computed for this series.",
            fix="Reload the data and try Step 6 again.",
            code="correlate.failed",
        )
    cg = store["payload"]

    ctx = _next_context(session, series, rec, cg, period)
    # Run the previews for the operations the rules propose.
    ctx.previews = _run_previews(series, ctx, period)

    ranked = rank(ctx)
    stage_complete = True
    session.objects[NEXT_ADVICE] = {
        "version": session.state.series_version,
        "ranked": ranked,
        "headline": headline(ranked, ctx),
        "facts": ctx.as_dict(),
    }
    session.mark_stage(Stage.S7_ADVISE_NEXT, "complete" if stage_complete else "error",
                       headline(ranked, ctx)[:180])

    report = AdvisorReport(
        stage=Stage.S7_ADVISE_NEXT,
        headline=headline(ranked, ctx),
        recommendations=[RuleOut(**{k: v for k, v in r.items()
                                    if k in RuleOut.model_fields}) for r in ranked
                         if r["kind"] != "blocker"],
        blockers=[RuleOut(**{k: v for k, v in r.items()
                             if k in RuleOut.model_fields}) for r in ranked
                  if r["kind"] == "blocker"],
        facts_used={k: v for k, v in ctx.as_dict().items()
                    if isinstance(v, (int, float, str, bool))},
        confidence=Confidence.HIGH if ctx.n_unit_root_tests >= 2 else Confidence.MEDIUM,
        next_stage_ready=bool(ranked),
        narrative=headline(ranked, ctx),
    )

    return {
        "stage": Stage.S7_ADVISE_NEXT.value,
        "status": "complete",
        "headline": headline(ranked, ctx),
        "ranked": ranked,
        "facts": ctx.as_dict(),
        "pattern": cg["pattern"],
        "stationarity": cg.get("stationarity"),
        "advisor": report.model_dump(mode="json"),
        "results": [_next_narrative(ctx, ranked).model_dump(mode="json")],
        "duration_ms": int((time.perf_counter() - t0) * 1000),
    }


class PreviewRequest(BaseModel):
    op: Literal["difference", "transform.log", "undo"]
    order: int | Literal["seasonal"] = 1
    rule_id: str | None = None


@router.post("/analysis/next/preview")
def post_preview(body: PreviewRequest, session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    "What would happen if I did this?" — computed on a copy; the session is untouched.
    """
    series = _require_series(session)
    period = _period(session)
    params = {"order": body.order} if body.op == "difference" else {}
    out = preview_action(series, body.op, period=period, params=params)
    out["rule_id"] = body.rule_id
    out["stage"] = Stage.S7_ADVISE_NEXT.value
    return out


class ApplyRequest(BaseModel):
    op: Literal["difference", "transform.log"]
    order: int | Literal["seasonal"] = 1
    rule_id: str | None = None
    reason: str = ""


@router.post("/analysis/next/apply")
def post_apply(body: ApplyRequest, session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    Commit a recommended action through the same versioned, reversible path as Step 2.

    Deliberately not a separate implementation: an action taken from the "what next" list and the
    same action taken from the Step 2 menu must produce the identical versioned record, or undo
    will diverge from what the user did.
    """
    from app.api.preprocess import OpRequest, apply_operation

    period = _period(session)

    # Both actions go through the *same* S3 machinery as the Step 2 menu. This is deliberate: an
    # action taken from "what next" and the same action taken from the menu must produce an
    # identical versioned record, or undo will not match what the user did.
    if body.op == "difference":
        order = period if body.order == "seasonal" else int(body.order)
        if not order or order < 1:
            raise ChronosError(
                plain="A difference needs a lag of at least 1.",
                fix="Choose a lag, or run Step 5 to detect the seasonal period.",
                code="next.bad_lag",
            )
        if body.order == "seasonal" and not period:
            raise ChronosError(
                plain="No seasonal period was detected, so there is no seasonal difference to take.",
                fix="Run Step 0/1 to detect the calendar, or take a first difference instead.",
                code="next.no_seasonal_period",
            )
        request = OpRequest(op="difference", params={"lag": int(order)},
                            rule_id=body.rule_id, reason=body.reason)
    else:
        request = OpRequest(op="transform.log", params={}, rule_id=body.rule_id,
                            reason=body.reason)

    resp = apply_operation(request, session=session)
    session.mark_stage(Stage.S7_ADVISE_NEXT, "complete", "Applied from the what-next list")
    return resp.model_dump(mode="json")


def _reason_for(rule_id: str | None) -> str:
    return {
        "next.diff.first": "A unit root was not rejected in Step 5 (rule next.diff.first)",
        "next.diff.seasonal": "A seasonal unit root was not rejected in Step 5 "
                              "(rule next.diff.seasonal)",
        "next.transform.log": "The spread grows with the level (rule next.transform.log)",
    }.get(rule_id or "", "")


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #
def _next_context(
    session: Session, series: pd.Series, rec: Any, cg: dict[str, Any], period: int | None
) -> NextContext:
    """Fill the context from the stored S4/S5/S6 results — nothing is invented here."""
    decomp = session.objects.get("decomposition") or {}
    shares = decomp.get("shares") or {}
    findings = {f["id"]: f for f in decomp.get("findings") or []}

    seasonal_finding = findings.get("stat.seasonality") or {}
    trend_finding = findings.get("stat.trend") or {}

    a_rows = cg["acf"]
    r1 = a_rows[0]["acf"] if a_rows else None
    pattern = cg["pattern"]
    lb10 = next((r for r in cg["ljung_box"] if r["lag"] == min(10, cg["max_lag"])), None)
    lb_seasonal = (next((r for r in cg["ljung_box"] if r["lag"] == period), None)
                   if period else None)

    vr = cg.get("variance_ratio")

    return NextContext(
        series_name=session.state.dataset.name if session.state.dataset else "your series",
        n=int(series.notna().sum()),
        period=period,
        r1=r1,
        max_lag=cg["max_lag"],
        n_significant=len([r for r in a_rows if r["significant"]]),
        decay_lags=int(next((i for i, r in enumerate(a_rows) if not r["significant"]),
                            len(a_rows))),
        pattern=pattern["kind"],
        ljung_box_p=lb10["p_value"] if lb10 else None,
        ljung_box_p_seasonal=lb_seasonal["p_value"] if lb_seasonal else None,
        seasonal_acf_significant=bool((cg.get("seasonal_acf") or {}).get("significant")),
        variance_ratio=vr["ratio"] if vr else None,
        variance_ratio_p=vr["p_value"] if vr else None,
        stationarity_verdict=rec["verdict"],
        differencing_needed=bool(rec["differencing_needed"]),
        seasonal_differencing_needed=bool(rec.get("seasonal_differencing_needed", False)),
        n_unit_root_tests=int(rec.get("n_unit_root_tests") or 0),
        trend_share=shares.get("trend"),
        seasonal_share=shares.get("seasonal"),
        seasonal_strength=decomp.get("seasonal_strength"),
        seasonal_growing=bool(seasonal_finding.get("caveats") and any(
            "swing is" in c for c in seasonal_finding.get("caveats", []))),
        trend_band=trend_finding.get("band", "none"),
        already_logged=any(
            (p.method or "").lower().startswith("log") for p in session.state.preprocessing),
        already_differenced=any(p.step == "difference" for p in session.state.preprocessing),
        already_seasonally_differenced=any(
            p.step == "difference" and (p.params or {}).get("lag", 1) not in (1, None)
            for p in session.state.preprocessing),
        n_transforms_applied=sum(
            1 for p in session.state.preprocessing if p.step in {"transform", "difference"}),
        variance_stable=None,
    )


def _run_previews(series: pd.Series, ctx: NextContext, period: int | None) -> dict[str, Any]:
    """
    Run every distinct operation the rules propose, once, and key the result by rule id.

    Cached by (op, params) so three rules proposing a first difference cost one computation.
    """
    from app.advisor.rules_next import RULES

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    out: dict[str, Any] = {}
    for rule in RULES:
        if rule.op is None or not rule.applies(ctx):
            continue
        key = (rule.op, str(sorted(rule.op_params.items())))
        if key not in cache:
            try:
                cache[key] = preview_action(series, rule.op, period=period, params=rule.op_params)
            except Exception as exc:                     # pragma: no cover - defensive
                cache[key] = {"verdict": "unavailable",
                              "plain": f"The preview could not run: {exc}"}
        out[rule.id] = cache[key]
    return out


def _next_narrative(ctx: NextContext, ranked: list[dict[str, Any]]) -> NarrativeResult:
    """The paragraph that goes into the report, built from the same numbers as the list."""
    top = ranked[0] if ranked else None
    paragraphs = [
        f"The correlogram of {ctx.series_name} is {ctx.pattern.replace('_', ' ')}: "
        f"{ctx.n_significant} of the first {ctx.max_lag} autocorrelations are outside the "
        f"significance band, and Ljung–Box gives {p_phrase(ctx.ljung_box_p)}."
        if ctx.ljung_box_p is not None else
        f"The correlogram of {ctx.series_name} is {ctx.pattern.replace('_', ' ')}.",
        f"Step 5 concluded that the series is {ctx.stationarity_verdict.replace('_', ' ')}, "
        f"and the next step is: {top['recommend'].lower() if top else 'no change'}. "
        + (f"Reason: {top['because']}" if top else ""),
    ]
    return NarrativeResult(
        id="narrative.next_steps",
        stage=Stage.S7_ADVISE_NEXT,
        title="What to do next, and why",
        plain="A written summary of the ranked action list, using the same numbers.",
        why="A ranked list without a paragraph is easy to act on and hard to defend in a write-up.",
        method="Composed from the S5 verdict, the S6 correlogram and the S7 rule outputs.",
        paragraphs=[p for p in paragraphs if p],
        numbers_cited={
            k: float(v) for k, v in {
                "r1": ctx.r1, "ljung_box_p": ctx.ljung_box_p,
                "variance_ratio": ctx.variance_ratio,
            }.items() if v is not None
        },
        caveats=[
            "Everything here assumes the series has not changed since Steps 4–6 ran. If it has, "
            "those diagnostics describe a different series.",
        ],
        reproducible_snippet=(
            "import pandas as pd\n"
            "from statsmodels.stats.diagnostic import acorr_ljungbox\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "print(acorr_ljungbox(s, lags=[10], return_df=True))\n"
        ),
    )


__all__ = ["router"]
