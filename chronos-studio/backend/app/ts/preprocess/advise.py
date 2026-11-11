"""
ts/preprocess/advise.py — the S2 stage: turn facts into an advisor report (§4.4, §5.3).

S2 does not touch the data. It reads what S1 computed, adds the diagnostics that
only matter for a preprocessing decision, runs the rules in `advisor/rules_prep.py`,
and emits:

  * one `AdvisorReport` (headline + ranked recommendations + blockers + facts_used),
  * a `TableResult` that shows the *evidence* behind every recommendation, so the
    user can disagree with the advice without opening a notebook,
  * one figure — the variance-vs-level diagnostic, because that is the picture that
    makes a log transform make sense to someone who has never heard of Box–Cox.

Grounding rule (R0.5): every number in `because` is in `facts_used`, and a test
asserts that. The AI assistant may only cite `facts_used`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.advisor.rules_prep import (
    STRONG_SEASONALITY,
    PrepContext,
    Rule,
    UNCONDITIONAL,
    rank,
)
from app.charts import primitives as P
from app.core.schemas import (
    AdvisorReport,
    Confidence,
    DataClass,
    FigureResult,
    Stage,
    TableColumn,
    TableResult,
)

SNIPPET_HEADER = (
    "# Paste into a notebook with `pip install pandas numpy scipy statsmodels`.\n"
    "import numpy as np, pandas as pd\n"
    "s = pd.read_csv('your_file.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
)


@dataclass
class PrepAdvice:
    report: AdvisorReport
    results: list[Any]
    facts: dict[str, float | int | str | bool]
    context: PrepContext


# --------------------------------------------------------------------------- #
# Diagnostics that only S2 needs
# --------------------------------------------------------------------------- #
def gap_runs(series: pd.Series) -> list[int]:
    """Lengths of every run of consecutive missing observations."""
    runs: list[int] = []
    current = 0
    for missing in series.isna().to_numpy():
        if missing:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def longest_stall_run(series: pd.Series) -> int:
    """Longest run of identical consecutive values (a stall, or a genuinely flat series)."""
    v = series.dropna().to_numpy(dtype=float)
    if len(v) < 2:
        return len(v)
    longest = current = 1
    for a, b in zip(v[:-1], v[1:]):
        current = current + 1 if a == b else 1
        longest = max(longest, current)
    return longest


def boxcox_lambda(series: pd.Series) -> tuple[float | None, tuple[float, float] | None]:
    """
    MLE λ and an approximate 95% interval from the profile likelihood.

    Returning an interval rather than a point is the whole point: λ = 0.12 and λ = 0.6
    both "estimate a λ", and only the interval tells you which transform is defensible.
    """
    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 12 or np.any(y <= 0):
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam = float(stats.boxcox_normmax(y, method="mle", brack=(-2.0, 2.0)))
        return lam, _boxcox_ci(y, lam)
    except Exception:  # pragma: no cover - degenerate input
        return None, None


def _boxcox_ci(y: np.ndarray, lam_hat: float, level: float = 0.95) -> tuple[float, float]:
    """
    Profile-likelihood interval: all λ whose log-likelihood is within χ²₁(0.95)/2.
    """
    n = len(y)
    grid = np.linspace(max(-2.0, lam_hat - 1.5), min(2.0, lam_hat + 1.5), 241)

    def loglik(lam: float) -> float:
        if abs(lam) < 1e-8:
            z = np.log(y)
        else:
            z = (np.power(y, lam) - 1.0) / lam
        s2 = np.var(z, ddof=1)
        if s2 <= 0:
            return -np.inf
        return -(n / 2.0) * np.log(s2) + (lam - 1.0) * np.sum(np.log(y))

    lls = np.array([loglik(l) for l in grid])
    best = float(np.max(lls))
    keep = grid[lls >= best - 0.5 * stats.chi2.ppf(level, 1)]
    if keep.size == 0:                            # pragma: no cover - extreme case
        return (lam_hat, lam_hat)
    return (float(keep.min()), float(keep.max()))


def variance_level_corr(series: pd.Series, bins: int = 8) -> float | None:
    """
    Park–Ruprecht style check: does the local spread grow with the local level?

    The series is cut into consecutive blocks; each block contributes (median, IQR).
    A positive correlation means the variance is a function of the level — which is
    exactly what a log transform assumes and fixes.
    """
    v = series.dropna()
    if len(v) < bins * 4:
        return None
    blocks = np.array_split(v.to_numpy(dtype=float), bins)
    lv = np.array([np.median(b) for b in blocks])
    sp = np.array([np.subtract(*np.percentile(b, [75, 25])) for b in blocks])
    if np.std(sp) == 0 or np.std(lv) == 0:
        return None
    return float(np.clip(np.corrcoef(lv, sp)[0, 1], -1.0, 1.0))


def remainder_diagnostics(series: pd.Series, period: int | None) -> dict[str, float | None]:
    """
    |ρ₁| of the STL remainder and the signal-to-noise summary used by the smoothing rule.
    """
    out: dict[str, float | None] = {
        "remainder_acf1": None, "remainder_sd": None, "signal_to_noise": None,
    }
    s = series.astype(float)
    if s.notna().sum() < 8:
        return out
    try:
        if period and period > 1 and s.notna().sum() >= 2 * period + 2:
            from statsmodels.tsa.seasonal import STL

            res = STL(s.interpolate(limit_direction="both"), period=period,
                      robust=True, seasonal=7).fit()
            resid = np.asarray(res.resid, dtype=float)
            sig = np.asarray(res.seasonal, dtype=float) + np.asarray(res.trend, dtype=float)
        else:
            trend = s.interpolate(limit_direction="both").rolling(5, center=True, min_periods=2).mean()
            resid = (s - trend).dropna().to_numpy(dtype=float)
            sig = trend.dropna().to_numpy(dtype=float)
        resid = resid[np.isfinite(resid)]
        sig = sig[np.isfinite(sig)]
        if resid.size > 3:
            sd_r = float(np.std(resid, ddof=1))
            out["remainder_sd"] = sd_r
            if sd_r > 0:
                r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if resid.size > 2 else 0.0
                out["remainder_acf1"] = abs(r1) if np.isfinite(r1) else None
                out["signal_to_noise"] = float(np.var(sig, ddof=1) / (sd_r**2)) if sig.size > 1 else None
    except Exception:  # pragma: no cover - very short series
        pass
    return out


def seasonal_diff_hint(series: pd.Series, period: int | None) -> bool | None:
    """
    Would a seasonal difference remove a persistent spike at the seasonal lag?

    Deliberately a *hint*, not a test: the seasonal unit-root battery belongs to S5.
    """
    if not period or period < 2:
        return None
    v = series.dropna()
    if len(v) < 3 * period:
        return None
    d = v.diff(period).dropna()
    if d.std() == 0:
        return None
    acf_d = d.autocorr(lag=period)
    acf_y = v.autocorr(lag=period)
    if acf_y is None or acf_d is None or not np.isfinite(acf_y) or not np.isfinite(acf_d):
        return None
    # seasonal difference "works" when the seasonal autocorrelation collapses
    return bool(acf_d < 0.35 * max(acf_y, 1e-9))


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #
def build_context(
    series: pd.Series,
    *,
    eda_facts: dict[str, Any],
    name: str,
    frequency_label: str,
    period: int | None,
    data_class: str,
    applied_ops: tuple[str, ...] = (),
    n_unique: int | None = None,
) -> PrepContext:
    s = series.astype(float)
    runs = gap_runs(s)
    lam, ci = boxcox_lambda(s) if data_class != "intermittent" else (None, None)
    rem = remainder_diagnostics(s, period)
    values = s.dropna()

    ctx = PrepContext(
        n=int(values.size),
        period=period,
        frequency_label=frequency_label,
        data_class=data_class,
        series_name=name,
        n_missing=int(s.isna().sum()),
        pct_zeros=float(eda_facts.get("pct_zeros", 0.0) or 0.0),
        # S1 supplies these, but S2 must not depend on S1 having been told the same
        # story: if a treatment introduced zeros, the count has to follow.
        n_zeros=int(eda_facts.get("n_zeros", int((values == 0).sum())) or 0),
        n_negatives=int(eda_facts.get("n_negatives", int((values < 0).sum())) or 0),
        n_gaps=int(eda_facts.get("n_gaps", 0) or 0),
        longest_gap=max(runs) if runs else int(eda_facts.get("longest_gap", 0) or 0),
        n_outliers=int(eda_facts.get("n_outliers", 0) or 0),
        n_level_shifts=int(eda_facts.get("n_level_shifts", 0) or 0),
        seasonal_strength=_as_float(eda_facts.get("seasonal_strength")),
        seasonal_amplitude=_as_float(eda_facts.get("seasonal_amplitude")),
        seasonal_amplitude_growing=_as_bool_or_none(eda_facts.get("seasonal_amplitude_growing")),
        seasonal_p=_as_float(eda_facts.get("seasonal_p")),
        cv=float(eda_facts.get("cv", 0.0) or 0.0),
        skewness=float(eda_facts.get("skewness", 0.0) or 0.0),
        mean=_as_float(eda_facts.get("mean")),
        min_value=_as_float(eda_facts.get("min_value")),
        trend_r2=float(eda_facts.get("trend_r2", 0.0) or 0.0),
        trend_pct_over_window=float(eda_facts.get("trend_pct_over_window", 0.0) or 0.0),
        variance_ratio=_as_float(eda_facts.get("variance_ratio")),
        n_unique=int(n_unique if n_unique is not None else values.nunique()),
        stall_run=longest_stall_run(s),
        gap_runs=runs,
        boxcox_lambda=lam,
        boxcox_ci=ci,
        variance_level_corr=variance_level_corr(s),
        remainder_acf1=rem["remainder_acf1"],
        remainder_sd=rem["remainder_sd"],
        signal_to_noise=rem["signal_to_noise"],
        seasonal_diff_needed=seasonal_diff_hint(s, period),
        is_positive=bool(values.size and values.min() > 0),
        already_applied=tuple(applied_ops),
    )
    return ctx


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _as_bool_or_none(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return None


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #
def advise_preprocessing(
    series: pd.Series,
    *,
    eda_facts: dict[str, Any],
    name: str = "your series",
    frequency_label: str = "observation",
    period: int | None = None,
    data_class: DataClass | str = DataClass.CONTINUOUS,
    applied_ops: tuple[str, ...] = (),
    audience: str = "beginner",
) -> PrepAdvice:
    """Run S2. Pure: takes facts and a series, returns advice. Stores nothing."""
    data_class_str = data_class.value if isinstance(data_class, DataClass) else str(data_class)
    ctx = build_context(
        series, eda_facts=eda_facts, name=name, frequency_label=frequency_label,
        period=period, data_class=data_class_str, applied_ops=applied_ops,
    )

    rules = rank(ctx)
    blockers = [r for r in rules if r.kind == "blocker"]
    recommendations = [r for r in rules if r.kind != "blocker"]

    facts_used = _facts_used(ctx)
    headline = _headline(ctx, blockers, recommendations)

    report = AdvisorReport(
        stage=Stage.S2_ADVISE_PREP,
        headline=headline,
        recommendations=[_rule_out(r, ctx) for r in recommendations],
        blockers=[_rule_out(r, ctx) for r in blockers],
        facts_used=facts_used,
        confidence=_overall_confidence(blockers, recommendations),
        next_stage_ready=not blockers,
        narrative=_narrative(ctx, blockers, recommendations),
    )

    results: list[Any] = [_evidence_table(ctx, report)]
    fig = variance_level_chart(series, ctx)
    if fig is not None:
        results.append(fig)

    return PrepAdvice(report=report, results=results, facts=facts_used, context=ctx)


def _rule_out(rule: Rule, ctx: PrepContext) -> Any:
    from app.core.schemas import RuleOut

    return RuleOut(**rule.to_out(ctx))


def _facts_used(c: PrepContext) -> dict[str, float | int | str | bool]:
    """Every number a recommendation is allowed to cite. Nothing else reaches the AI."""
    raw: dict[str, Any] = {
        "n": c.n,
        "period": c.period if c.period is not None else "none",
        "frequency": c.frequency_label,
        "data_class": c.data_class,
        "n_missing": c.n_missing,
        "longest_gap": c.longest_gap,
        "n_gap_runs": len(c.gap_runs),
        "n_zeros": c.n_zeros,
        "n_negatives": c.n_negatives,
        "pct_zeros": c.pct_zeros,
        "n_outliers": c.n_outliers,
        "n_level_shifts": c.n_level_shifts,
        "seasonal_strength": c.seasonal_strength,
        "seasonal_amplitude": c.seasonal_amplitude,
        "seasonal_growing": bool(c.seasonal_amplitude_growing) if c.seasonal_amplitude_growing is not None else "untested",
        "seasonal_p": c.seasonal_p,
        "strong_seasonality_threshold": STRONG_SEASONALITY,
        "cv": c.cv,
        "skewness": c.skewness,
        "trend_r2": c.trend_r2,
        "trend_pct_over_window": c.trend_pct_over_window,
        "variance_ratio": c.variance_ratio,
        # the sentence states growth in percent; register that number, do not compute it
        "variance_growth_pct": ((c.variance_ratio - 1.0) * 100.0
                                if c.variance_ratio is not None else None),
        "variance_level_corr": c.variance_level_corr,
        "boxcox_lambda": c.boxcox_lambda,
        # the sentences say "95% interval" — that confidence level is a computed choice,
        # so it belongs in facts_used like any other number
        "boxcox_ci_level_pct": 95 if c.boxcox_ci else None,
        "boxcox_ci_low": c.boxcox_ci[0] if c.boxcox_ci else None,
        "boxcox_ci_high": c.boxcox_ci[1] if c.boxcox_ci else None,
        "remainder_acf1": c.remainder_acf1,
        "signal_to_noise": c.signal_to_noise,
        "stall_run": c.stall_run,
        "n_unique": c.n_unique,
        "is_positive": c.is_positive,
        "seasonal_difference_helps": (
            bool(c.seasonal_diff_needed) if c.seasonal_diff_needed is not None else "untested"
        ),
    }
    out: dict[str, float | int | str | bool] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, (bool, str)):
            out[k] = v
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            fv = float(v)
            if math.isfinite(fv):
                out[k] = fv
    return out


def _headline(c: PrepContext, blockers: list[Rule], recs: list[Rule]) -> str:
    if blockers:
        return (
            f"{len(blockers)} decision{'s' if len(blockers) != 1 else ''} must be made before "
            "preprocessing can finish — the first is: " + blockers[0].recommend.lower() + "."
        )
    if not recs:
        return (
            f"Nothing needs doing: {c.n} observations, no gaps, and no transform is justified "
            "by the variance diagnostics."
        )
    top = recs[0]
    return f"{len(recs)} suggestion{'s' if len(recs) != 1 else ''} — the one that matters most is: {top.recommend.lower()}."


def _overall_confidence(blockers: list[Rule], recs: list[Rule]) -> Confidence:
    if blockers or any(r.confidence == Confidence.HIGH for r in recs):
        return Confidence.HIGH
    if any(r.confidence == Confidence.MEDIUM for r in recs):
        return Confidence.MEDIUM
    return Confidence.LOW


def _narrative(c: PrepContext, blockers: list[Rule], recs: list[Rule]) -> str:
    lines = [f"Read from {c.n} observations of {c.series_name}."]
    if blockers:
        lines.append(
            "Blocking: " + "; ".join(b.recommend.lower() for b in blockers) + "."
        )
    if recs:
        lines.append(
            "Suggested, in order of consequence: "
            + "; ".join(r.recommend.lower() for r in recs[:3]) + "."
        )
    lines.append(
        "Nothing has been changed yet — every suggestion is applied in Step 3, one click at a time, "
        "and can be undone."
    )
    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _evidence_table(c: PrepContext, report: AdvisorReport) -> TableResult:
    """The evidence line for each recommendation, in the same order as the advice."""
    rows: list[dict[str, Any]] = []

    def add(check: str, finding: str, evidence: str, action: str) -> None:
        rows.append({"check": check, "finding": finding, "evidence": evidence, "action": action})

    add(
        "Missing values",
        "none" if c.n_missing == 0 else f"{c.n_missing} missing in {len(c.gap_runs)} run(s)",
        f"longest run {c.longest_gap} of {c.n} observations",
        "nothing needed" if c.n_missing == 0 else "see Imputation below",
    )
    add(
        "Variance vs level",
        "spread grows with level" if (c.variance_ratio or 1) > 1.25 else "spread roughly stable",
        f"rolling SD ratio {_f(c.variance_ratio)}; level–spread correlation {_f(c.variance_level_corr)}",
        "transform likely" if (c.variance_ratio or 1) > 1.25 else "no transform needed for variance",
    )
    add(
        "Box–Cox λ",
        "log-like" if c.boxcox_lambda is not None and abs(c.boxcox_lambda) <= 0.25 else
        ("root-like" if c.boxcox_lambda is not None and 0.25 < c.boxcox_lambda <= 0.75 else
         ("no transform" if c.boxcox_lambda is not None and abs(c.boxcox_lambda - 1) <= 0.25 else "non-standard")),
        f"λ = {_f(c.boxcox_lambda)}"
        + (f" (95% CI {_f(c.boxcox_ci[0])} to {_f(c.boxcox_ci[1])})" if c.boxcox_ci else ""),
        "log" if c.boxcox_lambda is not None and abs(c.boxcox_lambda) <= 0.25 else
        ("square root" if c.boxcox_lambda is not None and 0.25 < c.boxcox_lambda <= 0.75 else "none"),
    )
    # `seasonal_diff_needed` is a *tri-state*: True, False, or None when the hint could not be
    # computed (no usable period, or fewer than three full cycles). Interpolating it directly
    # printed "seasonal difference helps: None" — Python's repr of a question that was never
    # answered, in a sentence addressed to a non-programmer.
    hint = (
        "untested — fewer than three full seasonal cycles"
        if c.seasonal_diff_needed is None
        else ("yes" if c.seasonal_diff_needed else "no")
    )
    add(
        "Seasonality",
        f"F_S {_f(c.seasonal_strength)}" + (" — strong" if (c.seasonal_strength or 0) >= STRONG_SEASONALITY else " — not strong"),
        f"Kruskal–Wallis p {_p(c.seasonal_p)}; seasonal difference helps: {hint}",
        "keep seasonal terms" if (c.seasonal_strength or 0) >= STRONG_SEASONALITY else "no seasonal terms",
    )
    add(
        "Outliers",
        f"{c.n_outliers} flagged" + (f" ({c.n_level_shifts} level shift)" if c.n_level_shifts else ""),
        "flagged in Step 1, nothing removed",
        "model as intervention dummies, do not delete",
    )
    add(
        "Smoothing",
        "noisy" if c.signal_to_noise is not None and c.signal_to_noise < 3 else "signal dominates",
        f"signal-to-noise {_f(c.signal_to_noise)}; |ρ₁| of remainder {_f(c.remainder_acf1)}",
        "presentation only — never before modelling",
    )
    add(
        "Stalled values",
        f"run of {c.stall_run}" if c.stall_run >= 12 else "none",
        f"{c.n_unique} distinct values in {c.n} observations",
        "check with the data owner" if c.stall_run >= 12 else "nothing needed",
    )

    return TableResult(
        id="table.prep.evidence",
        stage=Stage.S2_ADVISE_PREP,
        title="Evidence behind the advice",
        plain="One line per decision, with the number that triggered it.",
        why="So you can disagree with a recommendation on the evidence rather than on trust.",
        method=(
            "Missingness runs from the observed index; variance diagnostics from rolling SD and a "
            "blocked Park–Ruprecht correlation; λ from profile-likelihood Box–Cox; seasonality from S1."
        ),
        citations=["Box & Cox (1964)", "Park & Ruprecht (1987)", "Hyndman & Athanasopoulos (2021) §3.1"],
        reproducible_snippet=SNIPPET_HEADER + (
            "from scipy import stats\n"
            "y = s.dropna().values\n"
            "print('lambda', stats.boxcox_normmax(y, method='mle'))\n"
            "print('rolling sd ratio', s.rolling(12).std().iloc[-12:].mean() / s.rolling(12).std().iloc[:12].mean())\n"
        ),
        columns=[
            TableColumn(key="check", label="Decision", dtype="string"),
            TableColumn(key="finding", label="What the data shows", dtype="string"),
            TableColumn(key="evidence", label="The number", dtype="string"),
            TableColumn(key="action", label="What that implies", dtype="string"),
        ],
        rows=rows,
        caption=f"{len(report.recommendations) + len(report.blockers)} rule(s) fired on this series.",
        footnote="Every row is computed; none is a default. A row saying 'nothing needed' is a real result.",
    )


def _p(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def _f(x: float | None, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    ax = abs(x)
    if ax >= 1000:
        return f"{round(x):,}"
    if ax >= 100:
        return f"{round(x, 1):,}"
    return f"{round(x, digits):g}"


def variance_level_chart(series: pd.Series, ctx: PrepContext) -> FigureResult | None:
    """
    Scatter of local level vs local spread, with the fitted line — the picture that
    turns "λ = −0.03" into "the spread grows with the level, so take logs".
    """
    v = series.dropna()
    if len(v) < 24:
        return None
    blocks = np.array_split(v, min(10, max(4, len(v) // 12)))
    lv = np.array([float(np.median(b)) for b in blocks])
    sp = np.array([float(np.subtract(*np.percentile(b, [75, 25]))) for b in blocks])
    if np.all(sp == 0):
        return None

    chart = P.scatter_with_fit(
        x=lv, y=sp,
        title="Does the spread grow with the level?",
        subtitle=(
            "Each dot is a block of the series. If the dots slope upwards, a transform "
            "will steady the variance — a log is the usual choice."
        ),
        x_title="Typical level in the block",
        y_title="Spread within the block (interquartile range)",
        point_labels=[f"block {i + 1}" for i in range(len(lv))],
    )

    slope = corr = None
    if len(lv) >= 3 and np.std(lv) > 0 and np.std(sp) > 0:
        beta = np.polyfit(lv, sp, 1)
        slope = float(beta[0])
        corr = float(np.corrcoef(lv, sp)[0, 1])

    verdict = (
        "grows with the level, so the variance is not constant"
        if (corr or 0) > 0.5 else
        "does not grow with the level, so the variance looks stable"
        if abs(corr or 0) < 0.3 else
        "drifts with the level, weakly"
    )
    return FigureResult(
        id="fig.prep.variance_level",
        stage=Stage.S2_ADVISE_PREP,
        title="Variance vs level",
        plain=f"The spread {verdict}.",
        why="This single picture decides whether a transform is honest or decorative.",
        method=(
            "The series is cut into equal consecutive blocks; each contributes its median level "
            "and its interquartile range. Pearson correlation and an OLS line are computed on those."
        ),
        citations=["Park & Ruprecht (1987)"],
        caveats=[
            "Blocking smooths over regime changes; a series that is stable in each half but "
            "different between them can still show a slope here."
        ],
        reproducible_snippet=SNIPPET_HEADER + (
            "import numpy as np\n"
            "blocks = np.array_split(s.dropna().values, 10)\n"
            "print([(np.median(b), np.percentile(b,75)-np.percentile(b,25)) for b in blocks])\n"
        ),
        figure_json=chart.figure_json,
        alt_text=chart.alt_text,
        caption=(
            f"Correlation between block level and block spread: {_f(corr)}"
            + (f"; slope {_f(slope, 4)}" if slope is not None else "")
            + f". Box–Cox λ = {_f(ctx.boxcox_lambda)}."
        ),
    )


__all__ = [
    "PrepAdvice",
    "advise_preprocessing",
    "build_context",
    "gap_runs",
    "longest_stall_run",
    "boxcox_lambda",
    "variance_level_corr",
    "remainder_diagnostics",
    "seasonal_diff_hint",
]
