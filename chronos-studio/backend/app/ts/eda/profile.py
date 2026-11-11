"""
ts/eda/profile.py — the S1 stage: run EDA and emit contract-shaped results.

This is the module the API calls. It assembles the tables and figures built by
`descriptives`, `seasonal`, `outliers`, `figures` and `narrate` into
`TableResult` / `FigureResult` / `StatResult` / `NarrativeResult` objects.

Rules honoured here:
  * S1 runs automatically on load — no clicks (§5.2).
  * Every figure is exportable and carries generated alt text.
  * The advisor (Part 3) receives `facts`, a flat dict of computed numbers. Nothing
    is recommended at S1; the facts are the raw material for S2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from app.core.formatting import p_phrase

from app.core.schemas import (
    DataClass,
    FigureResult,
    NarrativeResult,
    StatResult,
    TableColumn,
    TableResult,
    Stage,
)
from app.ts.eda import figures as F
from app.ts.eda.descriptives import describe, rolling_cv_stability, runs_test
from app.ts.eda.narrate import NarrativeFacts, narrate
from app.ts.eda.outliers import detect_outliers
from app.ts.eda.outliers import Outlier
from app.ts.eda.seasonal import (
    labels_for,
    month_year_matrix,
    seasonal_profile,
    trend_hint,
)


@dataclass
class EdaOutput:
    results: list[Any]
    facts: dict[str, Any]
    caveats: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: The flagged observations themselves. Downstream models build intervention dummies from these;
    #: a count alone cannot say *where* the shock was. Empty list means "none flagged", and that is
    #: only trustworthy because the outlier step always returns a list rather than `None`.
    outliers: list[Any] = field(default_factory=list)


STAGE = Stage.S1_PROFILE
SNIPPET_HEADER = "# Reproduce this from the raw data with:\n#   pip install pandas statsmodels plotly\n"


def _base_kwargs(result_id: str, title: str, plain: str, why: str,
                 method: str, snippet: str, **extra: Any) -> dict[str, Any]:
    return dict(
        id=result_id, stage=STAGE, title=title, plain=plain, why=why,
        method=method, reproducible_snippet=snippet, inputs=extra.pop("inputs", {}),
        citations=extra.pop("citations", []), caveats=extra.pop("caveats", []),
        warnings=extra.pop("warnings", []), **extra,
    )


def run_eda(
    series: pd.Series,
    *,
    name: str,
    frequency_label: str = "",
    freq_code: str = "",
    period: int | None = None,
    data_class: DataClass = DataClass.CONTINUOUS,
    window: int | None = None,
    audience: str = "beginner",
) -> EdaOutput:
    """Produce every S1 result for one series."""
    s = series.copy()
    s.index = pd.DatetimeIndex(s.index)
    n = int(s.notna().sum())
    if n < 4:
        return EdaOutput(results=[], facts={},
                         warnings=["Too few observations for exploratory analysis."])

    results: list[Any] = []
    facts: dict[str, Any] = {"n": n, "n_missing": int(s.isna().sum()), "period": period}
    caveats: list[str] = []
    warnings: list[str] = []

    d = describe(s, frequency_label, period)
    profile = seasonal_profile(s, period, freq_code)
    outliers, outlier_summary = detect_outliers(s, period)
    trend = trend_hint(s)
    stab = rolling_cv_stability(s, window or max(3, (period or 4)))
    runs = runs_test(s)

    # ------------------------------------------------------------------ #
    # 1. The "just tell me what this is" card — always first
    # ------------------------------------------------------------------ #
    nfacts = NarrativeFacts(
        series_name=name, n=n, n_missing=d.n_missing,
        time_start=d.time_start, time_end=d.time_end,
        frequency_label=frequency_label or "observation", period=period,
        data_class=data_class,
        first=d.first, last=d.last, pct_change=d.pct_change,
        mean=d.mean, median=d.median, sd=d.sd, cv=d.cv,
        minimum=d.minimum, maximum=d.maximum, skewness=d.skewness,
        n_outliers=outlier_summary.get("n_flagged", 0),
        outlier_types=outlier_summary.get("counts_by_type", {}),
        n_zeros=d.n_zeros, pct_zeros=d.pct_zeros, n_negatives=d.n_negatives,
        n_gaps=int(s.isna().sum()), constant_run=d.n_ties,
        # the span dates are what a reader can verify against the card above; the
        # narration therefore cites the years, not the observation count twice
        start_year=d.time_start[:4], end_year=d.time_end[:4],
        seasonality={
            "available": profile.available,
            "strength": profile.strength,
            "peak_label": profile.peak_label,
            "trough_label": profile.trough_label,
            "amplitude": profile.amplitude,
            "amplitude_growing": profile.amplitude_growing,
            "kw_p": profile.kw_p_value,
        },
        variance_stability={
            "available": bool(stab.get("available")),
            "ratio": stab.get("ratio"),
            "early_sd": stab.get("early_sd"),
            "late_sd": stab.get("late_sd"),
        },
        trend_hint={k: v for k, v in trend.items() if k != "plain"},
        persistence={"runs_z": runs.get("z"), "runs_p": runs.get("p_value")}
        if runs.get("available") else {},
    )
    paragraphs = narrate(nfacts, audience=audience)
    results.append(NarrativeResult(**_base_kwargs(
        "narr.series_description",
        "What this series is",
        "A plain-language summary of the shape, seasonality and data quality of your series.",
        "Read this first: it is the fastest way to know whether the analysis that follows "
        "matches what you expected.",
        "Deterministic template filled from the computed descriptives, seasonal profile, "
        "outlier scan and variance-stability check (no language model).",
        SNIPPET_HEADER + "# See the numbers in the Descriptives and Seasonality cards below.",
        paragraphs=paragraphs, audience=audience, ai_drafted=False,
        numbers_cited=nfacts.numbers(),
    )))

    # ------------------------------------------------------------------ #
    # 2. Descriptives table + a stat result for the headline
    # ------------------------------------------------------------------ #
    results.append(TableResult(**_base_kwargs(
        "table.descriptives",
        "Descriptive statistics",
        "A one-page summary of level, spread, shape and change over the window.",
        "These are the numbers you will quote in the first paragraph of a results section — "
        "and the baseline that any model has to improve on.",
        "pandas/scipy on the observed values; SD uses n−1; CAGR uses calendar years elapsed.",
        SNIPPET_HEADER + (
            "import pandas as pd\n"
            "s = pd.read_csv('your_file.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "print(s.describe())\n"
            "print('skew', s.skew(), 'kurtosis', s.kurt())\n"
        ),
        columns=[
            TableColumn(key="quantity", label="Quantity", dtype="string"),
            TableColumn(key="value", label="Value", dtype="number", align="right"),
            TableColumn(key="note", label="Interpretation", dtype="string"),
        ],
        rows=d.as_rows(),
        caption="Descriptive statistics for the observed series.",
        footnote="SD = standard deviation (n−1). CAGR is only defined when both endpoints are positive.",
        caveats=[d.normality.get("caveat")] if d.normality.get("caveat") else [],
    )))

    if period and period > 1:
        results.append(StatResult(**_base_kwargs(
            "stat.kruskalwal1is_seasonal",
            "Do the seasons differ? (Kruskal–Wallis)",
            "Tests whether values differ systematically between calendar positions.",
            "A non-significant result means there is no seasonal pattern to model at this "
            "period — adding seasonal terms would be fitting noise.",
            "scipy.stats.kruskal on the linearly detrended series grouped by position within "
            "the cycle (detrending is essential: an undetrended rising series hides its own "
            "seasonality from this test).",
            SNIPPET_HEADER + (
                "from scipy import stats\n"
                "pos = range(len(s)) % " + str(period) + "\n"
                "groups = [s.values[i::" + str(period) + "] for i in pos]\n"
                "print(stats.kruskal(*groups))\n"
            ),
            value=float(profile.kw_statistic or 0.0),
            p_value=profile.kw_p_value,
            statistic_name="Kruskal–Wallis H",
            df=float(period - 1),
            interpretation=(
                f"{p_phrase(profile.kw_p_value, 4)}: "
                + ("the {p}-period pattern is statistically real.".format(p=period)
                   if (profile.kw_p_value or 1) < 0.05 else
                   f"no evidence of a {period}-period pattern.")
            ) if profile.kw_p_value is not None else "",
            band="strong" if (profile.kw_p_value or 1) < 0.001 else (
                "moderate" if (profile.kw_p_value or 1) < 0.05 else "none"),
            caveats=[
                "This tests whether positions differ, not whether a seasonal model will help. "
                "The variance share F_S below is the number to weigh for modelling.",
            ],
        )))

    # ------------------------------------------------------------------ #
    # 3. Seasonality table
    # ------------------------------------------------------------------ #
    if profile.available:
        rows = [
            {
                "position": profile.labels[i] if i < len(profile.labels) else f"P{i + 1}",
                "mean": round(profile.means[i], 4),
                "index": round(profile.indices[i], 4),
                "n_obs": profile.counts[i],
                "deviation_pct": round((profile.indices[i] - 1) * 100, 2),
            }
            for i in range(len(profile.means))
        ]
        results.append(TableResult(**_base_kwargs(
            "table.seasonal_profile",
            "Seasonal profile",
            "The typical value for each position in the seasonal cycle.",
            "This is the table behind the month-by-month chart, and the seasonal index a "
            "reader can apply by hand.",
            "Mean of the observed values at each position within the cycle; seasonal index = "
            "position mean ÷ overall mean.",
            SNIPPET_HEADER + (
                "pos = range(len(s)) % " + str(period) + "\n"
                "import pandas as pd\n"
                "print(pd.Series(s.values).groupby(pos).mean())\n"
            ),
            columns=[
                TableColumn(key="position", label="Position", dtype="string"),
                TableColumn(key="mean", label="Mean value", dtype="number", align="right"),
                TableColumn(key="index", label="Seasonal index", dtype="number", align="right"),
                TableColumn(key="n_obs", label="n", dtype="integer", align="right"),
                TableColumn(key="deviation_pct", label="vs average (%)", dtype="number", align="right"),
            ],
            rows=rows,
            caption=f"Seasonal profile for period {period}."
                     + (f" Peak: {profile.peak_label}; trough: {profile.trough_label}."),
            footnote="Index of 1.000 means the position is exactly at the series average.",
        )))

        results.append(FigureResult(**_base_kwargs(
            "fig.seasonal_profile",
            "Seasonal profile chart",
            "The average value for each month (or quarter, or weekday).",
            "Flat bars mean no seasonality; a clear peak-and-trough shape means seasonality "
            "must be modelled rather than ignored.",
            "Bar chart of the position means from the table above.",
            SNIPPET_HEADER + "print(pd.Series(s.values).groupby(range(len(s)) % "
            + str(period) + ").mean().plot.bar())",
            **_fig(F.seasonal_profile_chart(profile)),
        )))

        matrix = month_year_matrix(s, period, freq_code)
        results.append(FigureResult(**_base_kwargs(
            "fig.month_year_heatmap",
            "Calendar heatmap",
            "Every year as a row, every month (or quarter) as a column.",
            "This is where you see whether the seasonal pattern is stable over time or shifts — "
            "something the averaged profile cannot show.",
            "Pivot of the observed values by year × position, cell value = mean.",
            SNIPPET_HEADER + "# pivot your series by year and season, then plot as a heatmap",
            **_fig(F.calendar_heatmap_chart(matrix)),
        )))

        results.append(FigureResult(**_base_kwargs(
            "fig.seasonal_subseries",
            "Seasonal subseries plot",
            "Each month's history plotted as its own little line.",
            "Shows whether a season is drifting upward or downward relative to its own history — "
            "a stable pattern should look flat within each panel.",
            "Observed values split by position within the cycle.",
            SNIPPET_HEADER + "# group by position and plot each group as a line",
            **_fig(F.seasonal_subseries_chart(s, profile)),
        )))

    # ------------------------------------------------------------------ #
    # 4. Time plots
    # ------------------------------------------------------------------ #
    win = window or max(2, (period or 4))
    results.append(FigureResult(**_base_kwargs(
        "fig.series",
        "The series over time",
        "Your data as a line, with missing observations left as visible gaps.",
        "The single most informative view: level, trend, seasonality, shocks and gaps all "
        "appear here before any model does.",
        "Line plot of the observed series; no smoothing, no aggregation.",
        SNIPPET_HEADER + "s.plot(linewidth=1.5, marker='.', markersize=2)",
        **_fig(F.series_chart(s, name)),
    )))

    results.append(FigureResult(**_base_kwargs(
        "fig.rolling_stats",
        "Rolling mean and spread",
        f"A moving {win}-period average, and how the spread changes across the series.",
        "A moving average that ramps up or down indicates a trend; a widening spread indicates "
        "variance that is not constant, which most models assume it is.",
        f"Rolling mean and rolling SD with window {win} (min_periods = window/2).",
        SNIPPET_HEADER + f"s.rolling({win}).mean().plot()\ns.rolling({win}).std().plot()",
        caveats=[stab["plain"]] if stab.get("available") and stab.get("plain") else [],
        **_fig(F.rolling_chart(s, win)),
    )))

    if profile.available:
        results.append(FigureResult(**_base_kwargs(
            "fig.seasonal_box",
            "Distribution by season",
            "One box per month (or quarter), on the same scale.",
            "Separates a genuine seasonal level difference from year-to-year noise: boxes that "
            "barely overlap are a strong seasonal signal.",
            "Box plots of the observed values grouped by position within the cycle.",
            SNIPPET_HEADER + "# boxplot of your series grouped by month/quarter",
            **_fig(F.seasonal_box_chart(s, profile)),
        )))

    # ------------------------------------------------------------------ #
    # 5. Distribution
    # ------------------------------------------------------------------ #
    results.append(FigureResult(**_base_kwargs(
        "fig.distribution",
        "Distribution of the values",
        "Histogram, density, box plot, cumulative distribution and a normality check.",
        "Level and spread are not the whole story: heavy tails or strong skew change which "
        "models and which summaries are appropriate.",
        "Histogram with a Gaussian KDE overlay, box plot, ECDF and a normal Q–Q plot.",
        SNIPPET_HEADER + "s.hist(bins=30)\ns.plot(kind='kde')",
        caveats=[d.normality.get("plain", "")] if d.normality.get("plain") else [],
        **_fig(F.distribution_chart(s)),
    )))

    if period and period > 1:
        results.append(FigureResult(**_base_kwargs(
            "fig.lag_plots",
            "Lag plots",
            "Today's value plotted against its value 1, 2, 3… periods ago.",
            "Tight straight-line lag plots mean strong persistence; a scattered cloud means "
            "little memory. It is the visual form of the autocorrelation the next steps measure.",
            "Scatter of y(t) against y(t−k) for the first few lags.",
            SNIPPET_HEADER + "pd.plotting.lag_plot(s, lag=1)",
            **_fig(F.lag_plot_chart(s, period)),
        )))

    # ------------------------------------------------------------------ #
    # 6. Outliers
    # ------------------------------------------------------------------ #
    results.append(FigureResult(**_base_kwargs(
        "fig.outliers",
        "Outliers and structural changes",
        "Unusual observations, marked where they sit in the series.",
        "A level shift and a one-off spike need completely different treatment; deleting a "
        "level shift destroys real information.",
        f"Flagged from the {outlier_summary.get('basis', 'series')} using a modified z-score "
        f"threshold of {outlier_summary.get('z_cutoff', 3.5)}.",
        SNIPPET_HEADER + (
            "from statsmodels.tsa.seasonal import STL\n"
            "resid = STL(s.interpolate(), period=" + str(period or 1) + ", robust=True).fit().resid\n"
        ),
        **_fig(F.outlier_chart(s, outliers)),
    )))

    if outliers:
        results.append(TableResult(**_base_kwargs(
            "table.outliers",
            "Flagged observations",
            "Every unusual observation, with what type it looks like.",
            "The type decides the treatment: a spike can be corrected or given a pulse dummy, "
            "a level shift must be modelled, and neither should simply be deleted.",
            f"Detected on the {outlier_summary.get('basis')} with the Iglewicz–Hoaglin robust "
            "z-score; types assigned from the persistence of the deviation.",
            SNIPPET_HEADER + "# see the Outliers figure for the reproduction code",
            columns=[
                TableColumn(key="time", label="Date", dtype="date"),
                TableColumn(key="value", label="Observed", dtype="number", align="right"),
                TableColumn(key="expected", label="Expected", dtype="number", align="right"),
                TableColumn(key="robust_z", label="Robust z", dtype="number", align="right"),
                TableColumn(key="type", label="Type", dtype="string"),
                TableColumn(key="season_flag", label="Unusual for its season?", dtype="string"),
                TableColumn(key="comment", label="What it means", dtype="string"),
            ],
            rows=[
                {
                    "time": o.time[:10],
                    "value": round(o.value, 4),
                    "expected": round(o.expected, 4),
                    "robust_z": round(o.robust_z, 2),
                    "type": o.type.replace("_", " "),
                    "season_flag": "yes" if o.unusual_for_season else "no",
                    "comment": o.plain,
                }
                for o in outliers
            ],
            caption=outlier_summary.get("plain", ""),
            footnote="Expected = value predicted from trend and seasonality alone.",
            caveats=[
                "Outlier detection on a single series is inherently provisional: a level shift "
                "also looks like several outliers in a row. The structural-break tests in "
                "Step 5 are the formal check.",
            ],
        )))

    # ------------------------------------------------------------------ #
    # 7. Battery of small diagnostics
    # ------------------------------------------------------------------ #
    if runs.get("available"):
        results.append(StatResult(**_base_kwargs(
            "stat.runs_test",
            "Runs test (persistence)",
            "Checks whether the series switches direction more or less often than chance.",
            "Too few runs is the signature of a trend or slow cycle; too many is the signature "
            "of a sawtooth, which is also what over-differencing looks like.",
            "Wald–Wolfowitz runs test on the signs of the first differences (normal approximation).",
            SNIPPET_HEADER + "# count sign changes in np.diff(s) and compare to chance",
            value=float(runs["z"]),
            p_value=float(runs["p_value"]),
            statistic_name="Runs z",
            interpretation=runs["plain"],
            band="none" if float(runs["p_value"]) >= 0.05 else "moderate",
        )))

    # ------------------------------------------------------------------ #
    # Facts for the advisor (Part 3) — flat, numeric, no prose
    # ------------------------------------------------------------------ #
    facts.update({
        "cv": d.cv,
        "sd": d.sd,
        "mean": d.mean,
        "median": d.median,
        "skewness": d.skewness,
        "excess_kurtosis": d.excess_kurtosis,
        "pct_change": d.pct_change,
        "cagr": d.cagr if d.cagr is not None else 0.0,
        "n_zeros": d.n_zeros,
        "pct_zeros": d.pct_zeros,
        "n_negatives": d.n_negatives,
        "n_gaps": int(s.isna().sum()),
        "longest_gap": _longest_gap(s),
        "n_outliers": outlier_summary.get("n_flagged", 0),
        "n_level_shifts": outlier_summary.get("counts_by_type", {}).get("level_shift", 0),
        "seasonal_strength": profile.strength if profile.available else 0.0,
        "seasonal_amplitude": profile.amplitude if profile.available else 0.0,
        "seasonal_amplitude_growing": bool(profile.amplitude_growing)
        if profile.amplitude_growing is not None else False,
        "seasonal_p": profile.kw_p_value if profile.available else 1.0,
        "trend_r2": trend.get("r_squared", 0.0) if trend.get("available") else 0.0,
        "trend_pct_over_window": trend.get("pct_over_window", 0.0) if trend.get("available") else 0.0,
        "variance_ratio": stab.get("ratio", 1.0) if stab.get("available") else 1.0,
        "min_value": d.minimum,
        "max_value": d.maximum,
    })

    if d.pct_zeros >= 0.30:
        facts["intermittent"] = True
    if d.n_negatives:
        caveats.append(
            f"{d.n_negatives} {'observation is' if d.n_negatives == 1 else 'observations are'} negative, so a log or multiplicative "
            "transform is not available without shifting the data."
        )
    if period and n < 2 * period:
        warnings.append(
            f"Only {n} observations for a period of {period}: less than two full seasonal "
            "cycles, so any seasonal estimate is fragile."
        )

    return EdaOutput(results=results, facts=facts, caveats=caveats, warnings=warnings,
                     outliers=outliers)


def _fig(chart: F.ChartLike) -> dict[str, Any]:
    """Unpack a chart into FigureResult keyword arguments."""
    return {
        "figure_json": chart.figure_json,
        "alt_text": chart.alt_text,
        "caption": getattr(chart, "caption", ""),
    }


def _longest_gap(s: pd.Series) -> int:
    longest = run = 0
    for flag in s.isna().to_numpy():
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


__all__ = ["run_eda", "EdaOutput"]
