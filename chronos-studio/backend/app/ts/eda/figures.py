"""
ts/eda/figures.py — the EDA figure set (§5.2, group 1 of the §5.10 catalog).

Most functions here compose `app/charts/primitives.py`; the two bespoke figures
start from `app.charts.theme.apply`/`to_json`. Colours, fonts, fills and dash
patterns all come from the single theme. Each returns a `Chart` carrying figure
JSON, generated alt text and the data behind the plot.

Two rules worth stating, because they look like omissions:

*   The time plot is **not** smoothed by default. Smoothing the series you are
    about to model destroys autocorrelation structure — showing a smoothed line as
    *the* series is a beginner trap we refuse to set.
*   Missing observations are drawn as breaks, never interpolated silently.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.charts import primitives as P
from app.charts.theme import apply, colours as theme_colours, to_json, with_alpha
from app.ts.eda.outliers import Outlier
from app.ts.eda.seasonal import SeasonalProfile, labels_for


class ChartLike(Protocol):
    figure_json: dict[str, Any]
    alt_text: str
    data_behind: list[dict[str, Any]]


# --------------------------------------------------------------------------- #
def series_chart(series: pd.Series, name: str) -> P.Chart:
    """The observed series, gaps visible, with the rolling level as context."""
    return P.time_series(
        series,
        title=f"{name} over time",
        subtitle="Line breaks are missing observations — nothing has been filled in.",
        y_title="",
        overlays=[],
        height=340,
    )


def rolling_chart(series: pd.Series, window: int) -> P.Chart:
    return P.rolling_panel(
        series, window,
        title=f"Rolling mean and spread ({window}-period window)",
        subtitle=(
            "A rising or falling moving average indicates a trend; a widening band "
            "indicates variance that is not constant."
        ),
    )


def seasonal_profile_chart(profile: SeasonalProfile) -> P.Chart:
    """Bars of the position means, with the overall mean as the reference line."""
    overall = float(np.mean(profile.means)) if profile.means else 0.0
    deviations = [float(m - overall) for m in profile.means]
    c = theme_colours("light")
    bar_colours = [
        with_alpha(c["series"], 0.85) if d >= 0 else with_alpha(c["trend"], 0.85)
        for d in deviations
    ]
    return P.bars(
        profile.labels,
        [round(m, 6) for m in profile.means],
        title="Seasonal profile",
        subtitle=(
            f"Peak in {profile.peak_label}, trough in {profile.trough_label}; "
            f"swing of {profile.amplitude:,.4g} units"
            + (f" ({profile.amplitude_pct_of_mean:.0f}% of the average level)."
               if np.isfinite(profile.amplitude_pct_of_mean) else ".")
        ),
        y_title="Average value",
        reference=overall,
        colours_override=bar_colours,
        height=330,
    )


def seasonal_box_chart(series: pd.Series, profile: SeasonalProfile) -> P.Chart:
    """Box plot per calendar position — the spread question the means cannot answer."""
    c = theme_colours("light")
    s = series.dropna()
    pos = np.arange(len(s)) % profile.period
    fig = go.Figure()
    for i, label in enumerate(profile.labels):
        vals = s.to_numpy()[pos == i]
        if vals.size == 0:
            continue
        fig.add_trace(go.Box(
            y=vals, name=label, boxpoints="outliers",
            marker={"color": c["series"], "size": 3.5, "opacity": 0.7},
            line={"color": c["series"], "width": 1.1},
            fillcolor=with_alpha(c["series"], 0.13),
            hovertemplate=f"{label}<br>%{{y:,.4~g}}<extra></extra>",
        ))
    apply(fig, title="Distribution by season",
          subtitle="Boxes that barely overlap indicate a season that is genuinely different, not just noisy.",
          y_title="Observed value", height=330, legend=False, hovermode="closest")
    fig.update_layout(margin={"l": 56, "r": 16, "t": 62, "b": 40})
    alt = (
        f"Box plots of the series by {profile.period} seasonal positions. The median ranges from "
        f"{min(profile.means):,.4g} to {max(profile.means):,.4g} across positions."
    )
    return P.Chart(to_json(fig), alt, [])


def calendar_heatmap_chart(matrix: pd.DataFrame) -> P.Chart:
    """Year × position heatmap — where a drifting seasonal pattern becomes visible."""
    if matrix.empty:
        return P.Chart({"data": [], "layout": {}}, "No calendar structure to display.", [])
    return P.heatmap(
        matrix,
        title="Year × season heatmap",
        subtitle="Each row is a year; a season that shifts over time shows up as a diagonal streak.",
        x_title="Seasonal position",
        y_title="Year",
        value_label="Value",
        height=max(260, 26 * len(matrix.index) + 90),
    )


def seasonal_subseries_chart(series: pd.Series, profile: SeasonalProfile) -> P.Chart:
    """One small plot per position: is this month drifting against its own history?"""
    s = series.dropna()
    pos = np.arange(len(s)) % profile.period
    frames: list[tuple[str, pd.Series]] = []
    for i, label in enumerate(profile.labels):
        vals = s.to_numpy()[pos == i]
        if vals.size:
            frames.append((label, pd.Series(vals, index=np.arange(len(vals)))))
    n_cols = 6 if profile.period >= 10 else (4 if profile.period >= 6 else profile.period)
    return P.small_multiples(
        frames,
        title="Seasonal subseries plot",
        subtitle="Each panel is one calendar position across the years; a flat line means a stable season.",
        n_cols=n_cols,
        marker_mode=True,
        height_per_row=104,
    )


def distribution_chart(series: pd.Series) -> P.Chart:
    return P.distribution_panel(
        series,
        title="Distribution of the observed values",
        subtitle=(
            "The Q–Q plot compares your values with a normal distribution. "
            "Departures matter for intervals and for skewed data — not for the point forecast itself."
        ),
    )


def lag_plot_chart(series: pd.Series, period: int | None) -> P.Chart:
    """
    Lag plots for lags 1..6 plus one seasonal lag.

    A tight linear cloud at lag k means the value k periods ago predicts today.
    """
    c = theme_colours("light")
    s = series.dropna()
    n = len(s)
    # A lag needs at least three (past, now) pairs before its correlation means anything. The
    # calendar period can be far larger than the series — a 100-minute-long series has a nominal
    # daily period of 1440 — and appending it produced an empty panel annotated "r = nan", which
    # reads as a broken number rather than as "this lag cannot be measured here".
    lags = [k for k in (1, 2, 3, 4, 5, 6) if k < n - 2]
    if period and period > 6 and period < n - 2:
        lags.append(period)

    n_cols = 4
    n_rows = max(1, int(np.ceil(len(lags) / n_cols)))
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"lag {k}" for k in lags] or ["No usable lag"],
        vertical_spacing=0.14, horizontal_spacing=0.07,
    )
    for i, lag in enumerate(lags):
        r, q = divmod(i, n_cols)
        x = s.iloc[:-lag].to_numpy()
        y = s.iloc[lag:].to_numpy()
        # A constant window has zero variance, so np.corrcoef returns nan. That is a refusal, not
        # a correlation of zero: write the reason rather than the string "nan".
        corr: float | None = None
        if len(x) > 2 and float(np.std(x)) > 0 and float(np.std(y)) > 0:
            value = float(np.corrcoef(x, y)[0, 1])
            corr = value if np.isfinite(value) else None
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=f"lag {lag}",
            marker={"color": c["series"], "size": 4, "opacity": 0.65},
            showlegend=False,
            hovertemplate=f"lag {lag}<br>past %{{x:,.4~g}}<br>now %{{y:,.4~g}}<extra></extra>",
        ), row=r + 1, col=q + 1)
        fig.update_xaxes(gridcolor=c["grid"], tickfont={"size": 8.5}, row=r + 1, col=q + 1)
        fig.update_yaxes(gridcolor=c["grid"], tickfont={"size": 8.5}, row=r + 1, col=q + 1)
        fig.update_annotations(font_size=10)
        # annotate the correlation, so the reading is not left to the eye
        fig.add_annotation(
            xref=f"x{i + 1 if i else ''} domain", yref=f"y{i + 1 if i else ''} domain",
            x=0.98, y=0.04,
            text=f"r = {corr:.2f}" if corr is not None else "r = n/a — constant window",
            showarrow=False,
            font={"size": 9.5, "color": c["muted"]}, xanchor="right", row=r + 1, col=q + 1,
        )
    if not lags:
        fig.add_annotation(
            text="Too few observations for a lag plot.", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font={"size": 12, "color": c["muted"]},
        )

    apply(fig, title="Lag plots", subtitle="A tight diagonal cloud means the past predicts the present.",
          height=max(240, 150 * n_rows + 70), legend=False, hovermode="closest")
    fig.update_layout(margin={"l": 50, "r": 18, "t": 64, "b": 34})

    lags_txt = ", ".join(str(k) for k in lags) or "none"
    alt = (
        f"Lag plot matrix for lags {lags_txt}. Each panel plots the value at time t against "
        f"time t minus the lag; the correlation of each cloud is annotated in its corner, and a "
        f"lag with too few pairs or a constant window is omitted or marked 'not available'."
        + (f" Lags larger than the {n}-observation series were left out." if period and period >= n - 2 else "")
    )
    return P.Chart(to_json(fig), alt, [])


def outlier_chart(series: pd.Series, outliers: list[Outlier]) -> P.Chart:
    """The series with anomalies marked, and the type shown by marker symbol."""
    symbols = {
        "spike": "diamond",
        "level_shift": "square",
        "temporary_change": "triangle-up",
        "isolated_run": "circle",
    }
    overlays: list[dict[str, Any]] = []
    for kind, sym in symbols.items():
        pts = [o for o in outliers if o.type == kind]
        if not pts:
            continue
        idx = pd.DatetimeIndex([pd.Timestamp(o.time) for o in pts])
        overlays.append({
            "y": pd.Series([o.value for o in pts], index=idx),
            "name": kind.replace("_", " "),
            "kind": "markers",
            "symbol": sym,
            "size": 9,
        })
    subtitle = (
        f"{len(outliers)} flagged {'observation' if len(outliers) == 1 else 'observations'}; marker shape shows the type."
        if outliers else "No observations were flagged."
    )
    return P.time_series(
        series,
        title="Outliers and structural changes",
        subtitle=subtitle,
        overlays=overlays,
        height=340,
    )


__all__ = [
    "ChartLike", "series_chart", "rolling_chart", "seasonal_profile_chart",
    "seasonal_box_chart", "calendar_heatmap_chart", "seasonal_subseries_chart",
    "distribution_chart", "lag_plot_chart", "outlier_chart",
]
