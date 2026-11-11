"""
ts/decompose/figures.py — the decomposition figure set (§5.10, group 2).

The bespoke Plotly figures here start from `app.charts.theme.apply`/`to_json` and
take every colour, dash pattern and font from the single house theme; the bar
and primitive-based figures compose `app.charts.primitives`. No API handler
builds a figure by hand. Each function returns a `Chart` carrying figure JSON,
alt text that repeats the key numbers, and the data behind the plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.charts import primitives as P
from app.charts.theme import DASHES, PALETTE, TITLE_TOP_MARGIN, apply, colours, to_json
from app.ts.decompose.methods import Decomposition


def _component_rows(dec: Decomposition) -> list[dict[str, Any]]:
    """The data behind every decomposition chart: time, observed, and each component."""
    df = pd.DataFrame({
        "time": [str(t) for t in dec.observed.index],
        "observed": dec.observed.to_numpy(dtype=float),
        "trend": np.asarray(dec.trend, dtype=float),
        "seasonal": np.asarray(dec.seasonal, dtype=float),
        "remainder": np.asarray(dec.remainder, dtype=float),
    })
    return df.replace([np.inf, -np.inf], np.nan).to_dict("records")


def component_grid(dec: Decomposition, name: str) -> P.Chart:
    """
    Observed, trend, seasonal and remainder stacked — the canonical decomposition picture.

    The panels share an x-axis and are plotted in order, because the whole point is to see that
    the four add up: the eye should be able to subtract trend from observed and land on seasonal +
    remainder. Each panel keeps its own y-scale (the remainder is often an order of magnitude
    smaller) and the scales are printed in the titles.
    """
    c = colours("light")
    fig = go.Figure()
    # HP and CF produce a *cycle*, not a seasonal component. Naming the panel "Seasonal" taught
    # the opposite of what the method does, so the label follows the decomposition.
    second = "Cycle" if dec.method in {"hp", "cf"} else "Seasonal"
    panels = [
        ("Observed", dec.observed, c["series"]),
        ("Trend", dec.trend, c.get("trend", c["series"])),
        (second, dec.seasonal, c.get("seasonal", c["series"])),
        ("Remainder", dec.remainder, c.get("remainder", c["series"])),
    ]
    for i, (label, series, colour) in enumerate(panels):
        fig.add_trace(go.Scatter(
            x=dec.observed.index, y=np.asarray(series, dtype=float),
            mode="lines", name=label,
            line={"color": colour, "width": 1.5},
            xaxis=f"x{i + 1}" if i else "x", yaxis=f"y{i + 1}" if i else "y",
            hovertemplate=f"{label}: <b>%{{y:,.4~g}}</b><extra></extra>",
        ))

    n = len(panels)
    apply(fig, title=f"{name}: {dec.label} decomposition", mode="light",
          height=150 * n, legend=False, hovermode="x unified")
    layout: dict[str, Any] = {
        "height": 150 * n,
        "showlegend": False,
        "margin": {"l": 56, "r": 16, "t": TITLE_TOP_MARGIN, "b": 26},
        "grid": {"rows": n, "columns": 1, "pattern": "independent"},
        "hovermode": "x unified",
    }
    for i, (label, series, _c) in enumerate(panels):
        suffix = "" if i == 0 else str(i + 1)
        arr = np.asarray(series, dtype=float)
        layout[f"yaxis{suffix}"] = {
            "title": {"text": label, "font": {"size": 11}},
            "range": [float(np.nanmin(arr)), float(np.nanmax(arr))] if np.isfinite(arr).any() else None,
        }
    fig.update_layout(**layout)

    trend_sd = float(np.nanstd(np.asarray(dec.trend, dtype=float)))
    seas_sd = float(np.nanstd(np.asarray(dec.seasonal, dtype=float)))
    rem_sd = float(np.nanstd(np.asarray(dec.remainder, dtype=float)))
    return P.Chart(
        figure_json=to_json(fig),
        alt_text=(
            f"Four stacked panels for {name}: the observed series, then its {dec.label} "
            f"decomposition into trend (variation {trend_sd:.3g}), "
            f"{('cycle' if dec.method in {'hp', 'cf'} else 'seasonal')} "
            f"(variation {seas_sd:.3g}) and remainder (variation {rem_sd:.3g}). "
            f"The panels share a time axis."
        ),
        data_behind=_component_rows(dec),
    )


def side_by_side(decompositions: list[Decomposition], name: str) -> P.Chart:
    """
    Several methods' trend estimates on one axis.

    This is the figure that teaches the most: the choice of decomposition method changes the
    trend you see, and seeing them disagree is more honest than being shown one.
    """
    fig = go.Figure()
    for i, dec in enumerate(decompositions):
        fig.add_trace(go.Scatter(
            x=dec.trend.index, y=np.asarray(dec.trend, dtype=float), mode="lines",
            name=f"{dec.label} trend",
            line={"color": PALETTE[i % len(PALETTE)], "width": 1.6,
                  "dash": DASHES[i % len(DASHES)]},
            hovertemplate=f"{dec.label}: <b>%{{y:,.4~g}}</b><extra></extra>",
        ))
    apply(fig, title="Trend estimate by method", mode="light", height=340,
          legend=True, hovermode="x unified")
    fig.update_layout(
        height=340,
        margin={"l": 56, "r": 16, "t": TITLE_TOP_MARGIN, "b": 30},
        legend={"orientation": "h", "y": -0.18},
    )
    labels = [d.label for d in decompositions]
    rows = [
        {"time": str(t), **{d.label: float(np.asarray(d.trend, dtype=float)[i]) for d in decompositions}}
        for i, t in enumerate(decompositions[0].trend.index)
    ] if decompositions else []
    return P.Chart(
        figure_json=to_json(fig),
        alt_text=(
            f"Trend estimates for {name} from {len(decompositions)} methods ({', '.join(labels)}), "
            "overlaid. Where the lines separate, the trend is a choice rather than a fact."
        ),
        data_behind=rows,
    )


def seasonal_index_bars(shape: list[dict[str, Any]], period: int, name: str) -> P.Chart | None:
    """The seasonal shape as deviations from the average position — the 'index' a human reads."""
    if not shape:
        return None
    labels = [str(r["position"]) for r in shape]
    values = [float(r["effect"]) for r in shape]
    chart = P.bars(
        labels,
        values,
        title=f"Seasonal shape ({name})",
        subtitle=(
            f"Average deviation from the mean at each of the {period} positions. "
            "Positive bars are the periods that run above average."
        ),
        y_title="deviation from mean",
        height=300,
        reference=0.0,
    )
    chart.alt_text = (
        f"Bar chart of the seasonal shape over {period} positions for {name}; the largest "
        f"positive deviation is {max(values):.4g} at position {labels[int(np.argmax(values))]} and "
        f"the largest negative is {min(values):.4g} at position {labels[int(np.argmin(values))]}."
    )
    return chart


def periodogram_chart(series: pd.Series, period: int | None, top: list[dict[str, Any]]) -> P.Chart:
    """
    Power by period, on a log x-axis, with the seasonal period marked.

    Period (not frequency) on the x-axis because "every 12 months" is readable and "0.083 cycles
    per month" is not.
    """
    c = colours("light")
    y = np.asarray(series.dropna(), dtype=float)
    y = y - y.mean()
    power = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y))
    mask = freqs > 0
    periods = 1.0 / freqs[mask]
    vals = power[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=vals, mode="lines", name="Power",
        line={"color": c["series"], "width": 1.4},
        hovertemplate="period %{x:.2f}<br>power %{y:.4g}<extra></extra>",
    ))
    if period:
        fig.add_vline(
            x=period,
            line={"color": c["bad"], "width": 1.4, "dash": DASHES[1]},
            annotation_text=f"seasonal ({period})", annotation_position="top right",
            annotation_font={"size": 10, "color": c["muted"]},
        )
    apply(fig, title="Power by period", x_title="period (observations)", y_title="power",
          mode="light", height=300, legend=False, hovermode="closest")
    fig.update_xaxes(type="log")
    fig.update_layout(margin={"l": 56, "r": 16, "t": TITLE_TOP_MARGIN, "b": 46})
    peaks = ", ".join(f"period {p['period']:g}" for p in top[:3] if p.get("period"))
    return P.Chart(
        figure_json=to_json(fig),
        alt_text=(
            f"Periodogram of the detrended series. The tallest peaks are at {peaks}. "
            + (f"The dashed line marks the seasonal period of {period}." if period else "")
        ),
        data_behind=[{"period": round(float(p), 4), "power": float(v)}
                     for p, v in zip(periods[-200:], vals[-200:])],
    )


__all__ = [
    "component_grid", "side_by_side", "seasonal_index_bars", "periodogram_chart",
]
