"""
app/charts/primitives.py — the chart factory.

Small, composable builders. Every EDA/decomposition/diagnostic figure in Parts 2-18
is assembled from these, which is what keeps 35+ plots consistent and lets one
change (say, a new palette) propagate everywhere.

Each builder returns `(figure_json, alt_text, data_behind)` — the alt text is
generated from the numbers, not hand-written, so it never drifts from the plot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.charts.theme import (
    DASHES,
    FONT_STACK,
    MONO_STACK,
    PALETTE,
    TITLE_TOP_MARGIN,
    ThemeMode,
    apply,
    colours,
    to_json,
    with_alpha,
)


@dataclass
class Chart:
    """What a builder hands back to the result layer."""

    figure_json: dict[str, Any]
    alt_text: str
    data_behind: list[dict[str, Any]] = field(default_factory=list)


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    ax = abs(x)
    if ax != 0 and (ax < 0.01 or ax >= 1e6):
        return f"{x:.2e}"
    return f"{x:,.{digits}f}".rstrip("0").rstrip(".") if ax < 1000 else f"{x:,.0f}"


def _series_rows(s: pd.Series | None) -> list[dict[str, Any]]:
    if s is None:
        return []
    return [
        {"time": str(t), "value": (None if pd.isna(v) else float(v))}
        for t, v in s.items()
    ]


# --------------------------------------------------------------------------- #
# Time-series plot with optional overlays
# --------------------------------------------------------------------------- #
def time_series(
    series: pd.Series,
    *,
    title: str,
    subtitle: str = "",
    x_title: str = "",
    y_title: str = "",
    overlays: Sequence[dict[str, Any]] | None = None,
    shading: Sequence[dict[str, Any]] | None = None,
    markers: Sequence[dict[str, Any]] | None = None,
    mode: ThemeMode = "light",
    height: int = 360,
    connect_gaps: bool = False,
) -> Chart:
    """
    The workhorse: the observed series, plus any number of overlays.

    overlays: dicts with `y` (Series), `name`, `kind` in
      {line, dashed, band, step, markers, loess, rolling}, `colour`, optional
      `fill_to` (Series) for bands.
    shading: dicts with `x0`, `x1`, `label`, `colour` — used for holdout windows,
      structural-break candidates and evaluation folds.
    markers: dicts with `x`, `y`, `label`, `symbol` — outlier flags, break dates.
    """
    c = colours(mode)
    fig = go.Figure()

    if shading:
        for sh in shading:
            fig.add_vrect(
                x0=sh["x0"], x1=sh["x1"],
                fillcolor=sh.get("colour", with_alpha(c["neutral"], 0.10)),
                layer="below", line_width=0,
                annotation_text=sh.get("label", ""),
                annotation_position="top left",
                annotation_font={"size": 10, "color": c["muted"]},
            )

    # Observed series — drawn with gaps as real gaps unless told otherwise.
    y = series.astype(float)
    if connect_gaps:
        y = y.interpolate(limit_direction="both")
    fig.add_trace(go.Scatter(
        x=series.index, y=y, mode="lines", name="Observed",
        line={"color": c["series"], "width": 1.7},
        connectgaps=connect_gaps,
        hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:,.4~g}</b><extra></extra>",
    ))

    dash_i = 1
    for ov in overlays or []:
        kind = ov.get("kind", "line")
        colour = ov.get("colour") or PALETTE[dash_i % len(PALETTE)]
        name = ov.get("name", kind)
        ys = ov["y"]

        if kind == "band":
            lo, hi = ov["y_low"], ov["y_high"]
            fig.add_trace(go.Scatter(
                x=ys.index, y=hi, mode="lines", name=f"{name} band",
                line={"width": 0}, showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=ys.index, y=lo, mode="lines", name=name,
                line={"width": 0}, fill="tonexty",
                fillcolor=ov.get("fill", with_alpha(c["series"], 0.14)),
                showlegend=True, hoverinfo="skip",
            ))
        elif kind == "markers":
            fig.add_trace(go.Scatter(
                x=ys.index, y=ys, mode="markers", name=name,
                marker={"color": colour, "size": ov.get("size", 7),
                        "symbol": ov.get("symbol", "circle"),
                        "line": {"width": 0.8, "color": c["paper"]}},
                hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br><b>%{{y:,.4~g}}</b><extra></extra>",
            ))
        else:
            dash = {
                "line": DASHES[0], "dashed": DASHES[1], "dot": DASHES[2],
                "step": DASHES[0], "loess": DASHES[0], "rolling": DASHES[1],
            }[kind]
            fig.add_trace(go.Scatter(
                x=ys.index, y=ys, mode="lines", name=name,
                line={"color": colour, "width": ov.get("width", 1.6), "dash": dash},
                connectgaps=False,
                hovertemplate=f"{name}: %{{y:,.4~g}}<extra></extra>",
            ))
        dash_i += 1

    for m in markers or []:
        fig.add_trace(go.Scatter(
            x=[m["x"]], y=[m["y"]], mode="markers+text",
            name=m.get("label", "Point"),
            marker={"color": m.get("colour", c["bad"]), "size": 9, "symbol": m.get("symbol", "diamond")},
            text=[m.get("short", "")], textposition="top center",
            textfont={"size": 9.5, "color": c["muted"]},
            hovertext=[m.get("label", "")], hoverinfo="text",
        ))

    apply(fig, title=title, subtitle=subtitle, x_title=x_title, y_title=y_title, mode=mode,
          height=height, hovermode="x unified")

    return Chart(
        figure_json=to_json(fig),
        alt_text=_alt_time_series(series, title, overlays),
        data_behind=_series_rows(series),
    )


def _alt_time_series(
    series: pd.Series, title: str, overlays: Sequence[dict[str, Any]] | None
) -> str:
    v = series.dropna()
    if v.empty:
        return f"{title}: no observations to plot."
    n = len(v)
    first, last = float(v.iloc[0]), float(v.iloc[-1])
    direction = "rises" if last > first else ("falls" if last < first else "stays level")
    gaps = int(series.isna().sum())
    # A catalogue figure may legitimately use step numbers, counts or category
    # labels on x. Formatting every index as a date made those charts raise.
    start, end = _x_label(series.index[0]), _x_label(series.index[-1])
    bits = [
        f"{title}. Line chart of {n} observations from {start} "
        f"to {end}; the series {direction} from "
        f"{_fmt(first)} to {_fmt(last)}, ranging {_fmt(float(v.min()))} to {_fmt(float(v.max()))}."
    ]
    if gaps:
        bits.append(f"{gaps} observation(s) are missing and appear as breaks in the line.")
    if overlays:
        bits.append("Overlays: " + ", ".join(o.get("name", "layer") for o in overlays) + ".")
    return " ".join(bits)


# --------------------------------------------------------------------------- #
# Distribution
# --------------------------------------------------------------------------- #
def distribution_panel(
    values: pd.Series,
    *,
    title: str = "Distribution of the series",
    subtitle: str = "",
    mode: ThemeMode = "light",
) -> Chart:
    """Histogram + KDE on top, box plot + ECDF beneath — one figure, four readings."""
    c = colours(mode)
    v = values.dropna().astype(float)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Histogram and density", "Box plot", "Empirical CDF", "Q–Q vs normal"),
        vertical_spacing=0.17, horizontal_spacing=0.12,
    )
    if v.empty:
        apply(fig, title=title, mode=mode, height=560)
        return Chart(to_json(fig), f"{title}: no observations.", [])

    fig.add_trace(go.Histogram(
        x=v, name="Count", nbinsx=min(40, max(8, int(np.sqrt(len(v))))),
        marker={"color": c["series"], "opacity": 0.55, "line": {"width": 0}},
        hovertemplate="bin %{x}<br>count %{y}<extra></extra>",
    ), row=1, col=1)

    if len(v) > 3 and v.nunique() > 2:
        kde_x, kde_y = _gaussian_kde(v)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y, mode="lines", name="Density",
            line={"color": c["trend"], "width": 2}, yaxis="y2",
            hovertemplate="density at %{x:,.3~g}: %{y:.3~g}<extra></extra>",
        ), row=1, col=1)
        fig.update_layout(yaxis2={
            "overlaying": "y", "side": "right", "showgrid": False,
            "tickfont": {"size": 10, "color": c["muted"]}, "title": None,
        })

    fig.add_trace(go.Box(
        y=v, name="", boxpoints="outliers", marker={"color": c["series"], "size": 4},
        line={"color": c["series"]}, fillcolor=with_alpha(c["series"], 0.14),
    ), row=1, col=2)

    x_sorted = np.sort(v.to_numpy())
    ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    fig.add_trace(go.Scatter(
        x=x_sorted, y=ecdf, mode="lines", name="ECDF",
        line={"color": c["series"], "width": 1.7}, showlegend=False,
        hovertemplate="x %{x:,.4~g}<br>P(X ≤ x) %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    # Normal Q-Q, so a user can see departure from normality without a test.
    theoretical = _normal_quantiles(len(x_sorted))
    fig.add_trace(go.Scatter(
        x=theoretical, y=x_sorted, mode="markers", name="Observed",
        marker={"color": c["series"], "size": 4.5}, showlegend=False,
        hovertemplate="normal %{x:.2f}<br>observed %{y:,.4~g}<extra></extra>",
    ), row=2, col=2)
    mu, sd = float(np.mean(x_sorted)), float(np.std(x_sorted, ddof=1)) if len(x_sorted) > 1 else 0.0
    if sd > 0:
        fig.add_trace(go.Scatter(
            x=theoretical, y=mu + sd * theoretical, mode="lines", name="Normal reference",
            line={"color": c["bad"], "width": 1.1, "dash": DASHES[1]},
            showlegend=False, hoverinfo="skip",
        ), row=2, col=2)

    for r in (1, 2):
        for q in (1, 2):
            fig.update_xaxes(gridcolor=c["grid"], zeroline=False, row=r, col=q,
                             tickfont={"size": 10, "color": c["muted"]})
            fig.update_yaxes(gridcolor=c["grid"], zeroline=False, row=r, col=q,
                             tickfont={"size": 10, "color": c["muted"]})

    apply(fig, title=title, subtitle=subtitle, mode=mode, height=560, legend=False)
    fig.update_layout(margin={"l": 52, "r": 52, "t": 68 if title else 34, "b": 40})

    sk = float(pd.Series(v).skew())
    ku = float(pd.Series(v).kurt())
    alt = (
        f"{title}. {len(v)} values, mean {_fmt(float(v.mean()))}, median {_fmt(float(v.median()))}, "
        f"standard deviation {_fmt(float(v.std(ddof=1)))}. Skewness {sk:.2f}, excess kurtosis {ku:.2f}"
        + (" — noticeably skewed." if abs(sk) > 1 else ".")
    )
    return Chart(to_json(fig), alt, [{"value": float(x)} for x in v])


def _gaussian_kde(v: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Silverman-bandwidth Gaussian KDE, computed directly (no scipy.stats import cost)."""
    x = v.to_numpy(dtype=float)
    n = len(x)
    sd = float(np.std(x, ddof=1)) or 1e-9
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    a = min(sd, iqr / 1.349) if iqr > 0 else sd
    bw = 0.9 * a * n ** (-1 / 5) or 1e-9
    grid = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, 200)
    z = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    # Scale density to histogram counts so the two share an axis sensibly.
    return grid, dens * n * (x.max() - x.min()) / max(1, min(40, int(np.sqrt(n))))


def _normal_quantiles(n: int) -> np.ndarray:
    """Theoretical standard-normal quantiles for a Q-Q plot."""
    p = (np.arange(1, n + 1) - 0.5) / n
    return np.array([_erfinv(2 * pi - 1) * np.sqrt(2) for pi in p])


def _erfinv(y: float) -> float:
    """Winitzki's approximation — accurate to ~1e-4, plenty for a Q-Q reference."""
    if y <= -1:
        return -np.inf
    if y >= 1:
        return np.inf
    a = 0.147
    ln = np.log(1 - y * y)
    t = 2 / (np.pi * a) + ln / 2
    return float(np.sign(y) * np.sqrt(np.sqrt(t * t - ln / a) - t))


# --------------------------------------------------------------------------- #
# Autocorrelation-style stem plot (used by ACF/PACF in Part 6, too)
# --------------------------------------------------------------------------- #
def stem_plot(
    lags: Sequence[int],
    values: Sequence[float],
    *,
    title: str,
    subtitle: str = "",
    band: float | Sequence[float] | None = None,
    y_title: str = "Autocorrelation",
    mode: ThemeMode = "light",
    height: int = 330,
) -> Chart:
    """Stems with a confidence band drawn as a shaded region, never as colour alone."""
    c = colours(mode)
    fig = go.Figure()
    lags_arr = np.asarray(lags)
    vals = np.asarray(values, dtype=float)

    if band is not None:
        if np.isscalar(band):
            lo = np.full_like(vals, -float(band), dtype=float)
            hi = np.full_like(vals, float(band), dtype=float)
        else:
            hi = np.asarray(band, dtype=float)
            lo = -hi
        fig.add_trace(go.Scatter(
            x=np.r_[lags_arr, lags_arr[::-1]], y=np.r_[hi, lo[::-1]],
            fill="toself", fillcolor=with_alpha(c["reference"], 0.20),
            line={"width": 0}, name="95% band", hoverinfo="skip",
        ))

    for lag, val in zip(lags_arr, vals):
        fig.add_trace(go.Scatter(
            x=[lag, lag], y=[0, val], mode="lines",
            line={"color": c["series"], "width": 1.4}, showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=lags_arr, y=vals, mode="markers", name="Correlation",
        marker={"color": c["series"], "size": 6},
        hovertemplate="lag %{x}<br>r = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line={"color": c["muted"], "width": 0.9})

    apply(fig, title=title, subtitle=subtitle, y_title=y_title, mode=mode,
          height=height, hovermode="x unified")
    if band is not None and np.isscalar(band):
        alt = (
            f"{title}. Largest absolute correlation {np.max(np.abs(vals)):.2f} at lag "
            f"{int(lags_arr[np.argmax(np.abs(vals))])}; the shaded band marks the 95% "
            f"significance threshold ±{float(band):.3f}."
        )
    else:
        alt = f"{title}. Largest absolute correlation {np.max(np.abs(vals)):.2f}."
    return Chart(to_json(fig), alt, [{"lag": int(l), "value": float(v)} for l, v in zip(lags_arr, vals)])


# --------------------------------------------------------------------------- #
# Scatter with a fitted line — diagnostics like variance-vs-level
# --------------------------------------------------------------------------- #
def scatter_with_fit(
    *,
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    subtitle: str = "",
    x_title: str = "",
    y_title: str = "",
    point_labels: Sequence[str] | None = None,
    mode: ThemeMode = "light",
    height: int = 380,
) -> Chart:
    """
    Points plus an ordinary least squares line.

    Used where the *relationship between two summaries* is the evidence — variance
    against level for a log transform, spread against fitted value for diagnostics.
    The line is drawn only when there are enough points for a slope to mean anything.
    """
    c = colours(mode)
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    labels = [tl for tl, k in zip(point_labels or [""] * len(keep), keep) if k]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name="Blocks of the series",
        marker={"color": c["series"], "size": 9, "line": {"width": 1, "color": c["paper"]}},
        text=labels if labels and any(labels) else None,
        hovertemplate="%{text}<br>level %{x:,.4~g}<br>spread %{y:,.4~g}<extra></extra>",
    ))

    r: float | None = None
    beta: np.ndarray | None = None
    if xs.size >= 3 and np.std(xs) > 0:
        beta = np.polyfit(xs, ys, 1)
        xr = np.linspace(float(xs.min()), float(xs.max()), 50)
        fig.add_trace(go.Scatter(
            x=xr, y=np.polyval(beta, xr), mode="lines", name="Least-squares line",
            line={"color": c["trend"], "width": 2, "dash": DASHES[1]},
            hovertemplate="fitted %{y:,.4~g}<extra></extra>",
        ))
        if np.std(ys) > 0:
            r = float(np.corrcoef(xs, ys)[0, 1])

    apply(fig, title=title, subtitle=subtitle, x_title=x_title, y_title=y_title,
          mode=mode, height=height, hovermode="closest")
    fig.update_xaxes(gridcolor=c["grid"])
    fig.update_yaxes(gridcolor=c["grid"])

    trend_word = (
        "rises" if (beta is not None and beta[0] > 0) else
        "falls" if beta is not None else "cannot be judged"
    )
    alt = (
        f"{title}. {len(xs)} blocks plotted. The spread {trend_word} with the level"
        + (f", correlation {r:.2f}" if r is not None else "")
        + (f"; the fitted line has slope {beta[0]:.4g}." if beta is not None else ".")
    )
    return Chart(
        to_json(fig), alt,
        [{"block": (labels[i] if i < len(labels) else f"block {i + 1}"),
          "level": float(xs[i]), "spread": float(ys[i])} for i in range(len(xs))],
    )


# --------------------------------------------------------------------------- #
# Heatmaps (calendar, month × year, correlation)
# --------------------------------------------------------------------------- #
def heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    subtitle: str = "",
    x_title: str = "",
    y_title: str = "",
    value_label: str = "Value",
    diverging: bool = False,
    mode: ThemeMode = "light",
    height: int = 340,
    text_format: str = ".3~g",
) -> Chart:
    """Calendar heatmaps, month×year grids and correlation matrices."""
    c = colours(mode)
    scale = (
        [[0.0, c["bad"]], [0.5, c["paper"]], [1.0, c["series"]]]
        if diverging
        else [[0.0, with_alpha(c["series"], 0.06)], [0.5, c["reference"]], [1.0, c["series"]]]
    )
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(dtype=float),
        x=[str(x) for x in matrix.columns],
        y=[str(y) for y in matrix.index],
        colorscale=scale,
        zmid=0 if diverging else None,
        showscale=True,
        colorbar={"title": {"text": value_label, "font": {"size": 10}},
                  "thickness": 10, "tickfont": {"size": 9.5}, "outlinewidth": 0},
        hovertemplate=f"%{{y}} · %{{x}}<br><b>%{{z:{text_format}}}</b><extra></extra>",
        hoverongaps=False,
    ))
    apply(fig, title=title, subtitle=subtitle, x_title=x_title, y_title=y_title,
          mode=mode, height=height, legend=False, hovermode="closest")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    z = matrix.to_numpy(dtype=float)
    finite = z[np.isfinite(z)]
    if finite.size:
        alt = (
            f"{title}. A grid of {matrix.shape[0]} rows by {matrix.shape[1]} columns; values "
            f"range from {_fmt(float(finite.min()))} to {_fmt(float(finite.max()))}."
        )
    else:
        alt = f"{title}. No finite values to display."
    rows = [
        {"row": str(i), **{str(col): (None if not np.isfinite(matrix.loc[i, col]) else float(matrix.loc[i, col]))
                           for col in matrix.columns}}
        for i in matrix.index
    ]
    return Chart(to_json(fig), alt, rows)


# --------------------------------------------------------------------------- #
# Bars (seasonal profile, decomposition scorecard, IC comparison later)
# --------------------------------------------------------------------------- #
def bars(
    categories: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    subtitle: str = "",
    x_title: str = "",
    y_title: str = "",
    mode: ThemeMode = "light",
    height: int = 320,
    reference: float | None = None,
    horizontal: bool = False,
    colours_override: Sequence[str] | None = None,
    text: Sequence[str] | None = None,
) -> Chart:
    c = colours(mode)
    fig = go.Figure(go.Bar(
        x=values if horizontal else list(categories),
        y=list(categories) if horizontal else values,
        orientation="h" if horizontal else "v",
        marker={
            "color": list(colours_override) if colours_override else c["series"],
            "line": {"width": 0},
        },
        text=list(text) if text else None,
        textposition="outside" if text else "none",
        textfont={"size": 10, "color": c["muted"]},
        hovertemplate="%{x}<br><b>%{y}</b><extra></extra>" if horizontal
        else "%{x}<br><b>%{y:,.4~g}</b><extra></extra>",
    ))
    if reference is not None:
        if horizontal:
            fig.add_vline(x=reference, line={"color": c["bad"], "width": 1.2, "dash": DASHES[1]})
        else:
            fig.add_hline(y=reference, line={"color": c["bad"], "width": 1.2, "dash": DASHES[1]})
    apply(fig, title=title, subtitle=subtitle, x_title=x_title, y_title=y_title,
          mode=mode, height=height, legend=False, hovermode="closest")
    top_i = int(np.argmax(np.abs(np.asarray(values, dtype=float))))
    alt = f"{title}. {len(values)} categories; largest is {categories[top_i]} at {_fmt(float(values[top_i]))}."
    return Chart(to_json(fig), alt,
                 [{"category": str(cat), "value": float(v)} for cat, v in zip(categories, values)])


# --------------------------------------------------------------------------- #
# Small multiples (seasonal subseries, lag plots)
# --------------------------------------------------------------------------- #
def small_multiples(
    frames: Sequence[tuple[str, pd.Series]],
    *,
    title: str,
    subtitle: str = "",
    n_cols: int = 4,
    mode: ThemeMode = "light",
    height_per_row: int = 120,
    shared_y: bool = True,
    marker_mode: bool = False,
) -> Chart:
    """A grid of tiny series plots — seasonal subseries and lag plots live here."""
    c = colours(mode)
    n = max(1, len(frames))
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[name for name, _ in frames],
        vertical_spacing=min(0.09, 0.6 / max(n_rows, 1)),
        horizontal_spacing=0.06,
    )
    for i, (_name, s) in enumerate(frames):
        r, q = divmod(i, n_cols)
        if marker_mode:
            fig.add_trace(go.Scatter(
                x=np.arange(len(s)), y=s.to_numpy(dtype=float), mode="markers",
                marker={"color": c["series"], "size": 4, "opacity": 0.7},
                showlegend=False, hovertemplate="%{x}<br>%{y:,.4~g}<extra></extra>",
            ), row=r + 1, col=q + 1)
        else:
            fig.add_trace(go.Scatter(
                x=list(range(len(s))), y=s.to_numpy(dtype=float), mode="lines+markers",
                line={"color": c["series"], "width": 1.3},
                marker={"size": 3.5, "color": c["series"]},
                showlegend=False, hovertemplate="%{x}<br>%{y:,.4~g}<extra></extra>",
            ), row=r + 1, col=q + 1)
        fig.update_xaxes(showticklabels=False, gridcolor=c["grid"], row=r + 1, col=q + 1)
        fig.update_yaxes(gridcolor=c["grid"], tickfont={"size": 8.5, "color": c["muted"]},
                         row=r + 1, col=q + 1)

    apply(fig, title=title, subtitle=subtitle, mode=mode,
          height=max(200, height_per_row * n_rows + 70), legend=False, hovermode="closest")
    if shared_y:
        fig.update_yaxes(matches="y")
    fig.update_annotations(font_size=10)
    alt = f"{title}. {n} small panels: " + ", ".join(name for name, _ in frames[:8]) + (
        " …" if n > 8 else ""
    ) + "."
    return Chart(to_json(fig), alt, [])


# --------------------------------------------------------------------------- #
# Rolling statistics
# --------------------------------------------------------------------------- #
def rolling_panel(
    series: pd.Series,
    window: int,
    *,
    title: str,
    subtitle: str = "",
    mode: ThemeMode = "light",
) -> Chart:
    """The raw series with a rolling mean and a rolling standard deviation."""
    c = colours(mode)
    roll_mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
    roll_sd = series.rolling(window, min_periods=max(2, window // 2)).std()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Rolling mean (level)", f"Rolling standard deviation (spread, window {window})"),
        row_heights=[0.58, 0.42],
    )
    fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name="Observed",
                             line={"color": c["series"], "width": 1.1},
                             opacity=0.45,
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.4~g}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, mode="lines",
                             name=f"{window}-point mean",
                             line={"color": c["trend"], "width": 2.1},
                             hovertemplate="mean %{y:,.4~g}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=roll_sd.index, y=roll_sd, mode="lines",
                             name=f"{window}-point SD",
                             line={"color": c["seasonal"], "width": 1.8},
                             fill="tozeroy", fillcolor=with_alpha(c["seasonal"], 0.12),
                             hovertemplate="SD %{y:,.4~g}<extra></extra>"), row=2, col=1)

    apply(fig, title=title, subtitle=subtitle, mode=mode, height=430, hovermode="x unified")
    for r in (1, 2):
        fig.update_xaxes(gridcolor=c["grid"], row=r, col=1)
        fig.update_yaxes(gridcolor=c["grid"], row=r, col=1,
                         tickfont={"size": 10, "color": c["muted"]})
    fig.update_annotations(font_size=10.5)

    sd_valid = roll_sd.dropna()
    trend = "roughly constant" if sd_valid.empty else (
        "rising over time" if sd_valid.iloc[-max(1, len(sd_valid) // 4):].mean()
        > sd_valid.iloc[: max(1, len(sd_valid) // 4)].mean() * 1.25
        else "stable over time"
    )
    alt = (
        f"{title}. The {window}-observation rolling mean moves from "
        f"{_fmt(float(roll_mean.dropna().iloc[0]) if roll_mean.notna().any() else None)} to "
        f"{_fmt(float(roll_mean.dropna().iloc[-1]) if roll_mean.notna().any() else None)}, and the "
        f"rolling spread is {trend}."
    )
    return Chart(to_json(fig), alt,
                 [{"time": str(t), "rolling_mean": None if pd.isna(m) else float(m),
                   "rolling_sd": None if pd.isna(s) else float(s)}
                  for t, m, s in zip(series.index, roll_mean, roll_sd)])


# --------------------------------------------------------------------------- #
# Forecast fan chart (Part 14, §5.10 group 7 / §5.11)
# --------------------------------------------------------------------------- #
def fan_chart(
    observed: pd.Series,
    *,
    index: Sequence[Any],
    point: Sequence[float],
    bands: Sequence[dict[str, Any]],
    title: str,
    subtitle: str = "",
    y_title: str = "",
    mode: ThemeMode = "light",
    height: int = 400,
    history: int = 72,
    comparison: Sequence[dict[str, Any]] | None = None,
) -> Chart:
    """
    One fan: the observed tail, nested two-sided bands at every requested coverage,
    and the point path. Every band is drawn widest-first, so the reader sees 95%
    behind 90% behind 80%, with opacity increasing as the band narrows.

    `bands` are dicts with `coverage`, `lower`, `upper`, `construction`, `source`.
    Coverage is in the legend and in `data_behind`; the construction is carried in
    the subtitle and in the result's `method` (§6.8). Nothing about the statistics
    is computed here — this is a drawing function only.
    """
    c = colours(mode)
    fig = go.Figure()
    obs = pd.Series(observed).astype(float).dropna()
    hist = obs.iloc[-max(int(history), 8):] if len(obs) else obs
    x_hist = [_x_label(t) for t in hist.index]
    x_fc = [_x_label(t) for t in index]
    anchor = float(hist.iloc[-1]) if len(hist) else float(np.nan)
    x_band = ([x_hist[-1]] if x_hist else []) + x_fc

    fig.add_trace(go.Scatter(
        x=x_hist, y=hist.to_numpy(), mode="lines", name="Observed",
        line={"color": c["series"], "width": 1.8},
        hovertemplate="Observed %{x}: %{y:,.4~g}<extra></extra>",
    ))

    ordered = sorted(
        [dict(b) for b in bands], key=lambda b: float(b.get("coverage", 0)), reverse=True
    )
    band_colour = c["forecast"] if mode != "print" else c["series"]
    for i, b in enumerate(ordered):
        cov = float(b["coverage"])
        lo_raw = np.asarray(b["lower"], dtype=float)
        hi_raw = np.asarray(b["upper"], dtype=float)
        lo_raw, hi_raw = np.minimum(lo_raw, hi_raw), np.maximum(lo_raw, hi_raw)
        lo = np.concatenate([[anchor], lo_raw])
        hi = np.concatenate([[anchor], hi_raw])
        alpha = 0.09 + 0.22 * (i + 1) / max(len(ordered), 1)
        fill = with_alpha(band_colour, alpha)
        label = f"{cov:.0%} interval ({b.get('source', 'interval')})"
        fig.add_trace(go.Scatter(
            x=x_band, y=hi, mode="lines", name=f"{label} upper",
            line={"width": 0}, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x_band, y=lo, mode="lines", name=label,
            line={"width": 0}, fill="tonexty", fillcolor=fill,
            showlegend=True, hoverinfo="skip",
        ))

    pt = np.asarray(point, dtype=float)
    fig.add_trace(go.Scatter(
        x=x_band, y=np.concatenate([[anchor], pt]), mode="lines", name="Point forecast",
        line={"color": c["forecast"], "width": 2.2, "dash": DASHES[1]},
        hovertemplate="Point %{x}: %{y:,.4~g}<extra></extra>",
    ))
    for extra in comparison or []:
        vals = np.asarray(extra["y"], dtype=float)
        fig.add_trace(go.Scatter(
            x=x_band, y=np.concatenate([[anchor], vals]), mode="lines",
            name=str(extra.get("name", "Comparison")),
            line={"color": extra.get("colour", c["reference"]),
                  "width": 1.5, "dash": extra.get("dash", DASHES[2])},
            hovertemplate=f"{extra.get('name', 'Comparison')} %{{x}}: %{{y:,.4~g}}<extra></extra>",
        ))

    apply(fig, title=title, subtitle=subtitle, y_title=y_title, mode=mode,
          height=height, hovermode="x unified")
    fig.update_layout(
        xaxis={"type": "category", "tickangle": -35, "nticks": 10},
    )

    lo95 = ordered[0] if ordered else None
    alt = (
        f"{title}. {len(hist)} observed values followed by {len(x_fc)} forecast steps. "
        f"The point path runs from {_fmt(float(pt[0]) if pt.size else None)} to "
        f"{_fmt(float(pt[-1]) if pt.size else None)}. "
        + (
            f"The widest {float(lo95['coverage']):.0%} band reaches "
            f"{_fmt(float(np.min(lo95['lower'])))} to {_fmt(float(np.max(lo95['upper'])))} "
            "over the horizon. " if lo95 else ""
        )
        + "Nested bands: " + ", ".join(f"{float(b['coverage']):.0%}" for b in ordered) + "."
    )
    rows: list[dict[str, Any]] = []
    for j, t in enumerate(x_fc):
        row: dict[str, Any] = {
            "step": j + 1,
            "index": str(t),
            "point": float(pt[j]) if j < pt.size else None,
        }
        for b in ordered:
            key = f"{int(round(float(b['coverage']) * 100))}pct"
            row[f"lower_{key}"] = float(b["lower"][j])
            row[f"upper_{key}"] = float(b["upper"][j])
        rows.append(row)
    return Chart(to_json(fig), alt, rows)


def _x_label(value: Any) -> str:
    """One x representation for dates, numbers and strings: always a string."""
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except (ValueError, TypeError):  # pragma: no cover - exotic index
            pass
    return str(value)


# --------------------------------------------------------------------------- #
# Tables as exportable figures (Part 16's SVG/PDF table exports)
# --------------------------------------------------------------------------- #
def _table_cell(value: Any) -> str:
    """A cell that is safe for Plotly's table trace and for static PDF export."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return _fmt(value, 4)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)[:180]
    if isinstance(value, np.generic):
        return _table_cell(value.item())
    return str(value)[:180]


def table_figure(
    columns: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    subtitle: str = "",
    mode: ThemeMode = "light",
    max_rows: int = 120,
) -> Chart:
    """
    Render a stored table with the same house theme as every chart.

    Catalogue tables must be exportable to PNG/SVG/PDF as well as XLSX/CSV. A
    Plotly Table trace is the one shared renderer; the numbers themselves always
    travel in ``data_behind`` and in the spreadsheet export. Very long tables are
    truncated in the *picture* only and the truncation is stated in the alt text.
    """
    c = colours(mode)
    keys = [str(x.get("key") or x.get("label") or f"column {i + 1}") for i, x in enumerate(columns)]
    labels = [str(x.get("label") or x.get("key") or f"Column {i + 1}") for i, x in enumerate(columns)]
    shown = list(rows[:max_rows])
    cells = [[_table_cell(row.get(key)) for row in shown] for key in keys]
    fig = go.Figure(go.Table(
        header={
            "values": labels,
            "fill": {"color": c["paper"]},
            "font": {"color": c["text"], "size": 10.5, "family": FONT_STACK},
            "align": "left",
            "line": {"color": c["grid"], "width": 0.6},
            "height": 26,
        },
        cells={
            "values": cells,
            "fill": {"color": c["paper"]},
            "font": {"color": c["muted"], "size": 9.5, "family": FONT_STACK},
            "align": "left",
            "line": {"color": c["grid"], "width": 0.4},
            "height": 21,
        },
    ))
    height = max(220, min(760, 78 + 22 * len(shown)))
    apply(fig, title=title, subtitle=subtitle, mode=mode, height=height,
          legend=False, hovermode="closest")
    fig.update_layout(margin={"l": 12, "r": 12,
                             "t": TITLE_TOP_MARGIN if title else 18, "b": 12})
    truncated = len(rows) > len(shown)
    alt = (
        f"{title}. A table of {len(rows)} rows and {len(columns)} columns; "
        f"{len(shown)} rows are shown in the static picture"
        + (" and the remainder are in the data export." if truncated else ".")
    )
    return Chart(to_json(fig), alt, list(rows))


__all__ = [
    "Chart",
    "fan_chart",
    "time_series",
    "distribution_panel",
    "stem_plot",
    "heatmap",
    "bars",
    "small_multiples",
    "rolling_panel",
    "scatter_with_fit",
    "table_figure",
    "DASHES",
    "FONT_STACK",
    "MONO_STACK",
]
