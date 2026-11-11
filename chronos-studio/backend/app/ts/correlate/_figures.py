"""
ts/correlate/_figures.py — the correlation figure set (§5.10, group 4).

Built from `app/charts/primitives.py` and the theme, never from ad-hoc styling. Three figures:

* `acf_pacf_grid`   — the same series' ACF and PACF side by side, with the white-noise *and*
  Bartlett bands drawn, because the difference between them is the whole point.
* `acf_comparison_grid` — raw vs first-differenced vs seasonally-differenced, 2×3. This is the
  "teaching moment" plot the spec asks for: the eye sees a slowly decaying ACF flatten out after
  one difference, and can see over-differencing as the mirror image.
* `ccf_chart`       — cross-correlation over negative and positive lags, with the leading side
  shaded so "leads" and "lags" are visible without reading the axis backwards.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go

from app.charts import primitives as P
from app.charts.primitives import to_json
from app.charts.theme import DASHES, apply, colours, with_alpha
from app.ts.correlate._acf import ACF, PACF
from app.ts.correlate._ccf import CCF


def _stems(fig: go.Figure, x: Sequence[int], y: Sequence[float], colour: str,
           name: str, row: int = 1, col: int = 1, n_cols: int = 1) -> None:
    """A stem plot: one trace for the stems, one for the heads (keeps the legend clean)."""
    xi = list(x)
    yi = list(y)
    xs: list[float | None] = []
    ys: list[float | None] = []
    for a, b in zip(xi, yi):
        xs.extend([a, a, None])
        ys.extend([0.0, float(b), None])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": colour, "width": 1.3},
                             hoverinfo="skip", showlegend=False),
                  row=row, col=col)
    fig.add_trace(go.Scatter(x=xi, y=yi, mode="markers", name=name,
                             marker={"color": colour, "size": 5.5},
                             hovertemplate="lag %{x}<br>%{y:.3f}<extra></extra>"),
                  row=row, col=col)


def _band(fig: go.Figure, lags: Sequence[int], band: Sequence[float] | float,
          colour: str, name: str, row: int = 1, col: int = 1) -> None:
    lags = list(lags)
    if isinstance(band, (int, float)):
        hi = np.full(len(lags), float(band))
    else:
        hi = np.asarray(band, dtype=float)
    lo = -hi
    fig.add_trace(go.Scatter(
        x=lags + lags[::-1], y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor=with_alpha(colour, 0.16), line={"width": 0},
        name=name, hoverinfo="skip",
    ), row=row, col=col)


def acf_pacf_grid(a: ACF, p: PACF, *, name: str = "the series", height: int = 420) -> P.Chart:
    """
    ACF (left) and PACF (right), each with the white-noise band and the Bartlett band.

    Both bands are drawn on purpose. The white-noise band is the one most people have seen; the
    Bartlett band is the one that remains correct once the series is autocorrelated, and on a
    persistent series they differ by a factor of two or more. Showing both is the honest version of
    "is this lag significant".
    """
    c = colours("light")
    fig = go.Figure().set_subplots(rows=1, cols=2, horizontal_spacing=0.10)

    _band(fig, a.lags, a.band_white, c["warn"], "white-noise band", row=1, col=1)
    _band(fig, a.lags, a.band_bartlett, c["series"], "Bartlett band", row=1, col=1)
    _stems(fig, a.lags, a.values, c["series"], "ACF", row=1, col=1)

    _band(fig, p.lags, p.band_white, c["warn"], "white-noise band", row=1, col=2)
    _stems(fig, p.lags, p.values, c.get("trend", c["series"]), "PACF", row=1, col=2)

    fig.add_hline(y=0, line={"color": c["muted"], "width": 0.8}, row=1, col=1)
    fig.add_hline(y=0, line={"color": c["muted"], "width": 0.8}, row=1, col=2)
    for col, title in ((1, "Autocorrelation (ACF)"), (2, "Partial autocorrelation (PACF)")):
        fig.update_xaxes(title_text="Lag", row=1, col=col)
        fig.update_yaxes(title_text=title, range=[-1.05, 1.05], row=1, col=col)

    apply(fig, title=f"How much of {name} is explained by its own past",
          subtitle=(
              f"Lags 1–{a.max_lag} on {a.n} observations. The blue shaded band is the correct "
              f"significance band once the series is autocorrelated (Bartlett); the wider amber one "
              f"is the white-noise band ±{a.band_white:.3f}. Bars outside the band mean a real "
              "correlation at that lag."
          ),
          mode="light", height=height, legend=True, hovermode="x unified")

    sig = a.significant_lags()
    alt = (
        f"Autocorrelation and partial autocorrelation of {name} at lags 1 to {a.max_lag}. "
        f"{len(sig)} lags clear the Bartlett band"
        + (f" (the first is lag {sig[0]}, the largest is lag "
           f"{max(sig, key=lambda k: abs(a.at(k) or 0))})" if sig else "")
        + f". The largest absolute autocorrelation is {max(abs(v) for v in a.values):.2f}."
    )
    return P.Chart(to_json(fig), alt, [
        {"lag": lag, "acf": round(float(v), 4), "pacf": round(float(p.values[lag - 1]), 4)}
        for lag, v in zip(a.lags, a.values)
    ])


def acf_comparison_grid(
    frames: list[tuple[str, ACF, PACF]],
    *,
    name: str = "the series",
    height: int = 560,
) -> P.Chart:
    """
    ACF+PACF for up to three versions of the series side by side.

    Rows are versions (raw, first difference, seasonal difference); columns are ACF and PACF. The
    point is comparison: a slowly decaying top row that flattens in the middle row is what
    stationarity looks like, and a middle row whose lag 1 has become strongly negative is what
    over-differencing looks like.
    """
    c = colours("light")
    n_rows = max(1, min(len(frames), 3))
    fig = go.Figure().set_subplots(
        rows=n_rows, cols=2, horizontal_spacing=0.09, vertical_spacing=0.11,
        subplot_titles=[t for f in frames[:n_rows] for t in (f"ACF — {f[0]}", f"PACF — {f[0]}")],
    )
    for i, (label, a, p) in enumerate(frames[:n_rows], start=1):
        _band(fig, a.lags, a.band_bartlett, c["series"], f"Bartlett band ({label})", row=i, col=1)
        _stems(fig, a.lags, a.values, c["series"], f"ACF ({label})", row=i, col=1)
        _band(fig, p.lags, p.band_white, c["warn"], f"band ({label})", row=i, col=2)
        _stems(fig, p.lags, p.values, c.get("trend", c["series"]), f"PACF ({label})", row=i, col=2)
        for col in (1, 2):
            fig.add_hline(y=0, line={"color": c["muted"], "width": 0.8}, row=i, col=col)
            fig.update_yaxes(range=[-1.05, 1.05], row=i, col=col)
            fig.update_xaxes(title_text="Lag", row=i, col=col)

    apply(fig, title=f"What one difference does to {name}'s correlogram",
          subtitle=(
              "Each row is a version of the series, each column a correlogram. Read down the left "
              "column: a pattern that keeps decaying slowly means non-stationary; one that drops to "
              "the band means the difference did its job. A lag-1 bar that swings strongly negative "
              "means the difference went one step too far."
          ),
          mode="light", height=height, legend=False, hovermode="closest")

    alt_parts = []
    for label, a, _p in frames[:n_rows]:
        alt_parts.append(
            f"{label}: r₁ = {a.values[0]:.2f}, {len(a.significant_lags())} of {a.max_lag} lags "
            f"outside the band"
        )
    return P.Chart(to_json(fig),
                   f"Autocorrelation of {name} under each transformation. " + "; ".join(alt_parts) + ".",
                   [{"version": label, "lag": lag, "acf": round(float(v), 4)}
                    for label, a, _ in frames[:n_rows] for lag, v in zip(a.lags, a.values)])


def ccf_chart(c: CCF, *, height: int = 380) -> P.Chart:
    """
    Cross-correlation across negative and positive lags, with the leading half shaded.

    Negative lags are drawn on the left and shaded differently, with the axis labelled in words
    ("input leads ← | → input lags"). Reading a CCF backwards is the single most common error with
    this figure; the shading means the reader does not have to.
    """
    col = colours("light")
    fig = go.Figure()
    neg = [l for l in c.lags if l < 0]
    pos = [l for l in c.lags if l >= 0]
    fig.add_vrect(x0=min(c.lags) - 0.5, x1=-0.5, fillcolor=with_alpha(col["warn"], 0.10),
                  line_width=0, annotation_text=f"{c.x_name} leads", annotation_position="top left",
                  annotation_font={"size": 10, "color": col["muted"]})
    fig.add_vrect(x0=0.5, x1=max(c.lags) + 0.5, fillcolor=with_alpha(col["series"], 0.07),
                  line_width=0, annotation_text=f"{c.y_name} leads",
                  annotation_position="top right",
                  annotation_font={"size": 10, "color": col["muted"]})

    for l, v in zip(c.lags, c.values):
        colour = col["series"] if abs(v) > c.band else col["muted"]
        fig.add_trace(go.Scatter(x=[l, l], y=[0, v], mode="lines",
                                 line={"color": colour, "width": 1.3},
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=c.lags, y=c.values, mode="markers", name="Cross-correlation",
        marker={"color": [col["series"] if abs(v) > c.band else col["muted"] for v in c.values],
                "size": 5.5},
        hovertemplate="lag %{x}<br>r = %{y:.3f}<extra></extra>",
    ))
    for sign in (1, -1):
        fig.add_hline(y=sign * c.band, line={"color": col["warn"], "width": 1.0, "dash": DASHES[2]})
    fig.add_hline(y=0, line={"color": col["muted"], "width": 0.8})

    peak = c.peak()
    apply(fig, title=f"Does {c.x_name} move before or after {c.y_name}?",
          subtitle=(
              f"{'Pre-whitened: both series were filtered by an AR model, so their own memory is '
                 'removed.' if c.prewhitened else 'Raw: both series keep their own memory, so peaks '
                 'here can be shared trend rather than a relationship.'} "
              f"Dotted lines are the ±{c.band:.3f} band at {c.n} observations."
          ),
          mode="light", height=height, legend=False, hovermode="x unified",
          x_title="Lag (negative = input leads)", y_title="Cross-correlation")

    alt = (
        f"Cross-correlation of {c.y_name} against {c.x_name} at lags {min(c.lags)} to {max(c.lags)}. "
        f"The largest is {peak['value']:.2f} at lag {peak['lag']}, so {peak['leads_lags']}"
        + ("." if peak["significant"] else f" — inside the ±{c.band:.3f} band, so not significant.")
    )
    return P.Chart(to_json(fig), alt, c.rows())


__all__ = ["acf_pacf_grid", "acf_comparison_grid", "ccf_chart"]
