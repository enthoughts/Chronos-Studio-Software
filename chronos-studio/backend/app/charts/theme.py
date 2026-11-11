"""
app/charts/theme.py — the single source of chart styling (§5.10 consistency rule).

Every Plotly figure starts here: factories and the few bespoke figure modules call
`apply`/`to_json` and take palette, fonts, dash patterns and mode variants from
this module. No API handler builds its own figure and no chart module hardcodes a
colour, font or dash sequence.

Design constraints that are not negotiable:
  * colour-blind safe categorical palette (Okabe–Ito), never a rainbow,
  * no colour-only encoding — series differ by dash pattern too,
  * print mode: black/white-safe line styles and greys,
  * the UI's CSS variables and this palette agree, so a chart never clashes.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Literal

import plotly.graph_objects as go

ThemeMode = Literal["light", "dark", "print"]

# Plotly.js 3.8 anchors a horizontal legend *above* the plotting area by
# default. A two-line HTML title needs 58 px and each wrapped legend row needs
# about 21 px more, so the figure grows by exactly that amount. Keeping the
# calculation here preserves the single-source styling rule for both the UI and
# the static 300 dpi exports.
TITLE_TOP_MARGIN = 74
PLAIN_TOP_MARGIN = 18
# 0.94 keeps the first bold line inside the SVG: Plotly's 0.7em baseline shift
# otherwise lifts it by about 13 px. 74 px of top margin then leaves a clear band
# between the two-line title and a wrapped horizontal legend, including the
# 800 px-tall static export where the title's container-normalised y is larger.
TITLE_Y = 0.94
LEGEND_ROW_HEIGHT = 22
LEGEND_ITEMS_PER_ROW = 6

# Okabe–Ito: distinguishable under every common form of colour blindness.
PALETTE: list[str] = [
    "#0072B2",  # blue        — primary / observed series
    "#D55E00",  # vermillion  — accent / important
    "#009E73",  # green       — good / pass
    "#CC79A7",  # purple      — secondary
    "#E69F00",  # orange      — caution
    "#56B4E9",  # sky         — reference
    "#F0E442",  # yellow      — highlight (dark text only)
    "#666666",  # grey        — noise / remainder
]

SEMANTIC = {
    "series": PALETTE[0],
    "trend": PALETTE[1],
    "seasonal": PALETTE[2],
    "cycle": PALETTE[3],
    "remainder": PALETTE[7],
    "forecast": PALETTE[1],
    "interval": "rgba(213,94,0,0.18)",
    "reference": PALETTE[5],
    "threshold": "#B3261E",
    "ok": "#1A7F4B",
    "warn": "#A86B00",
    "bad": "#B3261E",
    "neutral": "#788796",  # 120,135,150 — grid ink without its alpha
    "grid": "rgba(120,135,150,0.22)",
}

# Same categorical order in every mode. Dark brightens the colours that would
# otherwise disappear on the dark paper; print collapses to distinguishable greys.
PALETTE_VARIANTS: dict[str, list[str]] = {
    "light": PALETTE,
    "dark": ["#56B4E9", "#FF8A65", "#3DBE8B", "#E599F7",
             "#FFC44D", "#8FD3F4", "#F7F071", "#AAB7C5"],
    "print": ["#000000", "#222222", "#555555", "#777777",
              "#999999", "#444444", "#BBBBBB", "#AAAAAA"],
}

_STATUS_VARIANTS: dict[str, dict[str, str]] = {
    "light": {"threshold": "#B3261E", "ok": "#1A7F4B", "warn": "#A86B00",
              "bad": "#B3261E", "neutral": "#788796"},
    "dark": {"threshold": "#FF8A80", "ok": "#5BD6A0", "warn": "#FFC44D",
             "bad": "#FF8A80", "neutral": "#A0AFBE"},
    "print": {"threshold": "#000000", "ok": "#555555", "warn": "#666666",
              "bad": "#000000", "neutral": "#000000"},
}

DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

FONT_STACK = (
    'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
)
MONO_STACK = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace'

_INK = {
    "light": {"text": "#16202c", "muted": "#4a5a6a", "paper": "#ffffff", "grid": "rgba(120,135,150,0.22)"},
    "dark": {"text": "#e8edf3", "muted": "#aab7c5", "paper": "#18212b", "grid": "rgba(160,175,190,0.22)"},
    "print": {"text": "#000000", "muted": "#333333", "paper": "#ffffff", "grid": "rgba(0,0,0,0.18)"},
}


def colours(mode: ThemeMode = "light") -> dict[str, str]:
    """Ink + semantic colours for a mode, with print collapsing to greys."""
    ink = _INK[mode]
    if mode == "dark":
        v = PALETTE_VARIANTS["dark"]
        return {
            **SEMANTIC, **_STATUS_VARIANTS["dark"],
            "series": v[0], "trend": v[1], "seasonal": v[2], "cycle": v[3],
            "remainder": v[7], "forecast": v[1], "reference": v[5],
            "interval": "rgba(255,138,101,0.18)", **ink,
        }
    if mode == "print":
        return {
            **SEMANTIC, **_STATUS_VARIANTS["print"],
            "series": "#000000",
            "trend": "#000000",
            "seasonal": "#555555",
            "cycle": "#888888",
            "remainder": "#AAAAAA",
            "forecast": "#000000",
            "interval": "rgba(0,0,0,0.10)",
            "reference": "#444444",
            **ink,
        }
    return {**SEMANTIC, **ink}


_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_RGB_RE = re.compile(
    r"^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*"
    r"(?:,\s*([01]?(?:\.\d+)?|\d+(?:\.\d+)?)\s*)?\)$"
)


def _parse_colour(value: Any) -> tuple[int, int, int, float] | None:
    """Return ``(r, g, b, alpha)`` for a hex/rgb/rgba string, else ``None``."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    m = _HEX_RE.match(s)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0
    m = _RGB_RE.match(s)
    if m:
        r, g, b = (min(255, int(m.group(i))) for i in (1, 2, 3))
        a = float(m.group(4)) if m.group(4) is not None else 1.0
        return r, g, b, max(0.0, min(1.0, a))
    return None


def with_alpha(colour: str, alpha: float) -> str:
    """One house helper for every translucent fill; no module writes its own rgba."""
    parsed = _parse_colour(colour)
    if parsed is None:
        return colour
    r, g, b, _ = parsed
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a:g})"



def _hex_blend(a: str, b: str, t: float) -> str:
    """Blend two theme colours into an openpyxl-safe ``RRGGBB`` string."""
    pa, pb = _parse_colour(a), _parse_colour(b)
    if pa is None or pb is None:
        return "FFFFFF"
    t = max(0.0, min(1.0, float(t)))
    rgb = tuple(round(pa[i] * (1 - t) + pb[i] * t) for i in range(3))
    return f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


# Spreadsheet exports use the same house ink; openpyxl needs plain hex, no alpha.
SPREADSHEET_FONT = "Calibri"
SPREADSHEET_HEADER_FILL = _hex_blend(_INK["light"]["paper"], SEMANTIC["neutral"], 0.10)
SPREADSHEET_HEADER_FONT = _INK["light"]["text"].lstrip("#").upper()


def _colour_remap(mode: ThemeMode) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    """Light-mode ink → the requested mode's equivalent RGB."""
    if mode == "light":
        return {}
    out: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    variants = PALETTE_VARIANTS[mode]
    for i, light in enumerate(PALETTE):
        lp, tp = _parse_colour(light), _parse_colour(variants[i])
        if lp and tp:
            out[lp[:3]] = tp[:3]
    semantic_light = {**SEMANTIC, **_STATUS_VARIANTS["light"]}
    semantic_target = colours(mode)
    for name, light in semantic_light.items():
        target = semantic_target.get(name, light)
        lp, tp = _parse_colour(light), _parse_colour(target)
        if lp and tp:
            out[lp[:3]] = tp[:3]
    for key in ("text", "muted", "paper"):
        lp, tp = _parse_colour(_INK["light"][key]), _parse_colour(_INK[mode][key])
        if lp and tp:
            out[lp[:3]] = tp[:3]
    # The categorical variants are authoritative: they must win over any
    # semantic key that happens to share the same light-mode RGB.
    for i, light in enumerate(PALETTE):
        lp, tp = _parse_colour(light), _parse_colour(variants[i])
        if lp and tp:
            out[lp[:3]] = tp[:3]
    return out


def _remap_colour_string(
    value: str, mode: ThemeMode,
    remap: dict[tuple[int, int, int], tuple[int, int, int]],
) -> str:
    parsed = _parse_colour(value)
    if parsed is None or mode == "light":
        return value
    r, g, b, a = parsed
    key = (r, g, b)
    if key in remap:
        tr, tg, tb = remap[key]
    elif mode == "print":
        # Any legacy colour unknown to the theme becomes a luminance-preserving grey.
        shade = max(0, min(210, int(0.2126 * r + 0.7152 * g + 0.0722 * b)))
        tr = tg = tb = shade
    else:
        # Dark mode: lift genuinely dark legacy ink; leave bright colours alone.
        if max(r, g, b) >= 110:
            return value
        tr, tg, tb = (min(255, int(v * 1.55 + 45)) for v in (r, g, b))
    if a >= 0.999:
        return f"#{tr:02X}{tg:02X}{tb:02X}"
    return f"rgba({tr},{tg},{tb},{a:g})"


def _recolour_node(
    node: Any, mode: ThemeMode,
    remap: dict[tuple[int, int, int], tuple[int, int, int]],
) -> Any:
    if isinstance(node, dict):
        return {k: _recolour_node(v, mode, remap) for k, v in node.items()}
    if isinstance(node, list):
        return [_recolour_node(v, mode, remap) for v in node]
    if isinstance(node, tuple):
        return tuple(_recolour_node(v, mode, remap) for v in node)
    if isinstance(node, str):
        return _remap_colour_string(node, mode, remap)
    return node


def _print_dashes(fig: go.Figure) -> None:
    """Print-safe line styles, so greys are not the only encoding left."""
    for i, trace in enumerate(fig.data):
        line = getattr(trace, "line", None)
        if line is None or not isinstance(getattr(trace, "mode", ""), str):
            continue
        if "lines" not in (getattr(trace, "mode", "") or ""):
            continue
        if getattr(line, "width", None) == 0 or getattr(line, "color", None) is None:
            continue
        if not getattr(line, "dash", None):
            line.dash = DASHES[i % len(DASHES)]


def base_layout(
    *,
    title: str = "",
    subtitle: str = "",
    x_title: str = "",
    y_title: str = "",
    mode: ThemeMode = "light",
    height: int = 360,
    legend: bool = True,
    hovermode: str = "x unified",
) -> dict[str, Any]:
    """
    The house style. Every figure starts here so margins, fonts, grid and tooltip
    behaviour are identical across 35+ charts.
    """
    c = colours(mode)
    heading = title
    if subtitle:
        heading = (
            f"<b>{title}</b>"
            f"<br><span style='font-size:11.5px;color:{c['muted']}'>{subtitle}</span>"
        )
    return {
        "title": {
            "text": heading,
            "x": 0.0,
            "xanchor": "left",
            "y": TITLE_Y,
            "yanchor": "top",
            "font": {"size": 13.5, "family": FONT_STACK, "color": c["text"]},
        },
        "font": {"family": FONT_STACK, "size": 11.5, "color": c["text"]},
        "paper_bgcolor": c["paper"],
        "plot_bgcolor": c["paper"],
        "height": height,
        "margin": {"l": 58, "r": 18,
                   "t": TITLE_TOP_MARGIN if title else PLAIN_TOP_MARGIN, "b": 46},
        "xaxis": {
            "title": {"text": x_title, "font": {"size": 11.5, "color": c["muted"]}}
            if x_title else {},
            "gridcolor": c["grid"],
            "zeroline": False,
            "linecolor": c["grid"],
            "ticks": "outside",
            "tickfont": {"size": 10.5, "color": c["muted"]},
            "showspikes": False,
        },
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 11.5, "color": c["muted"]}}
            if y_title else {},
            "gridcolor": c["grid"],
            "zeroline": False,
            "linecolor": "rgba(0,0,0,0)",
            "ticks": "outside",
            "tickfont": {"size": 10.5, "color": c["muted"]},
        },
        "hovermode": hovermode,
        "hoverlabel": {
            "bgcolor": c["paper"],
            "bordercolor": c["grid"],
            "font": {"size": 11.5, "family": MONO_STACK, "color": c["text"]},
        },
        "showlegend": legend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10.5, "color": c["muted"]},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "modebar": {"orientation": "v"},
    }


def _legend_rows(fig: go.Figure, legend_enabled: bool) -> int:
    """Estimate how many horizontal rows Plotly.js needs for the visible traces.

    The catalogue renders at a fixed card width; six named traces per row is a
    deliberately conservative estimate that also covers long CUSUM/IRF labels.
    Over-reserving a few pixels is preferable to a legend covering the subtitle.
    """
    if not legend_enabled or fig.layout.showlegend is False:
        return 0
    named = [
        trace for trace in fig.data
        if getattr(trace, "showlegend", None) is not False
        and getattr(trace, "name", None)
    ]
    return max(1, math.ceil(len(named) / LEGEND_ITEMS_PER_ROW)) if named else 0


def _restyle_layout(fig: go.Figure, mode: ThemeMode) -> go.Figure:
    """Recolour an existing figure without touching its geometry.

    A stored figure already carries its title, legend visibility, margins, height
    and axis ranges. Calling `base_layout` again would silently turn hidden
    legends back on and reset the top margin to 18 px, causing the two-line title
    to collide with the legend. Only the theme's ink is reapplied here.
    """
    c = colours(mode)
    fig.update_layout(
        font={"family": FONT_STACK, "size": 11.5, "color": c["text"]},
        paper_bgcolor=c["paper"],
        plot_bgcolor=c["paper"],
        hoverlabel={
            "bgcolor": c["paper"],
            "bordercolor": c["grid"],
            "font": {"size": 11.5, "family": MONO_STACK, "color": c["text"]},
        },
    )
    if fig.layout.title.text:
        fig.layout.title.x = 0.0
        fig.layout.title.xanchor = "left"
        fig.layout.title.y = TITLE_Y
        fig.layout.title.yanchor = "top"
        fig.layout.title.font.color = c["text"]
        fig.layout.title.font.family = FONT_STACK
    fig.update_xaxes(
        gridcolor=c["grid"], linecolor=c["grid"],
        tickfont={"size": 10.5, "color": c["muted"]},
        title_font={"size": 11.5, "color": c["muted"]},
    )
    fig.update_yaxes(
        gridcolor=c["grid"], linecolor="rgba(0,0,0,0)",
        tickfont={"size": 10.5, "color": c["muted"]},
        title_font={"size": 11.5, "color": c["muted"]},
    )
    fig.update_layout(legend={
        "font": {"size": 10.5, "color": c["muted"]},
        "bgcolor": "rgba(0,0,0,0)",
    })
    return fig


def apply(fig: go.Figure, **layout_kwargs: Any) -> go.Figure:
    """Apply the house layout to a figure at build time.

    `update_layout` merges nested axis objects, so an axis title already set by a
    builder wins over the theme's (usually empty) title. A horizontal legend sits
    above the plot in Plotly.js 3.8; reserve one row of top margin per six traces
    and grow the figure by the same amount so the plot area is unchanged.
    """
    layout = base_layout(**layout_kwargs)
    has_title = bool(layout_kwargs.get("title") or (fig.layout.title.text or ""))
    if not layout_kwargs.get("title") and has_title:
        layout.pop("title", None)
    rows = _legend_rows(fig, bool(layout.get("showlegend")))
    if rows:
        extra = rows * LEGEND_ROW_HEIGHT
        layout["margin"]["t"] = (
            TITLE_TOP_MARGIN if has_title else PLAIN_TOP_MARGIN
        ) + extra
        base_height = int(layout_kwargs.get("height") or fig.layout.height or 360)
        layout["height"] = base_height + extra
    # One merged `update_layout` call instead of one call per key. Plotly's
    # property-validation machinery dominates the EDA figure build on long
    # series; nine charts × ~14 keys was measurably slower than a single merge.
    # The title key was removed above when a builder already supplied one, so
    # explicitly set titles still win.
    fig.update_layout(layout)
    return fig


def restyled(figure_json: dict[str, Any], mode: ThemeMode = "light") -> dict[str, Any]:
    """Re-apply the single house theme to a stored figure in another mode.

    Used by Part 16's dark / light / print catalogue and its static exports.
    Layout ink, trace colours, translucent fills and heatmap scales are all
    remapped here; builders never need a second, mode-specific styling path.
    """
    import plotly.io as pio

    fig = pio.from_json(json.dumps(figure_json))
    _restyle_layout(fig, mode)
    # Subtitles are inline HTML so they can sit under a bold title. The inline
    # colour must follow the mode too, otherwise a dark figure keeps light-grey
    # subtitle text on a dark background.
    title = str(fig.layout.title.text or "")
    if "font-size:11.5px;color:" in title:
        fig.layout.title.text = re.sub(
            r"font-size:11\.5px;color:[^'\"]+",
            f"font-size:11.5px;color:{colours(mode)['muted']}",
            title,
        )
    if mode != "light":
        payload = fig.to_plotly_json()
        layout = payload.get("layout") or {}
        layout.pop("template", None)
        remap = _colour_remap(mode)
        # Recolour data and drawn shapes/annotations, but not the layout ink that
        # `apply` just set for this mode (e.g. the dark paper background).
        payload["data"] = _recolour_node(payload.get("data", []), mode, remap)
        for key in ("shapes", "annotations"):
            if key in layout:
                layout[key] = _recolour_node(layout[key], mode, remap)
        fig = go.Figure(payload)
        if mode == "print":
            _print_dashes(fig)
    return to_json(fig)


def to_json(fig: go.Figure) -> dict[str, Any]:
    """
    Figure → plain JSON for the wire.

    `plotly.io.to_json` keeps numpy arrays out of the payload and is what the
    frontend's Plotly.js consumes directly. We drop the template to keep the
    payload small (the frontend applies its own layout defaults).
    """
    import json

    payload = json.loads(fig.to_json())
    payload.get("layout", {}).pop("template", None)
    return payload


__all__ = [
    "PALETTE", "PALETTE_VARIANTS", "SEMANTIC", "DASHES", "FONT_STACK", "MONO_STACK",
    "ThemeMode", "colours", "with_alpha", "base_layout", "apply", "restyled", "to_json",
    "TITLE_TOP_MARGIN", "PLAIN_TOP_MARGIN", "TITLE_Y",
    "LEGEND_ROW_HEIGHT", "LEGEND_ITEMS_PER_ROW",
    "SPREADSHEET_FONT", "SPREADSHEET_HEADER_FILL", "SPREADSHEET_HEADER_FONT",
]
