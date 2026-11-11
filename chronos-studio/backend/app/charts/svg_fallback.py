"""Pure-Python, offline SVG renderer for Plotly catalogue figures.

The standalone package deliberately omits Kaleido and Chromium.  Without a
fallback, ``render_html`` catches the missing-renderer exception and replaces
every figure with a callout, so the advertised one-file HTML report has no
charts at all.  This module draws a static, vector version of the most common
catalogue figure types directly from the Plotly JSON: scatter (including
filled confidence bands), bars, boxes, histograms, heatmaps, grid subplots,
reference shapes and annotations.  It uses only NumPy from the runtime
wheelhouse and adds no scripts, external fonts, network references or browser
dependency to the exported SVG.
"""

from __future__ import annotations

import base64
import html
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.charts.theme import PALETTE, PALETTE_VARIANTS, colours

SVG_NS = "http://www.w3.org/2000/svg"
FONT = "Arial, Helvetica, sans-serif"
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 800
DASH_PATTERNS = {
    "solid": None,
    "dash": "6 4",
    "dot": "2 3",
    "dashdot": "8 3 2 3",
    "longdash": "12 5",
    "longdashdot": "12 4 2 4",
}
SYSTEM_BROWSERS = (
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
)


def _n(value: float) -> str:
    """A compact, finite number for SVG coordinates."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "0"
    if not math.isfinite(v):
        return "0"
    if abs(v) < 1e-9:
        v = 0.0
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _plain(value: Any) -> str:
    """Strip the small HTML subset used by Plotly titles and annotations."""
    text = re.sub(r"<br\s*/?>", " | ", str(value or ""), flags=re.I)
    text = re.sub(r"<[^>]*>", "", text)
    return " ".join(html.unescape(text).split())


def _decode(value: Any) -> Any:
    """Decode Plotly 7 typed arrays (base64 + numpy dtype) into plain Python."""
    if isinstance(value, dict) and "bdata" in value:
        try:
            dtype = np.dtype(str(value.get("dtype") or "f8"))
            if dtype.byteorder not in {"=", "|", "<"}:
                dtype = dtype.newbyteorder("<")
            array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtype)
            shape = value.get("shape")
            if shape:
                if isinstance(shape, str):
                    dimensions = tuple(int(x) for x in shape.split(",") if x.strip())
                else:
                    dimensions = tuple(int(x) for x in shape)
                array = array.reshape(dimensions)
            return array.tolist()
        except Exception:
            return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _flatten(value: Any) -> list[Any]:
    out: list[Any] = []
    for item in _decode(value):
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return float(int(value)) if isinstance(value, bool) else None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, str):
        try:
            v = float(value.strip())
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def _key(value: Any) -> str:
    n = _finite(value)
    if n is not None:
        return str(int(n)) if n.is_integer() else f"{n:.10g}"
    return str(value)


def _parse_hex(text: str) -> tuple[int, int, int, float] | None:
    """Parse a theme colour into 0..255 RGB plus alpha."""
    raw = text.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) not in {6, 8}:
        return None
    try:
        rgb = tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))
        alpha = int(raw[6:8], 16) / 255 if len(raw) == 8 else 1.0
        return (*rgb, alpha)
    except ValueError:
        return None


def _rgba(value: Any, default: str | None = None) -> tuple[int, int, int, float]:
    """Return 0..255 RGB plus alpha, without relying on a CSS engine."""
    fallback = default or PALETTE[0]
    candidates = value if isinstance(value, (list, tuple)) and len(value) > 1 else [value]
    text = ""
    for candidate in candidates:
        if isinstance(candidate, str):
            text = candidate.strip()
            break
    if not isinstance(text, str) or not text:
        text = fallback
    parsed = _parse_hex(text) if text.startswith("#") else None
    if parsed is not None:
        return parsed
    match = re.match(r"rgba?\(([^)]+)\)", text, flags=re.I)
    if match:
        parts = [x.strip() for x in match.group(1).split(",")]
        try:
            rgb = tuple(max(0, min(255, round(float(x)))) for x in parts[:3])
            alpha = float(parts[3]) if len(parts) > 3 else 1.0
            return (*rgb, max(0.0, min(1.0, alpha)))
        except (ValueError, TypeError):
            pass
    ink = colours("light")
    named = {
        "white": ink["paper"],
        "black": ink["text"],
        "grey": ink["muted"],
        "gray": ink["muted"],
        "red": ink["bad"],
        "green": ink["ok"],
        "blue": ink["series"],
    }
    named_value = named.get(text.lower(), fallback)
    return _parse_hex(named_value) or _parse_hex(fallback) or (0, 114, 178, 1.0)


def _hex(rgb: tuple[int, int, int, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb[:3])


def _paint(color: Any, kind: str = "fill", extra: float = 1.0, default: str | None = None) -> str:
    rgba = _rgba(color, default)
    alpha = max(0.0, min(1.0, rgba[3] * extra))
    return f'{kind}="{_hex(rgba)}" {kind}-opacity="{_n(alpha)}"'


def _stroke(color: Any, width: Any, dash: Any = None, extra: float = 1.0) -> str:
    rgba = _rgba(color)
    alpha = max(0.0, min(1.0, rgba[3] * extra))
    try:
        w = float(width)
    except (TypeError, ValueError):
        w = 1.4
    w = max(0.0, min(5.0, w))
    out = (
        f'stroke="{_hex(rgba)}" stroke-opacity="{_n(alpha)}" '
        f'stroke-width="{_n(w)}" stroke-linecap="round" stroke-linejoin="round"'
    )
    pattern = DASH_PATTERNS.get(str(dash or "solid"), DASH_PATTERNS["dash"])
    if pattern:
        out += f' stroke-dasharray="{pattern}"'
    return out


def _axis_key(ref: str, letter: str) -> str:
    return f"{letter}axis{ref[1:]}" if len(ref) > 1 else f"{letter}axis"


def _ref_number(ref: str) -> int:
    suffix = ref[1:]
    return int(suffix) if suffix.isdigit() else 1


def _is_ref(ref: Any, letter: str) -> bool:
    return isinstance(ref, str) and re.fullmatch(rf"{letter}\d*", ref.strip()) is not None


def _norm_ref(ref: Any, letter: str) -> str:
    if _is_ref(ref, letter):
        return str(ref).strip()
    return letter


def _canonical(ref: str, letter: str, layout: dict[str, Any]) -> str:
    seen: set[str] = set()
    current = ref
    while current not in seen:
        seen.add(current)
        spec = layout.get(_axis_key(current, letter)) or {}
        match = spec.get("matches")
        if not isinstance(match, str) or not _is_ref(match, letter) or match in seen:
            break
        current = match.strip()
    return current


def _layout_key(ref: str, letter: str) -> str:
    return _axis_key(ref, letter)


def _base_ref(ref: Any, letter: str) -> str:
    """Map 'paper', 'x domain' and 'x2 domain' to a concrete axis reference."""
    if not isinstance(ref, str):
        return letter
    text = ref.strip()
    if text.endswith(" domain"):
        text = text[: -len(" domain")].strip()
    return _norm_ref(text, letter)


def _is_data_ref(ref: Any, letter: str) -> bool:
    if not isinstance(ref, str):
        return True  # Plotly's default is a data axis.
    text = ref.strip()
    return text != "paper" and not text.endswith(" domain") and _is_ref(text, letter)


def _domain(ref: str, letter: str, layout: dict[str, Any]) -> tuple[float, float]:
    spec = layout.get(_layout_key(ref, letter)) or {}
    domain = spec.get("domain")
    if isinstance(domain, (list, tuple)) and len(domain) == 2:
        try:
            a, b = float(domain[0]), float(domain[1])
            return (min(a, b), max(a, b))
        except (TypeError, ValueError):
            pass
    grid = layout.get("grid")
    if isinstance(grid, dict):
        rows = max(1, int(grid.get("rows", 1) or 1))
        cols = max(1, int(grid.get("columns", 1) or 1))
        number = _ref_number(ref)
        row = (number - 1) // cols + 1
        col = (number - 1) % cols + 1
        xgap = float(grid.get("xgap", 0.08) or 0)
        ygap = float(grid.get("ygap", 0.06) or 0)
        cell_w = max(0.01, (1 - xgap * (cols - 1)) / cols)
        cell_h = max(0.01, (1 - ygap * (rows - 1)) / rows)
        if letter == "x":
            start = (col - 1) * (cell_w + xgap)
            return (start, start + cell_w)
        bottom = 1 - row * cell_h - (row - 1) * ygap
        return (bottom, bottom + cell_h)
    return (0.0, 1.0)


@dataclass
class _Axis:
    ref: str
    raw: list[Any] = field(default_factory=list)
    ranges: list[list[Any]] = field(default_factory=list)
    forced_category: bool = False
    log: bool = False
    reversed: bool = False
    categories: dict[str, int] = field(default_factory=dict)
    categorical: bool = False
    low: float = 0.0
    high: float = 1.0

    def build(self) -> None:
        numeric_strings = bool(self.forced_category)
        for value in self.raw:
            if value is None:
                continue
            if _finite(value) is None:
                numeric_strings = True
                break
        self.categorical = numeric_strings
        if self.categorical:
            for value in self.raw:
                if value is None:
                    continue
                key = _key(value)
                if key not in self.categories:
                    self.categories[key] = len(self.categories)
            for pair in self.ranges:
                for value in pair:
                    key = _key(value)
                    if key not in self.categories:
                        self.categories[key] = len(self.categories)
            numbers = [
                float(self.categories[_key(value)])
                for value in self.raw
                if value is not None
            ]
            for pair in self.ranges:
                numbers.extend(float(self._range_number(value)) for value in pair)
            if numbers:
                self.low, self.high = min(numbers) - 0.5, max(numbers) + 0.5
            else:
                self.low, self.high = -0.5, 0.5
            if self.ranges:
                values = [self._range_number(x) for x in self.ranges[0]]
                self.low, self.high = min(values), max(values)
            return

        numbers: list[float] = []
        for value in self.raw:
            n = _finite(value)
            if n is None:
                continue
            if self.log:
                if n > 0:
                    numbers.append(math.log10(n))
            else:
                numbers.append(n)
        for pair in self.ranges:
            for value in pair:
                n = _finite(value)
                if n is None:
                    continue
                numbers.append(math.log10(n) if self.log and n > 0 else n)
        if numbers:
            self.low, self.high = min(numbers), max(numbers)
        else:
            self.low, self.high = 0.0, 1.0
        if self.ranges:
            converted = []
            for value in self.ranges[0]:
                n = _finite(value)
                if n is not None and (not self.log or n > 0):
                    converted.append(math.log10(n) if self.log else n)
            if len(converted) == 2:
                self.low, self.high = min(converted), max(converted)
        elif self.high - self.low < 1e-12:
            self.low -= 0.5
            self.high += 0.5
        else:
            pad = (self.high - self.low) * 0.05
            self.low -= pad
            self.high += pad

    def _range_number(self, value: Any) -> float:
        key = _key(value)
        if key in self.categories:
            return float(self.categories[key])
        n = _finite(value)
        return n if n is not None else 0.0

    def number(self, value: Any) -> float | None:
        if value is None:
            return None
        if self.categorical:
            key = _key(value)
            if key in self.categories:
                return float(self.categories[key])
            n = _finite(value)
            return n if n is not None else 0.0
        n = _finite(value)
        if n is None:
            return None
        if self.log:
            return math.log10(n) if n > 0 else None
        return n

    def frac(self, value: Any) -> float | None:
        number = self.number(value)
        if number is None or abs(self.high - self.low) < 1e-12:
            return None
        fraction = (number - self.low) / (self.high - self.low)
        return 1 - fraction if self.reversed else fraction


@dataclass
class _Subplot:
    key: tuple[str, str]
    xref: str
    yref: str
    x0: float
    y0: float
    x1: float
    y1: float
    clip: str

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    span = high - low
    if not math.isfinite(span) or span <= 0:
        return []
    raw = span / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = magnitude
    for multiplier in (1, 2, 2.5, 5, 10):
        candidate = multiplier * magnitude
        if candidate >= raw:
            step = candidate
            break
    first = math.ceil(low / step) * step
    out: list[float] = []
    for index in range(20):
        value = first + index * step
        if value > high + step * 1e-6:
            break
        out.append(0.0 if abs(value) < 1e-12 else value)
    return out


def _format_tick(value: float, log: bool = False) -> str:
    if log:
        return f"{10 ** value:.3g}"
    if abs(value) >= 10000 or (0 < abs(value) < 0.001):
        return f"{value:.2e}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4g}"


def _short_label(value: str, limit: int = 15) -> str:
    text = str(value)
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        text = text[:10]
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _histogram_bars(trace: dict[str, Any]) -> tuple[list[float], list[float]]:
    xs = [n for n in (_finite(v) for v in _flatten(trace.get("x"))) if n is not None]
    if not xs:
        return [], []
    bins = trace.get("xbins") or {}
    if all(key in bins for key in ("start", "end", "size")):
        try:
            start, end, size = float(bins["start"]), float(bins["end"]), float(bins["size"])
            if size > 0 and end > start:
                edges = list(np.arange(start, end + size, size))
                counts, edges = np.histogram(xs, bins=edges)
                return [float(x) for x in edges], [float(x) for x in counts]
        except (TypeError, ValueError):
            pass
    try:
        count = max(1, min(60, int(trace.get("nbinsx") or 12)))
    except (TypeError, ValueError):
        count = 12
    counts, edges = np.histogram(xs, bins=count)
    return [float(x) for x in edges], [float(x) for x in counts]


def _trace_color(trace: dict[str, Any], index: int, palette: list[str]) -> str:
    line = trace.get("line") or {}
    marker = trace.get("marker") or {}
    color = line.get("color") or marker.get("color") or trace.get("fillcolor")
    if isinstance(color, dict):
        values = _flatten(color)
        color = values[0] if values else None
    elif isinstance(color, (list, tuple)):
        color = color[0] if color else None
    if isinstance(color, (int, float, np.integer, np.floating)) and not isinstance(color, bool):
        return palette[int(color) % len(palette)]
    if isinstance(color, str) and color:
        return color
    return palette[index % len(palette)]


def _color_at(trace: dict[str, Any], index: int, position: int, palette: list[str]) -> str:
    marker = trace.get("marker") or {}
    color = marker.get("color")
    if isinstance(color, dict):
        values = _flatten(color)
        if values:
            value = values[position % len(values)]
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)):
                return palette[int(value) % len(palette)]
    elif isinstance(color, (list, tuple)):
        if color:
            value = color[position % len(color)]
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)):
                return palette[int(value) % len(palette)]
    return _trace_color(trace, index, palette)


def _scale_color(scale: Any, value: float, palette: list[str]) -> str:
    stops: list[tuple[float, str]] = []
    if isinstance(scale, (list, tuple)):
        for item in scale:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    stops.append((float(item[0]), str(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, str) and item:
                stops.append((len(stops) / 3.0, item))
    if not stops:
        stops = [
            (0.0, colours("light")["paper"]),
            (0.5, palette[5 % len(palette)]),
            (1.0, palette[0]),
        ]
    stops.sort(key=lambda pair: pair[0])
    if value <= stops[0][0]:
        return stops[0][1]
    if value >= stops[-1][0]:
        return stops[-1][1]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return c1
            a, b = _rgba(c0), _rgba(c1)
            weight = (value - x0) / (x1 - x0)
            mixed = tuple(round(a[i] * (1 - weight) + b[i] * weight) for i in range(3))
            alpha = a[3] * (1 - weight) + b[3] * weight
            return f"rgba({mixed[0]},{mixed[1]},{mixed[2]},{alpha:.3g})"
    return stops[-1][1]


def _points(trace: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    xs = _flatten(trace.get("x"))
    ys = _flatten(trace.get("y"))
    if not xs and ys:
        xs = list(range(len(ys)))
    if not ys and xs:
        ys = list(range(len(xs)))
    n = min(len(xs), len(ys))
    return xs[:n], ys[:n]


def _error_svg(exc: Exception) -> bytes:
    ink = colours("light")
    body = (
        f'<svg xmlns="{SVG_NS}" width="{DEFAULT_WIDTH}" height="{DEFAULT_HEIGHT}" '
        f'viewBox="0 0 {DEFAULT_WIDTH} {DEFAULT_HEIGHT}">'
        f'<rect width="100%" height="100%" {_paint(ink["paper"])}/>'
        f'<text x="30" y="48" font-family="{_esc(FONT)}" font-size="17" '
        f'{_paint(ink["bad"], "fill")}>'
        "Static vector figure unavailable</text>"
        f'<text x="30" y="76" font-family="{_esc(FONT)}" font-size="12" '
        f'{_paint(ink["muted"], "fill")}>'
        f"{_esc(type(exc).__name__)}: {_esc(str(exc)[:180])}</text></svg>"
    )
    return body.encode("utf-8")


def render_svg(
    figure: dict[str, Any],
    *,
    mode: str = "light",
    title: str = "",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> bytes:
    """Return a standalone, browser-renderable SVG for a Plotly figure dict."""
    try:
        return _render_svg(figure, mode=mode, title=title, width=width, height=height)
    except Exception as exc:  # a broken catalogue item must not break the whole report
        return _error_svg(exc)


def _render_svg(
    figure: dict[str, Any],
    *,
    mode: str,
    title: str,
    width: int,
    height: int,
) -> bytes:
    figure = figure if isinstance(figure, dict) else {}
    layout = dict(figure.get("layout") or {})
    traces = [dict(t) for t in (figure.get("data") or []) if isinstance(t, dict)]
    ink = colours(mode if mode in {"light", "dark", "print"} else "light")
    palette = list(PALETTE_VARIANTS.get(mode, PALETTE_VARIANTS["light"]))
    text_color = str((layout.get("font") or {}).get("color") or ink["text"])
    muted = ink["muted"]
    grid_color = str((layout.get("xaxis") or {}).get("gridcolor") or ink["grid"])
    paper = str(layout.get("paper_bgcolor") or ink["paper"])
    plot_bg = str(layout.get("plot_bgcolor") or paper)
    margin = layout.get("margin") or {}
    left = max(16.0, float(margin.get("l", 60) or 60))
    right = max(12.0, float(margin.get("r", 20) or 20))
    top = max(36.0, float(margin.get("t", 70) or 70))
    bottom = max(28.0, float(margin.get("b", 50) or 50))
    plot_w = max(10.0, float(width) - left - right)
    plot_h = max(10.0, float(height) - top - bottom)

    shapes = [dict(s) for s in (layout.get("shapes") or []) if isinstance(s, dict)]
    annotations = [dict(a) for a in (layout.get("annotations") or []) if isinstance(a, dict)]

    x_raw: dict[str, list[Any]] = {"x": []}
    y_raw: dict[str, list[Any]] = {"y": []}
    x_forced: set[str] = set()
    y_forced: set[str] = set()
    used: set[tuple[str, str]] = {("x", "y")}

    def canon(ref: str, letter: str) -> str:
        return _canonical(ref, letter, layout)

    def collect(
        xref: str,
        yref: str,
        xvalues: list[Any],
        yvalues: list[Any],
        *,
        force_x: bool = False,
        force_y: bool = False,
    ) -> None:
        cx, cy = canon(xref, "x"), canon(yref, "y")
        x_raw.setdefault(cx, []).extend(v for v in xvalues if v is not None)
        y_raw.setdefault(cy, []).extend(v for v in yvalues if v is not None)
        if force_x:
            x_forced.add(cx)
        if force_y:
            y_forced.add(cy)
        used.add((xref, yref))

    for index, trace in enumerate(traces):
        kind = str(trace.get("type") or "scatter")
        xref = _norm_ref(trace.get("xaxis"), "x")
        yref = _norm_ref(trace.get("yaxis"), "y")
        orientation = str(trace.get("orientation") or "v")
        if kind == "heatmap":
            xs = _flatten(trace.get("x"))
            ys = _flatten(trace.get("y"))
            collect(xref, yref, xs, ys, force_x=True, force_y=True)
        elif kind == "histogram":
            xs = _flatten(trace.get("x"))
            edges, counts = _histogram_bars(trace)
            collect(xref, yref, xs + edges, counts + [0])
        elif kind == "bar":
            xs, ys = _points(trace)
            label_values = ys if orientation == "h" else xs
            number_values = xs if orientation == "h" else ys
            non_numeric = any(isinstance(v, str) and _finite(v) is None for v in label_values)
            if orientation == "h":
                collect(xref, yref, number_values + [0], label_values, force_y=non_numeric)
            else:
                collect(xref, yref, label_values, number_values + [0], force_x=non_numeric)
        elif kind == "box":
            values = _flatten(trace.get("y") if orientation != "h" else trace.get("x"))
            labels = _flatten(trace.get("x") if orientation != "h" else trace.get("y"))
            if not labels:
                labels = [trace.get("name") or f"series {index + 1}"]
            if orientation == "h":
                collect(xref, yref, values, labels, force_y=True)
            else:
                collect(xref, yref, labels, values, force_x=True)
        else:
            xs, ys = _points(trace)
            extra_y: list[Any] = []
            if str(trace.get("fill") or "") == "tozeroy":
                extra_y.append(0)
            collect(xref, yref, xs, ys + extra_y)

    for shape in shapes:
        xref = shape.get("xref") or "x"
        yref = shape.get("yref") or "y"
        bx = _base_ref(xref, "x")
        by = _base_ref(yref, "y")
        xvals: list[Any] = []
        yvals: list[Any] = []
        if _is_data_ref(xref, "x"):
            xvals = [shape.get("x0"), shape.get("x1")]
        if _is_data_ref(yref, "y"):
            yvals = [shape.get("y0"), shape.get("y1")]
        collect(bx, by, xvals, yvals)
    for annotation in annotations:
        xref = annotation.get("xref") or "x"
        yref = annotation.get("yref") or "y"
        bx = _base_ref(xref, "x")
        by = _base_ref(yref, "y")
        xvals = [annotation.get("x")] if _is_data_ref(xref, "x") else []
        yvals = [annotation.get("y")] if _is_data_ref(yref, "y") else []
        collect(bx, by, xvals, yvals)

    axis_specs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, value in layout.items():
        if not isinstance(value, dict):
            continue
        if re.fullmatch(r"xaxis\d*", key):
            ref = "x" + key[5:]
            axis_specs.setdefault((canon(ref, "x"), "x"), []).append(value)
        elif re.fullmatch(r"yaxis\d*", key):
            ref = "y" + key[5:]
            axis_specs.setdefault((canon(ref, "y"), "y"), []).append(value)

    def build_group(ref: str, letter: str) -> _Axis:
        specs = axis_specs.get((ref, letter), [])
        axis = _Axis(
            ref=ref,
            raw=list((x_raw if letter == "x" else y_raw).get(ref, [])),
            ranges=[list(spec["range"]) for spec in specs if isinstance(spec.get("range"), (list, tuple)) and len(spec["range"]) == 2],
            forced_category=ref in (x_forced if letter == "x" else y_forced)
            or any(str(spec.get("type") or "") == "category" for spec in specs),
            log=any(str(spec.get("type") or "") == "log" for spec in specs),
            reversed=any(str(spec.get("autorange") or "") == "reversed" for spec in specs),
        )
        axis.build()
        return axis

    x_groups = {ref: build_group(ref, "x") for ref in x_raw}
    y_groups = {ref: build_group(ref, "y") for ref in y_raw}

    subplots: dict[tuple[str, str], _Subplot] = {}

    def subplot_for(xref: str, yref: str) -> _Subplot:
        key = (xref, yref)
        if key in subplots:
            return subplots[key]
        dx0, dx1 = _domain(xref, "x", layout)
        dy0, dy1 = _domain(yref, "y", layout)
        x0 = left + dx0 * plot_w
        x1 = left + dx1 * plot_w
        y0 = top + (1 - dy1) * plot_h
        y1 = top + (1 - dy0) * plot_h
        sub = _Subplot(key, xref, yref, x0, y0, x1, y1, f"clip{len(subplots)}")
        subplots[key] = sub
        return sub

    for xref, yref in used:
        subplot_for(xref, yref)

    def group_for(ref: str, letter: str) -> _Axis:
        canonical = canon(ref, letter)
        groups = x_groups if letter == "x" else y_groups
        if canonical not in groups:
            groups[canonical] = build_group(canonical, letter)
        return groups[canonical]

    def x_px(value: Any, xref: str, sub: _Subplot) -> float | None:
        axis = group_for(xref, "x")
        fraction = axis.frac(value)
        if fraction is None:
            return None
        return sub.x0 + fraction * sub.width

    def y_px(value: Any, yref: str, sub: _Subplot) -> float | None:
        axis = group_for(yref, "y")
        fraction = axis.frac(value)
        if fraction is None:
            return None
        return sub.y1 - fraction * sub.height

    def coord_x(value: Any, ref: Any, base: str, sub: _Subplot) -> float | None:
        text = str(ref).strip() if isinstance(ref, str) else ""
        number = _finite(value)
        if text == "paper" and number is not None:
            return left + number * plot_w
        if text.endswith(" domain") and number is not None:
            return sub.x0 + number * sub.width
        return x_px(value, base, sub)

    def coord_y(value: Any, ref: Any, base: str, sub: _Subplot) -> float | None:
        text = str(ref).strip() if isinstance(ref, str) else ""
        number = _finite(value)
        if text == "paper" and number is not None:
            return top + (1 - number) * plot_h
        if text.endswith(" domain") and number is not None:
            return sub.y1 - number * sub.height
        return y_px(value, base, sub)

    parts: list[str] = [
        f'<svg xmlns="{SVG_NS}" width="{_n(width)}" height="{_n(height)}" '
        f'viewBox="0 0 {_n(width)} {_n(height)}" '
        'role="img" aria-label="Chronos Studio static vector figure">',
        "<desc>Offline pure-SVG fallback; no external scripts, fonts or images.</desc>",
        f'<rect x="0" y="0" width="{_n(width)}" height="{_n(height)}" {_paint(paper)}/>',
    ]
    title_value = layout.get("title") or {}
    title_text = title_value.get("text") if isinstance(title_value, dict) else title_value
    title_text = _plain(title_text or title)
    if title_text:
        lines = [x.strip() for x in title_text.split("|") if x.strip()][:2]
        for line_index, line in enumerate(lines):
            parts.append(
                f'<text x="{_n(left)}" y="{_n(22 + line_index * 19)}" '
                f'font-family="{_esc(FONT)}" font-size="{17 if line_index == 0 else 11.5}" '
                f'font-weight="{"700" if line_index == 0 else "400"}" '
                f'{_paint(text_color, "fill")}>{_esc(line[:150])}</text>'
            )

    if layout.get("showlegend") is not False:
        entries: list[tuple[str, str]] = []
        seen: set[str] = set()
        for index, trace in enumerate(traces):
            if trace.get("showlegend") is False:
                continue
            name = _plain(trace.get("name"))
            if not name or name in seen:
                continue
            seen.add(name)
            entries.append((name, _trace_color(trace, index, palette)))
            if len(entries) >= 8:
                break
        if entries:
            legend_y = max(30.0, top - 12.0)
            item_width = 118.0
            start_x = max(left, width - right - item_width * len(entries))
            for position, (name, color) in enumerate(entries):
                x = start_x + position * item_width
                parts.append(
                    f'<line x1="{_n(x)}" y1="{_n(legend_y)}" x2="{_n(x + 16)}" '
                    f'y2="{_n(legend_y)}" {_stroke(color, 2.2)}/>'
                )
                parts.append(
                    f'<text x="{_n(x + 21)}" y="{_n(legend_y + 4)}" '
                    f'font-family="{_esc(FONT)}" font-size="10.5" '
                    f'{_paint(muted, "fill")}>{_esc(_short_label(name, 16))}</text>'
                )

    parts.append('<defs>')
    for sub in subplots.values():
        parts.append(
            f'<clipPath id="{sub.clip}"><rect x="{_n(sub.x0)}" y="{_n(sub.y0)}" '
            f'width="{_n(sub.width)}" height="{_n(sub.height)}"/></clipPath>'
        )
    parts.append("</defs>")

    ordered = sorted(subplots.values(), key=lambda s: (s.y0, s.x0))
    min_x = min(s.x0 for s in ordered)
    max_y1 = max(s.y1 for s in ordered)
    for sub in ordered:
        parts.append(
            f'<rect x="{_n(sub.x0)}" y="{_n(sub.y0)}" width="{_n(sub.width)}" '
            f'height="{_n(sub.height)}" {_paint(plot_bg)}/>'
        )
        x_axis = group_for(sub.xref, "x")
        y_axis = group_for(sub.yref, "y")
        bottom = abs(sub.y1 - max_y1) < 1.0
        leftmost = abs(sub.x0 - min_x) < 1.0
        if x_axis.categorical and x_axis.categories:
            pairs = sorted(x_axis.categories.items(), key=lambda item: item[1])
            count = min(6, len(pairs))
            positions = sorted({round(i * (len(pairs) - 1) / max(1, count - 1)) for i in range(count)})
            ticks: list[tuple[float, str]] = [(float(pairs[i][1]), pairs[i][0]) for i in positions]
        else:
            ticks = [(value, _format_tick(value, x_axis.log)) for value in _nice_ticks(x_axis.low, x_axis.high)]
        for value, label in ticks:
            x = x_px(value, sub.xref, sub)
            if x is None:
                continue
            parts.append(
                f'<line x1="{_n(x)}" y1="{_n(sub.y0)}" x2="{_n(x)}" '
                f'y2="{_n(sub.y1)}" {_stroke(grid_color, 0.8)}/>'
            )
            if bottom:
                parts.append(
                    f'<text x="{_n(x)}" y="{_n(sub.y1 + 15)}" text-anchor="middle" '
                    f'font-family="{_esc(FONT)}" font-size="9.5" '
                    f'{_paint(muted, "fill")}>{_esc(_short_label(label, 13))}</text>'
                )
        if y_axis.categorical and y_axis.categories:
            pairs_y = sorted(y_axis.categories.items(), key=lambda item: item[1])
            count_y = min(5, len(pairs_y))
            positions_y = sorted({round(i * (len(pairs_y) - 1) / max(1, count_y - 1)) for i in range(count_y)})
            y_ticks: list[tuple[float, str]] = [(float(pairs_y[i][1]), pairs_y[i][0]) for i in positions_y]
        else:
            y_ticks = [(value, _format_tick(value, y_axis.log)) for value in _nice_ticks(y_axis.low, y_axis.high)]
        for value, label in y_ticks:
            y = y_px(value, sub.yref, sub)
            if y is None:
                continue
            parts.append(
                f'<line x1="{_n(sub.x0)}" y1="{_n(y)}" x2="{_n(sub.x1)}" '
                f'y2="{_n(y)}" {_stroke(grid_color, 0.8)}/>'
            )
            if leftmost:
                parts.append(
                    f'<text x="{_n(sub.x0 - 7)}" y="{_n(y + 3)}" text-anchor="end" '
                    f'font-family="{_esc(FONT)}" font-size="9.5" '
                    f'{_paint(muted, "fill")}>{_esc(_short_label(label, 12))}</text>'
                )
        parts.append(
            f'<line x1="{_n(sub.x0)}" y1="{_n(sub.y1)}" x2="{_n(sub.x1)}" '
            f'y2="{_n(sub.y1)}" {_stroke(grid_color, 1.0)}/>'
        )
        x_title = (layout.get(_layout_key(sub.xref, "x")) or {}).get("title") or {}
        x_title_text = _plain(x_title.get("text") if isinstance(x_title, dict) else x_title)
        if x_title_text and bottom:
            parts.append(
                f'<text x="{_n((sub.x0 + sub.x1) / 2)}" y="{_n(sub.y1 + 31)}" '
                f'text-anchor="middle" font-family="{_esc(FONT)}" font-size="10.5" '
                f'{_paint(muted, "fill")}>{_esc(_short_label(x_title_text, 40))}</text>'
            )
        y_title = (layout.get(_layout_key(sub.yref, "y")) or {}).get("title") or {}
        y_title_text = _plain(y_title.get("text") if isinstance(y_title, dict) else y_title)
        if y_title_text and leftmost:
            cx, cy = sub.x0 - 38, (sub.y0 + sub.y1) / 2
            parts.append(
                f'<text x="{_n(cx)}" y="{_n(cy)}" text-anchor="middle" '
                f'transform="rotate(-90 {_n(cx)} {_n(cy)})" '
                f'font-family="{_esc(FONT)}" font-size="10.5" '
                f'{_paint(muted, "fill")}>{_esc(_short_label(y_title_text, 40))}</text>'
            )

    below: dict[tuple[str, str], list[str]] = {key: [] for key in subplots}
    above: dict[tuple[str, str], list[str]] = {key: [] for key in subplots}
    for shape in shapes:
        xref = shape.get("xref") or "x"
        yref = shape.get("yref") or "y"
        bx = _base_ref(xref, "x")
        by = _base_ref(yref, "y")
        sub = subplot_for(bx, by)
        target = below if str(shape.get("layer") or "").lower() == "below" else above
        x0 = coord_x(shape.get("x0"), xref, bx, sub)
        x1 = coord_x(shape.get("x1"), xref, bx, sub)
        y0 = coord_y(shape.get("y0"), yref, by, sub)
        y1 = coord_y(shape.get("y1"), yref, by, sub)
        if None in (x0, x1, y0, y1):
            continue
        spec = shape.get("line") or {}
        if str(shape.get("type") or "line") == "rect":
            fill = shape.get("fillcolor")
            rect = (
                f'<rect x="{_n(min(x0, x1))}" y="{_n(min(y0, y1))}" '
                f'width="{_n(abs(x1 - x0))}" height="{_n(abs(y1 - y0))}" '
            )
            if fill:
                rect += _paint(fill) + " "
            if float(spec.get("width", 0) or 0) > 0:
                rect += _stroke(spec.get("color") or grid_color, spec.get("width"), spec.get("dash")) + " "
            target[sub.key].append(rect + "/>")
        else:
            width = float(spec.get("width", 1.2) or 0)
            if width > 0:
                target[sub.key].append(
                    f'<line x1="{_n(x0)}" y1="{_n(y0)}" x2="{_n(x1)}" '
                    f'y2="{_n(y1)}" {_stroke(spec.get("color") or muted, width, spec.get("dash"))}/>'
                )

    trace_parts: dict[tuple[str, str], list[str]] = {key: [] for key in subplots}
    previous: dict[tuple[str, str], tuple[list[Any], list[Any]]] = {}
    for index, trace in enumerate(traces):
        kind = str(trace.get("type") or "scatter")
        xref = _norm_ref(trace.get("xaxis"), "x")
        yref = _norm_ref(trace.get("yaxis"), "y")
        sub = subplot_for(xref, yref)
        output = trace_parts.setdefault(sub.key, [])
        color = _trace_color(trace, index, palette)
        line = trace.get("line") or {}
        marker = trace.get("marker") or {}
        opacity = float(trace.get("opacity") or 1.0)
        try:
            line_width = float(line.get("width", 1.5) if line.get("width") is not None else 1.5)
        except (TypeError, ValueError):
            line_width = 1.5
        line_width = max(0.0, min(5.0, line_width))

        if kind == "heatmap":
            xs = _flatten(trace.get("x"))
            ys = _flatten(trace.get("y"))
            z_raw = _decode(trace.get("z"))
            rows = [row if isinstance(row, (list, tuple)) else [row] for row in z_raw] if z_raw else []
            flat_z = [n for n in (_finite(v) for v in _flatten(trace.get("z"))) if n is not None]
            if not rows or not flat_z:
                continue
            zmin = _finite(trace.get("zmin"))
            zmax = _finite(trace.get("zmax"))
            low = zmin if zmin is not None else min(flat_z)
            high = zmax if zmax is not None else max(flat_z)
            if high - low < 1e-12:
                high = low + 1.0
            scale = trace.get("colorscale") or []
            for row_index, row in enumerate(rows):
                if row_index >= len(ys):
                    break
                for col_index, value in enumerate(row):
                    if col_index >= len(xs):
                        break
                    n = _finite(value)
                    if n is None:
                        continue
                    xa = x_px(col_index - 0.5, xref, sub)
                    xb = x_px(col_index + 0.5, xref, sub)
                    ya = y_px(row_index - 0.5, yref, sub)
                    yb = y_px(row_index + 0.5, yref, sub)
                    if None in (xa, xb, ya, yb):
                        continue
                    cell_color = _scale_color(scale, (n - low) / (high - low), palette)
                    output.append(
                        f'<rect x="{_n(min(xa, xb))}" y="{_n(min(ya, yb))}" '
                        f'width="{_n(abs(xb - xa))}" height="{_n(abs(yb - ya))}" '
                        f'{_paint(cell_color)}/>'
                    )
            previous[sub.key] = ([], [])
            continue

        if kind == "histogram":
            edges, counts = _histogram_bars(trace)
            for position, count in enumerate(counts):
                if position + 1 >= len(edges):
                    break
                xa = x_px(edges[position], xref, sub)
                xb = x_px(edges[position + 1], xref, sub)
                y_zero = y_px(0, yref, sub)
                y_value = y_px(count, yref, sub)
                if None in (xa, xb, y_zero, y_value):
                    continue
                output.append(
                    f'<rect x="{_n(min(xa, xb))}" y="{_n(min(y_zero, y_value))}" '
                    f'width="{_n(abs(xb - xa))}" height="{_n(abs(y_value - y_zero))}" '
                    f'{_paint(color, extra=float(marker.get("opacity") or 0.7))}/>'
                )
            previous[sub.key] = ([], [])
            continue

        if kind == "bar":
            xs, ys = _points(trace)
            orientation = str(trace.get("orientation") or "v")
            bar_width = max(2.0, min(34.0, sub.width / max(1.0, len(xs) * 1.6)))
            for position, (xv, yv) in enumerate(zip(xs, ys)):
                bar_color = _color_at(trace, index, position, palette)
                if orientation == "h":
                    x_zero = x_px(0, xref, sub)
                    x_value = x_px(xv, xref, sub)
                    y_center = y_px(yv, yref, sub)
                    if None in (x_zero, x_value, y_center):
                        continue
                    output.append(
                        f'<rect x="{_n(min(x_zero, x_value))}" y="{_n(y_center - bar_width / 2)}" '
                        f'width="{_n(abs(x_value - x_zero))}" height="{_n(bar_width)}" '
                        f'{_paint(bar_color, extra=float(marker.get("opacity") or 0.9))}/>'
                    )
                else:
                    x_center = x_px(xv, xref, sub)
                    y_zero = y_px(0, yref, sub)
                    y_value = y_px(yv, yref, sub)
                    if None in (x_center, y_zero, y_value):
                        continue
                    output.append(
                        f'<rect x="{_n(x_center - bar_width / 2)}" '
                        f'y="{_n(min(y_zero, y_value))}" width="{_n(bar_width)}" '
                        f'height="{_n(abs(y_value - y_zero))}" '
                        f'{_paint(bar_color, extra=float(marker.get("opacity") or 0.9))}/>'
                    )
            previous[sub.key] = ([], [])
            continue

        if kind == "box":
            data = _flatten(trace.get("y") if str(trace.get("orientation") or "v") != "h" else trace.get("x"))
            labels = _flatten(trace.get("x") if str(trace.get("orientation") or "v") != "h" else trace.get("y"))
            if not labels:
                labels = [trace.get("name") or f"series {index + 1}"]
            values = [n for n in (_finite(v) for v in data) if n is not None]
            if not values:
                continue
            q1, median, q3 = (float(x) for x in np.percentile(values, [25, 50, 75]))
            iqr = q3 - q1
            low_fence = q1 - 1.5 * iqr
            high_fence = q3 + 1.5 * iqr
            inside = [v for v in values if low_fence <= v <= high_fence]
            whisker_low = min(inside) if inside else min(values)
            whisker_high = max(inside) if inside else max(values)
            horizontal = str(trace.get("orientation") or "v") == "h"
            label = labels[0]
            centre = y_px(label, yref, sub) if horizontal else x_px(label, xref, sub)
            if centre is None:
                continue
            box_w = max(8.0, min(30.0, (sub.height if horizontal else sub.width) / 14.0))
            def value_px(v: float) -> float | None:
                return x_px(v, xref, sub) if horizontal else y_px(v, yref, sub)
            coords = {name: value_px(value) for name, value in {
                "lo": whisker_low, "q1": q1, "med": median, "q3": q3, "hi": whisker_high,
            }.items()}
            if any(v is None for v in coords.values()):
                continue
            if horizontal:
                y_top, y_bottom = centre - box_w / 2, centre + box_w / 2
                output.append(f'<line x1="{_n(coords["lo"])}" y1="{_n(centre)}" x2="{_n(coords["hi"])}" y2="{_n(centre)}" {_stroke(color, 1.2)}/>')
                output.append(f'<rect x="{_n(coords["q1"])}" y="{_n(y_top)}" width="{_n(abs(coords["q3"] - coords["q1"]))}" height="{_n(box_w)}" fill="none" {_stroke(color, 1.2)}/>')
                output.append(f'<line x1="{_n(coords["med"])}" y1="{_n(y_top)}" x2="{_n(coords["med"])}" y2="{_n(y_bottom)}" {_stroke(color, 2.0)}/>')
            else:
                x_left, x_right = centre - box_w / 2, centre + box_w / 2
                output.append(f'<line x1="{_n(centre)}" y1="{_n(coords["lo"])}" x2="{_n(centre)}" y2="{_n(coords["hi"])}" {_stroke(color, 1.2)}/>')
                output.append(f'<rect x="{_n(x_left)}" y="{_n(min(coords["q1"], coords["q3"]))}" width="{_n(box_w)}" height="{_n(abs(coords["q3"] - coords["q1"]))}" fill="none" {_stroke(color, 1.2)}/>')
                output.append(f'<line x1="{_n(x_left)}" y1="{_n(coords["med"])}" x2="{_n(x_right)}" y2="{_n(coords["med"])}" {_stroke(color, 2.0)}/>')
            for value in values:
                if value < low_fence or value > high_fence:
                    point = value_px(value)
                    if point is None:
                        continue
                    if horizontal:
                        output.append(f'<circle cx="{_n(point)}" cy="{_n(centre)}" r="1.8" {_paint(color, extra=0.7)}/>')
                    else:
                        output.append(f'<circle cx="{_n(centre)}" cy="{_n(point)}" r="1.8" {_paint(color, extra=0.7)}/>')
            previous[sub.key] = ([], [])
            continue

        xs, ys = _points(trace)
        if not xs or not ys:
            previous[sub.key] = (xs, ys)
            continue
        pixels: list[tuple[float, float]] = []
        for xv, yv in zip(xs, ys):
            x = x_px(xv, xref, sub)
            y = y_px(yv, yref, sub)
            if x is None or y is None:
                continue
            pixels.append((x, y))
        fill = str(trace.get("fill") or "")
        fill_color = trace.get("fillcolor") or color
        fill_alpha = 0.18 if not trace.get("fillcolor") else 1.0
        polygon: list[tuple[float, float]] = []
        if fill == "toself" and len(pixels) >= 3:
            polygon = pixels + [pixels[0]]
        elif fill == "tozeroy" and len(pixels) >= 2:
            y0 = y_px(0, yref, sub)
            if y0 is not None:
                polygon = [(pixels[0][0], y0)] + pixels + [(pixels[-1][0], y0)]
        elif fill == "tonexty":
            prev_xs, prev_ys = previous.get(sub.key, ([], []))
            if prev_ys:
                upper: list[tuple[float, float]] = []
                for position, (xv, prev_y) in enumerate(zip(xs, prev_ys)):
                    x = x_px(xv, xref, sub)
                    y = y_px(prev_y, yref, sub)
                    if x is not None and y is not None:
                        upper.append((x, y))
                if upper and pixels:
                    polygon = pixels + list(reversed(upper))
        if polygon:
            points = " ".join(f"{_n(x)},{_n(y)}" for x, y in polygon)
            output.append(f'<polygon points="{points}" {_paint(fill_color, extra=fill_alpha)} stroke="none"/>')

        mode_value = str(trace.get("mode") or "lines")
        if line_width > 0 and "none" not in mode_value and pixels:
            commands: list[str] = []
            last_valid = False
            for x, y in pixels:
                commands.append(("M" if not last_valid else "L") + f"{_n(x)} {_n(y)}")
                last_valid = True
            output.append(
                f'<path d="{" ".join(commands)}" fill="none" '
                f'{_stroke(line.get("color") or color, line_width, line.get("dash"), extra=opacity)}/>'
            )
        if "markers" in mode_value and pixels:
            try:
                radius = max(1.0, min(5.0, float(marker.get("size") or 3.5) / 2))
            except (TypeError, ValueError):
                radius = 1.8
            limit = min(len(pixels), 400)
            for position in range(limit):
                x, y = pixels[position]
                mark_color = _color_at(trace, index, position, palette)
                output.append(
                    f'<circle cx="{_n(x)}" cy="{_n(y)}" r="{_n(radius)}" '
                    f'{_paint(mark_color, extra=float(marker.get("opacity") or 0.85))}/>'
                )
        previous[sub.key] = (xs, ys)

    for sub in ordered:
        parts.append(f'<g clip-path="url(#{sub.clip})">')
        parts.extend(below.get(sub.key, []))
        parts.extend(trace_parts.get(sub.key, []))
        parts.extend(above.get(sub.key, []))
        parts.append("</g>")

    for annotation in annotations:
        xref = annotation.get("xref") or "x"
        yref = annotation.get("yref") or "y"
        bx = _base_ref(xref, "x")
        by = _base_ref(yref, "y")
        sub = subplot_for(bx, by)
        x = coord_x(annotation.get("x"), xref, bx, sub)
        y = coord_y(annotation.get("y"), yref, by, sub)
        text = _plain(annotation.get("text"))
        if x is None or y is None or not text:
            continue
        font = annotation.get("font") or {}
        align = str(annotation.get("align") or "").lower()
        paper_x = _finite(annotation.get("x"))
        anchor = (
            "middle"
            if align == "center"
            or (str(xref).strip() == "paper" and paper_x is not None and 0.4 <= paper_x <= 0.6)
            else "end" if align == "right" else "start"
        )
        try:
            size = max(8.0, min(18.0, float(font.get("size") or 10.5)))
        except (TypeError, ValueError):
            size = 10.5
        parts.append(
            f'<text x="{_n(x)}" y="{_n(y)}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" font-family="{_esc(FONT)}" font-size="{_n(size)}" '
            f'{_paint(font.get("color") or muted, "fill")}>{_esc(_short_label(text, 90))}</text>'
        )

    parts.append("</svg>")
    return "".join(parts).encode("utf-8")


__all__ = ["render_svg"]
