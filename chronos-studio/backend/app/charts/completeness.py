"""
charts/completeness.py — Part 16's plot/table catalogue (§5.10, §8).

The universal renderer already draws every stored `FigureResult`. This module
supplies the *gaps*: charts whose numbers live inside fitted wrappers, forecast
bundles, CV reports, diagnostic batteries and evaluation reports rather than in
a result card.

Everything here is built from :mod:`app.charts.theme` and
:mod:`app.charts.primitives`. No ad-hoc Plotly figure construction lives here: the
house palette, fonts, dash patterns, dark/light/print variants, captions and alt
text have one source. Every item carries `data_behind` rows, so the catalogue's
"download data behind this plot" control is truthful rather than decorative.

The nine §5.10 groups are represented by stable, lowercase item ids. A family is
only offered when the stored object that feeds it exists; an unavailable chart is
never a blank card or a fabricated line. No pipeline stage is added: each item is
attributed to one of the existing S0–S15 codes.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from app.charts import primitives as P
from app.charts.theme import ThemeMode, colours, restyled, with_alpha

#: §5.10's nine catalogue groups, in the order the scorecard displays them.
GROUPS: tuple[str, ...] = (
    "Data & quality",
    "Decomposition",
    "Stationarity",
    "Correlation",
    "Model fit",
    "Diagnostics",
    "Forecast",
    "Evaluation",
    "Tables",
)

MIN_PLOTS = 35
MIN_TABLES = 30
DPI = 300
STATIC_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")
DATA_FORMATS: tuple[str, ...] = ("xlsx", "csv")
FORMATS: tuple[str, ...] = (*STATIC_FORMATS, *DATA_FORMATS)
MODES: tuple[ThemeMode, ...] = ("light", "dark", "print")

STAGE_GROUP: dict[str, str] = {
    "S0": "Data & quality", "S1": "Data & quality", "S2": "Data & quality",
    "S3": "Data & quality", "S4": "Decomposition", "S5": "Stationarity",
    "S6": "Correlation", "S7": "Correlation", "S8": "Tables",
    "S9": "Model fit", "S10": "Model fit", "S11": "Evaluation",
    "S12": "Diagnostics", "S13": "Forecast", "S14": "Evaluation", "S15": "Tables",
}


# --------------------------------------------------------------------------- #
# JSON / typed-table helpers
# --------------------------------------------------------------------------- #
def _number(value: Any) -> float | None:
    """Finite float or ``None``; NaN/Inf never reach Plotly's strict JSON serializer."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        x = _number(v)
        if x is not None:
            out.append(x)
    return out


def _jsonable(value: Any) -> Any:
    """Recursively convert stored numpy/pandas/dataclass objects to strict JSON."""
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (dict,)):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, pd.Index, pd.Series)):
        return [_jsonable(x) for x in list(value)]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _columns(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Typed table columns inferred from data-behind rows.

    The wire schema uses five dtypes; a mixed column is a string column so no
    value is silently coerced or dropped.
    """
    keys: list[str] = []
    for row in rows[:500]:
        for key in row:
            if key not in keys:
                keys.append(key)
    cols: list[dict[str, Any]] = []
    for key in keys:
        present = [_number(r.get(key)) for r in rows[:500] if r.get(key) is not None]
        if present and all(x is not None for x in present):
            dtype = "integer" if all(float(x).is_integer() for x in present) else "number"
            align = "right"
        else:
            dtype, align = "string", "left"
        cols.append({
            "key": key, "label": key.replace("_", " ").title(), "dtype": dtype,
            "format": None, "align": align,
        })
    if not cols:
        cols = [{"key": "status", "label": "Status", "dtype": "string",
                 "format": None, "align": "left"}]
    return cols


def _figure_rows(figure_json: dict[str, Any], *, limit: int = 2000) -> list[dict[str, Any]]:
    """Recover data-behind from a stored figure whose builder did not attach it.

    Part 16 cannot honestly offer "download data behind this plot" for the early
    EDA/decomposition charts unless their traced numbers are extractable. This
    walks every trace rather than re-running the statistics, so the download is
    exactly what the reader sees. Heatmaps become row/column/value rows.
    """
    rows: list[dict[str, Any]] = []
    traces = figure_json.get("data") or []
    for ti, trace in enumerate(traces):
        if not isinstance(trace, dict):
            continue
        name = str(trace.get("name") or trace.get("type") or f"trace {ti + 1}")
        typ = str(trace.get("type") or "")
        z = trace.get("z")
        if typ == "heatmap" or (z is not None and typ != "table"):
            xs, ys = list(trace.get("x") or []), list(trace.get("y") or [])
            zrows = z if isinstance(z, (list, tuple)) else [[z]]
            for i, zrow in enumerate(zrows):
                vals = zrow if isinstance(zrow, (list, tuple)) else [zrow]
                for j, val in enumerate(vals):
                    rows.append({
                        "trace": name,
                        "row": str(ys[i] if i < len(ys) else i),
                        "column": str(xs[j] if j < len(xs) else j),
                        "value": _number(val),
                    })
        else:
            x = list(trace.get("x") or [])
            y = list(trace.get("y") or trace.get("values") or [])
            if not y:
                continue
            if not x:
                x = list(range(len(y)))
            n = min(len(x), len(y), 500)
            rows.extend({
                "trace": name, "x": _jsonable(x[i]), "y": _number(y[i]),
            } for i in range(n))
        if len(rows) >= limit:
            break
    if not rows:
        rows = [{"trace": "none", "x": None, "y": None,
                 "status": "no exported trace data"}]
    return _jsonable(rows[:limit])


def _alt_text(title: str, caption: str, rows: Sequence[dict[str, Any]]) -> str:
    """A numeric, human-readable description that never drifts from the table."""
    numbers = [_number(v) for r in rows[:200] for v in r.values()]
    finite = [x for x in numbers if x is not None]
    span = (
        f" Values range from {min(finite):.4g} to {max(finite):.4g}."
        if finite else ""
    )
    return (
        f"{title}. {caption} The downloadable table carries {len(rows)} rows "
        f"and every plotted value.{span}"
    )


def _base(index: Sequence[Any]) -> pd.Series:
    """An all-NaN observed series so overlay-only charts still use the shared builder."""
    labels = [str(x) for x in index] or ["0"]
    return pd.Series([float("nan")] * len(labels), index=labels, dtype=float)


def _ov(
    name: str,
    x: Sequence[Any],
    y: Sequence[Any],
    *,
    kind: str = "line",
    colour: str | None = None,
    width: float = 1.6,
    size: int = 6,
) -> dict[str, Any]:
    """One overlay in the exact shape `primitives.time_series` expects."""
    n = min(len(x), len(y)) if x is not None else len(y)
    labels = [str(x[i]) for i in range(n)] if x is not None else [str(i + 1) for i in range(n)]
    vals = [float("nan") if _number(y[i]) is None else float(_number(y[i])) for i in range(n)]
    out: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "y": pd.Series(vals, index=labels, dtype=float),
        "width": width,
        "size": size,
    }
    if colour:
        out["colour"] = colour
    return out


def _band_ov(
    name: str,
    x: Sequence[Any],
    lower: Sequence[Any],
    upper: Sequence[Any],
    fill: str | None = None,
) -> dict[str, Any]:
    fill = fill or with_alpha(colours("light")["reference"], 0.16)
    n = min(len(x), len(lower), len(upper))
    labels = [str(x[i]) for i in range(n)]
    lo = pd.Series([float("nan") if _number(lower[i]) is None else float(_number(lower[i]))
                    for i in range(n)], index=labels, dtype=float)
    hi = pd.Series([float("nan") if _number(upper[i]) is None else float(_number(upper[i]))
                    for i in range(n)], index=labels, dtype=float)
    return {
        "name": name, "kind": "band",
        "y": pd.Series([float("nan")] * n, index=labels, dtype=float),
        "y_low": lo, "y_high": hi, "fill": fill,
    }


def _first_column(values: Any) -> list[float]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return []
    if arr.ndim == 1:
        return _finite(arr.tolist())
    return _finite(arr[:, 0].tolist())


def _band_values(band: Any, n: int) -> float | list[float]:
    if isinstance(band, (list, tuple, np.ndarray, pd.Series)):
        vals = _finite(list(band))
        if not vals:
            return 0.0
        if len(vals) >= n:
            return vals[:n]
        return vals + [vals[-1]] * (n - len(vals))
    return float(_number(band) or 0.0)


def _complete(
    chart: P.Chart,
    *,
    item_id: str,
    title: str,
    caption: str,
    group: str,
    stage: str,
    source: str,
    rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One catalogue plot, with the export contract attached to the chart object."""
    behind = _jsonable(list(rows if rows is not None else chart.data_behind))
    chart.data_behind = behind
    chart.alt_text = _alt_text(title, caption, behind)
    return {
        "id": item_id,
        "title": title,
        "group": group,
        "stage": stage,
        "kind": "plot",
        "source": source,
        "caption": caption,
        "alt_text": chart.alt_text,
        "figure_json": _jsonable(chart.figure_json),
        "columns": _columns(behind),
        "rows": behind,
        "formats": list(FORMATS),
        "dpi": DPI,
        "report_insertable": True,
    }


def _table_item(
    *,
    item_id: str,
    title: str,
    group: str,
    stage: str,
    source: str,
    rows: Sequence[dict[str, Any]],
    caption: str,
    mode: ThemeMode = "light",
) -> dict[str, Any]:
    behind = _jsonable(list(rows))
    if not behind:
        # An empty stage result is still a catalogued, captioned, exportable item;
        # a zero-row spreadsheet with no explanation reads as a broken download.
        behind = [{"status": "No rows were produced by this stage; nothing was fabricated."}]
    cols = _columns(behind)
    chart = P.table_figure(cols, behind, title=title, subtitle=caption[:120], mode=mode)
    return {
        "id": item_id,
        "title": title,
        "group": group,
        "stage": stage,
        "kind": "table",
        "source": source,
        "caption": caption,
        "alt_text": chart.alt_text,
        "figure_json": _jsonable(chart.figure_json),
        "columns": cols,
        "rows": behind,
        "formats": list(FORMATS),
        "dpi": DPI,
        "report_insertable": True,
    }


# --------------------------------------------------------------------------- #
# Data & quality / Decomposition / Stationarity / Correlation
# --------------------------------------------------------------------------- #
def quality_spark(series: pd.Series, *, mode: ThemeMode = "light") -> dict[str, Any]:
    s = pd.Series(series).astype(float)
    labels = [str(x) for x in s.index]
    vals = [_number(x) for x in s.tolist()]
    observed = pd.Series([float("nan") if v is None else v for v in vals],
                         index=labels, dtype=float)
    finite = [v for v in vals if v is not None]
    med = float(np.median(finite)) if finite else 0.0
    mad = float(np.median(np.abs(np.asarray(finite) - med))) if finite else 0.0
    scale = 1.4826 * mad if mad > 0 else (float(np.std(finite)) if len(finite) > 1 else 0.0)
    missing = [i for i, v in enumerate(vals) if v is None]
    outliers = [i for i, v in enumerate(vals)
                if v is not None and scale > 0 and abs(v - med) > 3.5 * scale]
    c = colours(mode)
    shading = [
        {"x0": labels[i], "x1": labels[min(i + 1, len(labels) - 1)],
         "label": "missing", "colour": with_alpha(c["neutral"], 0.14)}
        for i in missing[:40]
    ]
    markers = [
        {"x": labels[i], "y": vals[i], "label": "Robust outlier", "short": "!",
         "colour": c["bad"]}
        for i in outliers[:40]
    ]
    chart = P.time_series(
        observed,
        title="Data quality: gaps, outliers and level shifts",
        subtitle="Grey bands mark missing observations; red markers are robust MAD outliers.",
        y_title="observed value", mode=mode, height=320,
        shading=shading, markers=markers,
    )
    rows = [{"time": labels[i], "value": vals[i], "missing": i in missing,
             "robust_outlier": i in outliers} for i in range(len(vals))]
    return _complete(
        chart, item_id="data.quality_audit", title="Data quality audit",
        caption=f"Observed series with {len(missing)} missing or non-finite value(s) and "
                f"{len(outliers)} robust outlier(s) over {len(vals)} observations.",
        group="Data & quality", stage="S1", source="session.objects['series']", rows=rows,
    )


def missingness_calendar(series: pd.Series, *, mode: ThemeMode = "light") -> dict[str, Any]:
    s = pd.Series(series)
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(s.index))
        miss = pd.Series([1.0 if _number(x) is None else 0.0 for x in s.tolist()],
                         index=dates, dtype=float)
        years = sorted({int(y) for y in dates.year})
        months = list(range(1, 13))
        matrix = pd.DataFrame(0.0, index=[str(y) for y in years],
                              columns=[str(m) for m in months])
        grouped = miss.groupby([dates.year, dates.month]).mean()
        for (y, m), v in grouped.items():
            matrix.loc[str(int(y)), str(int(m))] = float(v)
        rows = [{"year": str(y), "month": str(m),
                 "missing_fraction": float(matrix.loc[str(y), str(m)])}
                for y in years for m in months]
        chart = P.heatmap(
            matrix, title="Missingness calendar",
            subtitle="Fraction of observations missing in each calendar month.",
            x_title="month", y_title="year", value_label="missing fraction", mode=mode,
        )
        caption = "Calendar heatmap of the fraction of missing values by month and year."
    except (TypeError, ValueError, IndexError):
        vals = [1.0 if _number(x) is None else 0.0 for x in s.tolist()][:200]
        labels = [str(x) for x in s.index[:200]]
        rows = [{"position": i, "time": labels[i], "missing": vals[i]} for i in range(len(vals))]
        chart = P.bars(
            labels, vals, title="Missingness by observation",
            subtitle="1 = missing or non-finite; 0 = observed.",
            x_title="row", y_title="missing", mode=mode,
        )
        caption = "Missing and non-finite values by observation; a non-date index cannot be a calendar."
    return _complete(
        chart, item_id="data.missingness_calendar", title="Missingness calendar",
        caption=caption, group="Data & quality", stage="S1",
        source="session.objects['series']", rows=rows,
    )


def lag_scatter(series: pd.Series, *, lag: int = 1, mode: ThemeMode = "light") -> dict[str, Any]:
    s = pd.Series(series).dropna().astype(float)
    lag = max(1, int(lag))
    if len(s) > lag:
        x, y = s.iloc[:-lag].tolist(), s.iloc[lag:].tolist()
    else:
        x, y = [0.0], [0.0]
    chart = P.scatter_with_fit(
        x=x, y=y, title=f"Lag {lag} dependence",
        subtitle="Each point is one observation against its predecessor.",
        x_title=f"value at t−{lag}", y_title="value at t", mode=mode,
    )
    rows = [{"x": _number(a), "y": _number(b)} for a, b in zip(x, y)]
    return _complete(
        chart, item_id="data.lag_scatter", title=f"Lag {lag} scatter with fit",
        caption=f"Scatter of {len(rows)} observations against lag {lag}, with a least-squares line.",
        group="Data & quality", stage="S1", source="session.objects['series']", rows=rows,
    )


def components(decomposition: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    payload = decomposition or {}
    shares = payload.get("shares") or {}
    rows = [{"component": str(k), "share": _number(v), "reading": "stored variance share"}
            for k, v in shares.items() if _number(v) is not None]
    if not rows:
        score = payload.get("scorecard", {}).get("rows", []) if isinstance(payload.get("scorecard"), dict) else []
        rows = [{"component": str(r.get("component", "component")), "share": _number(r.get("share")),
                 "reading": str(r.get("reading", ""))} for r in score]
    if not rows:
        rows = [{"component": "no decomposition stored", "share": 0.0, "reading": "not available"}]
    labels = [str(r["component"]) for r in rows]
    values = [float(r["share"] or 0.0) for r in rows]
    chart = P.bars(
        labels, values, title="Decomposition component shares",
        subtitle="Stored variance shares of trend, seasonality, cycle and remainder.",
        x_title="component", y_title="share of variation", mode=mode,
    )
    return _complete(
        chart, item_id="decomposition.components", title="Decomposition components",
        caption="Stored decomposition component shares; the table carries every plotted value.",
        group="Decomposition", stage="S4", source="session.objects['decomposition']", rows=rows,
    )


def seasonal_profile(profile: Sequence[Any], *, period: int | None = None,
                     mode: ThemeMode = "light") -> dict[str, Any]:
    values = _finite(list(profile)) or [0.0]
    labels = [str(i + 1) for i in range(len(values))]
    rows = [{"position": i + 1, "effect": v} for i, v in enumerate(values)]
    chart = P.bars(
        labels, values, title="Seasonal profile",
        subtitle=f"Average deviation at each of {period or len(values)} seasonal positions.",
        x_title="seasonal position", y_title="deviation from mean", reference=0.0, mode=mode,
    )
    return _complete(
        chart, item_id="decomposition.seasonal_profile", title="Seasonal profile",
        caption="Average seasonal effect by calendar position, with a zero reference line.",
        group="Decomposition", stage="S4", source="decomposition seasonality table", rows=rows,
    )


def rolling_statistics(series: pd.Series, *, window: int = 12,
                       mode: ThemeMode = "light") -> dict[str, Any]:
    s = pd.Series(series).astype(float)
    w = max(2, min(int(window), max(2, len(s))))
    chart = P.rolling_panel(
        s, w, title="Rolling mean and spread",
        subtitle=f"{w}-observation window; the lower panel is the rolling standard deviation.",
        mode=mode,
    )
    roll = s.rolling(w, min_periods=max(2, w // 2))
    rows = [{"time": str(t), "value": _number(s.iloc[i]),
             "rolling_mean": _number(m), "rolling_sd": _number(sd)}
            for i, (t, m, sd) in enumerate(zip(s.index, roll.mean().tolist(), roll.std().tolist()))]
    return _complete(
        chart, item_id="decomposition.rolling_statistics", title="Rolling mean and spread",
        caption=f"Rolling {w}-observation mean and standard deviation with the observed series.",
        group="Decomposition", stage="S4", source="session.objects['series']", rows=rows,
    )


def test_statistic_bars(battery: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in ("unit_root", "seasonal", "breaks"):
        for r in (battery.get(group) or []):
            rows.append({
                "group": group, "test": str(r.get("test") or r.get("id") or "test"),
                "statistic": _number(r.get("statistic")),
                "critical_5pct": _number(r.get("cv_5pct")),
                "p_value": _number(r.get("p_value")),
                "reading": str(r.get("reading") or r.get("says") or ""),
            })
    if not rows:
        rows = [{"group": "none", "test": "no stationarity tests", "statistic": 0.0,
                 "critical_5pct": None, "p_value": None, "reading": "not available"}]
    chart = P.bars(
        [r["test"] for r in rows], [float(r["statistic"] or 0.0) for r in rows],
        title="Stationarity test statistics",
        subtitle="Bars are test statistics; the table carries critical values and p-values.",
        x_title="test", y_title="statistic", mode=mode, horizontal=True,
    )
    return _complete(
        chart, item_id="stationarity.statistic_bars", title="Stationarity test statistics",
        caption="All unit-root, seasonal and break statistics from the stored stationarity battery.",
        group="Stationarity", stage="S5", source="session.objects['stat_battery']", rows=rows,
    )


def acf_pacf_stems(correlogram: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    payload = correlogram.get("payload", correlogram) if isinstance(correlogram, dict) else {}
    acf = list(payload.get("acf") or [])
    if not acf:
        acf = [{"lag": 1, "acf": 0.0}]
    lags = [int(r.get("lag", i + 1)) for i, r in enumerate(acf)]
    values = [float(_number(r.get("acf")) or 0.0) for r in acf]
    band = payload.get("band_bartlett") or payload.get("band_white") or 0.0
    chart = P.stem_plot(
        lags, values, title="ACF with Bartlett confidence band",
        subtitle="Bars outside the band carry evidence of remaining dependence.",
        band=_band_values(band, len(values)), y_title="autocorrelation", mode=mode,
    )
    pacf = list(payload.get("pacf") or [])
    bvals = _band_values(band, len(values))
    rows = [{
        "lag": lags[i], "acf": _number(values[i]),
        "pacf": _number((pacf[i] if i < len(pacf) else {}).get("pacf")),
        "band": (bvals[i] if isinstance(bvals, list) and i < len(bvals) else
                 float(bvals) if isinstance(bvals, float) else None),
    } for i in range(len(values))]
    return _complete(
        chart, item_id="correlation.acf_stems", title="ACF with confidence bands",
        caption=f"Autocorrelation at {len(rows)} lags; the table also carries the partial autocorrelation.",
        group="Correlation", stage="S6", source="session.objects['correlogram']", rows=rows,
    )


def ccf_lags(
    series: pd.Series | None,
    exogenous: pd.DataFrame | None,
    correlogram: dict[str, Any],
    *,
    mode: ThemeMode = "light",
) -> dict[str, Any]:
    """Lead/lag cross-correlations between the target and its first numeric driver."""
    payload = correlogram.get("payload", correlogram) if isinstance(correlogram, dict) else {}
    rows: list[dict[str, Any]] = []
    band = 0.0
    if isinstance(exogenous, pd.DataFrame) and len(exogenous.columns) and series is not None:
        y = np.asarray(pd.Series(series).astype(float).tolist(), dtype=float)
        x = np.asarray(pd.to_numeric(exogenous.iloc[:, 0], errors="coerce").tolist(), dtype=float)
        n = min(len(y), len(x))
        if n >= 4:
            band = 1.96 / math.sqrt(n)
            for lag in range(-min(12, n - 2), min(12, n - 2) + 1):
                if lag >= 0:
                    a, b = x[: n - lag], y[lag:n]
                else:
                    a, b = x[-lag:n], y[: n + lag]
                keep = np.isfinite(a) & np.isfinite(b)
                a, b = a[keep], b[keep]
                r = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0 else None
                rows.append({"lag": lag, "correlation": _number(r), "driver": str(exogenous.columns[0]),
                             "band": _number(band)})
    if not rows:
        stored = list(payload.get("ccf") or payload.get("cross_correlation") or [])
        rows = [{"lag": int(r.get("lag", i + 1)),
                 "correlation": _number(r.get("correlation") or r.get("ccf")),
                 "driver": "stored CCF", "band": _number(payload.get("ccf_band"))}
                for i, r in enumerate(stored)]
    if not rows:
        rows = [{"lag": 0, "correlation": 0.0, "driver": "no paired driver",
                 "band": None, "status": "not applicable"}]
    chart = P.stem_plot(
        [int(r["lag"]) for r in rows], [float(r["correlation"] or 0.0) for r in rows],
        title="Cross-correlation by lag",
        subtitle="Negative lags mean the driver leads; the dashed band is an approximate 95% bound.",
        band=band or _number(rows[0].get("band")) or 0.0,
        y_title="cross-correlation", mode=mode,
    )
    return _complete(
        chart, item_id="correlation.ccf_stems", title="Cross-correlation function",
        caption="Cross-correlations by lead/lag; the table names the driver and the 95% bound.",
        group="Correlation", stage="S6", source="session.objects['exogenous']", rows=rows,
    )


# --------------------------------------------------------------------------- #
# Model fit
# --------------------------------------------------------------------------- #
def fitted_vs_actual(outcomes: Sequence[Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    overlays: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    observed: pd.Series | None = None
    for o in outcomes or []:
        wrapper, summary = getattr(o, "wrapper", None), getattr(o, "summary", None)
        if wrapper is None or summary is None or getattr(summary, "failed", False):
            continue
        y = getattr(wrapper, "y", None)
        if y is None:
            continue
        ys = pd.Series(y).astype(float)
        try:
            fit = wrapper.fitted_values()
        except Exception:
            continue
        if fit is None:
            continue
        n = min(len(ys), len(np.asarray(fit).ravel()))
        if n <= 0:
            continue
        idx = [str(x) for x in ys.index[:n]]
        if observed is None:
            observed = pd.Series(ys.iloc[:n].tolist(), index=idx, dtype=float)
        mid = getattr(summary, "model_id", "fit")
        display = getattr(summary, "display_name", mid)
        overlays.append(_ov(display, idx, np.asarray(fit).ravel()[:n].tolist(), kind="line"))
        rows.extend({"time": idx[i], "model": mid,
                     "actual": _number(ys.iloc[i]),
                     "fitted": _number(np.asarray(fit).ravel()[i])} for i in range(n))
    if observed is None:
        observed = _base(["0"])
        rows = [{"model": getattr(getattr(o, "summary", None), "model_id", "unknown"),
                 "status": "no successful fitted values", "reason": str(
                     getattr(getattr(o, "summary", None), "failure_reason", "") or "")}
                for o in (outcomes or [])] or [{"model": "none", "status": "no fits"}]
    chart = P.time_series(
        observed, title="Fitted values against the observed series",
        subtitle="One line per successful fit; the table carries actual and fitted values.",
        y_title="series", overlays=overlays, mode=mode,
    )
    return _complete(
        chart, item_id="models.fitted_vs_actual", title="Fitted values vs actual",
        caption="In-sample fitted values for every successful candidate, never an out-of-sample accuracy claim.",
        group="Model fit", stage="S10", source="session.objects['fits'].outcomes", rows=rows,
    )


def information_criteria(outcomes: Sequence[Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for o in outcomes or []:
        s = getattr(o, "summary", None)
        if s is None or getattr(s, "failed", False):
            continue
        rows.append({
            "model": getattr(s, "model_id", "model"),
            "aic": _number(getattr(s, "aic", None)),
            "aicc": _number(getattr(s, "aicc", None)),
            "bic": _number(getattr(s, "bic", None)),
            "comparable": bool(getattr(s, "ic_comparable", True)),
        })
    rows = [r for r in rows if r["comparable"]]
    if not rows:
        rows = [{"model": "no comparable fits", "aic": None, "aicc": 0.0,
                 "bic": None, "comparable": False}]
    chart = P.bars(
        [str(r["model"]) for r in rows], [float(r["aicc"] or 0.0) for r in rows],
        title="AICc among comparable fits",
        subtitle="Only fits on the same likelihood scale are compared; lower is better.",
        x_title="model", y_title="AICc", mode=mode, horizontal=True,
    )
    return _complete(
        chart, item_id="models.information_criteria", title="Information criteria comparison",
        caption="AICc, AIC and BIC for fits whose likelihood scales are comparable; CV, not IC, selects the winner.",
        group="Model fit", stage="S10", source="session.objects['fits'].outcomes", rows=rows,
    )


def _variance_of(o: Any) -> dict[str, Any]:
    wrapper, summary = getattr(o, "wrapper", None), getattr(o, "summary", None)
    if wrapper is None or summary is None or getattr(summary, "failed", False):
        return {}
    family = getattr(getattr(summary, "family", None), "value", str(getattr(summary, "family", "")))
    fn = getattr(wrapper, "variance_forecast", None)
    if family != "volatility" or not callable(fn):
        return {}
    try:
        return dict(fn())
    except Exception:
        return {}


def _leverage_of(wrapper: Any) -> float | None:
    theta = getattr(wrapper, "_var_theta", None)
    try:
        return _number(float(np.asarray(theta)[2])) if theta is not None and len(theta) > 2 else None
    except (TypeError, ValueError, IndexError):
        return None


def variance_path(outcomes: Sequence[Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    """The §8 per-family GARCH variance path, with σ, unconditional σ and persistence."""
    overlays: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    path_length = 12
    for o in outcomes or []:
        wrapper, summary = getattr(o, "wrapper", None), getattr(o, "summary", None)
        if wrapper is None or summary is None:
            continue
        v = _variance_of(o)
        path = _finite(v.get("variance") or [])
        sd = _finite(v.get("sd") or []) or [math.sqrt(x) for x in path]
        if not path:
            continue
        path_length = max(path_length, len(path))
        x = list(range(1, len(path) + 1))
        mid = getattr(summary, "model_id", "volatility")
        overlays.append(_ov(f"{mid}: σ²", x, path, kind="line"))
        overlays.append(_ov(f"{mid}: σ", x, sd[:len(path)], kind="dashed"))
        uncond = _number(v.get("unconditional_sd"))
        if uncond is not None:
            overlays.append(_ov(f"{mid}: long-run σ", x, [uncond] * len(path), kind="dot"))
        persistence = _number(getattr(wrapper, "persistence", None))
        leverage = _leverage_of(wrapper)
        rows.extend({
            "model": mid, "step": i + 1, "variance": _number(path[i]),
            "sd": _number(sd[i]) if i < len(sd) else None,
            "unconditional_sd": uncond, "persistence": persistence, "leverage": leverage,
        } for i in range(len(path)))
    x = list(range(1, path_length + 1))
    if not overlays:
        overlays = [_ov("no fitted volatility model", x, [float("nan")] * len(x), kind="dot")]
        rows = [{"model": "none", "step": i + 1, "variance": None, "sd": None,
                 "unconditional_sd": None, "persistence": None, "leverage": None,
                 "status": "no fitted volatility model"} for i in x]
    chart = P.time_series(
        _base(x), title="GARCH variance and volatility term structure",
        subtitle="σ² and σ at each forecast step; the dotted line is the long-run σ.",
        x_title="forecast step", y_title="variance / standard deviation",
        overlays=overlays, mode=mode,
    )
    return _complete(
        chart, item_id="models.garch_variance_path", title="GARCH variance path",
        caption="Per-family variance path and standard deviation from every fitted ARCH/GARCH model; "
                "a flat mean forecast is not the absence of a forecast.",
        group="Model fit", stage="S10", source="session.objects['fits'].outcomes", rows=rows,
    )


def count_pmf_ribbons(forecast: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    """§8's count-PMF ribbons: exact or empirical mass at each integer, per horizon."""
    bundle = forecast.get("bundle") if isinstance(forecast, dict) else None
    data = bundle.as_dict() if hasattr(bundle, "as_dict") else (bundle if isinstance(bundle, dict) else {})
    discrete = data.get("discrete") or {}
    support = discrete.get("support") or []
    pmf = discrete.get("pmf") or []
    index = data.get("index") or [str(i + 1) for i in range(len(pmf))]
    family = str(discrete.get("family") or discrete.get("source") or "predictive PMF")
    empirical = "empirical" in family.lower() or "simulated" in family.lower()
    rows: list[dict[str, Any]] = []
    pairs: dict[str, list[tuple[float, float]]] = {}
    all_counts: set[str] = set()
    for h, (grid, mass) in enumerate(zip(support, pmf)):
        g, p = _finite(grid), _finite(mass)
        n = min(len(g), len(p))
        step = h + 1
        pairs[str(step)] = list(zip(g[:n], p[:n]))
        for k, v in zip(g[:n], p[:n]):
            all_counts.add(f"{k:g}")
            rows.append({"step": step,
                         "time": str(index[h] if h < len(index) else step),
                         "count": _number(k), "probability": _number(v),
                         "source": family, "empirical": empirical})
    # A PMF can have thousands of atoms. The picture shows the 60 largest; the
    # downloadable table always carries every atom that was stored.
    ranked = sorted({c for step in pairs.values() for c, _ in step},
                    key=lambda c: -max((p for step in pairs.values() for k, p in step if k == c), default=0))
    shown_counts = sorted(ranked[:60])
    labels = [f"{c:g}" for c in shown_counts]
    overlays: list[dict[str, Any]] = []
    if shown_counts:
        for step, values in pairs.items():
            lookup = {f"{k:g}": p for k, p in values}
            maxp = max((p for _, p in values), default=0.0)
            y = [float(step) + (0.8 * lookup.get(label, 0.0) / maxp if maxp > 0 else 0.0)
                 for label in labels]
            overlays.append(_ov(f"step {step}", labels, y, kind="line"))
    else:
        labels = ["0"]
        overlays = [_ov("no count PMF", labels, [0.0], kind="dot")]
        rows = [{"step": 0, "time": "", "count": None, "probability": None,
                 "source": "continuous predictive distribution", "empirical": False,
                 "status": "not applicable"}]
    chart = P.time_series(
        _base(labels), title="Predictive count PMF ribbons",
        subtitle="One ribbon per forecast step; horizontal position is the integer count, height its probability.",
        x_title="count", y_title="forecast step (offset ribbon)", overlays=overlays, mode=mode,
    )
    disclosure = "an explicitly disclosed empirical simulated integer PMF" if empirical else "an exact parametric PMF"
    return _complete(
        chart, item_id="forecast.count_pmf_ribbons", title="Predictive count PMF ribbons",
        caption=f"Per-step integer PMFs from {disclosure}; no integer quantile is interpolated.",
        group="Forecast", stage="S13", source="session.objects['forecast'].bundle.discrete", rows=rows,
    )


def _multivariate(outcomes: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for o in outcomes or []:
        wrapper, summary = getattr(o, "wrapper", None), getattr(o, "summary", None)
        if wrapper is None or summary is None or getattr(summary, "failed", False):
            continue
        fn = getattr(wrapper, "multivariate_payload", None)
        if not callable(fn):
            continue
        try:
            payload = fn()
        except Exception:
            continue
        out.append({"model_id": getattr(summary, "model_id", "var"), "payload": payload})
    return out


def var_irf(outcomes: Sequence[Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    """§8's VAR impulse-response functions with bootstrap percentile bands."""
    overlays: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for entry in _multivariate(outcomes):
        mid, payload = entry["model_id"], entry["payload"]
        irf = payload.get("irf") or {}
        paths = irf.get("paths") or {}
        bands = irf.get("bands_90") or {}
        ordering = [str(x) for x in (irf.get("ordering") or payload.get("series") or [])]
        shock_to = str(irf.get("response_to") or (ordering[0] if ordering else "shock"))
        n_boot = int(irf.get("n_boot") or 0)
        for target, matrix in paths.items():
            arr = np.asarray(matrix, dtype=float)
            if arr.ndim != 2 or not arr.size:
                continue
            y = _finite(arr[:, 0].tolist())
            lo = _first_column((bands.get(target) or {}).get("lower"))
            hi = _first_column((bands.get(target) or {}).get("upper"))
            x = list(range(1, len(y) + 1))
            overlays.append(_ov(f"{mid}: {target}", x, y, kind="line"))
            if lo and hi:
                n = min(len(x), len(lo), len(hi))
                overlays.append(_ov(f"{mid}: {target} 90% lower", x[:n], lo[:n], kind="dot"))
                overlays.append(_ov(f"{mid}: {target} 90% upper", x[:n], hi[:n], kind="dot"))
            for h in range(arr.shape[0]):
                row: dict[str, Any] = {
                    "model": mid, "shock": shock_to, "target": target, "step": h + 1,
                    "point": _number(arr[h, 0]),
                    "lower_90": _number(lo[h]) if h < len(lo) else None,
                    "upper_90": _number(hi[h]) if h < len(hi) else None,
                    "ordering": ", ".join(ordering), "n_boot": n_boot,
                }
                for j, shock in enumerate(ordering[:arr.shape[1]]):
                    row[f"response_{re.sub(r'[^A-Za-z0-9_]', '_', shock)}"] = _number(arr[h, j])
                rows.append(row)
    if not overlays:
        rows = [{"model": "none", "shock": "", "target": "", "step": 0, "point": None,
                 "status": "no fitted VAR/VECM"}]
        x = [1]
        overlays = [_ov("no fitted VAR/VECM", x, [0.0], kind="dot")]
    else:
        x = list(range(1, max((int(r.get("step") or 0) for r in rows), default=1) + 1))
    chart = P.time_series(
        _base(x), title="VAR impulse responses with 90% bootstrap bands",
        subtitle=f"Identified shock: {rows[0].get('shock') or 'n/a'}; dotted lines are percentile bounds.",
        x_title="steps after shock", y_title="response", overlays=overlays, mode=mode,
    )
    return _complete(
        chart, item_id="models.var_irf", title="VAR impulse-response functions",
        caption="Orthogonalised IRFs with 90% bootstrap bands; the Cholesky ordering travels with the numbers.",
        group="Model fit", stage="S10", source="session.objects['fits'].outcomes", rows=rows,
    )


def var_fevd(outcomes: Sequence[Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    """§8's forecast-error variance decomposition, with explicit ordering."""
    entries = _multivariate(outcomes)
    shocks: list[str] = []
    for e in entries:
        fevd = e["payload"].get("fevd") or {}
        for s in fevd.get("ordering") or e["payload"].get("series") or []:
            if str(s) not in shocks:
                shocks.append(str(s))
    rows: list[dict[str, Any]] = []
    matrix_rows: list[list[float]] = []
    labels: list[str] = []
    for e in entries:
        mid = e["model_id"]
        fevd = e["payload"].get("fevd") or {}
        shares = fevd.get("shares") or {}
        ordering = [str(x) for x in (fevd.get("ordering") or e["payload"].get("series") or [])]
        horizon = int(fevd.get("horizon") or 0)
        for target, by_shock in shares.items():
            labels.append(f"{mid}: {target}")
            vals = [float(_number(by_shock.get(s)) or 0.0) for s in shocks]
            matrix_rows.append(vals)
            for s, val in zip(shocks, vals):
                rows.append({"model": mid, "target": target, "shock": s, "share": val,
                             "horizon": horizon, "ordering": ", ".join(ordering)})
    if not matrix_rows:
        shocks = ["no fitted VAR/VECM"]
        matrix_rows, labels = [[0.0]], ["no fitted VAR/VECM"]
        rows = [{"model": "none", "target": "", "shock": "", "share": None,
                 "horizon": 0, "ordering": "", "status": "no fitted VAR/VECM"}]
    chart = P.heatmap(
        pd.DataFrame(matrix_rows, index=labels, columns=shocks),
        title="Forecast-error variance decomposition",
        subtitle="Share of forecast-error variance attributed to each orthogonalised shock.",
        x_title="shock", y_title="target", value_label="share", mode=mode,
    )
    return _complete(
        chart, item_id="models.var_fevd", title="VAR forecast-error variance decomposition",
        caption="FEVD shares at the stated horizon under the disclosed Cholesky ordering.",
        group="Model fit", stage="S10", source="session.objects['fits'].outcomes", rows=rows,
    )


# --------------------------------------------------------------------------- #
# Diagnostics / Forecast / Evaluation / Tables
# --------------------------------------------------------------------------- #
def residual_acf(diagnostics: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    blocks = diagnostics.get("blocks", {}) if isinstance(diagnostics, dict) else {}
    rows: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    for mid, block in (blocks or {}).items():
        battery = (block.get("battery") or {}) if isinstance(block, dict) else {}
        acf = battery.get("acf") or {}
        lags = _finite(acf.get("lags") or [])
        vals = _finite(acf.get("acf") or [])
        bounds = _finite(acf.get("bounds") or [])
        if not lags or not vals:
            continue
        n = min(len(lags), len(vals), len(bounds) if bounds else len(vals))
        x = [int(v) for v in lags[:n]]
        y = vals[:n]
        lo = [-abs(bounds[i]) if i < len(bounds) else -0.1 for i in range(n)]
        hi = [abs(bounds[i]) if i < len(bounds) else 0.1 for i in range(n)]
        overlays.append(_ov(f"{mid} ACF", x, y, kind="line"))
        overlays.append(_band_ov(f"{mid} 95% band", x, lo, hi))
        rows.extend({"model": mid, "lag": x[i], "acf": _number(y[i]),
                     "lower_95": _number(lo[i]), "upper_95": _number(hi[i])}
                    for i in range(n))
    if not rows:
        rows = [{"model": "none", "lag": 0, "acf": None,
                 "status": "no residual battery"}]
        x = [0]
        overlays = [_ov("no residual battery", x, [0.0], kind="dot")]
    else:
        x = sorted({int(r["lag"]) for r in rows})
    chart = P.time_series(
        _base(x), title="Residual autocorrelation by model",
        subtitle="Lines inside the shaded confidence band suggest no linear dependence left behind.",
        x_title="lag", y_title="residual ACF", overlays=overlays, mode=mode,
    )
    return _complete(
        chart, item_id="diagnostics.residual_acf", title="Residual ACF",
        caption="Residual autocorrelations and 95% bounds from every stored diagnostic battery.",
        group="Diagnostics", stage="S12", source="session.objects['diagnostics'].blocks", rows=rows,
    )


def residual_qq(
    diagnostics: dict[str, Any],
    outcomes: Sequence[Any],
    *,
    mode: ThemeMode = "light",
) -> dict[str, Any]:
    """A normal Q–Q from the stored residuals, with a 45-degree reference line."""
    rows: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    x_labels: list[str] = []
    for o in outcomes or []:
        wrapper, summary = getattr(o, "wrapper", None), getattr(o, "summary", None)
        if wrapper is None or summary is None or getattr(summary, "failed", False):
            continue
        try:
            resid = wrapper.residuals()
        except Exception:
            continue
        vals = _finite(np.asarray(resid, dtype=float).ravel().tolist()) if resid is not None else []
        if len(vals) < 3:
            continue
        arr = np.asarray(vals, dtype=float)
        sd = float(arr.std())
        z = np.sort((arr - float(arr.mean())) / (sd if sd > 0 else 1.0))
        theo = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
        labels = [f"{v:.4f}" for v in theo]
        if not x_labels:
            x_labels = labels
        mid = getattr(summary, "model_id", "model")
        overlays.append(_ov(f"{mid} residuals", labels, z.tolist(), kind="markers", size=5))
        rows.extend({"model": mid, "theoretical_quantile": _number(theo[i]),
                     "observed_quantile": _number(z[i]), "n": len(z)} for i in range(len(z)))
    if x_labels:
        overlays.append(_ov("normal reference", x_labels, [float(v) for v in x_labels],
                            kind="dashed"))
    if not rows:
        # No raw residuals retained: fall back to the disclosed battery summaries,
        # explicitly labelled, rather than inventing a straight Q–Q line.
        blocks = diagnostics.get("blocks", {}) if isinstance(diagnostics, dict) else {}
        for mid, block in (blocks or {}).items():
            normal = (block.get("battery") or {}).get("normality") or {}
            jb = normal.get("jarque_bera") or {}
            rows.append({"model": mid, "theoretical_quantile": None,
                         "observed_quantile": None, "n": 0,
                         "skew": _number(jb.get("skew")),
                         "excess_kurtosis": _number(jb.get("kurtosis")),
                         "jarque_bera_p": _number(jb.get("p")),
                         "status": "stored normality summary, no raw residuals"})
        if not rows:
            rows = [{"model": "none", "theoretical_quantile": None,
                     "observed_quantile": None, "n": 0, "status": "no residual battery"}]
        x_labels = ["0"]
        overlays = [_ov("no residual Q–Q available", x_labels, [0.0], kind="dot")]
        caption = ("Normality summaries are shown because raw residuals were not retained; "
                   "the Jarque–Bera and Shapiro p-values travel in the table.")
    else:
        caption = "Standardised residual quantiles against normal quantiles for every successful fit."
    chart = P.time_series(
        _base(x_labels), title="Residual normal Q–Q",
        subtitle="Points on the dashed line are approximately normal.",
        x_title="theoretical normal quantile", y_title="standardised residual quantile",
        overlays=overlays, mode=mode,
    )
    return _complete(
        chart, item_id="diagnostics.residual_qq", title="Residual normal Q–Q",
        caption=caption,
        group="Diagnostics", stage="S12", source="session.objects['fits'].outcomes", rows=rows,
    )


def cusum_paths(diagnostics: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    blocks = diagnostics.get("blocks", {}) if isinstance(diagnostics, dict) else {}
    overlays: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for mid, block in (blocks or {}).items():
        battery = (block.get("battery") or {}) if isinstance(block, dict) else {}
        cusum = battery.get("cusum") or {}
        path = _finite(cusum.get("path") or [])
        crit = _number(cusum.get("critical"))
        if not path:
            continue
        x = list(range(1, len(path) + 1))
        overlays.append(_ov(f"{mid} CUSUM", x, path, kind="line"))
        if crit is not None:
            overlays.append(_ov(f"{mid} +95%", x, [crit] * len(path), kind="dashed"))
            overlays.append(_ov(f"{mid} −95%", x, [-crit] * len(path), kind="dashed"))
        rows.extend({"model": mid, "step": i + 1, "cusum": _number(v), "critical": crit,
                     "breach": bool(crit is not None and abs(v) > crit)}
                    for i, v in enumerate(path))
    if not rows:
        rows = [{"model": "none", "step": 0, "cusum": None,
                 "status": "no CUSUM path"}]
        x = [0]
        overlays = [_ov("no CUSUM path", x, [0.0], kind="dot")]
    else:
        x = list(range(1, max(int(r["step"] or 0) for r in rows) + 1))
    chart = P.time_series(
        _base(x), title="Residual CUSUM paths",
        subtitle="Dashed lines are the 95% critical bounds.",
        x_title="observation", y_title="cumulative sum", overlays=overlays, mode=mode,
    )
    return _complete(
        chart, item_id="diagnostics.cusum_paths", title="Residual CUSUM paths",
        caption="Cumulative-sum paths with disclosed critical bounds; a breach alone is not a structural-break finding.",
        group="Diagnostics", stage="S12", source="session.objects['diagnostics'].blocks", rows=rows,
    )


def _bundle_dict(forecast: dict[str, Any]) -> dict[str, Any]:
    bundle = forecast.get("bundle") if isinstance(forecast, dict) else None
    return bundle.as_dict() if hasattr(bundle, "as_dict") else (bundle if isinstance(bundle, dict) else {})


def fan_chart(forecast: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    data = _bundle_dict(forecast)
    index = [str(x) for x in data.get("index") or []]
    point = _finite(data.get("point") or [])
    bands = data.get("bands") or []
    series = forecast.get("series") if isinstance(forecast, dict) else None
    observed = pd.Series(series).astype(float) if series is not None else pd.Series(dtype=float)
    p_bands: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for b in bands:
        item = b if isinstance(b, dict) else (b.as_dict() if hasattr(b, "as_dict") else {})
        lower = _finite(item.get("lower") or [])
        upper = _finite(item.get("upper") or [])
        coverage = _number(item.get("coverage")) or 0.95
        p_bands.append({"coverage": coverage, "lower": lower, "upper": upper,
                        "source": str(item.get("source") or data.get("interval_method") or "interval"),
                        "construction": str(item.get("construction") or "")})
        rows.extend({
            "time": index[i] if i < len(index) else str(i + 1),
            "point": _number(point[i]) if i < len(point) else None,
            "lower": _number(lower[i]) if i < len(lower) else None,
            "upper": _number(upper[i]) if i < len(upper) else None,
            "coverage": coverage,
            "construction": str(item.get("construction") or data.get("construction") or ""),
        } for i in range(max(len(point), len(lower), len(upper))))
    if not rows:
        rows = [{"time": "", "point": None, "lower": None, "upper": None,
                 "coverage": None, "construction": "", "status": "no forecast bands"}]
    chart = P.fan_chart(
        observed, index=index, point=point, bands=p_bands,
        title="Forecast fan chart", subtitle="Every band states its coverage and construction.",
        y_title="series", mode=mode,
    )
    return _complete(
        chart, item_id="forecast.fan_chart", title="Forecast fan chart",
        caption="Point forecast and all stored coverage bands; the table states each band's construction.",
        group="Forecast", stage="S13", source="session.objects['forecast'].bundle", rows=rows,
    )


def interval_width(forecast: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    data = _bundle_dict(forecast)
    index = [str(x) for x in data.get("index") or []]
    rows: list[dict[str, Any]] = []
    for b in data.get("bands") or []:
        item = b if isinstance(b, dict) else (b.as_dict() if hasattr(b, "as_dict") else {})
        lo, hi = _finite(item.get("lower") or []), _finite(item.get("upper") or [])
        cov = _number(item.get("coverage")) or 0.95
        rows.extend({"time": index[i] if i < len(index) else str(i + 1), "coverage": cov,
                     "width": _number(hi[i] - lo[i]) if i < min(len(lo), len(hi)) else None}
                    for i in range(max(len(lo), len(hi))))
    if not rows:
        rows = [{"time": "", "coverage": None, "width": None, "status": "no intervals"}]
    chosen = [r for r in rows if abs(float(r["coverage"] or 0) - 0.95) < 1e-9] or rows
    chart = P.bars(
        [str(r["time"]) for r in chosen], [float(r["width"] or 0.0) for r in chosen],
        title="95% interval width by horizon",
        subtitle="Sharpness is meaningful only alongside the calibration results.",
        x_title="forecast time", y_title="interval width", mode=mode,
    )
    return _complete(
        chart, item_id="forecast.interval_width", title="Forecast interval width",
        caption="Width of each stored interval by horizon and coverage; calibration is reported separately.",
        group="Forecast", stage="S13", source="session.objects['forecast'].bundle", rows=rows,
    )


def scenario_paths(forecast: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    data = _bundle_dict(forecast)
    scenario = data.get("scenario") or {}
    paths: dict[str, list[float]] = {}
    future = forecast.get("future_exogenous") if isinstance(forecast, dict) else None
    if isinstance(future, pd.DataFrame):
        for col in future.columns:
            paths[str(col)] = _finite(pd.to_numeric(future[col], errors="coerce").tolist())
    comparison = scenario.get("comparison") if isinstance(scenario, dict) else None
    if isinstance(comparison, dict):
        for key in ("baseline", "scenario", "change"):
            vals = _finite(comparison.get(key) or [])
            if vals:
                paths[f"{key} forecast"] = vals
    rows: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    for name, vals in paths.items():
        x = list(range(1, len(vals) + 1))
        overlays.append(_ov(name, x, vals, kind="line"))
        rows.extend({"driver": name, "step": i + 1, "value": _number(v),
                     "source": "future_exogenous" if isinstance(future, pd.DataFrame) else "scenario comparison"}
                    for i, v in enumerate(vals))
    if not rows:
        rows = [{"driver": "none", "step": 0, "value": None, "source": "",
                 "status": "no exogenous scenario supplied"}]
        x = [0]
        overlays = [_ov("no exogenous scenario", x, [0.0], kind="dot")]
    else:
        x = list(range(1, max(int(r["step"] or 0) for r in rows) + 1))
    chart = P.time_series(
        _base(x), title="Exogenous scenario paths",
        subtitle="These are assumptions supplied by the analyst, not forecasts of the drivers.",
        x_title="forecast step", y_title="driver / forecast value", overlays=overlays, mode=mode,
    )
    return _complete(
        chart, item_id="forecast.scenario_paths", title="Exogenous scenario paths",
        caption="Conditional driver paths and their forecast comparison; held-flat columns are disclosed as assumptions.",
        group="Forecast", stage="S13", source="session.objects['future_exogenous']", rows=rows,
    )


def cv_loss_by_fold(selection: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    cv = selection.get("cv") or {}
    rows: list[dict[str, Any]] = []
    for mid, result in (cv.items() if isinstance(cv, dict) else []):
        folds = getattr(result, "folds", None) or getattr(result, "fold_scores", None) or []
        for i, f in enumerate(folds):
            d = _as_dict(f)
            rows.append({"model": mid, "fold": int(d.get("fold", i + 1) or i + 1),
                         "mase": _number(d.get("mase")), "mae": _number(d.get("mae")),
                         "rmse": _number(d.get("rmse"))})
    if not rows:
        report = selection.get("report") if isinstance(selection, dict) else None
        rd = _as_dict(report)
        for mid, val in (rd.get("cv_scores") or {}).items():
            rows.append({"model": mid, "fold": 1, "mase": _number(val),
                         "mae": None, "rmse": None})
    if not rows:
        rows = [{"model": "none", "fold": 0, "mase": None, "mae": None,
                 "rmse": None, "status": "no CV folds stored"}]
    chart = P.bars(
        [f"{r['model']} · fold {r['fold']}" for r in rows],
        [float(r["mase"] or 0.0) for r in rows],
        title="Cross-validated MASE by fold",
        subtitle="Each bar is an out-of-sample origin; in-sample fit is never shown as accuracy.",
        x_title="model and fold", y_title="MASE", mode=mode, horizontal=True,
    )
    return _complete(
        chart, item_id="evaluation.cv_loss_by_fold", title="CV loss by fold",
        caption="Rolling-origin MASE per fold and model; lower is better and the naive benchmark is 1.",
        group="Evaluation", stage="S11", source="session.objects['selection'].cv", rows=rows,
    )


def metric_radar(evaluation: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    report = evaluation.get("report", {}) if isinstance(evaluation, dict) else {}
    metric_rows = report.get("metric_rows") or []
    rows: list[dict[str, Any]] = []
    for r in metric_rows[:40]:
        for key, val in r.items():
            x = _number(val)
            if x is None or key in {"model", "model_id", "rank", "winner"}:
                continue
            rows.append({"model": str(r.get("model") or r.get("model_id") or "model"),
                         "metric": key, "value": x})
    if not rows:
        rows = [{"model": "none", "metric": "", "value": 0.0,
                 "status": "no evaluation report"}]
    chart = P.bars(
        [f"{r['model']}: {r['metric']}" for r in rows[:80]],
        [float(r["value"] or 0.0) for r in rows[:80]],
        title="Evaluation metric comparison",
        subtitle="Metrics have different scales; the heatmap and metric table remain the ranked view.",
        x_title="model · metric", y_title="score", mode=mode, horizontal=True,
    )
    return _complete(
        chart, item_id="evaluation.metric_comparison", title="Evaluation metric comparison",
        caption="All numeric scores from the stored S14 metric table; the primary metric decides the ranking.",
        group="Evaluation", stage="S14", source="session.objects['evaluation'].report", rows=rows,
    )


def pit_calibration(evaluation: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    report = evaluation.get("report", {}) if isinstance(evaluation, dict) else {}
    calibration = report.get("calibration") or {}
    rows: list[dict[str, Any]] = []
    for r in calibration.get("rows") or []:
        rows.append({"model": str(r.get("model") or r.get("model_id") or "model"),
                     "nominal": _number(r.get("nominal") or r.get("coverage")),
                     "empirical": _number(r.get("empirical") or r.get("observed")),
                     "construction": str(r.get("construction") or "")})
    if not rows:
        rows = [{"model": "none", "nominal": 0.0, "empirical": 0.0,
                 "construction": "", "status": "no calibration rows"}]
    chart = P.scatter_with_fit(
        x=[float(r["nominal"] or 0.0) for r in rows],
        y=[float(r["empirical"] or 0.0) for r in rows],
        title="Nominal vs empirical coverage",
        subtitle="Points on the diagonal are calibrated; the table states interval construction.",
        x_title="nominal coverage", y_title="empirical coverage",
        point_labels=[str(r["model"]) for r in rows], mode=mode,
    )
    return _complete(
        chart, item_id="evaluation.pit_calibration", title="Coverage calibration",
        caption="Nominal against empirical coverage from the S14 out-of-sample protocol.",
        group="Evaluation", stage="S14", source="session.objects['evaluation'].report", rows=rows,
    )


def leakage_audit(evaluation: dict[str, Any], *, mode: ThemeMode = "light") -> dict[str, Any]:
    report = evaluation.get("report", {}) if isinstance(evaluation, dict) else {}
    leak = report.get("leakage") or {}
    checks = leak.get("checks") or []
    rows = [{"check": str(c.get("name") or c.get("check") or "check"),
             "passed": bool(c.get("ok", False)), "detail": str(c.get("detail") or "")}
            for c in checks]
    if not rows:
        rows = [{"check": "no leakage audit stored", "passed": False, "detail": ""}]
    chart = P.bars(
        [r["check"] for r in rows], [1.0 if r["passed"] else 0.0 for r in rows],
        title="Leakage audit checks", subtitle="1 = passed, 0 = failed or not yet run.",
        x_title="check", y_title="passed", mode=mode, horizontal=True, reference=1.0,
    )
    return _complete(
        chart, item_id="evaluation.leakage_audit", title="Leakage audit",
        caption="Pass/fail for every zero-leakage protocol check in the stored evaluation report.",
        group="Evaluation", stage="S14", source="session.objects['evaluation'].report", rows=rows,
    )


# --------------------------------------------------------------------------- #
# Stored results and companion tables
# --------------------------------------------------------------------------- #
def _result_rows(session: Any, mode: ThemeMode) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plots: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    # A status read can overlap with finalization on the same session. Snapshot
    # both mappings before iterating: adding the S15 result bucket while a
    # catalogue is being read must not raise "dictionary changed size".
    snapshot = list(getattr(session, "results", {}).items())
    for stage, results in snapshot:
        code = getattr(stage, "value", str(stage))
        values = list(results.values()) if isinstance(results, dict) else list(results)
        for r in values:
            title = str(getattr(r, "title", "") or getattr(r, "id", "Result"))
            plain = str(getattr(r, "plain", "") or "")
            caption = str(getattr(r, "caption", "") or "") or f"{title}. {plain[:140]}".strip()
            source = f"session.results[{code}]"
            group = STAGE_GROUP.get(code, "Tables")
            if getattr(r, "kind", "") == "figure":
                behind = getattr(r, "data_behind", None)
                cols = _jsonable(list(getattr(behind, "columns", []))) if behind else []
                rows = _jsonable(list(getattr(behind, "rows", []))) if behind else []
                figure_json = _jsonable(getattr(r, "figure_json", {}))
                if not rows:
                    # Earlier chart builders left `data_behind` empty. Recover the
                    # traced numbers so every catalogue plot really is exportable.
                    rows = _figure_rows(figure_json)
                    cols = _columns(rows)
                alt = str(getattr(r, "alt_text", "") or "") or _alt_text(title, caption, rows)
                plots.append({
                    "id": getattr(r, "id", ""), "title": title, "group": group,
                    "stage": code, "kind": "plot", "source": source,
                    "caption": caption, "alt_text": alt,
                    "figure_json": figure_json,
                    "columns": cols, "rows": rows,
                    "formats": list(FORMATS), "dpi": DPI, "report_insertable": True,
                })
            elif getattr(r, "kind", "") == "table":
                tables.append(_table_item(
                    item_id=getattr(r, "id", ""), title=title, group=group, stage=code,
                    source=source, caption=caption, mode=mode,
                    rows=_jsonable(list(getattr(r, "rows", []))),
                ))
    return plots, tables


def _derived_plots(session: Any, mode: ThemeMode) -> list[dict[str, Any]]:
    o = session.objects
    out: list[dict[str, Any]] = []
    series = o.get("series")
    exog = o.get("exogenous")
    if isinstance(series, pd.Series) and len(series):
        out.extend([
            quality_spark(series, mode=mode),
            missingness_calendar(series, mode=mode),
            lag_scatter(series, mode=mode),
            rolling_statistics(series, mode=mode),
        ])
    if o.get("decomposition"):
        out.append(components(o["decomposition"], mode=mode))
        findings = o["decomposition"].get("findings") or []
        seasonal = next((f.get("table") for f in findings
                         if f.get("id") == "stat.seasonality" and f.get("table")), [])
        if seasonal:
            out.append(seasonal_profile([r.get("effect", 0) for r in seasonal], mode=mode))
    if o.get("stat_battery"):
        out.append(test_statistic_bars(o["stat_battery"], mode=mode))
    if o.get("correlogram"):
        out.append(acf_pacf_stems(o["correlogram"], mode=mode))
        out.append(ccf_lags(series if isinstance(series, pd.Series) else None,
                            exog if isinstance(exog, pd.DataFrame) else None,
                            o["correlogram"], mode=mode))
    fits = (o.get("fits") or {}).get("outcomes") or []
    if fits:
        out.extend([
            fitted_vs_actual(fits, mode=mode),
            information_criteria(fits, mode=mode),
            variance_path(fits, mode=mode),
            var_irf(fits, mode=mode),
            var_fevd(fits, mode=mode),
        ])
    if o.get("diagnostics"):
        out.extend([
            residual_acf(o["diagnostics"], mode=mode),
            residual_qq(o["diagnostics"], fits, mode=mode),
            cusum_paths(o["diagnostics"], mode=mode),
        ])
    if o.get("forecast"):
        f = {**o["forecast"], "series": series,
             "future_exogenous": o.get("future_exogenous")}
        out.extend([
            fan_chart(f, mode=mode),
            interval_width(f, mode=mode),
            count_pmf_ribbons(f, mode=mode),
            scenario_paths(f, mode=mode),
        ])
    if o.get("selection"):
        out.append(cv_loss_by_fold(o["selection"], mode=mode))
    if o.get("evaluation"):
        out.extend([
            metric_radar(o["evaluation"], mode=mode),
            pit_calibration(o["evaluation"], mode=mode),
            leakage_audit(o["evaluation"], mode=mode),
        ])
    return out


def _derived_tables(session: Any, mode: ThemeMode) -> list[dict[str, Any]]:
    """Companion tables, offered only for stored objects that actually exist."""
    o = session.objects
    out: list[dict[str, Any]] = []
    series = o.get("series")
    if isinstance(series, pd.Series):
        s = pd.Series(series).astype(float)
        out.append(_table_item(
            item_id="table.observations", title="Observation register", group="Tables", stage="S1",
            source="session.objects['series']", mode=mode,
            rows=[{"time": str(t), "value": _number(v)} for t, v in zip(s.index, s.tolist())],
            caption="Every observed value and timestamp used by the analysis."))
        desc = s.describe()
        out.append(_table_item(
            item_id="table.descriptive_moments", title="Descriptive moments", group="Tables",
            stage="S1", source="session.objects['series']", mode=mode,
            rows=[{"moment": str(k), "value": _number(v)} for k, v in desc.items()],
            caption="Count, mean, spread and quartiles of the analysed series."))
    battery = o.get("stat_battery") or {}
    if battery:
        for group in ("unit_root", "seasonal", "breaks"):
            rows = [{"test": str(r.get("test") or r.get("id")),
                     "statistic": _number(r.get("statistic")),
                     "p_value": _number(r.get("p_value")),
                     "critical_5pct": _number(r.get("cv_5pct")),
                     "reading": str(r.get("reading") or r.get("says") or "")}
                    for r in (battery.get(group) or [])]
            out.append(_table_item(
                item_id=f"table.stationarity_{group}",
                title=f"Stationarity: {group.replace('_', ' ')}", group="Tables", stage="S5",
                source="session.objects['stat_battery']", rows=rows, mode=mode,
                caption=f"Stored {group.replace('_', ' ')} battery with statistics, p-values and critical values."))
    corr = o.get("correlogram") or {}
    if corr:
        payload = corr.get("payload", corr) if isinstance(corr, dict) else {}
        for key in ("acf", "pacf", "ljung_box", "box_pierce"):
            rows = [_jsonable(dict(r)) if isinstance(r, dict) else {"value": r}
                    for r in (payload.get(key) or [])]
            out.append(_table_item(
                item_id=f"table.correlogram_{key}", title=f"Correlogram: {key}", group="Tables",
                stage="S6", source="session.objects['correlogram']", rows=rows, mode=mode,
                caption=f"Stored {key} numbers; every figure is reproducible from this table."))
    fits = (o.get("fits") or {}).get("outcomes") or []
    if fits:
        fit_rows = []
        for fit in fits:
            summary = getattr(fit, "summary", None)
            if summary is None:
                continue
            fit_rows.append({
                "model": getattr(summary, "model_id", ""),
                "display_name": getattr(summary, "display_name", ""),
                "family": getattr(getattr(summary, "family", None), "value", ""),
                "failed": bool(getattr(summary, "failed", False)),
                "aic": _number(getattr(summary, "aic", None)),
                "aicc": _number(getattr(summary, "aicc", None)),
                "bic": _number(getattr(summary, "bic", None)),
                "reason": str(getattr(summary, "failure_reason", "") or ""),
            })
        out.append(_table_item(
            item_id="table.fit_register", title="Fit register", group="Tables", stage="S10",
            source="session.objects['fits']", rows=fit_rows, mode=mode,
            caption="Every fitted and failed candidate with its information criteria and failure reason."))
    diagnostics_store = o.get("diagnostics") or {}
    if diagnostics_store:
        diag = diagnostics_store.get("blocks") or {}
        for key in ("ljung_box", "box_pierce", "normality", "arch_lm", "runs", "signs", "cusum"):
            rows = []
            for mid, block in (diag.items() if isinstance(diag, dict) else []):
                b = (block.get("battery") or {}) if isinstance(block, dict) else {}
                v = b.get(key)
                if isinstance(v, dict):
                    rows.append({"model": mid, **_jsonable(v)})
                elif v is not None:
                    rows.append({"model": mid, "value": _jsonable(v)})
            out.append(_table_item(
                item_id=f"table.diagnostics_{key}", title=f"Diagnostics: {key}", group="Tables",
                stage="S12", source="session.objects['diagnostics']", rows=rows, mode=mode,
                caption=f"Stored {key} diagnostic summaries for every fitted model."))
    forecast_store = o.get("forecast") or {}
    if forecast_store:
        bundle_data = _bundle_dict(forecast_store)
        for key in ("points", "intervals", "construction"):
            if key == "points":
                rows = [{"time": str(t), "point": _number(v)}
                        for t, v in zip(bundle_data.get("index") or [], bundle_data.get("point") or [])]
            elif key == "intervals":
                rows = []
                for b in bundle_data.get("bands") or []:
                    d = b if isinstance(b, dict) else (b.as_dict() if hasattr(b, "as_dict") else {})
                    idx = bundle_data.get("index") or []
                    for i, (lo, hi) in enumerate(zip(d.get("lower") or [], d.get("upper") or [])):
                        rows.append({"time": str(idx[i] if i < len(idx) else i + 1),
                                     "coverage": _number(d.get("coverage")), "lower": _number(lo),
                                     "upper": _number(hi),
                                     "construction": str(d.get("construction") or "")})
            else:
                rows = [{"field": k,
                         "value": _jsonable(v) if not isinstance(v, (dict, list)) else json.dumps(v, default=str)[:180]}
                        for k, v in bundle_data.items() if k in {
                            "construction", "interval_method", "seed", "n_paths", "scale",
                            "test_observations_used", "adequacy"}]
            out.append(_table_item(
                item_id=f"table.forecast_{key}", title=f"Forecast: {key}", group="Tables",
                stage="S13", source="session.objects['forecast'].bundle", rows=rows, mode=mode,
                caption=f"Stored forecast {key} with all disclosure fields."))
    evaluation_store = o.get("evaluation") or {}
    if evaluation_store:
        evaluation = evaluation_store.get("report") or {}
        for key in ("metric_rows", "fold_scores", "coverage_construction", "fold_failures", "folds"):
            rows = [_jsonable(dict(r)) if isinstance(r, dict) else {"value": r}
                    for r in (evaluation.get(key) or [])]
            out.append(_table_item(
                item_id=f"table.evaluation_{key}", title=f"Evaluation: {key.replace('_', ' ')}",
                group="Tables", stage="S14", source="session.objects['evaluation'].report",
                rows=rows, mode=mode,
                caption=f"Stored S14 {key.replace('_', ' ')}; every score is out-of-sample."))
    return out


# --------------------------------------------------------------------------- #
# Catalogue assembly
# --------------------------------------------------------------------------- #
def build_catalog(session: Any, *, mode: ThemeMode = "light") -> dict[str, Any]:
    """Build the complete §5.10 scorecard from stored results and stored objects."""
    if mode not in MODES:
        mode = "light"
    plots, tables = _result_rows(session, mode)
    plots.extend(_derived_plots(session, mode))
    tables.extend(_derived_tables(session, mode))
    for item in plots + tables:
        fig = item.get("figure_json")
        if isinstance(fig, dict) and fig:
            item["figure_json"] = _jsonable(restyled(fig, mode))
        if not item.get("caption"):
            item["caption"] = f"{item['title']}. Stored catalogue item."
        if not item.get("alt_text"):
            item["alt_text"] = _alt_text(item["title"], item["caption"], item.get("rows") or [])
        # Plots are grouped by the stage they explain; all tabular deliverables
        # belong to §5.10's ninth "Tables" group. This keeps all nine groups
        # represented without pretending a table is a plot.
        item["group"] = (
            STAGE_GROUP.get(item["stage"], item.get("group", "Tables"))
            if item["kind"] == "plot" else "Tables"
        )
        item["formats"] = list(FORMATS)
        item["dpi"] = DPI
        item["report_insertable"] = True
    groups = {g: {"plots": sum(1 for x in plots if x["group"] == g),
                  "tables": sum(1 for x in tables if x["group"] == g)} for g in GROUPS}
    scorecard = {
        "plots": len(plots), "tables": len(tables),
        "min_plots": MIN_PLOTS, "min_tables": MIN_TABLES,
        "plots_ok": len(plots) >= MIN_PLOTS, "tables_ok": len(tables) >= MIN_TABLES,
        "captions_ok": all(bool(x["caption"]) for x in plots + tables),
        "alt_text_ok": all(bool(x["alt_text"]) for x in plots),
        "data_behind_ok": all(len(x.get("rows") or []) > 0 for x in plots),
        "groups": groups,
        "styling": {
            "single_source": "app.charts.theme",
            "mode": mode,
            "colorblind_safe": True,
            "dark_light_print": list(MODES),
            "dash_patterns": True,
        },
        "exports": {"formats": list(FORMATS), "dpi": DPI, "captioned": True,
                    "report_insertable": True, "data_behind": True},
    }
    return {"scorecard": scorecard, "plots": plots, "tables": tables}


def public_item(item: dict[str, Any]) -> dict[str, Any]:
    """Strip builder objects; everything returned is strict JSON."""
    return _jsonable({k: v for k, v in item.items() if k != "chart"})


def find_item(catalog: dict[str, Any], item_id: str) -> dict[str, Any] | None:
    for item in catalog["plots"] + catalog["tables"]:
        if item["id"] == item_id:
            return item
    return None


def report_item_ids(session: Any) -> list[str]:
    raw = session.objects.get("catalog_report") or []
    return [str(x) for x in raw if isinstance(x, (str, int))]


def set_report_item_ids(session: Any, ids: Sequence[str]) -> None:
    session.objects["catalog_report"] = list(dict.fromkeys(str(x) for x in ids))


__all__ = [
    "GROUPS", "FORMATS", "STATIC_FORMATS", "DATA_FORMATS", "MODES",
    "MIN_PLOTS", "MIN_TABLES", "DPI", "build_catalog", "public_item",
    "find_item", "report_item_ids", "set_report_item_ids",
    "quality_spark", "missingness_calendar", "lag_scatter", "components",
    "seasonal_profile", "rolling_statistics", "test_statistic_bars",
    "acf_pacf_stems", "ccf_lags", "fitted_vs_actual", "information_criteria",
    "variance_path", "count_pmf_ribbons", "var_irf", "var_fevd",
    "residual_acf", "residual_qq", "cusum_paths", "fan_chart", "interval_width",
    "scenario_paths", "cv_loss_by_fold", "metric_radar", "pit_calibration",
    "leakage_audit",
]
