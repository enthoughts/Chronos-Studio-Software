"""
charts/evaluation.py — the S14 evaluation figures (§5.10, §5.12).

Every figure here goes through `app.charts.theme.apply` / `to_json` and the shared
`Chart` contract: `figure_json`, generated `alt_text`, and `data_behind` rows.
No API handler builds a Plotly figure. The primitives in `charts/primitives.py`
supply the heatmap and time-series split; the evaluation-specific builders below
supply boxplots, QQ points, PIT histograms, loss curves and calibration.

The alt text always quotes the numbers that were plotted, with `p_number` /
`p_phrase` for p-values, so a screen reader gets the same verdict as the chart.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.charts.primitives import Chart, heatmap, time_series
from app.charts.theme import DASHES, PALETTE, apply, colours, to_json, with_alpha
from app.core.formatting import p_number, p_phrase

_HIGHER_BETTER = {
    "hit_rate", "pit", "rqr", "randomized_pit", "kupiec", "christoffersen", "mcs_p",
}
_METRIC_LABELS = {
    "mae": "MAE", "rmse": "RMSE", "mase": "MASE", "rmsse": "RMSSE",
    "mpe": "MPE", "mape": "MAPE", "smape": "sMAPE", "gmrae": "GMRAE",
    "mdrae": "MdRAE", "rel_mae": "relative MAE", "rel_rmse": "relative RMSE",
    "crps": "CRPS", "pinball": "Pinball", "log_score": "Log score",
    "winkler": "Winkler", "coverage": "Coverage deviation", "calibration": "Calibration",
    "pit": "PIT CvM p", "rqr": "RQR KS p", "discrete_crps": "Discrete CRPS",
    "randomized_pit": "Randomised PIT CvM p", "qlike": "QLIKE",
    "mse_variance": "MSE variance", "kupiec": "Kupiec p",
    "christoffersen": "Christoffersen p", "hit_rate": "Directional hit rate",
}


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    if abs(x) != 0 and (abs(x) < 0.01 or abs(x) >= 1e6):
        return f"{x:.2e}"
    return f"{x:,.{digits}f}".rstrip("0").rstrip(".") if abs(x) < 1000 else f"{x:,.0f}"


def _finite(values: Sequence[Any]) -> list[float | None]:
    out: list[float | None] = []
    for v in values:
        try:
            x = float(v)
            out.append(x if math.isfinite(x) else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _line_chart(
    x: Sequence[Any],
    frames: Sequence[tuple[str, Sequence[Any]]],
    *,
    title: str,
    subtitle: str = "",
    x_title: str = "",
    y_title: str = "",
    height: int = 360,
) -> Chart:
    c = colours("light")
    fig = go.Figure()
    rows: list[dict[str, Any]] = []
    all_values: list[float] = []
    for i, (name, raw) in enumerate(frames):
        vals = _finite(raw)
        y = [v if v is not None else float("nan") for v in vals[: len(x)]]
        if len(y) < len(x):
            y += [float("nan")] * (len(x) - len(y))
        colour = PALETTE[i % len(PALETTE)]
        dash = DASHES[i % len(DASHES)]
        fig.add_trace(go.Scatter(
            x=list(x), y=y, mode="lines+markers", name=name,
            line={"color": colour, "width": 1.8, "dash": dash},
            marker={"size": 5, "color": colour},
            hovertemplate=f"{name}: %{{y:,.4~g}}<extra></extra>",
        ))
        for xv, yv in zip(x, y):
            rows.append({"x": xv, "series": name, "y": yv if math.isfinite(yv) else None})
            if math.isfinite(yv):
                all_values.append(float(yv))
    apply(fig, title=title, subtitle=subtitle, x_title=x_title, y_title=y_title,
          mode="light", height=height, legend=len(frames) > 1, hovermode="x unified")
    if all_values:
        alt = (f"{title}. {len(frames)} series over {len(x)} x-values; plotted values range "
               f"from {_fmt(min(all_values))} to {_fmt(max(all_values))}.")
    else:
        alt = f"{title}. No finite values were available to plot."
    return Chart(to_json(fig), alt, rows)


def metric_heatmap(
    metric_rows: Sequence[dict[str, Any]],
    *,
    policy: dict[str, Any],
    calibration_by_model: dict[str, dict[str, Any]] | None = None,
    title: str = "Metric comparison heatmap",
    subtitle: str = "Relative rank within each column: 1 = best",
) -> Chart:
    models = [str(r.get("model", "")) for r in metric_rows]
    metrics = [m for m in policy.get("metrics", []) if m in _METRIC_LABELS]
    if "hit_rate" not in metrics:
        metrics.append("hit_rate")
    raw = pd.DataFrame(index=models, columns=metrics, dtype=float)
    for r in metric_rows:
        mid = str(r.get("model", ""))
        for m in metrics:
            if m == "coverage" and calibration_by_model:
                rows = (calibration_by_model.get(mid, {}).get("calibration") or {}).get("rows") or []
                devs = [abs(float(x.get("empirical")) - float(x.get("nominal")))
                        for x in rows if x.get("empirical") is not None and x.get("nominal") is not None]
                raw.loc[mid, m] = float(np.mean(devs)) if devs else np.nan
            else:
                try:
                    v = float(r.get(m))
                    raw.loc[mid, m] = v if math.isfinite(v) else np.nan
                except (TypeError, ValueError):
                    raw.loc[mid, m] = np.nan
    score = pd.DataFrame(index=models, columns=metrics, dtype=float)
    for m in metrics:
        vals = raw[m].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0 or np.nanmax(finite) == np.nanmin(finite):
            score[m] = 1.0 if finite.size else 0.0
            continue
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if m in _HIGHER_BETTER:
            score[m] = (vals - lo) / (hi - lo)
        else:
            score[m] = (hi - vals) / (hi - lo)
        score[m] = score[m].fillna(0.0).clip(0.0, 1.0)
    chart = heatmap(
        score, title=title, subtitle=subtitle, x_title="Metric", y_title="Model",
        value_label="Relative rank (1 = best)", text_format=".2f", height=360,
    )
    primary = str(policy.get("primary") or "mase")
    best_model, best_value = "—", None
    if primary in raw.columns:
        vals = raw[primary].dropna()
        if not vals.empty:
            best_model = str(vals.idxmin() if primary not in _HIGHER_BETTER else vals.idxmax())
            best_value = float(vals.min() if primary not in _HIGHER_BETTER else vals.max())
    chart.alt_text = (
        f"{title}. {len(models)} models by {len(metrics)} metrics; relative rank 1 is best in each "
        f"column. On the primary metric {_METRIC_LABELS.get(primary, primary)}, {best_model} leads "
        f"at {_fmt(best_value)}. Coverage is shown as absolute deviation from nominal, so lower is better."
    )
    rows: list[dict[str, Any]] = []
    for mid in models:
        for m in metrics:
            rows.append({
                "model": mid, "metric": m,
                "metric_label": _METRIC_LABELS.get(m, m),
                "raw_value": None if not math.isfinite(float(raw.loc[mid, m])) else float(raw.loc[mid, m]),
                "relative_rank": None if not math.isfinite(float(score.loc[mid, m])) else float(score.loc[mid, m]),
                "higher_is_better": m in _HIGHER_BETTER,
            })
    chart.data_behind = rows
    return chart


def mase_by_horizon_chart(
    mase_by_horizon: dict[str, Sequence[Any]], horizon: int
) -> Chart:
    steps = list(range(1, int(horizon) + 1))
    frames = [(mid, list(vals)) for mid, vals in mase_by_horizon.items()]
    return _line_chart(
        steps, frames,
        title="MASE by forecast horizon",
        subtitle="Training-window naive / seasonal-naive denominator; lower is better",
        x_title="Horizon step h", y_title="MASE", height=360,
    )


def fold_boxplots(
    fold_scores: Sequence[dict[str, Any]], *, metric: str = "mase"
) -> Chart:
    c = colours("light")
    fig = go.Figure()
    rows: list[dict[str, Any]] = []
    by_model: dict[str, list[float]] = {}
    for r in fold_scores:
        mid = str(r.get("model_id", ""))
        try:
            v = float(r.get(metric))
            if not math.isfinite(v):
                continue
        except (TypeError, ValueError):
            continue
        by_model.setdefault(mid, []).append(v)
        rows.append({"model": mid, "fold": r.get("fold"), "metric": metric, "value": v})
    for i, (mid, vals) in enumerate(by_model.items()):
        colour = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Box(
            y=vals, name=mid, boxpoints="all", jitter=0.35, pointpos=0,
            marker={"color": colour, "size": 5},
            line={"color": colour, "width": 1.4},
            fillcolor=with_alpha(colour, 0.14),
            hovertemplate=f"{mid}<br>{_METRIC_LABELS.get(metric, metric)}: %{{y:,.4~g}}<extra></extra>",
        ))
    apply(fig, title=f"Fold-level {_METRIC_LABELS.get(metric, metric)}",
          subtitle="Each dot is one rolling-origin fold; a wide box is an unstable ranking",
          x_title="Model", y_title=_METRIC_LABELS.get(metric, metric), mode="light",
          height=360, legend=False, hovermode="closest")
    if by_model:
        medians = ", ".join(f"{m} {_fmt(float(np.median(v)))}" for m, v in by_model.items())
        alt = f"Boxplots of {metric} across folds for {len(by_model)} models. Medians: {medians}."
    else:
        alt = f"Boxplots of {metric}: no finite fold scores were available."
    return Chart(to_json(fig), alt, rows)


def loss_curves_chart(
    loss_curves: dict[str, dict[str, Sequence[Any]]], horizon: int
) -> Chart:
    steps = list(range(1, int(horizon) + 1))
    n_models = max(1, len(loss_curves))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["CRPS by horizon", "Pinball loss by horizon"],
                        horizontal_spacing=0.10)
    rows: list[dict[str, Any]] = []
    all_crps: list[float] = []
    for i, (mid, curves) in enumerate(loss_curves.items()):
        colour = PALETTE[i % len(PALETTE)]
        dash = DASHES[i % len(DASHES)]
        crps = _finite(curves.get("crps") or [])[:len(steps)]
        pin = _finite(curves.get("pinball") or [])[:len(steps)]
        dcrps = _finite(curves.get("discrete_crps") or [])[:len(steps)]
        crps += [None] * (len(steps) - len(crps))
        pin += [None] * (len(steps) - len(pin))
        dcrps += [None] * (len(steps) - len(dcrps))
        for col, vals, label in ((1, crps, "CRPS"), (2, pin, "Pinball")):
            y = [v if v is not None else float("nan") for v in vals]
            fig.add_trace(go.Scatter(
                x=steps, y=y, mode="lines+markers", name=mid,
                legendgroup=mid, showlegend=(col == 1),
                line={"color": colour, "width": 1.7, "dash": dash},
                marker={"size": 4.5, "color": colour},
                hovertemplate=f"{mid} · {label}: %{{y:,.4~g}}<extra></extra>",
            ), row=1, col=col)
        if any(v is not None and math.isfinite(float(v)) for v in dcrps):
            dy = [v if v is not None else float("nan") for v in dcrps]
            fig.add_trace(go.Scatter(
                x=steps, y=dy, mode="lines", name=f"{mid} · PMF CRPS",
                legendgroup=mid, showlegend=False,
                line={"color": colour, "width": 1.2, "dash": DASHES[2]},
                hovertemplate=f"{mid} · discrete PMF CRPS: %{{y:,.4~g}}<extra></extra>",
            ), row=1, col=1)
        for j, step in enumerate(steps):
            cval = crps[j] if j < len(crps) else None
            pval = pin[j] if j < len(pin) else None
            dval = dcrps[j] if j < len(dcrps) else None
            rows.append({"model": mid, "step": step, "crps": cval, "pinball": pval,
                         "discrete_crps": dval, "source": curves.get("source", "")})
            if cval is not None and math.isfinite(float(cval)):
                all_crps.append(float(cval))
    apply(fig, title="Distributional loss by horizon",
          subtitle="CRPS and mean pinball loss are lower-is-better; each step is averaged across folds",
          mode="light", height=380, legend=n_models > 1, hovermode="x unified")
    fig.update_xaxes(title_text="Horizon step h", row=1, col=1)
    fig.update_xaxes(title_text="Horizon step h", row=1, col=2)
    fig.update_yaxes(title_text="CRPS", row=1, col=1)
    fig.update_yaxes(title_text="Pinball loss", row=1, col=2)
    pmf_curves = sum(1 for c in loss_curves.values() if c.get("discrete_crps"))
    alt = (f"CRPS and pinball loss curves for {len(loss_curves)} models over {len(steps)} horizons. "
           + (f"CRPS ranges from {_fmt(min(all_crps))} to {_fmt(max(all_crps))}." if all_crps else
              "No finite CRPS values were available.")
           + (f" {pmf_curves} dotted curve(s) add ranked-probability CRPS from a discrete PMF."
              if pmf_curves else ""))
    return Chart(to_json(fig), alt, rows)


def pit_histograms(pit_by_model: dict[str, dict[str, Any]]) -> Chart:
    items = [(mid, p) for mid, p in pit_by_model.items() if p.get("hist")]
    n = max(1, len(items))
    cols = 2 if n > 1 else 1
    rows_n = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows_n, cols=cols,
                        subplot_titles=[f"{mid} PIT" for mid, _ in items],
                        vertical_spacing=min(0.13, 0.7 / max(rows_n, 1)),
                        horizontal_spacing=0.10)
    c = colours("light")
    labels = [f"{i * 10}–{(i + 1) * 10}%" for i in range(10)]
    behind: list[dict[str, Any]] = []
    for i, (mid, p) in enumerate(items):
        r, q = divmod(i, cols)
        hist = _finite(p.get("hist") or [])
        expected = float(np.nansum([v for v in hist if v is not None])) / 10.0
        colour = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Bar(
            x=labels, y=[v if v is not None else 0 for v in hist],
            marker={"color": colour, "line": {"width": 0}},
            name=mid, showlegend=False,
            hovertemplate=f"{mid}<br>%{{x}}<br>count %{{y}}<extra></extra>",
        ), row=r + 1, col=q + 1)
        if expected > 0:
            fig.add_hline(y=expected, line={"color": c["muted"], "width": 1.0, "dash": DASHES[2]},
                          row=r + 1, col=q + 1)
        for j, label in enumerate(labels):
            behind.append({
                "model": mid, "bin": label, "count": hist[j] if j < len(hist) else None,
                "expected_uniform": expected,
                "ks_p": p.get("ks_p"), "cvm_p": p.get("cvm_p"),
                "reading": p.get("reading", ""), "kind": p.get("kind", ""),
                "caveat": p.get("caveat", ""),
            })
        fig.update_xaxes(tickfont={"size": 8.5}, row=r + 1, col=q + 1)
        fig.update_yaxes(title_text="Count", tickfont={"size": 8.5}, row=r + 1, col=q + 1)
    apply(fig, title="PIT histograms by model",
          subtitle="Dotted line = uniform expectation. U-shape = too narrow; hill = too wide; skew = bias.",
          mode="light", height=max(300, 230 * rows_n + 70), legend=False, hovermode="closest")
    fig.update_annotations(font_size=10)
    cvm_ps = [float(p.get("cvm_p")) for _, p in items if p.get("cvm_p") is not None and math.isfinite(float(p.get("cvm_p")))]
    ps_phrase = p_phrase(min(cvm_ps)) if cvm_ps else "p-values unavailable"
    alt = (f"PIT histograms for {len(items)} models, ten bins each, with the uniform expectation. "
           f"The smallest Cramér–von Mises uniformity {ps_phrase}. "
           "A U-shape means intervals are too narrow, a hill too wide, and a skew a biased centre.")
    return Chart(to_json(fig), alt, behind)


def calibration_curve(rows: Sequence[dict[str, Any]]) -> Chart:
    c = colours("light")
    fig = go.Figure()
    # A report built from an older payload may omit `nominal` or carry a string.
    # Keep the row, draw no finite calibration point, and say so in the alt text
    # rather than raising a TypeError or inventing a coverage level.
    nominal = _finite([r.get("nominal") for r in rows])
    empirical = _finite([r.get("empirical") for r in rows])
    ps = _finite([r.get("p") for r in rows])
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line={"color": c["muted"], "width": 1.2, "dash": DASHES[2]},
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=nominal, y=empirical, mode="lines+markers", name="Empirical coverage",
        line={"color": PALETTE[0], "width": 2.0},
        marker={"size": 8, "color": PALETTE[1]},
        customdata=[[p_number(p), r.get("construction", "")] for p, r in zip(ps, rows)],
        hovertemplate="Nominal %{x:.0%}<br>Empirical %{y:.1%}<br>%{customdata[0]}<extra></extra>",
    ))
    apply(fig, title="Coverage vs nominal calibration",
          subtitle="Sharpness subject to calibration: only calibrated intervals earn a width comparison",
          x_title="Nominal coverage", y_title="Empirical coverage", mode="light",
          height=360, legend=True, hovermode="closest")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    behind = [{
        "nominal": r.get("nominal"), "coverage_pct": r.get("coverage_pct"),
        "empirical": r.get("empirical"), "covered": r.get("covered"), "n": r.get("n"),
        "p": r.get("p"), "p_display": p_number(ps[i]),
        "mean_width": r.get("mean_width"),
        "construction": r.get("construction"), "source": r.get("source"),
        "note": r.get("note", ""),
    } for i, r in enumerate(rows)]
    finite_p = [p for p in ps if p is not None]
    worst = p_phrase(min(finite_p)) if finite_p else "p-values unavailable"
    finite_emp = [v for v in empirical if v is not None]
    widths = [w for w in _finite([r.get("mean_width") for r in rows]) if w is not None]
    alt = (f"Coverage-vs-nominal calibration at {len(rows)} nominal levels. "
           + (f"Empirical coverage ranges from {_fmt(min(finite_emp))} to {_fmt(max(finite_emp))}; "
              if finite_emp else "Empirical coverage is unavailable; ")
           + f"smallest binomial {worst}. "
           + (f"Mean widths range from {_fmt(min(widths))} to {_fmt(max(widths))}." if widths else "Widths unavailable."))
    return Chart(to_json(fig), alt, behind)


def dm_heatmap(dm: dict[str, Any]) -> Chart:
    ids = list(dm.get("ids") or [])
    raw_matrix = dm.get("matrix") or []
    mat = pd.DataFrame(np.full((len(ids), len(ids)), np.nan), index=ids, columns=ids, dtype=float)
    for i, row in enumerate(raw_matrix[:len(ids)]):
        for j, value in enumerate(list(row)[:len(ids)]):
            try:
                mat.iloc[i, j] = float(value) if value is not None and math.isfinite(float(value)) else np.nan
            except (TypeError, ValueError):
                mat.iloc[i, j] = np.nan
    chart = heatmap(
        mat, title="Diebold–Mariano test matrix",
        subtitle=f"Cell = raw DM p, HAC lag {dm.get('lags', 0)}; Holm-adjusted p-values are in the table",
        x_title="Column model", y_title="Row model", value_label="DM p",
        text_format=".3~g", height=360,
    )
    rows: list[dict[str, Any]] = []
    for r in dm.get("rows") or []:
        rows.append({
            "row_model": r.get("model_a"), "col_model": r.get("model_b"),
            "p": r.get("p"), "p_display": r.get("p_display") or p_number(r.get("p")),
            "p_holm": r.get("p_holm"),
            "p_holm_display": r.get("p_holm_display") or p_number(r.get("p_holm")),
            "statistic": r.get("statistic"), "n": r.get("n"),
            "better": r.get("better"), "plain": r.get("plain", ""), "ok": r.get("ok"),
        })
    finite_p = [float(r.get("p")) for r in dm.get("rows") or []
                if r.get("p") is not None and math.isfinite(float(r.get("p")))]
    chart.alt_text = (
        f"Diebold–Mariano p-value matrix for {len(ids)} models. "
        + (f"Smallest {p_phrase(min(finite_p))}." if finite_p else "No finite paired tests were available.")
        + " Lower loss is better; Holm-adjusted p-values for the family of unique pairs are "
          "reported in the table data."
    )
    chart.data_behind = rows
    return chart


def confusion_heatmap(hit: dict[str, Any]) -> Chart:
    conf = hit.get("confusion") or {}
    mat = pd.DataFrame(
        [[conf.get("predicted_up", {}).get("actual_up", 0), conf.get("predicted_up", {}).get("actual_down", 0)],
         [conf.get("predicted_down", {}).get("actual_up", 0), conf.get("predicted_down", {}).get("actual_down", 0)]],
        index=["Predicted up", "Predicted down"], columns=["Actual up", "Actual down"], dtype=float,
    )
    chart = heatmap(
        mat, title="Directional accuracy: hit-rate confusion",
        subtitle="Direction is judged from the last observed value at each forecast origin",
        x_title="Observed direction", y_title="Forecast direction", value_label="Count",
        text_format=".0f", height=320,
    )
    behind = [
        {"predicted": "up", "actual": "up", "count": mat.loc["Predicted up", "Actual up"]},
        {"predicted": "up", "actual": "down", "count": mat.loc["Predicted up", "Actual down"]},
        {"predicted": "down", "actual": "up", "count": mat.loc["Predicted down", "Actual up"]},
        {"predicted": "down", "actual": "down", "count": mat.loc["Predicted down", "Actual down"]},
        {"hit_rate": hit.get("rate"), "p": hit.get("p"), "p_display": p_number(hit.get("p")),
         "n": hit.get("n"), "note": hit.get("note", "")},
    ]
    chart.alt_text = (
        f"Directional confusion matrix over {hit.get('n', 0)} out-of-sample points. "
        f"Hit rate {_fmt(hit.get('rate'), 3)}; exact binomial {p_phrase(hit.get('p'))}. "
        "This test has low power in short evaluation windows."
    )
    chart.data_behind = behind
    return chart


def rqr_qq(rqr_by_model: dict[str, dict[str, Any]]) -> Chart:
    c = colours("light")
    fig = go.Figure()
    behind: list[dict[str, Any]] = []
    all_theo: list[float] = []
    for i, (mid, r) in enumerate(rqr_by_model.items()):
        theo = [float(v) for v in (r.get("qq_theoretical") or []) if v is not None and math.isfinite(float(v))]
        samp = [float(v) for v in (r.get("qq_sample") or []) if v is not None and math.isfinite(float(v))]
        n = min(len(theo), len(samp))
        if not n:
            continue
        all_theo.extend(theo[:n])
        colour = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=theo[:n], y=samp[:n], mode="markers", name=mid,
            marker={"color": colour, "size": 5.5, "opacity": 0.8},
            hovertemplate=f"{mid}<br>theoretical %{{x:.3f}}<br>residual %{{y:.3f}}<extra></extra>",
        ))
        for j in range(n):
            behind.append({
                "model": mid, "theoretical": theo[j], "sample": samp[j],
                "combined_p": r.get("combined_p"), "ks_p": r.get("ks_p"),
                "reps": r.get("reps"), "stability": r.get("stability", ""),
                "caveat": r.get("caveat", ""),
            })
    lo = float(min(all_theo)) if all_theo else -3.0
    hi = float(max(all_theo)) if all_theo else 3.0
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="N(0,1) reference",
        line={"color": c["muted"], "width": 1.2, "dash": DASHES[2]}, hoverinfo="skip",
    ))
    apply(fig, title="Randomized quantile residual QQ",
          subtitle="One seeded inferential draw; repetitions report stability, not an average",
          x_title="Theoretical normal quantile", y_title="RQR (one draw)", mode="light",
          height=370, legend=True, hovermode="closest")
    combined = [float(r.get("combined_p")) for r in rqr_by_model.values()
                if r.get("combined_p") is not None and math.isfinite(float(r.get("combined_p")))]
    caveat = next((str(r.get("caveat", "")) for r in rqr_by_model.values() if r.get("caveat")), "")
    alt = (f"Dunn–Smyth randomized quantile residual QQ for {len(rqr_by_model)} models. "
           + (f"Smallest one-draw KS p {p_phrase(min(combined))}." if combined else
              "One-draw KS p-values unavailable.")
           + " Randomization is disclosed: one seeded draw is plotted and tested; extra draws "
             "measure stability and are not Fisher-combined. "
           + caveat)
    return Chart(to_json(fig), alt, behind)


def split_chart(
    series: pd.Series,
    folds: Sequence[dict[str, Any]],
    *,
    n: int,
    frequency: str,
    evaluation_window: str,
) -> Chart:
    palette_fill = [with_alpha(PALETTE[i], 0.10) for i in range(4)]
    shading: list[dict[str, Any]] = []
    for i, f in enumerate(folds):
        ts, te = int(f.get("test_start", 0)), int(f.get("test_end", 0))
        if te >= len(series.index):
            continue
        shading.append({
            "x0": series.index[ts], "x1": series.index[te],
            "label": f"Fold {int(f.get('fold', 0)) + 1} · OOS only",
            "colour": palette_fill[i % len(palette_fill)],
        })
    chart = time_series(
        series,
        title="Rolling-origin train / test split",
        subtitle=f"n = {n} · {frequency} · evaluation {evaluation_window} · shaded zones are out-of-sample only",
        y_title="Observed series", shading=shading, height=380,
    )
    behind: list[dict[str, Any]] = [
        {"kind": "observation", "time": str(row.get("time")), "value": row.get("value")}
        for row in chart.data_behind
    ]
    for f in folds:
        behind.append({
            "kind": "fold", "fold": f.get("fold"),
            "train_start": f.get("train_start"), "train_end": f.get("train_end"),
            "test_start": f.get("test_start"), "test_end": f.get("test_end"),
            "train_window": f.get("train_window"), "test_window": f.get("test_window"),
            "n_train": f.get("n_train"), "n_test": f.get("n_test"),
            "zone": f.get("zone", "out-of-sample only"),
            "d": f.get("d"), "D": f.get("D"), "lambda": f.get("lambda"),
            "benchmark": f.get("benchmark"), "decision_digest": f.get("decision_digest"),
        })
    chart.alt_text = (
        f"{chart.alt_text} {len(folds)} rolling-origin folds are shaded as out-of-sample-only zones "
        f"from {evaluation_window}; training decisions were made before each shaded block."
    )
    chart.data_behind = behind
    return chart


__all__ = [
    "calibration_curve", "confusion_heatmap", "dm_heatmap", "fold_boxplots",
    "loss_curves_chart", "mase_by_horizon_chart", "metric_heatmap",
    "pit_histograms", "rqr_qq", "split_chart",
]
