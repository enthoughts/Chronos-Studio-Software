"""
ts/metrics/comparative.py — model comparison on the SAME held-out observations (§5.12).

Everything here takes loss/error series already computed out of sample. No
information criteria, no in-sample residuals, no mixing of differencing orders.
Nested models get the canonical Clark–West MSPE-adjusted test and
Mincer–Zarnowitz; all pairs get Diebold–Mariano with a Newey–West HAC variance
whose lag count matches the `horizon − 1` overlap of h-step-ahead errors.

The DM matrix reports both raw and Holm-adjusted p-values across the family of
unique model pairs. This does not turn a descriptive scan into a pre-specified
experiment, but it stops the table from presenting 20 uncorrected p-values as
20 independent confirmations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.core.formatting import p_number, p_phrase
from app.ts.select.tests_compare import (
    MIN_PAIRS,
    clark_west,
    diebold_mariano,
    mincer_zarnowitz,
)

_CITATIONS = [
    "Diebold & Mariano (1995), Comparing Predictive Accuracy, JBES 13(3)",
    "Clark & West (2007), Approximately normal tests for equal predictive accuracy in nested models, JBES 25(1)",
    "Mincer & Zarnowitz (1969), Evaluation of economic forecasts",
    "Hansen, Lunde & Nason (2011), The Model Confidence Set, Econometrica 79(2)",
]


@dataclass
class HitRate:
    n: int = 0
    n_total: int = 0
    n_ties: int = 0
    correct: int = 0
    rate: float | None = None
    p: float | None = None
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    note: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "n_total": self.n_total, "n_ties": self.n_ties,
            "correct": self.correct, "rate": self.rate, "p": self.p,
            "tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn,
            "confusion": {
                "predicted_up": {"actual_up": self.tp, "actual_down": self.fp},
                "predicted_down": {"actual_up": self.fn, "actual_down": self.tn},
            },
            "note": self.note, "warnings": list(self.warnings),
        }


def directional_accuracy(
    baseline: Sequence[float], actual: Sequence[float], forecast: Sequence[float]
) -> HitRate:
    """Did the forecast call the direction of change from the forecast origin?

    A tie — the actual or the forecast exactly equals the origin value — has no
    direction. It is excluded and counted, not silently booked as "down", which
    would inflate the hit rate on flat series.
    """
    from scipy import stats as st

    b = np.asarray(baseline, dtype=float).reshape(-1)
    y = np.asarray(actual, dtype=float).reshape(-1)
    f = np.asarray(forecast, dtype=float).reshape(-1)
    n_total = int(min(b.size, y.size, f.size))
    finite = (
        np.isfinite(b[:n_total]) & np.isfinite(y[:n_total]) & np.isfinite(f[:n_total])
    )
    b, y, f = b[:n_total][finite], y[:n_total][finite], f[:n_total][finite]
    tie = np.isclose(f, b, rtol=0.0, atol=1e-12) | np.isclose(y, b, rtol=0.0, atol=1e-12)
    bb, yy, ff = b[~tie], y[~tie], f[~tie]
    pred_up, actual_up = ff > bb, yy > bb
    tp, tn = int((pred_up & actual_up).sum()), int((~pred_up & ~actual_up).sum())
    fp, fn = int((pred_up & ~actual_up).sum()), int((~pred_up & actual_up).sum())
    n = tp + tn + fp + fn
    rate = (tp + tn) / n if n else None
    p = float(st.binomtest(tp + tn, n, 0.5).pvalue) if n >= 10 else None
    n_ties = int(tie.sum())
    return HitRate(
        n=n, n_total=int(finite.sum()), n_ties=n_ties,
        correct=tp + tn, rate=rate, p=p, tp=tp, tn=tn, fp=fp, fn=fn,
        note=(
            "Direction of change relative to the last observed value before each forecast origin; "
            f"a coin-flip null (0.5) exact binomial {p_phrase(p)}. "
            f"{n_ties} tied or non-finite origin(s) were excluded, not counted as 'down'."
        ),
        warnings=(
            ["Directional accuracy has low power with a short evaluation window; a hit rate above "
             "0.5 is not proof of forecast skill."] if n < 30 else []
        ) + (
            [f"{n_ties} direction tie(s) were excluded from the exact binomial test."] if n_ties else []
        ),
    )


def _pooled(folds: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray([float(e) for fold in folds for e in fold], dtype=float)


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    """Add family-wise Holm p-values across UNIQUE unordered DM pairs."""
    groups: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        p = row.get("p")
        if p is None or not math.isfinite(float(p)):
            continue
        key = tuple(sorted((str(row["model_a"]), str(row["model_b"]))))
        groups.setdefault(key, []).append(i)
    if not groups:
        return
    keys = list(groups)
    raw = [min(float(rows[i]["p"]) for i in groups[k]) for k in keys]
    order = sorted(range(len(keys)), key=lambda j: raw[j])
    adjusted = [0.0] * len(keys)
    running = 0.0
    m = len(keys)
    for rank, j in enumerate(order):
        running = max(running, (m - rank) * raw[j])
        adjusted[j] = min(1.0, running)
    for j, key in enumerate(keys):
        value = round(float(adjusted[j]), 4)
        for i in groups[key]:
            rows[i]["p_holm"] = value
            rows[i]["p_holm_display"] = p_number(value)


def dm_matrix(
    errors_by_model: dict[str, Sequence[Sequence[float]]], horizon: int = 1
) -> dict[str, Any]:
    """
    Pairwise Diebold–Mariano p-values. `errors_by_model[id]` is a list of per-fold
    error vectors; the pooled vector preserves fold order for the HAC lag choice.
    """
    ids = list(errors_by_model)
    pooled = {k: _pooled(v) for k, v in errors_by_model.items()}
    rows: list[dict[str, Any]] = []
    matrix: list[list[float | None]] = []
    for a in ids:
        row: list[float | None] = []
        for b in ids:
            if a == b:
                row.append(None)
                continue
            ea, eb = pooled[a], pooled[b]
            n = min(ea.size, eb.size)
            mask = np.isfinite(ea[:n]) & np.isfinite(eb[:n])
            if int(mask.sum()) < MIN_PAIRS:
                row.append(None)
                rows.append({"model_a": a, "model_b": b, "p": None, "p_holm": None,
                             "n": int(mask.sum()), "ok": False,
                             "note": "Too few paired out-of-sample errors for DM."})
                continue
            res = diebold_mariano(
                ea[:n][mask], eb[:n][mask],
                name_a=a, name_b=b, horizon=max(int(horizon), 1), loss="squared",
            )
            p = float(res.p_value) if res.ok and res.p_value is not None else None
            row.append(p)
            rows.append({
                "model_a": a, "model_b": b, "p": p, "p_holm": None,
                "p_display": p_number(p),
                "statistic": res.statistic,
                "n": res.n,
                "horizon": int(horizon),
                "lags": getattr(res, "lags", max(int(horizon), 1) - 1),
                "better": res.better,
                "plain": res.note,
                "ok": bool(res.ok),
            })
        matrix.append(row)
    _holm_adjust(rows)
    return {
        "ids": ids, "matrix": matrix, "rows": rows,
        "horizon": int(horizon),
        "lags": max(int(horizon), 1) - 1,
        "note": (
            "Cell p tests equal squared-error predictive accuracy of the row model against the "
            f"column model; HAC lag = horizon − 1 = {max(int(horizon), 1) - 1}. "
            "p_holm is Holm-adjusted across unique model pairs; use it when more than one pair "
            "was inspected. A pre-specified single pair remains the strongest reading."
        ),
        "citations": [_CITATIONS[0], "Holm (1979), A simple sequentially rejective multiple test procedure"],
    }


def nested_tests(
    *,
    actual: Sequence[float],
    forecast: Sequence[float],
    benchmark_forecast: Sequence[float],
    benchmark_errors: Sequence[float],
    model_errors: Sequence[float],
    horizon: int = 1,
    model_name: str = "the model",
) -> dict[str, Any]:
    """Canonical Clark–West and Mincer–Zarnowitz against the nested benchmark forecast."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    f = np.asarray(forecast, dtype=float).reshape(-1)
    bf = np.asarray(benchmark_forecast, dtype=float).reshape(-1)
    be = np.asarray(benchmark_errors, dtype=float).reshape(-1)
    me = np.asarray(model_errors, dtype=float).reshape(-1)
    n = min(y.size, f.size, bf.size, be.size, me.size)
    mask = (
        np.isfinite(y[:n]) & np.isfinite(f[:n]) & np.isfinite(bf[:n])
        & np.isfinite(be[:n]) & np.isfinite(me[:n])
    )
    out: dict[str, Any] = {"n": int(mask.sum()), "citations": _CITATIONS[1:3],
                           "model": model_name}
    if out["n"] < MIN_PAIRS:
        out["note"] = "At least 8 paired observations are needed for the nested comparison."
        return out
    # f_small − f_large, supplied explicitly so the adjustment cannot be inferred
    # from a different pairing or silently ignored.
    forecast_gap = bf[:n][mask] - f[:n][mask]
    cw = clark_west(
        be[:n][mask], me[:n][mask], name_small="naive / seasonal-naive",
        name_large=model_name, horizon=max(int(horizon), 1), loss="squared",
        forecast_gap=forecast_gap,
    )
    out["clark_west"] = {
        "ok": bool(cw.ok),
        "statistic": cw.statistic,
        "p": cw.p_value,
        "p_display": p_number(cw.p_value),
        "adjusted_mean": cw.adjusted_mean,
        "adjustment": cw.adjustment,
        "tail": cw.tail,
        "plain": cw.note,
        "assumptions": list(getattr(cw, "assumptions", [])),
        "note": "Canonical Clark & West (2007) MSPE-adjusted statistic: e_small² − e_large² "
                "+ (f_small − f_large)², one-sided p-value, HAC lag = horizon − 1.",
    }
    mz = mincer_zarnowitz(y[:n][mask], f[:n][mask], name=model_name, horizon=max(int(horizon), 1))
    out["mincer_zarnowitz"] = {
        "ok": bool(mz.ok),
        "intercept": mz.intercept,
        "slope": mz.slope,
        "joint_chi2": mz.joint_chi2,
        "p": mz.joint_p,
        "p_display": p_number(mz.joint_p),
        "plain": mz.verdict,
        "note": mz.note or "Joint Wald test of (intercept, slope) = (0, 1). A rejection is a "
                           "calibration failure, not a skill ranking.",
    }
    out["note"] = (
        "Nested comparison uses the training-window naive / seasonal-naive benchmark as the "
        "restricted forecast. Mincer–Zarnowitz tests calibration of the unrestricted forecast."
    )
    return out


def model_confidence_set(
    errors_by_model: dict[str, Sequence[Sequence[float]]], *, alpha: float = 0.10
) -> dict[str, Any]:
    """
    Transparent range-statistic MCS with a stationary bootstrap over the
    per-observation loss matrix. Refuses short samples rather than quoting a
    decorative p-value.
    """
    pooled = {k: _pooled(v) for k, v in errors_by_model.items()}
    if len(pooled) < 2:
        return {"applicable": False, "note": "MCS needs at least two models.", "citations": [_CITATIONS[3]]}
    n = min(v.size for v in pooled.values())
    if n < 30:
        return {
            "applicable": False,
            "note": f"Only {n} pooled out-of-sample losses; the MCS bootstrap is not reported "
                    "below 30 because its p-values would be decorative.",
            "citations": [_CITATIONS[3]],
        }
    ids = list(pooled)
    losses = np.column_stack([pooled[k][:n] ** 2 for k in ids])
    usable = np.isfinite(losses).all(axis=1)
    if usable.sum() < 30:
        return {
            "applicable": False,
            "note": f"Only {int(usable.sum())} rows have a finite loss for every model; MCS needs 30.",
            "citations": [_CITATIONS[3]],
        }
    losses = losses[usable]
    n = losses.shape[0]
    observed_means = losses.mean(axis=0)
    rng = np.random.default_rng(20260101)
    boots = 1000
    mean_block = max(4.0, np.sqrt(n))
    reproduced = np.empty((boots, len(ids)))
    for b in range(boots):
        idx = np.empty(n, dtype=int)
        i, j = int(rng.integers(0, n)), 0
        while j < n:
            idx[j] = i
            j += 1
            if rng.random() < 1.0 / mean_block:
                i = int(rng.integers(0, n))
            else:
                i = (i + 1) % n
        reproduced[b] = losses[idx].mean(axis=0)
    boot_dev = reproduced - observed_means[None, :]

    survivors = list(ids)
    removed: list[dict[str, Any]] = []
    p_last = 1.0
    while len(survivors) > 1:
        cols = [ids.index(k) for k in survivors]
        means = observed_means[cols]
        worst_pos = int(np.argmax(means))
        target = survivors[worst_pos]
        target_col = ids.index(target)
        target_pos = cols.index(target_col)
        d_obs = means - means[target_pos]
        stat_obs = float(np.max(np.abs(d_obs - d_obs.mean())))
        bd = boot_dev[:, cols]
        d_boot = bd - bd[:, [target_pos]]
        stat_boot = np.max(np.abs(d_boot - d_boot.mean(axis=1, keepdims=True)), axis=1)
        p = float(np.mean(stat_boot >= stat_obs))
        p_last = p
        if p < alpha:
            removed.append({"model": target, "p": round(p, 4)})
            survivors.pop(worst_pos)
        else:
            break
    return {
        "applicable": True,
        "survivors": survivors,
        "removed": removed,
        "mcs_p": round(p_last, 4),
        "alpha": alpha,
        "n": int(n), "boot": boots,
        "note": "Range statistic with a stationary bootstrap (mean block √n, 1,000 resamples, "
                "seed 20260101). At each step the bootstrapped and observed loss differentials "
                "are centred the same way; model loss is squared OOS error.",
        "citations": [_CITATIONS[3]],
    }


__all__ = [
    "HitRate", "directional_accuracy", "dm_matrix", "model_confidence_set", "nested_tests",
]
