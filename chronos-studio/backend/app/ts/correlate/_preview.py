"""
ts/correlate/_preview.py — "what would happen if I did this?" (§5.6).

The spec calls the preview mandatory, and it is the single most useful thing in the S7 advisor: it
is what a textbook cannot do. Every candidate action is applied to a **copy** of the series, the
diagnostics are recomputed, and the result is scored against the current state.

The scoring is deliberately simple and stated in words, because a preview that reports a
mysterious score is no better than no preview:

* **better**  — the ACF loses its slow decay *and* the portmanteau p-value improves, *or* the
  variance ratio moves toward 1 without the lag-1 correlation becoming strongly negative.
* **worse**   — lag 1 goes below −0.5 (the over-differencing fingerprint), or variance rises while
  the ACF keeps decaying, or the portmanteau p-value falls substantially.
* **neutral** — anything else. Neutral is a real answer and is reported as one.

Nothing here decides anything. It reports, the advisor ranks, and the user commits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.ts.correlate._acf import acf, default_max_lag, ljung_box, read_pattern, variance_ratio


def _summarise(series: pd.Series, period: int | None) -> dict[str, Any]:
    """The compact diagnostic fingerprint used to compare before and after."""
    y = series.dropna()
    a = acf(y, period=period)
    lb = ljung_box(a, [min(10, a.max_lag)])
    vr = None
    try:
        vr = variance_ratio(y, q=min(2, max(2, len(y) // 20)))
    except ValueError:
        vr = None
    # Level vs spread: a log transform is supposed to break this link. Measuring it is the only
    # honest way to judge a variance-stabilising transform, because the ACF is *not supposed* to
    # change much — judging a log by its correlogram would mark a correct transform as a failure.
    win = max(5, len(y) // 8)
    roll_mean = y.rolling(win, min_periods=max(2, win // 2)).mean()
    roll_sd = y.rolling(win, min_periods=max(2, win // 2)).std(ddof=1)
    pair = pd.concat([roll_mean, roll_sd], axis=1).dropna()
    level_spread = float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) > 4 else float("nan")
    # Does a seasonal spike survive? For a first difference this is the difference between "fixed"
    # and "fixed the trend but left the season".
    seasonal_left = False
    if period and period >= 2 and period <= a.max_lag:
        seasonal_left = bool(a.significant(period))
    return {
        "n": int(len(y)),
        "sd": float(y.std(ddof=1)) if len(y) > 1 else float("nan"),
        "level_spread": level_spread,
        "seasonal_left": seasonal_left,
        "r1": float(a.values[0]),
        "decay": int(next((i for i, (v, b) in enumerate(zip(a.values, a.band_bartlett))
                           if abs(v) <= b), len(a.values))),
        "n_sig": len(a.significant_lags()),
        "max_lag": a.max_lag,
        "lb_p": float(lb.p_values[0]),
        "lb_stat": float(lb.stats[0]),
        "vr": None if vr is None else float(vr["ratio"]),
        "band": a.band_white,
    }


def _verdict(before: dict[str, Any], after: dict[str, Any], op: str) -> tuple[str, str]:
    """
    Score the change and write the sentence that explains it.

    **The criterion depends on what the operation is for.** A difference is judged on the
    correlogram and the variance ratio; a log transform is judged on the link between the level and
    the spread, because a log is not supposed to change the ACF much and marking it "worse" for
    that would be exactly the kind of confidently-wrong output this app exists to avoid.
    """
    r1, r1b = after["r1"], before["r1"]

    # ---------------- log: judge variance stability, not autocorrelation ------
    if op == "transform.log":
        ls_b, ls_a = before["level_spread"], after["level_spread"]
        if np.isfinite(ls_b) and np.isfinite(ls_a):
            if ls_b > 0.5 and ls_a < ls_b - 0.15:
                return (
                    "better",
                    f"The link between the level and the spread weakens: their rank correlation "
                    f"falls from {ls_b:.2f} to {ls_a:.2f}, so the series is closer to having one "
                    f"variance. The ACF is deliberately not the criterion here — a log is not "
                    f"meant to change it (lag 1 moves from {r1b:.3f} to {r1:.3f}).",
                )
            if ls_a > ls_b + 0.15:
                return (
                    "worse",
                    f"The level-to-spread correlation rises from {ls_b:.2f} to {ls_a:.2f}, so the "
                    "log did not stabilise the variance here.",
                )
            return (
                "neutral",
                f"The spread was already roughly constant against the level (rank correlation "
                f"{ls_b:.2f} before, {ls_a:.2f} after), so there is little for a log to do.",
            )
        return ("neutral", "The variance stability could not be measured before and after.")

    decay_improved = after["decay"] < before["decay"] and after["decay"] <= max(2, before["decay"] * 0.5)
    lb_both_reject = before["lb_p"] < 0.001 and after["lb_p"] < 0.001
    lb_better = (after["lb_p"] > before["lb_p"] * 1.5) or (after["lb_p"] > 0.05 >= before["lb_p"])
    lb_worse = (after["lb_p"] * 1.5 < before["lb_p"]) and after["lb_p"] < 0.05
    var_rose = after["sd"] > before["sd"] * 1.05
    season = (" A seasonal spike at the seasonal lag is still outside its band afterwards, so this "
              "change does not address the season on its own." if after.get("seasonal_left") else "")

    if op.startswith("difference") and r1 < -0.5:
        return (
            "worse",
            f"Lag 1 falls to {r1:.3f}, which is the signature of differencing a series that was "
            "already stationary (differencing white noise produces exactly −0.5 at lag 1 and zero "
            f"after). The spread went from {before['sd']:.4g} to {after['sd']:.4g} and the variance "
            f"ratio is {_f(after['vr'])}." + season,
        )
    if op.startswith("difference") and var_rose and r1 < -0.3:
        return (
            "worse",
            f"The spread grew after the difference ({before['sd']:.4g} → {after['sd']:.4g}) while "
            f"lag 1 went to {r1:.3f}. That combination means the difference removed less structure "
            "than it added noise." + season,
        )
    if decay_improved and not lb_worse:
        lb_text = (
            "the portmanteau still rejects at both lengths, so other lags remain"
            if lb_both_reject else
            f"the portmanteau p-value moves from {before['lb_p']:.4f} to {after['lb_p']:.4f}"
        )
        return (
            "better",
            f"The slow decay is gone: the ACF is now significant only through lag {after['decay']} "
            f"instead of {before['decay']}, and lag 1 falls from {r1b:.3f} to {r1:.3f}; {lb_text}."
            + season,
        )
    if lb_worse and not decay_improved:
        return (
            "worse",
            f"Neither the decay nor the portmanteau improves: the ACF is still significant through "
            f"lag {after['decay']} and the Ljung–Box p-value falls from {before['lb_p']:.4f} to "
            f"{after['lb_p']:.4f}." + season,
        )
    if after["decay"] > before["decay"]:
        return (
            "worse",
            f"The ACF becomes *more* significant, not less: it is outside its band through lag "
            f"{after['decay']} where it stopped at {before['decay']} before. This change adds "
            "structure rather than removing it." + season,
        )
    if abs(r1 - r1b) < 0.05 and after["decay"] == before["decay"]:
        return (
            "neutral",
            f"This changes almost nothing: lag 1 moves from {r1b:.3f} to {r1:.3f} and the ACF is "
            f"still significant through lag {after['decay']}. It is not the operation the "
            "correlogram is asking for." + season,
        )
    return (
        "neutral",
        f"The correlogram changes but does not clearly improve: significance now ends at lag "
        f"{after['decay']} (was {before['decay']}) and lag 1 moves from {r1b:.3f} to {r1:.3f}."
        + season,
    )


def preview_action(
    series: pd.Series,
    op: str,
    *,
    period: int | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run `op` on a copy and describe what the correlogram and tests would look like afterwards.

    Returns a dict with `verdict` ∈ {better, neutral, worse, unavailable}, a `plain` sentence, and
    the before/after numbers. An operation that cannot be computed returns `unavailable` with the
    reason — never a fabricated verdict.
    """
    params = dict(params or {})
    try:
        before = _summarise(series, period)
    except ValueError as exc:
        return {"verdict": "unavailable", "plain": f"The starting series cannot be summarised: {exc}",
                "before": None, "after": None}

    try:
        y = series.dropna()
        if op == "difference":
            order = params.get("order", 1)
            if order == "seasonal":
                if not period or period < 2:
                    return _unavailable("No seasonal period was detected, so a seasonal difference "
                                        "has nothing to remove.")
                after_series = y.diff(period)
                label = f"seasonal difference (lag {period})"
            else:
                after_series = y.diff(int(order))
                label = f"difference of order {int(order)}"
        elif op == "undo":
            after_series = None
            return _unavailable(
                "Undo is not previewable here: it restores the previous version of the series in "
                "the session, so the 'after' state is that earlier version's own diagnostics."
            )
        elif op == "transform.log":
            if (y <= 0).any():
                n_bad = int((y <= 0).sum())
                return _unavailable(
                    f"A log is undefined for zero or negative values, and {n_bad} of the "
                    f"{len(y)} observations are one or the other. Shift the series first, or use a "
                    "Box–Cox transform with an offset."
                )
            after_series = np.log(y)
            label = "log transform"
        else:
            return _unavailable(f"'{op}' is not an operation the preview can run.")
    except Exception as exc:                            # pragma: no cover - defensive
        return _unavailable(f"The preview could not run: {type(exc).__name__}: {exc}")

    after_series = after_series.dropna()
    if len(after_series) < 10:
        return _unavailable("The result would be too short to re-diagnose.")

    after = _summarise(after_series, period)
    verdict, plain = _verdict(before, after, op)

    # The pattern reading after the change, so the user sees the new shape in words, not just numbers.
    a2 = acf(after_series, period=period)
    from app.ts.correlate._acf import pacf
    pattern_after = read_pattern(a2, pacf(after_series, period=period), period=period,
                                 differenced=op.startswith("difference"))

    return {
        "verdict": verdict,
        "op": op,
        "label": label,
        "plain": plain,
        "pattern_after": pattern_after.kind,
        "pattern_after_label": pattern_after.label,
        "pattern_after_plain": pattern_after.plain,
        "before": before,
        "after": after,
        "n_lost": int(before["n"] - after["n"]),
    }


def _unavailable(why: str) -> dict[str, Any]:
    return {"verdict": "unavailable", "plain": why, "before": None, "after": None}


def _f(x: float | None) -> str:
    return "not computed" if x is None else f"{x:.2f}"


__all__ = ["preview_action"]
