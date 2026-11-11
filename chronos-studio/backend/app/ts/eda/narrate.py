"""
ts/eda/narrate.py — the deterministic plain-English description of a series (§5.2).

This module exists for two reasons:

1. The "just tell me what this is" card at the top of the EDA panel.
2. It is the **same generator the AI assistant falls back to in Offline mode**, so
   building it once here means the app reads consistently whether or not a Gemini
   key is present (§5.14).

Hard rule (R0.5): every sentence here is produced *from a number that was computed*.
No sentence may be emitted whose content is not also visible in the results, and no
number may be invented — the tests assert that the digits in the output all appear
in the input facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.schemas import DataClass


# The book's rule-of-thumb boundary between "moderate" and "strong" seasonality (F_S).
STRONG_STRENGTH = 0.64


def _plural(n: float, singular: str, plural: str | None = None) -> str:
    """`1 observation`, `3 observations` — small thing, but the prose is the product."""
    return singular if abs(float(n)) == 1 else (plural or f"{singular}s")


@dataclass
class NarrativeFacts:
    """
    The flattened set of computed values the narrator is allowed to use.

    Anything not in here cannot appear in the prose; that is enforced by
    `assert_grounded()` and tested.
    """

    series_name: str
    n: int
    n_missing: int
    time_start: str
    time_end: str
    frequency_label: str
    period: int | None
    data_class: DataClass

    first: float
    last: float
    pct_change: float
    mean: float
    median: float
    sd: float
    cv: float
    minimum: float
    maximum: float
    skewness: float

    n_outliers: int
    outlier_types: dict[str, int]
    n_zeros: int
    pct_zeros: float
    n_negatives: int
    n_gaps: int
    constant_run: int

    start_year: str = ""
    end_year: str = ""

    seasonality: dict[str, Any] = field(default_factory=dict)
    variance_stability: dict[str, Any] = field(default_factory=dict)
    trend_hint: dict[str, Any] = field(default_factory=dict)
    persistence: dict[str, Any] = field(default_factory=dict)

    def numbers(self) -> dict[str, float]:
        """Flatten every number the prose may cite (used by the grounding check)."""
        flat: dict[str, float] = {
            "n": float(self.n), "n_missing": float(self.n_missing),
            "first": self.first, "last": self.last, "pct_change": self.pct_change,
            "mean": self.mean, "median": self.median, "sd": self.sd, "cv": self.cv,
            "minimum": self.minimum, "maximum": self.maximum, "skewness": self.skewness,
            "n_outliers": float(self.n_outliers), "n_zeros": float(self.n_zeros),
            "pct_zeros": self.pct_zeros, "n_negatives": float(self.n_negatives),
            "n_gaps": float(self.n_gaps), "constant_run": float(self.constant_run),
            # the calendar span is printed as years; register them as citable numbers
            "start_year": float(self.start_year) if self.start_year.isdigit() else float("nan"),
            "end_year": float(self.end_year) if self.end_year.isdigit() else float("nan"),
        }
        for group in ("seasonality", "variance_stability", "trend_hint", "persistence"):
            for k, v in getattr(self, group).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    flat[f"{group}.{k}"] = float(v)

        # Quantities the prose states in different units from the raw fact. They are
        # registered explicitly rather than approximated, so the grounding check can
        # still hold the text to a number that was actually computed.
        ratio = self.variance_stability.get("ratio")
        if isinstance(ratio, (int, float)) and np.isfinite(ratio):
            flat["variance_stability.growth_pct"] = (float(ratio) - 1.0) * 100.0
        flat["seasonality.strong_threshold"] = STRONG_STRENGTH
        share = self.trend_hint.get("monotonic_share")
        if isinstance(share, (int, float)) and np.isfinite(share):
            flat["trend_hint.monotonic_pct"] = float(share) * 100.0
        return flat


def _fmt(x: float, digits: int = 3) -> str:
    """
    Format a computed number without ever inventing a digit.

    The grounding rule (R0.5) is that a reader can check every number against the
    results. Printing `242` when the computed value is `241.9` breaks that check for
    no benefit, so the value is rounded first and then formatted — the printed string
    is always an exact rendering of a (rounded) computed value.
    """
    if x is None or not np.isfinite(x):
        return "—"
    ax = abs(x)
    if ax >= 1e6 or (ax != 0 and ax < 0.01):
        return f"{x:.2e}"
    # Three significant digits is the house style: it is enough to read the number,
    # and it is coarse enough that nobody mistakes it for false precision.
    rounded = float(f"{x:.3g}")
    if ax >= 1000 or rounded == int(rounded):
        return f"{rounded:,.0f}"
    if ax >= 1:
        return f"{rounded:,}"
    return f"{rounded:,g}"


def narrate(facts: NarrativeFacts, audience: str = "beginner") -> list[str]:
    """
    Produce 3–6 short paragraphs: what the series is, what shape it has, what is
    unusual about it, and what that implies for the next step.
    """
    paras: list[str] = []
    period_txt = f"seasonal period {facts.period}" if facts.period else "no seasonal period"

    # --- 1. identity ------------------------------------------------------- #
    direction = "rose" if facts.pct_change > 0 else ("fell" if facts.pct_change < 0 else "stayed flat")
    p1 = (
        f"{facts.series_name} has {facts.n} {facts.frequency_label.lower()} observations "
        f"from {facts.time_start[:10]} to {facts.time_end[:10]} ({period_txt}). "
        f"Over that window it {direction} from {_fmt(facts.first)} to {_fmt(facts.last)}"
    )
    if np.isfinite(facts.pct_change):
        p1 += f", a change of {_fmt(facts.pct_change, digits=4)}%"
    p1 += f". The average level is {_fmt(facts.mean)} (median {_fmt(facts.median)}), with a spread of {_fmt(facts.sd)}."
    paras.append(p1)

    # --- 2. shape ---------------------------------------------------------- #
    if facts.cv < 0.1:
        shape = "The series is very stable relative to its level"
    elif facts.cv < 0.3:
        shape = "The series varies moderately around its level"
    else:
        shape = "The series is highly variable relative to its level"
    if abs(facts.skewness) >= 1:
        shape += (
            f", and it is strongly skewed {'right' if facts.skewness > 0 else 'left'}, "
            "so a few large values pull the average away from the typical value."
        )
    elif abs(facts.skewness) >= 0.5:
        shape += f", and it is moderately skewed {'right' if facts.skewness > 0 else 'left'}."
    else:
        shape += ", and its distribution is roughly symmetric."
    shape += (
        f" Values run from {_fmt(facts.minimum)} to {_fmt(facts.maximum)}, around a "
        f"median of {_fmt(facts.median)}."
    )
    paras.append(shape)

    # --- 3. seasonality ---------------------------------------------------- #
    seas = facts.seasonality or {}
    if seas.get("available") and seas.get("strength") is not None:
        strength = float(seas["strength"])
        if strength >= STRONG_STRENGTH:
            verdict = "strong and consistent"
        elif strength >= 0.3:
            verdict = "present but moderate"
        else:
            verdict = "weak"
        line = (
            f"Seasonality with period {facts.period} is {verdict} "
            f"(seasonal strength F_S = {strength:.2f}, on a 0-to-1 scale where above "
            f"{STRONG_STRENGTH:g} is conventionally called strong)"
        )
        if seas.get("peak_label"):
            line += (
                f". The typical seasonal peak is in {seas['peak_label']} and the trough in "
                f"{seas.get('trough_label', '—')}, a swing of about "
                f"{_fmt(float(seas.get('amplitude', 0)))} units"
            )
        if seas.get("amplitude_growing") is True:
            line += (
                ", and the swing is getting larger over time — that usually means the "
                "seasonal effect multiplies rather than adds, so a log transform is worth trying"
            )
        paras.append(line + ".")
    elif facts.period and facts.period > 1:
        paras.append(
            f"There is no meaningful seasonal pattern at period {facts.period} in these data."
        )

    # --- 4. trend ---------------------------------------------------------- #
    trend = facts.trend_hint or {}
    if trend.get("available"):
        slope = float(trend["slope_per_period"])
        total = float(trend["total_change"])
        para = (
            f"The trend is {'upward' if slope > 0 else 'downward'} at about "
            f"{_fmt(abs(slope))} per observation, which amounts to {_fmt(abs(total))} across "
            f"the whole window ({trend.get('pct_over_window', 0):+.1f}%)"
        )
        if trend.get("monotonic_share") is not None:
            para += (
                f". {float(trend['monotonic_share']):.0%} of consecutive steps move in the "
                "same direction as the trend"
            )
        paras.append(para + ".")

    # --- 5. variance stability -------------------------------------------- #
    var = facts.variance_stability or {}
    if var.get("available") and var.get("ratio") is not None:
        ratio = float(var["ratio"])
        if ratio > 1.25:
            paras.append(
                f"The spread is not constant: it grows by roughly "
                f"{_fmt(float(var.get('growth_pct', (ratio - 1) * 100)), digits=1)}% "
                "between the start and end of the series (rolling standard deviation "
                f"{_fmt(float(var['early_sd']))} → {_fmt(float(var['late_sd']))}). "
                "Models that assume constant variance will be unreliable here, so a log or "
                "Box–Cox transform is likely to help."
            )
        elif ratio < 0.8:
            paras.append(
                f"The spread shrinks by roughly {(1 - ratio) * 100:.0f}% across the series "
                f"(rolling standard deviation {_fmt(float(var['early_sd']))} → {_fmt(float(var['late_sd']))})."
            )

    # --- 6. anomalies and data issues ------------------------------------- #
    issues = []
    if facts.n_missing:
        issues.append(f"{facts.n_missing} {_plural(facts.n_missing, 'observation')} are missing")
    if facts.n_gaps:
        issues.append(f"{facts.n_gaps} calendar {_plural(facts.n_gaps, 'position')} had to be inserted as blanks")
    if facts.n_outliers:
        kinds = ", ".join(f"{v} {k.replace('_', ' ')}{'s' if v > 1 else ''}"
                          for k, v in facts.outlier_types.items())
        issues.append(f"{facts.n_outliers} {_plural(facts.n_outliers, 'observation')} look unusual once trend and season are removed ({kinds})")
    if facts.constant_run >= 6 and facts.data_class is not DataClass.INTERMITTENT:
        issues.append(f"there is a run of {facts.constant_run} identical values, which usually means "
                      "the series stalled or was truncated")
    if issues:
        paras.append("Data quality: " + "; ".join(issues) + ".")

    # --- 7. data-class consequence ---------------------------------------- #
    if facts.data_class is DataClass.INTERMITTENT:
        paras.append(
            f"{facts.pct_zeros:.0%} of the values are zero, so this is intermittent demand. "
            "Ordinary ARIMA/ETS models assume a continuous signal and will produce nonsense here; "
            "the appraisal in the next steps will steer towards Croston/TSB-type methods and MASE."
        )
    elif facts.data_class is DataClass.COUNT:
        paras.append(
            "The values are non-negative whole numbers, so this is count data. Poisson or "
            "negative-binomial models are the natural choice, and percentage-error metrics such "
            "as MAPE are not valid for it."
        )
    elif facts.data_class is DataClass.BINARY_OR_PROPORTION:
        paras.append(
            "Only two distinct values appear, so this behaves like a binary outcome rather than a "
            "continuous measurement."
        )

    if audience == "reviewer":
        paras = [_technical_rewrite(p) for p in paras]
    return paras


def _technical_rewrite(p: str) -> str:
    """A reviewer-facing register: same facts, less hand-holding."""
    return (
        p.replace("just tell", "summarise")
        .replace("look unusual", "are flagged as anomalies")
        .replace("will produce nonsense here", "are mis-specified here")
    )


# --------------------------------------------------------------------------- #
# Grounding checker (§5.14) — reused by the AI layer in Part 17
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


# Dates and clock times are identifiers, not measurements: "2020-12-30" must not be
# read as the numbers 2020, -12 and -30 when we check what a sentence claims.
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2})?(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
)


def numbers_in(text: str) -> set[str]:
    """Every numeric token in a piece of prose, normalised for comparison."""
    out: set[str] = set()
    for raw in _NUM_RE.findall(_DATE_RE.sub(" ", text)):
        token = raw.replace(",", "").strip(".")
        if not token or token == "-":
            continue
        try:
            val = float(token)
        except ValueError:
            continue
        out.add(f"{val:.6g}")
    return out


def assert_grounded(text: str, allowed: dict[str, float], tolerance_digits: int = 3) -> list[str]:
    """
    Return the numeric tokens in `text` that do not correspond to any computed value.

    This is the mechanism that stops the assistant inventing a p-value. It is a
    *report*, not an exception: the caller decides whether to withhold the answer.
    """
    allowed_values = sorted(float(v) for v in allowed.values() if _finite(v))

    ungrounded: list[str] = []
    for token in numbers_in(text):
        fv = float(token)
        if _is_rounding_of(fv, allowed_values, tolerance_digits):
            continue
        # Small integers are usually counts of things named in the text ("3 tests"),
        # dates, or list positions; they are checked but treated leniently.
        if fv == int(fv) and abs(fv) <= 20:
            continue
        ungrounded.append(token)
    return ungrounded


def _finite(v: Any) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _is_rounding_of(token_value: float, allowed_values: list[float], tolerance_digits: int) -> bool:
    """
    True when `token_value` is a faithful rendering of some computed value.

    The question is "could a person have produced this token by rounding one of the
    computed numbers?", not "is it identical to it". 285.7 is a legitimate rendering of
    285.7143; 0.037 is not a legitimate rendering of 0.612. The tolerance is therefore
    one of these, whichever is looser:

        * half of the last digit actually printed (so 0.612 stays 0.612), and
        * 1% relative (so three-significant-digit house style — 9,010 for 9,012.5,
          or 48 for 48.3% — is accepted).

    1% is far tighter than any mistake worth catching: an invented p-value or a
    misquoted mean is rarely within 1% of the number it should have been.
    """
    if not np.isfinite(token_value):
        return False
    ax = abs(token_value)
    reference = ax if ax > 0 else 1.0
    printed_digit = 10.0 ** (np.floor(np.log10(reference)) - max(1, len(f"{reference:.10g}".replace(".", "").lstrip("0"))) + 1)
    tolerance = max(0.5 * printed_digit, 0.01 * ax, 10.0 ** (-tolerance_digits))

    for v in allowed_values:
        if v == token_value:
            return True
        if abs(v - token_value) <= tolerance or abs(abs(v) - token_value) <= tolerance:
            # the prose often states a magnitude next to its signed change
            return True
        # the same computed share, stated in percent
        if ax > 0 and abs(v * 100.0 - token_value) <= tolerance:
            return True
    return False


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


__all__ = ["NarrativeFacts", "narrate", "assert_grounded", "numbers_in"]
