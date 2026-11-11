"""
advisor/rules_decompose.py — S4: what the decomposition means for the model stage.

A decomposition produces pictures. This file turns them into decisions, and the decisions are
narrow on purpose: a decomposition does not choose a model, it tells you which *terms* the model
needs and whether the obvious choice of model is wrong.

The rules that matter most:

* `dec.seasonal.needs_season` — the seasonal component scores above the book's 0.64 threshold,
  so a model without a seasonal term is wrong, not merely suboptimal.
* `dec.remainder.not_noise` — if the remainder is not white, the decomposition did its job and
  the *model* still has work to do. This is the single most common beginner misreading
  ("the remainder is small so the series is solved"), and the rule says the opposite.
* `dec.trend.no_separate_term` — when one method's trend absorbs everything and another's does
  not, the honest advice is that the trend is a choice; a model with a trend term should be
  compared against one without, not chosen on the strength of one decomposition.
* `dec.mult.not_possible` — the guardrail, surfaced as advice rather than only as an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from app.core.formatting import p_phrase

from app.core.schemas import Confidence

#: Hyndman's threshold for a component worth modelling.
STRONG = 0.64
WEAK = 0.30


@dataclass
class DecompContext:
    """All a decomposition rule may know: numbers from S4, nothing raw."""

    series_name: str = "your series"
    n: int = 0
    period: int | None = None
    method: str = "stl"
    model: str = "additive"

    trend_strength: float | None = None
    seasonal_strength: float | None = None
    trend_share: float | None = None
    seasonal_share: float | None = None
    remainder_share: float | None = None

    trend_p: float | None = None
    trend_slope: float | None = None
    seasonal_kw_p: float | None = None
    seasonal_amplitude_pct: float | None = None
    seasonal_growing: bool = False

    remainder_lb_p: float | None = None
    remainder_arch_p: float | None = None
    remainder_jb_p: float | None = None

    cycle_p: float | None = None
    cycle_period: float | None = None
    cycle_on_calendar: bool = False

    nested_seasonality: bool = False
    multiplicative_blocked: bool = False
    n_nonpositive: int = 0
    methods_compared: int = 0
    trend_spread: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class DecompRule:
    id: str
    stage: str
    recommend: str
    when: Callable[[DecompContext], bool]
    because: Callable[[DecompContext], str]
    confidence: Confidence = Confidence.MEDIUM
    provenance: list[str] = field(default_factory=list)
    alternative: str | None = None
    risk_if_ignored: str = ""
    how_to: str = ""
    kind: str = "recommend"          # recommend | blocker | note

    def applies(self, c: DecompContext) -> bool:
        try:
            return bool(self.when(c))
        except Exception:                                # pragma: no cover
            return False

    def to_out(self, c: DecompContext) -> dict[str, Any]:
        return {
            "id": self.id, "stage": self.stage, "recommend": self.recommend,
            "because": self.because(c), "confidence": self.confidence.value,
            "provenance": list(self.provenance), "alternative": self.alternative,
            "risk_if_ignored": self.risk_if_ignored, "how_to": self.how_to, "kind": self.kind,
        }


def _num(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "not computed"
    return f"{x:.{digits}f}"


def _pct(x: float | None) -> str:
    """
    A share as a percentage, without rounding a real share away.

    "0% of the variance" was being printed for a remainder that held 0.3% — small, but the whole
    point of the sentence is to say how small. Anything under 0.1% is shown as "<0.1%", and
    anything under 10% keeps one decimal.
    """
    if x is None:
        return "not computed"
    v = 100 * x
    if v == 0:
        return "0%"
    if abs(v) < 0.1:
        return "<0.1%"
    return f"{v:.0f}%" if abs(v) >= 10 else f"{v:.1f}%"


RULES: list[DecompRule] = [
    DecompRule(
        id="dec.seasonal.needs_season",
        stage="S4",
        recommend="Include a seasonal term in the model",
        when=lambda c: c.seasonal_strength is not None and c.seasonal_strength >= STRONG,
        because=lambda c: (
            f"The seasonal component explains F_S = {_num(c.seasonal_strength)} of the variation "
            f"around the remainder and swings {_num(c.seasonal_amplitude_pct, 1)}% of the series "
            f"mean each cycle. With F_S above the usual 0.64 threshold, a model without a "
            f"period-{c.period} term is not merely worse — it is misspecified."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.seasonality"],
        alternative="If the pattern is drifting rather than fixed, use a model whose seasonal "
                    "term is allowed to evolve (ETS with a damped seasonal, or TBATS).",
        risk_if_ignored="Residuals keep a seasonal shape, so prediction intervals are too narrow "
                        "at the peaks and troughs of every cycle.",
        how_to="Step 8 → choose a model with a seasonal component and set the period.",
    ),
    DecompRule(
        id="dec.seasonal.weak_omit",
        stage="S4",
        kind="note",
        recommend="A seasonal term is optional here",
        when=lambda c: c.seasonal_strength is not None and c.seasonal_strength < WEAK,
        because=lambda c: (
            f"Seasonal strength is only F_S = {_num(c.seasonal_strength)}, below the {WEAK} "
            "threshold where a seasonal term earns its parameter. Adding one mostly adds "
            "estimation noise; leaving it out loses almost nothing."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.seasonality"],
        alternative="If the calendar matters for reasons outside the data (holidays, billing "
                    "cycles), keep the term anyway and say why.",
        risk_if_ignored="A spurious seasonal term inflates the parameter count and can absorb "
                        "trend, making both components harder to interpret.",
        how_to="Step 8 → compare a model with and without the seasonal term before deciding.",
    ),
    DecompRule(
        id="dec.seasonal.growing",
        stage="S4",
        recommend="Use a multiplicative form, or a model with an evolving seasonal pattern",
        when=lambda c: bool(c.seasonal_growing),
        because=lambda c: (
            f"The seasonal swing widens over the window — the amplitude at the end is larger than "
            f"at the start, and today it is {_num(c.seasonal_amplitude_pct, 1)}% of the mean "
            "level. A single fixed additive seasonal term will under-predict recent swings."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.seasonality"],
        alternative="A log transform makes a growing seasonal swing constant, which lets an "
                    "additive model work — but only if the series is positive.",
        risk_if_ignored="Forecasts systematically under-shoot the recent peaks and over-shoot the "
                        "recent troughs.",
        how_to="Step 2 → transform (log) before modelling, or Step 8 → multiplicative model.",
    ),
    DecompRule(
        id="dec.trend.include",
        stage="S4",
        recommend="Include a trend term",
        when=lambda c: (c.trend_strength is not None and c.trend_strength >= STRONG
                        and (c.trend_p is None or c.trend_p < 0.05)),
        because=lambda c: (
            f"The trend carries F_T = {_num(c.trend_strength)} of the variation "
            f"({_pct(c.trend_share)} of the total), and it is statistically real: Mann–Kendall "
            f"{p_phrase(c.trend_p)} on the seasonally adjusted series with Sen's slope "
            f"{_num(c.trend_slope, 4)} per period."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.trend"],
        alternative="A damped trend is safer when a straight line cannot continue — it slows the "
                    "forecast instead of projecting the slope forever.",
        risk_if_ignored="A model with no trend term forecasts a flat line through a rising series.",
        how_to="Step 8 → model specification: trend = additive (or damped).",
    ),
    DecompRule(
        id="dec.trend.weak_or_uncertain",
        stage="S4",
        kind="note",
        recommend="Treat the trend as optional",
        when=lambda c: (c.trend_strength is None or c.trend_strength < WEAK
                        or (c.trend_p is not None and c.trend_p >= 0.05)),
        because=lambda c: (
            f"Trend strength is F_T = {_num(c.trend_strength)}"
            + (f" and its Mann–Kendall {p_phrase(c.trend_p)}" if c.trend_p is not None else "")
            + ". Neither is strong enough to justify a trend parameter on evidence alone — fit "
            "both with and without one and compare on held-out error."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.trend"],
        alternative="Keep the trend term if the domain says the level will keep drifting "
                    "(prices, populations), even when this window cannot prove it.",
        risk_if_ignored="A trend term fitted to noise extrapolates that noise, which is how "
                        "forecasts end up outside any plausible range.",
        how_to="Step 13 → compare the two specifications side by side.",
    ),
    DecompRule(
        id="dec.remainder.not_noise",
        stage="S4",
        recommend="Do not read a small remainder as 'the series is solved'",
        when=lambda c: c.remainder_lb_p is not None and c.remainder_lb_p < 0.05,
        because=lambda c: (
            f"The remainder holds only {_pct(c.remainder_share)} of the variance, which looks "
            f"reassuring — but Ljung–Box on it gives {p_phrase(c.remainder_lb_p)}, so it still "
            "has serial dependence. The decomposition removed the trend and season; it did not "
            "remove the predictability, and that predictability is exactly what the model has to "
            "capture."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.remainder"],
        alternative="If the remaining dependence is at the seasonal lag, a seasonal ARIMA term "
                    "handles it; if it is short-run, AR/MA terms do.",
        risk_if_ignored="Treating an autocorrelated remainder as noise makes a misspecified model "
                        "look adequate and its intervals look trustworthy.",
        how_to="Step 8 → add AR/MA terms; Step 13 → check the residual ACF after fitting.",
    ),
    DecompRule(
        id="dec.remainder.arch",
        stage="S4",
        recommend="Model the variance, not just the level",
        when=lambda c: c.remainder_arch_p is not None and c.remainder_arch_p < 0.05,
        because=lambda c: (
            f"ARCH-LM on the remainder gives {p_phrase(c.remainder_arch_p)}: the size of the "
            "misses changes over time. That is structure — volatility clustering — and a "
            "constant-variance model will produce intervals that are too wide in calm periods and "
            "too narrow in turbulent ones."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.remainder"],
        alternative="If the clustering is caused by a few known events, an intervention dummy for "
                    "those dates may be enough.",
        risk_if_ignored="Interval coverage is wrong exactly when it matters most — during volatile "
                        "stretches.",
        how_to="Step 8 → volatility models (GARCH family), or Step 2 → stabilise the variance.",
    ),
    DecompRule(
        id="dec.cycle.not_season",
        stage="S4",
        kind="note",
        recommend="Do not model a cycle with a seasonal term",
        when=lambda c: (c.cycle_p is not None and c.cycle_p < 0.01
                        and not c.cycle_on_calendar and c.cycle_period is not None),
        because=lambda c: (
            f"There is a real periodicity of about {_num(c.cycle_period, 1)} observations "
            f"(Fisher's g {p_phrase(c.cycle_p)}), which is not the calendar period "
            f"({c.period if c.period else 'none'}). A cycle has no fixed position in the year: a "
            "seasonal term would force it to land on the same dates every time."
        ),
        confidence=Confidence.LOW,
        provenance=["stat.cycle"],
        alternative="A longer-memory model (ARIMA with a high-order AR term), a band-pass filtered "
                    "component, or a structural model with a cycle component.",
        risk_if_ignored="A seasonal term fitted to a non-seasonal cycle produces forecasts that "
                        "repeat a shape the series does not repeat.",
        how_to="Step 8 → leave the seasonal period at the calendar value; do not 'round' the peak.",
    ),
    DecompRule(
        id="dec.mult.not_possible",
        stage="S4",
        kind="blocker",
        recommend="Use additive decomposition, or make the series positive first",
        when=lambda c: bool(c.multiplicative_blocked),
        because=lambda c: (
            f"A multiplicative decomposition divides by the level, so it is undefined wherever "
            f"the series is zero or negative — and {c.n_nonpositive} of the {c.n} observations "
            "are. This is a definition, not a preference: the app will not substitute additive "
            "silently, because that would answer a different question than the one asked."
        ),
        confidence=Confidence.HIGH,
        provenance=["fig.decomposition"],
        alternative="A log transform (positive data) or a shifted log turns a multiplicative "
                    "pattern into an additive one, exactly.",
        risk_if_ignored="Silently switching to additive makes a growing seasonal swing look like a "
                        "constant one, and the seasonal index is then wrong in the units the user "
                        "reads.",
        how_to="Step 2 → transform: log (or log(y + c)); then re-run this step.",
    ),
    DecompRule(
        id="dec.method.disagree",
        stage="S4",
        kind="note",
        recommend="Check the method menu before quoting the trend",
        when=lambda c: bool(c.nested_seasonality),
        because=lambda c: (
            "This method's trend absorbs part of what another method calls seasonal"
            + (f", and the {c.methods_compared} methods compared so far disagree by up to "
               f"{_num(c.trend_spread, 3)} at a single date"
               if c.methods_compared and c.trend_spread is not None else "")
            + ". Run the same series under STL and under classical and compare: where the two "
              "trends differ, the difference is the method, not the data."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["table.decomp_methods"],
        alternative="STL is the default for a reason: it keeps the ends and tolerates outliers. "
                    "Classical is kept because it is what textbooks show.",
        risk_if_ignored="Quoting one method's trend as if it were the series' trend is the easiest "
                        "way to be confidently wrong in this step.",
        how_to="Step 4 → 'Compare methods' — the trends are overlaid in one chart.",
    ),
]

UNCONDITIONAL: list[str] = [
    "A decomposition is a description, not a model: it has no likelihood and cannot be forecast.",
    "The remainder holding a small share does not mean the series is explained; check whether it is white.",
    "Changing the decomposition method changes the trend you see. That is a property of decompositions, not a bug.",
]

_FILE_ORDER = {r.id: i for i, r in enumerate(RULES)}


def rank(c: DecompContext) -> list[DecompRule]:
    hits = [r for r in RULES if r.applies(c)]
    conf_rank = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
    hits.sort(key=lambda r: (
        0 if r.kind == "blocker" else (2 if r.kind == "note" else 1),
        conf_rank.get(r.confidence, 3),
        _FILE_ORDER.get(r.id, 99),
    ))
    return hits


__all__ = ["DecompContext", "DecompRule", "RULES", "UNCONDITIONAL", "rank", "STRONG", "WEAK"]
