"""
advisor/rules_next.py — S7: the "what next" advisor (§5.6).

This is the last stage before modelling, and the one where the app can do the most damage or the
most good. The classic disaster it exists to prevent is "differenced it three times": each
difference *looks* like progress on the ACF, so nothing in the loop tells the user to stop.

Two design decisions do the work:

* **Every action is ranked against "leave it alone".** `next.none` is a real recommendation with a
  real `because`, not the absence of one. When the correlogram is white and the tests agree, the
  honest top-ranked action is to change nothing, and the app says so first.

* **Every action carries a mandatory preview.** The API can run each candidate on a copy of the
  series and report what the ACF and the tests would look like — before the user commits. A rule
  whose preview makes things *worse* is demoted and says so in its `because`, because that is the
  evidence the user needs and cannot get from a textbook.

Rule ids are stable — they are cited in reports and in the results' `provenance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from app.core.formatting import p_phrase

from app.core.schemas import Confidence


@dataclass
class NextContext:
    """Every number an S7 rule is allowed to know. All computed upstream."""

    # identity
    series_name: str = "your series"
    n: int = 0
    period: int | None = None

    # S6 — the correlogram
    r1: float | None = None
    max_lag: int = 0
    n_significant: int = 0
    decay_lags: int = 0
    pattern: str = "unclear"
    ljung_box_p: float | None = None
    ljung_box_p_seasonal: float | None = None
    seasonal_acf_significant: bool = False
    variance_ratio: float | None = None
    variance_ratio_p: float | None = None

    # S5 — the verdict the hints must agree with
    stationarity_verdict: str = "inconclusive"
    differencing_needed: bool = False
    seasonal_differencing_needed: bool = False
    n_unit_root_tests: int = 0

    # S4 — the decomposition
    trend_share: float | None = None
    seasonal_share: float | None = None
    seasonal_strength: float | None = None
    seasonal_growing: bool = False
    trend_band: str = "none"

    # S2/S3 — what has already been done
    already_logged: bool = False
    already_differenced: bool = False
    already_seasonally_differenced: bool = False
    n_transforms_applied: int = 0
    variance_stable: bool | None = None

    # preview results keyed by rule id (filled by the API after running the candidate)
    previews: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "previews"}


@dataclass
class NextRule:
    id: str
    stage: str
    recommend: str
    when: Callable[[NextContext], bool]
    because: Callable[[NextContext], str]
    confidence: Confidence = Confidence.MEDIUM
    provenance: list[str] = field(default_factory=list)
    alternative: str | None = None
    risk_if_ignored: str = ""
    how_to: str = ""
    kind: str = "recommend"                            # recommend | blocker | note | protect
    #: Optional override for confidence given the context; consumed by `confidence_for`.
    confidence_when: Callable[[NextContext], Confidence] | None = None
    #: The preprocessing op this rule proposes, if it proposes one. `None` = no change.
    op: str | None = None
    op_params: dict[str, Any] = field(default_factory=dict)
    #: Higher ranks first when several rules fire. Leave-alone carries a deliberately high base.
    weight: float = 0.0

    def applies(self, c: NextContext) -> bool:
        try:
            return bool(self.when(c))
        except Exception:                              # pragma: no cover - a rule must never crash
            return False

    def confidence_for(self, c: NextContext) -> Confidence:
        """Rule-specific confidence. Default is the static one."""
        if self.confidence_when is None:
            return self.confidence
        try:
            return self.confidence_when(c)
        except Exception:                              # pragma: no cover
            return self.confidence

    def to_out(self, c: NextContext) -> dict[str, Any]:
        preview = c.previews.get(self.id)
        because = self.because(c)
        if preview is not None and preview.get("verdict") == "worse":
            because += (
                " **The preview says this makes things worse**: "
                + str(preview.get("plain") or "the ACF gets worse and the tests do not improve.")
            )
        return {
            "id": self.id,
            "stage": self.stage,
            "recommend": self.recommend,
            "because": because,
            "confidence": self.confidence_for(c).value,
            "provenance": list(self.provenance),
            "alternative": self.alternative,
            "risk_if_ignored": self.risk_if_ignored,
            "how_to": self.how_to,
            "kind": self.kind,
            "op": self.op,
            "op_params": dict(self.op_params),
            "preview": preview,
            "rank": self.weight + _preview_weight(preview),
        }


def _preview_weight(preview: dict[str, Any] | None) -> float:
    """
    Nudge the ranking by what the preview actually showed.

    This is the mechanism that stops a textbook-correct rule from being applied to data it does not
    fit: the rule still appears, with its reasoning, but it no longer outranks the alternative.
    """
    if not preview:
        return 0.0
    return {"better": 0.6, "neutral": 0.0, "worse": -1.2}.get(str(preview.get("verdict")), 0.0)


def _num(x: float | None, digits: int = 3) -> str:
    """Citable numbers carry no thousands separators (R0.5)."""
    if x is None:
        return "not computed"
    return f"{x:.{digits}f}"


def _pct(x: float | None) -> str:
    if x is None:
        return "not computed"
    v = 100 * x
    if v == 0:
        return "0%"
    if abs(v) < 0.1:
        return "<0.1%"
    return f"{v:.0f}%" if abs(v) >= 10 else f"{v:.1f}%"


S = "S7"


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
RULES: list[NextRule] = [

    # ---- 1. leave it alone ------------------------------------------------ #
    NextRule(
        id="next.none.already_adequate",
        stage=S,
        recommend="Nothing needs changing — go to modelling as it stands",
        when=lambda c: (
            c.pattern == "white_noise"
            and not c.differencing_needed
            and not c.seasonal_differencing_needed
            and (c.ljung_box_p is None or c.ljung_box_p > 0.05)
        ),
        because=lambda c: (
            f"The autocorrelation leaves its band at {c.n_significant} of the first {c.max_lag} lags "
            f"and Ljung–Box does not reject ({p_phrase(c.ljung_box_p)}), so there is no "
            "dependence left to model with AR or MA terms. Every transformation from here adds "
            "estimation noise and takes observations away, so the correct action is none."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.ljung_box", "stat.acf"],
        alternative="Fit the white-noise baseline and let a more complex model earn its place out "
                    "of sample.",
        risk_if_ignored="Differencing a white-noise series creates an MA(1) with coefficient −1, "
                        "wider prediction intervals and worse forecasts.",
        how_to="Step 7 → Readiness gate",
        kind="recommend",
        op=None,
        weight=2.2,
    ),

    # ---- 2. the guardrail: stationarity says do not difference ------------ #
    NextRule(
        id="next.diff.guardrail",
        stage=S,
        recommend="Do not difference this series — the tests say it is already stationary",
        when=lambda c: (
            c.stationarity_verdict in {"stationary", "stationary_around_break"}
            and not c.differencing_needed
            and (c.pattern == "nonstationary_decay" or (c.r1 is not None and c.r1 > 0.7))
        ),
        because=lambda c: (
            f"The correlogram looks persistent (r₁ = {_num(c.r1)}), which would normally argue for a "
            f"difference — but Step 5's tests say {c.stationarity_verdict.replace('_', ' ')}. "
            "Persistence is not the same as a unit root: a trend, a break or long memory all produce "
            "a slowly decaying ACF, and only the last of those is fixed by differencing."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.acf", "stat.adf", "stat.kpss"],
        alternative="If the series has a trend, model the trend (or detrend) instead of differencing.",
        risk_if_ignored="An unnecessary difference introduces a moving-average unit root, inflates "
                        "interval width and destroys the level information a forecast needs.",
        how_to="Step 5 → read the verdict; Step 7 → fit with d = 0",
        kind="protect",
        op=None,
        weight=2.4,
    ),

    # ---- 3. first difference ---------------------------------------------- #
    NextRule(
        id="next.diff.first",
        stage=S,
        recommend="Take a first difference (d = 1)",
        when=lambda c: (
            c.differencing_needed
            and not c.already_differenced
            # A *seasonal* unit root with no regular one is not fixed by diff(1): Step 5's own text
            # says a plain difference just moves the problem into the seasonal lags. Firing here
            # anyway would put the app in the position of contradicting its own verdict.
            and not c.seasonal_differencing_needed
        ),
        because=lambda c: (
            f"A unit root was not rejected in Step 5"
            + (f" ({c.n_unit_root_tests} tests ran)" if c.n_unit_root_tests else "")
            + f", and the ACF decays slowly (r₁ = {_num(c.r1)}, still outside its band at lag "
            f"{c.decay_lags}). Differencing once is the operation that removes a unit root."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.adf", "stat.kpss", "stat.acf"],
        alternative="Detrend instead if Step 4 found a strong trend and Step 5 blamed a break, not a "
                    "unit root.",
        risk_if_ignored="ARIMA fitted to a non-stationary series gives spuriously good in-sample fits "
                        "and wide, miscalibrated intervals.",
        how_to="Step 7 → Preview → Apply (d = 1)",
        kind="recommend",
        op="difference",
        op_params={"order": 1},
        weight=1.6,
    ),

]


def _seasonal_rule() -> NextRule:
    return NextRule(
        id="next.diff.seasonal",
        stage=S,
        recommend="Take a seasonal difference at the detected period",
        when=lambda c: bool(c.seasonal_differencing_needed and c.period and c.period >= 2),
        because=lambda c: (
            f"Step 5's seasonal unit-root test (OCSB) did not reject, and the ACF spikes every "
            f"{c.period} observations (lag {c.period} is outside its band). A first difference does "
            f"not remove a seasonal unit root — the difference has to be at lag {c.period}."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.ocsb", "stat.acf"],
        alternative="Use a seasonal model term (P or Q at the seasonal lag) instead of "
                    "differencing, if the seasonal pattern is stable.",
        risk_if_ignored="Differencing at lag 1 leaves the seasonal unit root in place, so the model "
                        "will need many seasonal terms to compensate.",
        how_to="Step 7 → Preview seasonal difference → Apply",
        kind="recommend",
        op="difference",
        op_params={"order": "seasonal"},
        weight=1.5,
    )


# ---- 4b. the seasonal unit root, on its own ------------------------------- #
RULES.append(NextRule(
    id="next.diff.seasonal_only",
    stage=S,
    recommend="Take a seasonal difference at the detected period — not a first difference",
    # Fires whenever a seasonal unit root is present — including when a regular one is too. In that
    # combined case the seasonal difference is still the *first* one to take, and the separate
    # `next.note.both_differences` rule supplies the ordering.
    when=lambda c: bool(
        c.seasonal_differencing_needed
        and c.period and c.period >= 2 and not c.already_seasonally_differenced
    ),
    because=lambda c: (
        f"Step 5 found a seasonal unit root: the pattern does not return to a fixed seasonal shape "
        f"after a shock. A first difference does not remove that — it moves the problem into the "
        f"seasonal lags — so the difference has to be at lag {c.period}."
        + (" A regular unit root is present too, so the order matters: seasonal first, then regular."
           if c.differencing_needed else
           " The ACF supports it: lag " + str(c.period) + " is still outside its band.")
    ),
    confidence=Confidence.HIGH,
    provenance=["stat.ocsb", "stat.acf"],
    alternative="A model with a seasonal difference built in (D = 1 at the detected lag) achieves "
                "the same thing without preprocessing.",
    risk_if_ignored=f"Differencing at lag 1 leaves the seasonal unit root in place, so the model "
                    f"needs many seasonal terms to compensate and the forecast keeps the drift.",
    how_to="Step 7 → Preview seasonal difference → Apply",
    kind="recommend",
    op="difference",
    op_params={"order": "seasonal"},
    weight=2.3,
))


# ---- 5. log / variance stabilisation -------------------------------------- #
RULES.append(NextRule(
    id="next.transform.log",
    stage=S,
    recommend="Take logs before modelling (variance stabilisation)",
    when=lambda c: (
        not c.already_logged
        and (
            (c.seasonal_growing and (c.seasonal_share or 0) > 0.05)
            or (c.variance_stable is False and (c.trend_share or 0) > 0.3)
        )
    ),
    because=lambda c: (
        "The spread grows with the level: the seasonal swing is widening"
        + (f" (seasonal share {_pct(c.seasonal_share)} of the variation"
           + (", and Step 2 flagged the variance as unstable" if c.variance_stable is False else "")
           + ")")
        + ". A log turns a multiplicative pattern into an additive one, which is what ARIMA, ETS "
          "and the decomposition all assume."
    ),
    confidence=Confidence.MEDIUM,
    provenance=["eda.variance_stability", "dec.seasonal.growing"],
    alternative="A Box–Cox transform with an estimated λ, if a log over-corrects (λ far from 0).",
    risk_if_ignored="An ARIMA model on a series whose variance grows will under-predict the early "
                    "spread and over-predict the late one; interval coverage collapses at one end.",
    how_to="Step 2 → Transform → Log, then re-run Steps 4–6",
    kind="recommend",
    op="transform.log",
    op_params={},
    weight=1.3,
))

# ---- 6. over-differencing repair ------------------------------------------ #
RULES.append(NextRule(
    id="next.diff.undo_overdifferenced",
    stage=S,
    recommend="Undo the last difference — the series looks over-differenced",
    when=lambda c: (
        c.pattern == "overdifferenced"
        or (c.r1 is not None and c.r1 < -0.5)
        or (c.variance_ratio is not None and c.variance_ratio < 0.8
            and (c.already_differenced or c.already_seasonally_differenced))
    ),
    because=lambda c: (
        f"Lag 1 is {_num(c.r1)} — negative enough to be the fingerprint of differencing a series "
        f"that did not need it"
        + (f", and the variance ratio is {_num(c.variance_ratio)}, meaning the differences are "
           "negatively autocorrelated" if c.variance_ratio is not None else "")
        + (". A difference has already been applied, so this is repairable by undoing it."
           if (c.already_differenced or c.already_seasonally_differenced) else
           ". No difference has been applied yet, so this may simply be an MA(1) process.")
    ),
    # Confidence depends on whether a difference was actually applied — that is the tie-breaker
    # between "genuine MA(1)" and "over-differenced", and only the context knows it.
    confidence=Confidence.MEDIUM,
    confidence_when=lambda c: (
        Confidence.HIGH if (c.already_differenced or c.already_seasonally_differenced)
        else Confidence.LOW
    ),
    provenance=["stat.acf", "stat.variance_ratio"],
    alternative="If a difference is genuinely needed, fit ARIMA(0,d,1) and let the negative MA "
                "coefficient absorb the alternation.",
    risk_if_ignored="Over-differenced series produce forecasts that oscillate and intervals that are "
                    "far too wide; the level is lost entirely.",
    how_to="Step 2 → Undo, then re-run Steps 4–6",
    kind="protect",
    op="undo",
    op_params={},
    weight=2.0,
))

# ---- 7. seasonal terms instead of a seasonal difference ------------------- #
RULES.append(NextRule(
    id="next.season.model_terms",
    stage=S,
    recommend="Handle the seasonality with model terms, not a difference",
    when=lambda c: (
        c.seasonal_acf_significant
        and not c.seasonal_differencing_needed
        and (c.seasonal_strength is None or c.seasonal_strength < 0.64)
    ),
    because=lambda c: (
        f"The ACF spikes at lag {c.period}, but the seasonal pattern is not strong enough to need "
        f"its own difference"
        + (f" (F_S = {_num(c.seasonal_strength, 2)})" if c.seasonal_strength is not None else "")
        + ". Seasonal AR or MA terms capture a stable pattern with fewer parameters and without the "
          "risk of a seasonal difference overshooting."
    ),
    confidence=Confidence.MEDIUM,
    provenance=["stat.acf", "stat.seasonal_strength"],
    alternative="A seasonal difference at the detected lag if Step 5's seasonal unit-root test "
                "rejects and the pattern keeps changing.",
    risk_if_ignored="A seasonal difference on a weakly seasonal series removes real signal and makes "
                    "the forecast flat.",
    how_to="Step 9 → include seasonal AR/MA terms",
    kind="recommend",
    op=None,
    weight=0.9,
))

# ---- 8. the "you have already transformed this" note --------------------- #
RULES.append(NextRule(
    id="next.note.transform_chain",
    stage=S,
    recommend="Stop adding transformations — compare what you have",
    when=lambda c: c.n_transforms_applied >= 2,
    because=lambda c: (
        f"{c.n_transforms_applied} transformations have already been applied to this series. Each "
        "one is reversible on its own, but the chain is no longer something any of the tests in "
        "Steps 4–6 were computed on, so their verdicts describe an earlier version of the data."
    ),
    confidence=Confidence.HIGH,
    provenance=["preprocessing.records"],
    alternative="Undo back to the version whose diagnostics you trust, and fit there.",
    risk_if_ignored="Diagnostics that describe a different series than the one being modelled are "
                    "worse than no diagnostics: they look authoritative and are wrong.",
    how_to="Step 2 → Undo, or Step 7 → Re-run diagnostics",
    kind="note",
    op=None,
    weight=1.1,
))

# ---- 9. detrend instead of difference ----------------------------------- #
RULES.append(NextRule(
    id="next.trend.detrend",
    stage=S,
    recommend="Remove a deterministic trend instead of differencing",
    when=lambda c: (
        (c.trend_share or 0) > 0.6
        and c.trend_band in {"strong", "moderate"}
        and not c.differencing_needed
        and not c.already_differenced
    ),
    because=lambda c: (
        f"The trend is {_pct(c.trend_share)} of the variation and scores {c.trend_band}, while "
        "Step 5 found no unit root. A trend like that is a deterministic drift, so subtracting it "
        "keeps the level information that differencing throws away."
    ),
    confidence=Confidence.MEDIUM,
    provenance=["dec.trend.include", "stat.adf"],
    alternative="Hold the trend out of sample; if it stops, a difference would have been safer.",
    risk_if_ignored="Extrapolating a fitted trend outside the observed range is the main way "
                    "forecasts end up implausibly high or low.",
    how_to="Step 2 → detrend (or fit with a trend term in Step 9)",
    kind="recommend",
    op=None,
    weight=0.7,
))

# ---- 10. the seasonal unit root needs a seasonal difference ------------- #
RULES.append(NextRule(
    id="next.note.both_differences",
    stage=S,
    recommend="Both differences are needed, and the order matters",
    when=lambda c: bool(c.differencing_needed and c.seasonal_differencing_needed
                        and not c.already_differenced and not c.already_seasonally_differenced),
    because=lambda c: (
        "Step 5 found both a regular and a seasonal unit root. The conventional order is the "
        f"seasonal difference first (lag {c.period}), then a regular difference — the other order "
        "leaves a seasonal pattern in the residuals that the ACF will show as a spike at "
        f"lag {c.period}."
    ),
    confidence=Confidence.MEDIUM,
    provenance=["stat.adf", "stat.ocsb"],
    alternative="A seasonal model with both D = 1 and d = 1 terms achieves the same thing inside "
                "the model.",
    risk_if_ignored="Two differences applied in the wrong order can leave a seasonal unit root "
                    "and make the model appear to need far more terms than it does.",
    how_to="Step 7 → Preview both orders before applying either",
    kind="note",
    op=None,
    weight=0.8,
))


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def rank(ctx: NextContext) -> list[dict[str, Any]]:
    """
    Every applicable rule, best first.

    Ties are broken by rule id so a re-run produces the same order — a ranked list that shuffles
    between runs is not evidence of anything.
    """
    out = [r.to_out(ctx) for r in RULES if r.applies(ctx)]
    out.sort(key=lambda d: (-d["rank"], d["id"]))
    return out


def headline(ranked: list[dict[str, Any]], ctx: NextContext) -> str:
    """The one sentence above the list. It is the top-ranked recommendation, named."""
    if not ranked:
        return (
            "Nothing in the diagnostics forces a change. The safest next step is to fit a simple "
            "model and let the out-of-sample comparison decide whether more is needed."
        )
    top = ranked[0]
    head = str(top["recommend"])
    if top["id"] == "next.none.already_adequate":
        return (
            f"No transformation is needed: the correlogram of {ctx.series_name} is white-noise-like "
            f"and Ljung–Box does not reject ({p_phrase(ctx.ljung_box_p)}). Go to modelling."
        )
    if top["id"] == "next.diff.guardrail":
        return (
            f"Resist the temptation to difference: the ACF looks persistent (r₁ = {_num(ctx.r1)}), "
            f"but Step 5 says {ctx.stationarity_verdict.replace('_', ' ')}."
        )
    return f"Next step: {head[0].lower() + head[1:] if head else head}."


__all__ = ["NextContext", "NextRule", "RULES", "rank", "headline"]
