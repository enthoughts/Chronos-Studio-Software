"""
ts/models/suggest.py — the suggestion engine (§5.8).

The whole engine is: build facts, ask each registry row whether it applies, rank the compatible ones,
and say why. There is no per-model branch here, and `test_adding_a_row_offers_a_model` proves it by
registering a model at runtime and watching it appear with the right reasons.

Ranking is deliberately not a score out of 100. A score invites the reader to compare 74 with 81 when
the difference is one branch of an `if`. Instead each suggestion carries a **reason it is suggested**
and a **condition under which it would win**, both written from the measured facts, so the user can
disagree with the app on the merits.

Two things the engine refuses to do:

* offer a model whose assumptions the data visibly violates without saying so;
* hide the models it rejected. The excluded list is part of the output, with the number that
  excluded each one — "why can't I use Prophet?" is a question a user will ask out loud.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.formatting import p_phrase
from app.core.schemas import (
    DataClass,
    ModelFamily,
    ModelSpec,
    ModelSuggestion,
    ModelSuggestionSet,
)
from app.ts.models.facts import DataFacts
from app.ts.models.registry import (
    BASELINE_IDS,
    all_specs,
    evaluate_spec,
    is_implemented,
)

#: The families whose members are alternatives to one another. Within a family, the engine keeps the
#: best-fitting few; across families it keeps the best of each, because the point of a candidate set
#: is that the families disagree and the data chooses.
FAMILY_ORDER: tuple[ModelFamily, ...] = (
    ModelFamily.BASELINE,
    ModelFamily.SMOOTHING,
    ModelFamily.ARMA,
    ModelFamily.COUNT,
    ModelFamily.INTERMITTENT,
    ModelFamily.STRUCTURAL,
    ModelFamily.EXOGENOUS,
    ModelFamily.VOLATILITY,
    ModelFamily.MULTIVARIATE,
    ModelFamily.NONLINEAR,
    ModelFamily.POINT_PROCESS,
    ModelFamily.ML,
    ModelFamily.COMBINATION,
)


def _fmt(x: float | None, dp: int = 2) -> str:
    return "—" if x is None else f"{x:.{dp}f}"


@dataclass
class _Scored:
    spec: ModelSpec
    rank_key: tuple
    because: str
    would_win_if: str
    warnings: list[str]
    order_hint: str | None
    expected_seconds: float | None


def _because(spec: ModelSpec, f: DataFacts, warnings: list[str]) -> str:
    """
    One sentence, built from the facts, saying why *this* model is on the list for *this* series.

    Every branch cites a measurement. A sentence that would read identically for any dataset is not a
    reason, it is a description, and the UI already shows the description.
    """
    mid = spec.model_id
    season = (
        f"seasonal strength {_fmt(f.seasonal_strength)} at period {f.period}"
        if f.period and f.seasonal_strength is not None
        else ("a detected period of " + str(f.period) if f.period else "no detected seasonality")
    )

    if mid == "mean":
        return (
            f"The floor: {f.n_non_missing} observations with a mean of {_fmt(f.mean)}. "
            "Any model that cannot beat this has explained nothing."
        )
    if mid == "naive":
        return (
            f"Lag-1 autocorrelation is {_fmt(f.autocorr_lag1)}"
            + (", so the last value carries real information" if (f.autocorr_lag1 or 0) > 0.3 else "")
            + ". Beating the last value is the minimum bar for every model below."
        )
    if mid == "seasonal_naive":
        return (
            f"A period of {f.period} was detected, so 'the same season last year' is a benchmark the "
            "user can check by eye."
        )
    if mid == "drift":
        return (
            f"The series moves from {_fmt(f.min_value, 1)} to {_fmt(f.max_value, 1)}; a single "
            "straight-line slope is the cheapest way to extrapolate that, so it sets the bar for "
            "'found a trend'."
        )
    if mid == "white_noise":
        return (
            "The null hypothesis with a forecast attached. It wins outright when the residual "
            "autocorrelation battery finds nothing — and then no other model should be used."
        )
    if mid in {"arima"}:
        return (
            f"d is set to {f.d_hint} by the stationarity verdict"
            + (f" ({f.stationarity_verdict.replace('_', ' ')})." if f.stationarity_verdict else ".")
            + f" Autocorrelation at lag 1 is {_fmt(f.autocorr_lag1)}, which is what an ARMA structure "
            "is for."
        )
    if mid == "sarima":
        return (
            f"{season.capitalize()}"
            + (f" and D is set to {f.D_hint} by the seasonal unit-root test" if f.D_hint else "")
            + f"; {f.n_non_missing} observations give {f.n_non_missing / f.period:.1f} cycles, which is "
            "enough to estimate seasonal terms."
        )
    if mid == "arfima":
        return (
            f"Autocorrelation at lag 1 is {_fmt(f.autocorr_lag1)} and the series is long enough "
            f"({f.n_non_missing}) for the GPH estimator to say whether the memory is fractional "
            "rather than a full unit root."
        )
    if mid == "ets":
        return (
            f"{season.capitalize()}"
            + (f", trend R² {_fmt(f.trend_r2)}" if f.trend_r2 is not None else "")
            + ". Exponential smoothing needs no order search, so it is the safest strong model on a "
            "sample this size."
        )
    if mid == "theta":
        return (
            "Theta wins disproportionately on short, strongly seasonal series — it was the best "
            f"method in the M3 and M4 competitions. Here n = {f.n_non_missing}"
            + (f" with a period of {f.period}." if f.period else " with no season to adjust for.")
        )
    if mid == "poisson":
        return (
            f"Count data with variance {_fmt(f.dispersion_ratio)}× the mean"
            + (", which is close enough to Poisson for the rate model to apply."
               if (f.dispersion_ratio or 0) <= 1.5 else
               " — overdispersed, so read the warning before choosing this one.")
        )
    if mid == "neg_binomial":
        return (
            f"Count data with variance {_fmt(f.dispersion_ratio)}× the mean: the extra dispersion "
            "parameter is exactly what a Poisson cannot represent."
        )
    if mid == "zero_inflated":
        return (
            f"{f.n_zeros} zeros ({f.pct_zeros * 100:.0f}% of the series) — "
            + (f"{_fmt(f.zero_excess)}× what a plain count model would produce, so the zeros look "
               "structural." if (f.zero_excess or 0) >= 2 else
               "close to what a plain count model would produce, so the hurdle is unlikely to earn "
               "its extra parameters.")
        )
    if mid == "croston":
        nonzero = f.n - f.n_zeros
        return (
            f"{f.pct_zeros * 100:.0f}% of values are zero and only {nonzero} are not, so a "
            "level-and-trend model would forecast phantom demand in every empty period."
        )
    if mid == "tsb":
        return (
            f"{f.pct_zeros * 100:.0f}% zeros, and TSB's occurrence probability falls when demand "
            "stops — the right shape if this part may be going obsolete."
        )
    if mid in {"ingarch", "inar"}:
        acf1 = _fmt(f.autocorr_lag1)
        disp = _fmt(f.dispersion_ratio)
        persistent = f.autocorr_lag1 is not None and abs(f.autocorr_lag1) >= 0.20
        # The sentence has to agree with the measured autocorrelation. Saying "the rate carries into
        # the next period" next to an acf of 0.06 would be prose the number contradicts.
        if mid == "ingarch":
            head = (
                f"Lag-1 autocorrelation of the counts is {acf1}"
                + (", so a busy period genuinely raises the next rate" if persistent else
                   ", so persistence is weak")
                + f"; the variance is {disp}× the mean."
            )
            return head + (
                " The recursion lets the rate carry between periods, which a Poisson regression with "
                "fixed seasonality cannot express."
            )
        interval = (
            f"{f.n_non_missing / max(f.n - f.n_zeros, 1):.1f}" if f.n > f.n_zeros else "—"
        )
        return (
            f"With lag-1 autocorrelation {acf1}, {f.n_zeros} zero periods and one non-zero "
            f"period every {interval}, the thinning model attributes persistence to a share of "
            "cases surviving one period — a quantity worth reporting even if the fit is no better."
        )
    if mid == "adida":
        cv = f.cv_of_sizes
        return (
            f"The {f.n - f.n_zeros} non-zero periods have a size CV of {_fmt(cv)}"
            + (
                ", so one smoothed average cannot describe them — aggregating into blocks first "
                "steadies the series that gets smoothed."
                if cv is not None and cv >= 0.5
                else (
                    ", which is modest: block aggregation is offered for completeness, but plain "
                    "Croston should be enough here."
                )
            )
        )
    if mid == "croston_sy":
        interval = f.n_non_missing / max(f.n - f.n_zeros, 1)
        above = interval > 1.32
        return (
            f"{f.pct_zeros * 100:.0f}% zeros give a mean demand interval of {interval:.2f} periods, "
            + (
                "above the 1.32 cut-off, so the Syntetos–Boylan correction applies here."
                if above
                else "below the 1.32 cut-off, so this variant deliberately skips the correction "
                "and should land on plain Croston's answer."
            )
        )
    if mid in {"arch", "garch", "garch_t", "gjr", "gjr_t", "egarch", "egarch_t", "arima_garch"}:
        p_resid = f.arch_lm_resid_p
        # `str.capitalize()` lowercases the rest of the string, which turned "ARCH-LM" into
        # "Arch-lm". Write the sentence already capitalised instead of fixing it up afterwards.
        measured = (
            f"ARCH-LM on the mean-adjusted series returns {p_phrase(p_resid, 3)}"
            if p_resid is not None
            else "The ARCH-LM test could not be run"
        )
        if mid == "arch":
            return (
                f"{measured}, so squared shocks carry information. ARCH(1) is the "
                "shortest-memory version and the one to beat."
            )
        if mid in {"garch", "garch_t"}:
            return (
                f"{measured}, so the variance has memory of its own; GARCH(1,1) is what "
                "turns that into a forecast of uncertainty rather than a forecast of the value."
                + (
                    " The t version is offered alongside because the tail is what a Gaussian variance "
                    "model gets wrong first."
                    if mid == "garch_t"
                    else ""
                )
            )
        if mid in {"gjr", "gjr_t"}:
            return (
                f"{measured}, and asymmetry is the next question: does a fall raise "
                "future variance more than a rise of the same size?"
            )
        if mid in {"egarch", "egarch_t"}:
            return (
                f"{measured}. EGARCH models the log variance, so unlike GARCH the fitted "
                "path cannot go negative and the leverage coefficient has a sign you can read."
            )
        return (
            f"{measured} and the mean of this series is not white noise, so an AR mean "
            "equation is estimated jointly with the variance equation — two forecasts from one fit."
        )
    if mid == "renewal":
        return (
            f"The input is {f.n_non_missing} event periods of dates rather than values, so the "
            "observable quantity is the gap between events — a renewal model reports that gap and "
            "whether the hazard rises or falls."
        )
    if mid == "hawkes":
        return (
            "Event dates arrive in bursts rather than at a constant rate, which is what a "
            "self-exciting kernel measures: the branching ratio is the share of activity that "
            "earlier events generated."
        )
    if not is_implemented(spec.model_id):
        return (
            f"Suggested because {spec.triggered_when.lower()} This build can recommend it but cannot "
            "fit it yet."
        )
    return spec.triggered_when or "Included by the registry."


def _volatility_would_win(mid: str, f: DataFacts) -> str:
    """
    The falsifiable condition for a variance model.

    The `_because` text says *why it is offered*; this says *what would have to be true for it to win*,
    and for a variance model that is almost never "a low AICc" — it is "the standardised residuals stop
    clustering" or "the leverage coefficient is materially non-zero". Both are checkable in the fit
    table this app prints.
    """
    if mid == "arch":
        return (
            "the clustering is short-lived, so a single lag of squared shocks is enough and the extra "
            "persistence parameters earn nothing."
        )
    if mid == "garch_t":
        return (
            "the standardised residuals from the Gaussian GARCH are fat-tailed — the fitted nu is low and "
            "the AICc is lower than plain GARCH's."
        )
    if mid in {"gjr", "gjr_t", "egarch", "egarch_t"}:
        return (
            "the asymmetry coefficient is materially away from zero in the direction the leverage story "
            "predicts, and the AICc beats the symmetric fit — a gamma that cannot be distinguished from "
            "zero means the simpler GARCH is the honest answer."
        )
    if mid == "arima_garch":
        return (
            "the mean equation's AR terms are significant, so the value path beats a constant mean on the "
            "holdout while the variance path stays informative."
        )
    return (
        "the standardised residuals stop showing clustering and the variance path dies out to the "
        "long-run level, which is what a correctly specified GARCH implies."
    )


def _would_win_if(spec: ModelSpec, f: DataFacts) -> str:
    """The falsifiable condition. If the user cannot check it, it is not a prediction, it is a hope."""
    mid = spec.model_id
    table = {
        "mean": "the series has no usable trend, season or autocorrelation — that is, every other "
                "model's in-sample fit is no better than the average.",
        "naive": "the series is a random walk: one difference leaves nothing predictable.",
        "seasonal_naive": "the seasonal pattern repeats exactly and the level does not move.",
        "drift": "the series moves in a straight line with no curvature and no season.",
        "white_noise": "the residual autocorrelation battery fails to reject at every lag — in which "
                       "case no model is needed and this one says so.",
        "arima": "the series is stationary after the chosen differencing and its autocorrelation dies "
                 "out within a few lags with no seasonal structure left.",
        "sarima": f"the seasonal pattern at period {f.period} survives differencing and the seasonal "
                  "AR/MA terms pick it up better than harmonic terms do.",
        "arfima": "the GPH estimate of d lands strictly between 0 and 0.5 with a standard error small "
                  "enough to exclude both endpoints.",
        "ets": "the series is smooth enough that a weighted average beats an explicit ARMA structure — "
               "usual when the trend is stable.",
        "theta": "the season is stable and the trend is roughly linear — Theta's two-line average "
                 "beats a fitted seasonal model on short samples.",
        "poisson": f"variance stays near the mean (currently {_fmt(f.dispersion_ratio)}×) and the "
                   "counts are small.",
        "neg_binomial": "the overdispersion is real (not one or two large counts) and persists in the "
                        "holdout — then its wider intervals will be the calibrated ones.",
        "zero_inflated": "the zeros are structural: a promotion with no sales at all is a different "
                         "event from a promotion with few sales.",
        "croston": "demand stays intermittent and the rate is roughly constant — Croston's constant-rate "
                   "forecast is then exactly right.",
        "tsb": "demand stops for long stretches, so that decaying the occurrence probability beats "
               "holding the last rate.",
    }
    if mid in table:
        return table[mid]
    if mid in {"ingarch", "inar"}:
        acf1 = _fmt(f.autocorr_lag1)
        if mid == "ingarch":
            return (
                f"the lag-1 autocorrelation of {acf1} is genuine dependence rather than noise, so the "
                "gap between a Poisson regression's residuals stays autocorrelated after it is fitted "
                "— and INGARCH's recursion is what removes it."
            )
        return (
            "the persistence turns out to be a survival process rather than rate memory — then its "
            "thinning probability is the interpretable one, and it will also fit better on a "
            "short holdout."
        )
    if mid in {"adida", "croston_sy"}:
        # The condition has to be checkable *after* the fit, so it names a comparison the user will see
        # in the fit table rather than the property that got the model onto the list.
        return (
            "its forecast beats plain Croston on the holdout MASE — the only thing block aggregation "
            "and a bias correction buy is a lower error, so a tie means the simpler row wins."
        )
    if mid in {"arch", "garch", "garch_t", "gjr", "gjr_t", "egarch", "egarch_t", "arima_garch"}:
        return _volatility_would_win(mid, f)
    if mid == "renewal":
        return (
            "the gaps are close to memoryless once fitted, so the Weibull shape lands near 1 and the "
            "extra shape parameter earns nothing over the exponential."
        )
    if mid == "hawkes":
        return (
            "the estimated branching ratio is materially above zero and the fit beats renewal on AICc, "
            "which is the only evidence that events are causing events rather than the background rate "
            "varying."
        )
    if not is_implemented(spec.model_id):
        return "not applicable yet — this model cannot be fitted in this build."
    return spec.triggered_when or "the data matches this model's assumptions."


def _expected_seconds(spec: ModelSpec, f: DataFacts, warnings: list[str]) -> float | None:
    """A rough cost estimate, so 'fit everything' is an informed click rather than a surprise."""
    base = {
        ModelFamily.BASELINE: 0.05,
        ModelFamily.SMOOTHING: 1.0,
        ModelFamily.ARMA: 1.5,
        ModelFamily.COUNT: 1.0,
        ModelFamily.INTERMITTENT: 0.2,
        ModelFamily.VOLATILITY: 1.5,
    }.get(spec.family)
    if base is None:
        return None
    if spec.model_id == "sarima":
        base *= max(1.0, f.n_non_missing / 144.0) * (2.0 if f.n_non_missing >= 200 else 1.0)
    if spec.model_id == "arfima":
        base *= 1.5
    if spec.model_id == "ets":
        base *= 1.5
    return round(base, 2)


def suggest(f: DataFacts, *, max_per_family: int = 2) -> ModelSuggestionSet:
    """
    Build the candidate set.

    Ordering is by family first, then by a within-family preference: baselines in a fixed order (they
    are a *sequence* of increasing sophistication, not alternatives), then the models that answer the
    data class' main question, then the rest.
    """
    suggestions: list[ModelSuggestion] = []
    excluded: list[ModelSuggestion] = []
    scored: list[_Scored] = []

    for spec in all_specs():
        verdict = evaluate_spec(spec, f)
        if not verdict.compatible:
            excluded.append(
                ModelSuggestion(
                    model_id=spec.model_id,
                    display_name=spec.display_name,
                    family=spec.family,
                    is_baseline=spec.is_baseline,
                    compatible=False,
                    incompatibilities=[_as_sentence(r) for r in verdict.reasons],
                    because="; ".join(verdict.reasons),
                    would_win_if="not fitted — see the reasons it was excluded",
                    plain_description=spec.plain_description,
                    advanced=spec.feature_flag == "advanced",
                )
            )
            continue
        warnings = list(verdict.warnings)
        scored.append(
            _Scored(
                spec=spec,
                rank_key=_rank_key(spec, f),
                because=_because(spec, f, warnings),
                would_win_if=_would_win_if(spec, f),
                warnings=warnings,
                order_hint=_order_hint(spec, f),
                expected_seconds=_expected_seconds(spec, f, warnings),
            )
        )

    # Keep every baseline, then the best `max_per_family` of each other family.
    by_family: dict[ModelFamily, list[_Scored]] = {}
    for s in scored:
        by_family.setdefault(s.spec.family, []).append(s)

    keep: list[_Scored] = []
    for fam in FAMILY_ORDER:
        items = sorted(by_family.get(fam, []), key=lambda s: s.rank_key)
        if fam is ModelFamily.BASELINE:
            # Baselines are all kept, in the registry's own order — they form a ladder, and dropping
            # the middle rung would make the comparison table unreadable.
            keep.extend(sorted(items, key=lambda s: BASELINE_IDS.index(s.spec.model_id)
                               if s.spec.model_id in BASELINE_IDS else 99))
        else:
            chosen = items[:max_per_family]
            # Unknown-to-the-engine models are always kept as well: the cap exists to stop near
            # duplicates, and a model the ranking map cannot score is not a duplicate of anything.
            chosen += [i for i in items[max_per_family:] if i.spec.model_id not in _KNOWN_IDS]
            keep.extend(chosen)
        # Anything compatible but not kept is still listed, with an honest reason.
        dropped = [
            i for i in (items[max_per_family:] if fam is not ModelFamily.BASELINE else [])
            if i.spec.model_id in _KNOWN_IDS
        ]
        for s in dropped:
            excluded.append(
                ModelSuggestion(
                    model_id=s.spec.model_id,
                    display_name=s.spec.display_name,
                    family=s.spec.family,
                    compatible=False,
                    incompatibilities=[
                        f"It applies to this data, but only the top {max_per_family} of its family are "
                        "offered by default — this one adds cost without adding a distinct answer."
                    ],
                    because="a near-duplicate of a suggestion above",
                    would_win_if="the higher-ranked members of its family fail to fit",
                    plain_description=s.spec.plain_description,
                    advanced=s.spec.feature_flag == "advanced",
                )
            )

    for rank, s in enumerate(keep):
        suggestions.append(
            ModelSuggestion(
                model_id=s.spec.model_id,
                display_name=s.spec.display_name,
                family=s.spec.family,
                rank=rank,
                is_baseline=s.spec.is_baseline,
                recommended=rank == _recommended_index(keep, f.data_class),
                because=s.because,
                would_win_if=s.would_win_if,
                plain_description=s.spec.plain_description,
                advanced=s.spec.feature_flag == "advanced",
                compatible=True,
                assumption_warnings=s.warnings,
                order_hint=s.order_hint,
                expected_seconds=s.expected_seconds,
            )
        )

    default_ids = [s.model_id for s in suggestions if s.compatible]
    headline, plain = _headline(suggestions, excluded, f)

    return ModelSuggestionSet(
        data_class=f.data_class,
        headline=headline,
        plain=plain,
        suggestions=suggestions,
        excluded=excluded,
        default_fit_ids=default_ids,
        facts_used=f.as_dict(),
        n_registry=len(all_specs()),
        caveats=[
            w
            for s in suggestions
            for w in ([f"{s.display_name}: {w}" for w in s.assumption_warnings])
        ],
    )


_KNOWN_IDS = frozenset({
    "mean", "naive", "seasonal_naive", "drift", "white_noise",
    "arima", "sarima", "arfima", "ets", "theta",
    "poisson", "neg_binomial", "zero_inflated", "croston", "tsb",
    "arimax", "ucm", "var", "lightgbm", "combination",
    "arch", "garch", "garch_t", "gjr", "gjr_t", "egarch", "egarch_t", "arima_garch",
})


def _as_sentence(reason: str) -> str:
    """Capitalise and terminate a filter reason so it survives being concatenated by a renderer."""
    r = reason.strip()
    if not r:
        return r
    r = r[0].upper() + r[1:]
    return r if r[-1] in ".!?" else r + "."


def _rank_key(spec: ModelSpec, f: DataFacts) -> tuple:
    """
    Within-family preference, as an ordered tuple rather than a score.

    The first element encodes "does this model answer the question the data is asking" — a count
    series wants a count model first, an intermittent series wants Croston first. The second is the
    model-specific tie-break.
    """
    data_priority = {
        DataClass.COUNT: (ModelFamily.COUNT, ModelFamily.BASELINE, ModelFamily.INTERMITTENT),
        DataClass.INTERMITTENT: (ModelFamily.INTERMITTENT, ModelFamily.BASELINE),
        DataClass.CONTINUOUS: (ModelFamily.BASELINE, ModelFamily.SMOOTHING, ModelFamily.ARMA),
        DataClass.BINARY_OR_PROPORTION: (ModelFamily.BASELINE, ModelFamily.COUNT),
        DataClass.EVENT_TIMESTAMP: (ModelFamily.POINT_PROCESS, ModelFamily.BASELINE),
    }.get(f.data_class, (ModelFamily.BASELINE,))
    fam_rank = (
        data_priority.index(spec.family) if spec.family in data_priority else len(data_priority) + 1
    )
    # A fixed within-family order, stated once. Alphabetical order would put ARFIMA before ARIMA,
    # which is exactly backwards: the exotic model should follow the workhorse, not precede it.
    within = {
        "poisson": 0, "neg_binomial": 1, "zero_inflated": 2,
        "croston": 0, "tsb": 1, "adida": 2, "croston_sy": 3,
        "ingarch": 0, "inar": 1,
        "renewal": 0, "hawkes": 1,
        "garch": 0, "garch_t": 1, "arch": 2, "gjr": 3, "gjr_t": 4, "egarch": 5, "egarch_t": 6,
        "arima_garch": 7,
        "ets": 0, "theta": 1,
        "arima": 0, "sarima": 1, "arfima": 2,
    }
    extra = 0
    if spec.model_id == "neg_binomial" and (f.dispersion_ratio or 0) > 1.5:
        extra = -2  # overdispersion makes the NB the first count model, not the second
    if spec.model_id == "poisson" and (f.dispersion_ratio or 0) <= 1.5:
        extra = -1
    if spec.model_id == "tsb" and f.n_zeros and f.pct_zeros >= 0.5:
        extra = -1  # very sparse: obsolescence is the live question, so TSB leads
    if spec.model_id == "sarima" and f.period and f.D_hint:
        extra = -1  # a seasonal unit root makes SARIMA the first ARMA model, not the second
    # A model the map has never heard of sorts after the known ones in its family but is *kept*: the
    # map is a preference, not a whitelist, and a whitelist is what turns "add a row" back into
    # "edit the engine".
    return (fam_rank, extra, within.get(spec.model_id, 9), spec.display_name)


def _recommended_index(keep: list[_Scored], data_class: DataClass | None = None) -> int:
    """
    The one model to *start* with — never the one to end with.

    The recommendation is the most sophisticated non-baseline model in the data class' own family,
    because that is the one whose assumptions the data actually supports. If there is none (a
    continuous series, where smoothing and ARMA are both plausible), the recommendation goes to the
    first non-baseline suggestion and the UI says to let the comparison decide.
    """
    priority = {
        DataClass.COUNT: (ModelFamily.COUNT,),
        DataClass.INTERMITTENT: (ModelFamily.INTERMITTENT,),
        DataClass.EVENT_TIMESTAMP: (ModelFamily.POINT_PROCESS,),
        DataClass.BINARY_OR_PROPORTION: (ModelFamily.COUNT,),
    }.get(data_class)
    for i, s in enumerate(keep):
        if s.spec.is_baseline or not is_implemented(s.spec.model_id):
            continue
        # "Start here" must be a model of the *right family*. An INGARCH is a count model, and on an
        # intermittent series it is offered as a contrast, not as the place to begin — the first
        # non-baseline row happens to be a count model only because of family ordering.
        if priority and s.spec.family not in priority:
            continue
        return i
    for i, s in enumerate(keep):
        if not s.spec.is_baseline and is_implemented(s.spec.model_id):
            return i
    return 0


def _order_hint(spec: ModelSpec, f: DataFacts) -> str | None:
    """The order the data implies, for the rows that carry orders."""
    if spec.model_id == "arima":
        return f"d = {f.d_hint}"
    if spec.model_id == "sarima":
        return f"d = {f.d_hint}, D = {f.D_hint}, m = {f.period}"
    if spec.model_id == "arfima":
        return "d estimated (GPH)"
    if spec.model_id == "ets":
        bits = ["A" if f.data_class is not DataClass.COUNT else "M"]
        return "error/trend/season chosen by AICc"
    if spec.model_id == "theta":
        return f"season {f.period or 'none'}"
    return None


def _headline(
    suggestions: list[ModelSuggestion], excluded: list[ModelSuggestion], f: DataFacts
) -> tuple[str, str]:
    n_ok = len([s for s in suggestions if s.compatible])
    classes = f.data_class.value.replace("_", " ")
    if classes == "continuous":
        classes = "continuous-valued"
    # Events are counted, not sampled on a grid, so "series of N observations" would misdescribe them:
    # N is the number of *periods* the event dates were bucketed into.
    if f.data_class is DataClass.EVENT_TIMESTAMP:
        subject = f"a list of event dates over {f.n_non_missing} periods"
    else:
        subject = f"a {classes} series of {f.n_non_missing} observations"
    headline = f"{n_ok} candidate models for {subject}, {len(excluded)} excluded"
    parts = [
        f"The registry holds rows for every family, and {n_ok} of them apply here. "
        "Every candidate is listed with the reason it is suggested and the condition under which it "
        "would win, so the list can be argued with rather than accepted."
    ]
    if f.data_class is DataClass.COUNT and f.dispersion_ratio is not None:
        parts.append(
            f"Variance is {f.dispersion_ratio:.2f}× the mean, which "
            + ("makes the negative binomial the honest count model and the Poisson the optimistic one."
               if f.dispersion_ratio > 1.5 else
               "is consistent with a Poisson rate model.")
        )
    if f.data_class is DataClass.INTERMITTENT:
        parts.append(
            f"{f.pct_zeros * 100:.0f}% of the values are zero, so the intermittent branch is promoted "
            "and MAPE is refused outright."
        )
    if any(s.model_id == "white_noise" for s in suggestions):
        parts.append(
            "White noise is in the set as the null hypothesis: if it wins, the correct output is "
            "'no model needed', and the app will say so."
        )
    return headline, " ".join(parts)


__all__ = ["suggest"]
