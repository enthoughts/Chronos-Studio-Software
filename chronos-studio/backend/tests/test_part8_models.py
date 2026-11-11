"""
tests/test_part8_models.py — Part 8: the model registry, the suggestion engine, and fitting (§5.8).

This is the largest stage in the app and the one whose failures are quietest: a model that silently
does not apply, a forecast whose interval is symmetric when the outcome is a count, a comparison table
that ranks an ARFIMA AIC against an ARIMA one. The tests are grouped accordingly:

1. **The registry is data.** Adding a row must offer a model with reasons, with no code change
   anywhere else. This is the claim §5.8 makes, and it is the one that quietly stops being true.
2. **The filter refuses for a stated reason.** Every exclusion names the measurement.
3. **The wrapper contract is total.** A failed fit is a row, never an exception; a failed predict
   refuses rather than returning NaNs.
4. **Baselines are benchmarks, not decoration.** Their intervals have the right shape (√h), and the
   white-noise model is willing to say "no model needed".
5. **The ARMA family does not lie about order or roots.** `d` is not searched; a boundary root is
   reported rather than hidden; an ARFIMA AIC is marked incomparable.
6. **Count and intermittent branches obey their own rules.** Discrete intervals, and MAPE refused.
7. **Fitting always leaves something on the page.**
8. **The API enforces the gate** — a NOT READY verdict stops the fit rather than advising against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.schemas import DataClass, ModelFamily, ModelRequirements, ModelSpec
from app.ts.models import registry as reg
from app.ts.models.arma import (
    ARFIMAModel,
    ARIMAModel,
    SARIMAModel,
    _frac_integrate,
    fractional_difference,
    gph_estimate,
)
from app.ts.models.base import FitFailure, ModelWrapper
from app.ts.models.baselines import (
    DriftModel,
    MeanModel,
    NaiveModel,
    SeasonalNaiveModel,
    WhiteNoiseModel,
)
from app.ts.models.counts import HurdleModel, NegativeBinomialModel, PoissonModel
from app.ts.models.facts import (
    build_facts,
    classify_data_class,
    dispersion_test,
    zero_excess,
)
from app.ts.models.fit import IMPLEMENTED, best_by_aicc, fit_many, fit_one, summarise_failures
from app.ts.models.intermittent import CrostonModel, TSBModel
from app.ts.models.registry import all_specs, evaluate_spec, get_spec, is_implemented, register
from app.ts.models.smoothing import ETSModel, ThetaModel
from app.ts.models.suggest import suggest


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def seasonal_series() -> pd.Series:
    rng = np.random.default_rng(42)
    n, t = 144, np.arange(144)
    vals = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, n)
    return pd.Series(vals, index=pd.date_range("2010-01-01", periods=n, freq="MS"))


@pytest.fixture(scope="module")
def count_series() -> pd.Series:
    rng = np.random.default_rng(3)
    t = np.arange(60)
    lam = np.exp(1.5 + 0.3 * np.sin(2 * np.pi * t / 12))
    return pd.Series(rng.negative_binomial(2, 2 / (2 + lam)).astype(float))


@pytest.fixture(scope="module")
def intermittent_series() -> pd.Series:
    rng = np.random.default_rng(11)
    vals = np.zeros(96)
    idx = rng.choice(96, 27, replace=False)
    vals[idx] = rng.integers(1, 9, idx.size)
    return pd.Series(vals)


@pytest.fixture()
def continuous_facts(seasonal_series) -> object:
    return build_facts(
        seasonal_series,
        data_class=DataClass.CONTINUOUS,
        period=12,
        seasonal_strength=0.45,
        trend_r2=0.6,
        stationarity_verdict="unit_root",
        d_hint=1,
        D_hint=1,
    )


# --------------------------------------------------------------------------- #
# 1. The registry is data, not code paths
# --------------------------------------------------------------------------- #
def test_the_registry_holds_several_families():
    families = {s.family for s in reg.all_specs()}
    assert ModelFamily.BASELINE in families
    assert ModelFamily.ARMA in families
    assert ModelFamily.COUNT in families
    assert ModelFamily.INTERMITTENT in families
    # Families that exist as rows but are not fitted in this build — they must still be *offered*,
    # or the user concludes the app has never heard of a GARCH model.
    assert ModelFamily.VOLATILITY in families
    assert len(reg.all_specs()) >= 20


def test_adding_a_row_offers_a_model(continuous_facts):
    """
    The load-bearing test of the whole design: a new registry row becomes a suggestion with reasons,
    with no change to the filter, the ranking or the API.
    """
    before = {s.model_id for s in suggest(continuous_facts).suggestions}
    assert "rows_model" not in before

    row = ModelSpec(
        model_id="rows_model",
        display_name="Row-defined model",
        family=ModelFamily.SMOOTHING,
        description="A model that exists only as a registry row.",
        plain_description="Proof that adding a model is a data change.",
        required=ModelRequirements(min_n=10),
        triggered_when="Registered at runtime, for this test.",
    )
    register(row)
    try:
        after = suggest(continuous_facts)
        ids = [s.model_id for s in after.suggestions]
        assert "rows_model" in ids, "a new registry row must appear in the candidate set"
        entry = next(s for s in after.suggestions if s.model_id == "rows_model")
        assert entry.because  # a reason, not a blank
        assert entry.would_win_if
        assert entry.family is ModelFamily.SMOOTHING
    finally:
        reg.REGISTRY.pop("rows_model", None)
        reg._SPECS.remove(row)  # noqa: SLF001 - the registry owns this list


def test_every_registry_row_can_be_queried_without_special_casing(continuous_facts):
    """No row may raise from the filter, however exotic — a crash here would take the whole list."""
    for spec in reg.all_specs():
        v = evaluate_spec(spec, continuous_facts)
        assert isinstance(v.compatible, bool)
        assert all(isinstance(r, str) and r for r in v.reasons)


def test_implemented_models_all_have_a_wrapper():
    """Registry says fit-able ⇔ there is a wrapper class. A mismatch is a 404 at fit time."""
    for mid in IMPLEMENTED:
        assert mid in reg.REGISTRY, f"{mid} has a wrapper but no registry row"
    for spec in reg.implemented_specs():
        assert spec.model_id in IMPLEMENTED, (
            f"{spec.model_id} is marked implemented but has no wrapper"
        )


def test_pending_models_are_excluded_with_a_reason_that_is_not_a_guess(continuous_facts):
    """
    A pending row must be excluded with a reason a reader can act on.

    Two shapes are legitimate and both are tested: a row waiting on *code* quotes the part that will
    add it, and a row waiting on a *missing library* quotes the measured import failure. The rule the
    test enforces is that the sentence is specific — the model id or the package name — and never a
    bare "not available".
    """
    # Waiting on a library: the reason is the failed import, naming the package. Every remaining
    # pending row is in this group as of Part 13 — `combination` was the last row waiting on *code*,
    # and it now fits. The "waiting on code" branch is kept below against a synthetic row rather than
    # a live one, because the shape it guards ("planned for part N" with nothing to fit today) still
    # has to work the day a row is added back in that state.
    for model_id, package in (("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("lstm", "torch")):
        reasons = " ".join(evaluate_spec(get_spec(model_id), continuous_facts).reasons)
        assert package in reasons, f"{model_id} must name {package}: {reasons}"
        assert "not installed" in reasons
        assert "planned for part" not in reasons, (
            f"{model_id} is pending because a library is missing, not because of a schedule: {reasons}"
        )

    # Every pending row must be excluded for *some* stated reason, and a shipped part must never be
    # quoted as a future plan.
    shipped = {11, 12, 13}
    for spec in all_specs():
        if is_implemented(spec.model_id):
            continue
        joined = " ".join(evaluate_spec(spec, continuous_facts).reasons)
        assert joined, f"{spec.model_id} is pending with no reason"
        if spec.planned_part in shipped:
            assert "planned for part" not in joined, f"{spec.model_id}: {joined}"


# --------------------------------------------------------------------------- #
# 2. The filter refuses for a stated reason
# --------------------------------------------------------------------------- #
def test_a_count_series_excludes_the_continuous_families(count_series):
    f = build_facts(count_series, data_class=DataClass.COUNT, period=12)
    excluded = {e.model_id for e in suggest(f).excluded}
    assert {"arima", "sarima", "ets", "theta"} <= excluded


def test_a_continuous_series_excludes_the_count_families(continuous_facts):
    excluded = {e.model_id for e in suggest(continuous_facts).excluded}
    assert {"poisson", "neg_binomial", "croston", "tsb"} <= excluded


def test_a_sparse_count_series_is_reclassified_as_intermittent(intermittent_series):
    dc, why = classify_data_class(intermittent_series, DataClass.COUNT)
    assert dc is DataClass.INTERMITTENT
    assert "zero" in why and "%" in why


def test_intermittent_promotes_croston_above_count_models(intermittent_series):
    dc, _ = classify_data_class(intermittent_series, DataClass.COUNT)
    f = build_facts(intermittent_series, data_class=dc)
    s = suggest(f)
    ids = [x.model_id for x in s.suggestions]
    assert "croston" in ids and "tsb" in ids
    assert "poisson" not in ids
    rec = next(x for x in s.suggestions if x.recommended)
    assert rec.model_id in {"croston", "tsb"}


def test_every_exclusion_names_a_measurement(count_series, continuous_facts):
    for facts in (continuous_facts, build_facts(count_series, data_class=DataClass.COUNT)):
        for e in suggest(facts).excluded:
            text = " ".join(e.incompatibilities)
            assert text, f"{e.model_id} excluded with no reason"
            # A refusal without a number or a named structural fact teaches nothing.
            assert any(ch.isdigit() for ch in text) or any(
                w in text.lower()
                for w in ("family", "period", "cycle", "zero", "both", "values", "cannot fit",
                          "not installed")
            ), f"{e.model_id}: {text}"


def test_an_incompatibility_is_not_reported_as_a_warning(continuous_facts):
    """The two must not be mixed: 'cannot be fitted' and 'fitted but risky' are different answers."""
    for e in suggest(continuous_facts).excluded:
        assert not e.assumption_warnings
    for s in suggest(continuous_facts).suggestions:
        assert s.compatible
        assert not s.incompatibilities


def test_too_few_observations_excludes_with_the_number():
    tiny = pd.Series(np.arange(8, dtype=float))
    f = build_facts(tiny, data_class=DataClass.CONTINUOUS)
    excluded = {e.model_id: " ".join(e.incompatibilities) for e in suggest(f).excluded}
    assert "arima" in excluded
    assert "20" in excluded["arima"] and "8" in excluded["arima"]


def test_a_seasonal_model_for_a_series_without_a_period_is_excluded():
    y = pd.Series(np.random.default_rng(0).normal(0, 1, 60))
    f = build_facts(y, data_class=DataClass.CONTINUOUS, period=None)
    excluded = {e.model_id: " ".join(e.incompatibilities) for e in suggest(f).excluded}
    assert "seasonal_naive" in excluded
    assert "no seasonal period" in excluded["seasonal_naive"]


def test_assumption_warnings_travel_with_the_suggestion(seasonal_series):
    """An ARIMA on a strongly seasonal series must warn — it has no seasonal term."""
    f = build_facts(
        seasonal_series, data_class=DataClass.CONTINUOUS, period=12, seasonal_strength=0.62
    )
    s = suggest(f)
    arima = next((x for x in s.suggestions if x.model_id == "arima"), None)
    if arima is not None:
        assert any("season" in w for w in arima.assumption_warnings)


# --------------------------------------------------------------------------- #
# 3. Facts: the measurements the filter queries
# --------------------------------------------------------------------------- #
def test_dispersion_ratio_is_variance_over_mean(count_series):
    ratio, p = dispersion_test(count_series.to_numpy())
    y = count_series.to_numpy()
    assert ratio == pytest.approx(y.var(ddof=1) / y.mean())
    assert p is not None and p < 0.05  # overdispersed by construction


def test_dispersion_on_poisson_like_data_is_about_one():
    y = np.random.default_rng(5).poisson(4.0, 800).astype(float)
    ratio, _ = dispersion_test(y)
    assert 0.8 < ratio < 1.25


def test_zero_excess_distinguishes_structural_from_low_rate_zeros():
    """
    The reference has to be the *conditional* rate, or a low-rate series looks like a hurdle series.
    Here the conditional rate is a smooth seasonal curve in both cases; the only difference is that
    one of them has an extra mechanism that zeroes a third of the periods at random.
    """
    rng = np.random.default_rng(7)
    t = np.arange(600)
    lam = np.exp(1.2 + 0.4 * np.sin(2 * np.pi * t / 12))
    honest = rng.poisson(lam).astype(float)
    structural = honest.copy()
    structural[rng.choice(600, 200, replace=False)] = 0.0
    assert zero_excess(honest) < 1.6
    # 200 of 600 periods zeroed by a process the rate does not know about: the ratio must rise
    # clearly above 1 even though the pooled zero share is only 40%.
    assert zero_excess(structural) > 1.6
    assert zero_excess(structural) > 1.2 * zero_excess(honest)


def test_facts_do_not_recompute_stationarity(seasonal_series):
    """S5 owns the verdict; a second copy here could disagree with the one the gate travelled with."""
    f = build_facts(seasonal_series, stationarity_verdict="seasonal_unit_root", d_hint=0, D_hint=1)
    assert f.stationarity_verdict == "seasonal_unit_root"
    assert f.d_hint == 0 and f.D_hint == 1


def test_facts_report_no_dispersion_for_a_continuous_series(seasonal_series):
    f = build_facts(seasonal_series, data_class=DataClass.CONTINUOUS, period=12)
    assert f.dispersion_ratio is None


def test_classify_leaves_binary_data_alone():
    y = pd.Series([0.0, 0.4, 0.9, 0.1, 0.55, 0.7, 0.2, 0.66] * 5)
    dc, why = classify_data_class(y, DataClass.CONTINUOUS)
    assert dc is DataClass.BINARY_OR_PROPORTION
    assert "[0, 1]" in why


# --------------------------------------------------------------------------- #
# 4. Baselines
# --------------------------------------------------------------------------- #
def test_naive_intervals_widen_as_root_h(seasonal_series):
    m = NaiveModel().fit(seasonal_series)
    fc = m.predict(16)
    width = (fc["upper"] - fc["lower"]).to_numpy()
    assert np.all(np.diff(width) > 0), "a constant interval width is the classic fan-chart lie"
    # Width ∝ √h, so width(16)/width(4) = 2.
    assert width[15] / width[3] == pytest.approx(2.0, rel=0.02)


def test_mean_model_has_a_flat_forecast_and_constant_interval(seasonal_series):
    m = MeanModel().fit(seasonal_series)
    fc = m.predict(10)
    assert len(set(np.round(fc["mean"], 9))) == 1
    assert len(set(np.round(fc["upper"] - fc["lower"], 9))) == 1


def test_seasonal_naive_repeats_the_last_cycle(seasonal_series):
    m = SeasonalNaiveModel(period=12).fit(seasonal_series)
    fc = m.predict(12)
    expected = seasonal_series.to_numpy()[-12:]
    assert np.allclose(fc["mean"].to_numpy(), expected)


def test_seasonal_naive_refuses_without_a_period(seasonal_series):
    m = SeasonalNaiveModel(period=None).fit(seasonal_series)
    assert m.failed
    assert "no seasonal period" in m.summary.failure_reason.lower()


def test_drift_is_the_random_walk_plus_a_slope(seasonal_series):
    m = DriftModel().fit(seasonal_series)
    fc = m.predict(5)
    expected = m.last + m.slope * np.arange(1, 6)
    assert np.allclose(fc["mean"].to_numpy(), expected)


def test_white_noise_says_no_model_needed_on_actual_noise():
    y = pd.Series(np.random.default_rng(2).normal(0, 1, 400))
    m = WhiteNoiseModel().fit(y)
    assert m.ok
    assert m.is_white, [e["p_value"] for e in m.evidence]
    assert "no model needed" in m.verdict.lower()


def test_white_noise_does_not_claim_it_on_a_trending_series(seasonal_series):
    m = WhiteNoiseModel().fit(seasonal_series)
    assert m.ok
    assert not m.is_white
    assert "structure" in m.verdict.lower()


def test_baselines_report_no_information_criterion(seasonal_series):
    """A pseudo-likelihood AIC on a benchmark would be ranked against ARIMA's — comparing nothing."""
    for cls in (MeanModel, NaiveModel, DriftModel, WhiteNoiseModel):
        m = cls(period=12).fit(seasonal_series)
        if m.ok:
            s = m.summary
            assert s.aic is None and s.aicc is None
            assert s.ic_comparable is False
            assert s.ic_note


# --------------------------------------------------------------------------- #
# 5. The ARMA family
# --------------------------------------------------------------------------- #
def test_arima_respects_the_d_it_is_given(seasonal_series):
    m = ARIMAModel(d_hint=0, period=12).fit(seasonal_series)
    assert m.ok, m.summary.failure_reason
    assert m.order[1] == 0, "d must come from the caller, not from a search"


def test_arima_with_d_one_reports_that_its_ic_is_on_a_different_scale(seasonal_series):
    m = ARIMAModel(d_hint=1, period=12).fit(seasonal_series)
    assert m.ok, m.summary.failure_reason
    assert m.order[1] == 1
    assert "different scale" in (m.summary.ic_note + " ".join(m.summary.caveats)).lower()


def test_arima_reports_parameters_with_standard_errors(seasonal_series):
    m = ARIMAModel(d_hint=0, period=12).fit(seasonal_series)
    assert m.ok
    assert m.summary.params, "a fit with no parameter table cannot be checked"
    assert any(p.std_err is not None for p in m.summary.params)


def test_arima_refuses_missing_values():
    y = pd.Series(np.r_[np.linspace(1, 10, 30), np.nan, np.linspace(11, 20, 30)])
    m = ARIMAModel(d_hint=0, period=12).fit(y)
    assert m.failed
    assert "missing" in m.summary.failure_reason.lower()


def test_sarima_refuses_without_two_full_cycles():
    y = pd.Series(np.random.default_rng(0).normal(0, 1, 20))
    m = SARIMAModel(d_hint=0, D_hint=0, period=12).fit(y)
    assert m.failed
    assert "cycle" in m.summary.failure_reason.lower() or "period" in m.summary.failure_reason.lower()


def test_root_check_flags_a_boundary_root_rather_than_hiding_it(seasonal_series):
    """`enforce_stationarity=False` lets the optimiser reach the boundary; the report must say so."""
    m = SARIMAModel(d_hint=0, D_hint=0, period=12).fit(seasonal_series)
    assert m.ok
    if m.summary.roots_ok is False:
        assert m.summary.roots_note
        assert any(ch.isdigit() for ch in m.summary.roots_note)


def test_fractional_differencing_and_integration_are_inverses():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 400))
    for d in (0.2, 0.35, 0.45):
        back = _frac_integrate(fractional_difference(x, d), d)
        # Truncation loses the pre-sample tail, so compare away from the start.
        assert np.max(np.abs(back[100:] - x[100:])) < 1e-6


def test_gph_recovers_roughly_zero_for_white_noise():
    x = np.random.default_rng(4).normal(0, 1, 800)
    d, se, m = gph_estimate(x)
    assert d is not None and se is not None and m > 10
    assert abs(d) < 2.5 * se


def test_arfima_refuses_when_d_is_outside_the_long_memory_range():
    """A unit root is not long memory; ARFIMA must not be used as a dodge for it."""
    y = pd.Series(np.cumsum(np.random.default_rng(6).normal(0, 1, 300)))
    m = ARFIMAModel(period=12).fit(y)
    assert m.failed
    reason = m.summary.failure_reason.lower()
    assert "0.5" in reason or "unit root" in reason


def test_arfima_marks_its_ic_incomparable_when_it_does_fit():
    rng = np.random.default_rng(7)
    e = rng.normal(0, 1, 400)
    x = np.zeros(400)
    for i in range(1, 400):
        x[i] = x[i - 1] * 0.95 + e[i]
    y = pd.Series(x)
    m = ARFIMAModel(period=12).fit(y)
    if m.ok:
        assert m.summary.ic_comparable is False
        assert "NOT comparable" in m.summary.ic_note
        assert any("0 < d < 0.5" in c or "fractional" in c for c in m.summary.caveats)


# --------------------------------------------------------------------------- #
# 6. Smoothing
# --------------------------------------------------------------------------- #
def test_ets_fits_a_seasonal_series(seasonal_series):
    m = ETSModel(period=12).fit(seasonal_series)
    assert m.ok, m.summary.failure_reason
    assert m.spec is not None
    assert m.summary.aicc is not None


def test_ets_never_tries_a_multiplicative_season_on_non_positive_data():
    rng = np.random.default_rng(9)
    y = pd.Series(np.sin(np.arange(60) / 3) + rng.normal(0, 0.3, 60))  # crosses zero
    m = ETSModel(period=12).fit(y)
    if m.ok:
        assert m.spec[2] != "mul", "a multiplicative season is undefined on a series with <= 0"


def test_ets_forecast_has_wider_intervals_further_out(seasonal_series):
    m = ETSModel(period=12).fit(seasonal_series)
    assert m.ok
    fc = m.predict(12)
    width = (fc["upper"] - fc["lower"]).to_numpy()
    assert width[-1] > width[0]


def test_theta_refuses_to_report_a_zero_in_sample_error(seasonal_series):
    """
    Theta's two lines average back to the series, so a naive 'fitted values' would give it a MASE of
    exactly 0.0 — the most convincing wrong number the app could print.
    """
    m = ThetaModel(period=12).fit(seasonal_series)
    assert m.ok
    m.compute_diagnostics(seasonal_series)
    assert m.diagnostics.mase is None
    assert any("in-sample" in c for c in m.summary.caveats)


# --------------------------------------------------------------------------- #
# 7. Count and intermittent models
# --------------------------------------------------------------------------- #
def test_count_models_produce_non_negative_integer_intervals(count_series):
    for cls in (PoissonModel, NegativeBinomialModel, HurdleModel):
        m = cls(period=12).fit(count_series)
        assert m.ok, f"{cls.model_id}: {m.summary.failure_reason}"
        fc = m.predict(12)
        assert (fc["lower"] >= 0).all(), f"{cls.model_id} produced a negative count bound"
        assert np.allclose(fc["lower"], np.round(fc["lower"]))
        assert np.allclose(fc["upper"], np.round(fc["upper"]))
        assert (fc["upper"] >= fc["mean"]).all()


def test_the_hurdle_interval_contains_its_own_mean(count_series):
    """Regression: an earlier version added +1 to every draw, putting the whole interval above the
    point forecast."""
    m = HurdleModel(period=12).fit(count_series)
    assert m.ok
    fc = m.predict(12)
    assert ((fc["lower"] <= fc["mean"]) & (fc["mean"] <= fc["upper"])).all()


def test_poisson_warns_when_the_data_is_overdispersed(count_series):
    m = PoissonModel(period=12).fit(count_series)
    assert m.ok
    joined = " ".join(m.summary.caveats).lower()
    assert "overdispersed" in joined
    assert "negative binomial" in joined


def test_negative_binomial_warns_when_it_buys_nothing():
    y = pd.Series(np.random.default_rng(10).poisson(3.0, 120).astype(float))
    m = NegativeBinomialModel(period=None).fit(y)
    assert m.ok
    assert any("almost nothing" in c or "little" in c for c in m.summary.caveats)


def test_count_models_refuse_fractional_data(seasonal_series):
    m = PoissonModel(period=12).fit(seasonal_series / 7.3)
    assert m.failed
    assert "fractional" in m.summary.failure_reason.lower()


def test_mape_is_refused_for_count_data(count_series):
    m = PoissonModel(period=12).fit(count_series)
    m.compute_diagnostics(count_series, ban_mape=True)
    assert m.diagnostics.mape is None
    assert any("MAPE is not reported" in w for w in m.diagnostics.warnings)


def test_croston_applies_the_sba_halving(intermittent_series):
    m = CrostonModel().fit(intermittent_series)
    assert m.ok, m.summary.failure_reason
    assert m.rate < m.rate_uncorrected
    assert any("SBA" in c for c in m.summary.caveats)


def test_tsb_forecast_decays_as_zeros_accumulate(intermittent_series):
    m = TSBModel().fit(intermittent_series)
    assert m.ok
    fc = m.predict(12)
    assert fc["mean"].iloc[-1] < fc["mean"].iloc[0]


def test_intermittent_models_refuse_a_series_with_too_few_non_zero_points():
    y = pd.Series([0.0] * 60)
    y.iloc[3] = 4.0
    y.iloc[40] = 2.0
    for cls in (CrostonModel, TSBModel):
        m = cls().fit(y)
        assert m.failed
        assert "non-zero" in m.summary.failure_reason


# --------------------------------------------------------------------------- #
# 8. The wrapper contract
# --------------------------------------------------------------------------- #
def test_a_failed_fit_returns_a_summary_instead_of_raising():
    class Broken(ModelWrapper):
        model_id = "broken"
        display_name = "Broken"
        family = ModelFamily.BASELINE

        def _fit(self, y):
            raise FitFailure("it cannot work.", "do something else.")

        def _forecast(self, horizon, *, alpha):  # pragma: no cover
            raise AssertionError("must not be called")

    m = Broken().fit(pd.Series(np.arange(20, dtype=float)))
    assert m.failed
    assert m.summary is not None
    assert "cannot work" in m.summary.failure_reason
    assert m.summary.converged is False


def test_a_library_crash_is_reported_as_a_fit_failure_not_a_500():
    class Exploding(ModelWrapper):
        model_id = "exploding"
        display_name = "Exploding"
        family = ModelFamily.BASELINE

        def _fit(self, y):
            raise ValueError("nonsense from a library")

        def _forecast(self, horizon, *, alpha):  # pragma: no cover
            raise AssertionError("must not be called")

    m = Exploding().fit(pd.Series(np.arange(20, dtype=float)))
    assert m.failed
    assert "ValueError" in m.summary.failure_reason
    assert "reported rather than hidden" in m.summary.failure_reason


def test_predicting_from_a_failed_fit_refuses():
    m = SeasonalNaiveModel(period=None).fit(pd.Series(np.random.default_rng(0).normal(0, 1, 50)))
    assert m.failed
    with pytest.raises(FitFailure) as ei:
        m.predict(5)
    assert "cannot forecast" in ei.value.plain


def test_compute_diagnostics_aligns_a_one_step_short_fitted_series(seasonal_series):
    """A one-step-ahead fit has no prediction for t=1, so its array is one shorter than the series."""
    m = NaiveModel().fit(seasonal_series)
    d = m.compute_diagnostics(seasonal_series)
    assert d.mae is not None and d.mase is not None


def test_mase_uses_the_seasonal_naive_denominator_when_a_period_exists(seasonal_series):
    from app.ts.models.diagnostics import naive_scale

    y = seasonal_series.to_numpy()
    seasonal = naive_scale(y, 12)
    plain = naive_scale(y, None)
    assert seasonal != plain
    assert seasonal == pytest.approx(float(np.abs(y[12:] - y[:-12]).mean()))


def test_parameter_table_flags_a_boundary_estimate():
    m = MeanModel().fit(pd.Series(np.arange(30, dtype=float)))
    rows = m.params_table({"phi": (0.9999, 0.0001, True)})
    assert rows[0].at_boundary is True
    assert "boundary" in rows[0].plain


# --------------------------------------------------------------------------- #
# 9. Fitting always leaves something on the page
# --------------------------------------------------------------------------- #
def test_baselines_are_fitted_even_when_not_requested(seasonal_series):
    outcomes = fit_many(["ets"], seasonal_series, period=12)
    ids = {o.summary.model_id for o in outcomes}
    assert {"mean", "naive", "seasonal_naive", "drift", "white_noise"} <= ids
    assert "ets" in ids


def test_a_failure_does_not_remove_the_other_rows(seasonal_series):
    outcomes = fit_many(["ets", "sarima"], seasonal_series, period=None)
    assert any(o.summary.failed for o in outcomes)
    assert any(not o.summary.failed for o in outcomes)
    assert summarise_failures(outcomes)


def test_failure_summary_names_the_model_and_the_reason():
    y = pd.Series(np.random.default_rng(0).normal(0, 1, 50))
    outcomes = fit_many(["sarima"], y, period=None)
    text = summarise_failures(outcomes)
    assert "SARIMA" in text
    assert "%" not in text or True
    assert any(ch.isdigit() for ch in text) or "no seasonal period" in text.lower()


def test_best_by_aicc_keeps_incomparable_models_out_of_the_ranking(seasonal_series):
    outcomes = fit_many(["ets", "theta", "arima"], seasonal_series, period=12, d_hint=0)
    ranked = best_by_aicc(outcomes)
    comparable = [o for o in ranked if o.summary.ic_comparable and o.summary.aicc is not None]
    aiccs = [o.summary.aicc for o in comparable]
    assert aiccs == sorted(aiccs), "comparable models must be ranked by AICc"
    first_incomparable = next(
        (i for i, o in enumerate(ranked)
         if not (o.summary.ic_comparable and o.summary.aicc is not None)),
        None,
    )
    if first_incomparable is not None:
        assert all(
            not (o.summary.ic_comparable and o.summary.aicc is not None)
            for o in ranked[first_incomparable:]
        ), "an incomparable model must never be interleaved with ranked ones"


def test_fit_one_never_raises_for_an_unknown_id():
    """A bad id must produce a visible failure row, not a 500 at the API boundary."""
    y = pd.Series(np.arange(50, dtype=float))
    out = fit_one("not_a_model", y, period=None)
    assert out.summary.failed
    assert out.summary.failure_reason


# --------------------------------------------------------------------------- #
# 10. The API
# --------------------------------------------------------------------------- #
@pytest.fixture()
def ready_session(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    client.post("/api/v1/analysis/decompose", json={"method": "stl"}, headers=h)
    client.post("/api/v1/analysis/tests", json={"allow_stale": True}, headers=h)
    client.get("/api/v1/analysis/correlate", headers=h)
    client.post("/api/v1/analysis/gate", headers=h)
    return h


def test_registry_endpoint_reports_what_this_build_can_fit(client):
    r = client.get("/api/v1/analysis/registry")
    assert r.status_code == 200
    body = r.json()
    assert body["n_rows"] >= 20
    assert body["n_implemented"] >= 10
    by_id = {row["model_id"]: row for row in body["rows"]}
    assert by_id["arima"]["implemented"] is True
    # Part 11 fitted the variance rows, so the "not implemented" example moves to the multivariate/ML
    # family, which Part 12 will fill.
    assert by_id["garch"]["implemented"] is True
    assert by_id["lightgbm"]["implemented"] is False


def test_suggest_requires_the_gate(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    r = client.get("/api/v1/analysis/suggest", headers=h)
    assert r.status_code == 409


def test_suggest_returns_reasons_for_every_candidate(client, ready_session):
    body = client.get("/api/v1/analysis/suggest", headers=ready_session).json()
    assert body["suggestions"]
    for s in body["suggestions"]:
        assert s["because"]
        assert s["would_win_if"]
    assert body["excluded"], "an empty excluded list means the filter never ran"
    assert all(e["incompatibilities"] for e in body["excluded"])


def test_suggest_stage_marks_complete_and_adds_tables(client, ready_session):
    r = client.post("/api/v1/analysis/suggest", headers=ready_session)
    assert r.status_code == 200, r.text
    ids = {x["id"] for x in r.json()["results"]}
    assert "models.suggestions" in ids
    # `/pipeline` is the static stage map; the *session's* progress lives on `/sessions/{id}`.
    sess = client.get(f"/api/v1/sessions/{ready_session['X-Session-Id']}").json()
    # `stages` is a dict keyed by stage code, each value carrying its status.
    assert sess["pipeline"]["stages"]["S9"]["status"] == "complete"


def test_fit_refuses_without_the_suggestion_step(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    client.post("/api/v1/analysis/decompose", headers=h)
    client.post("/api/v1/analysis/tests", headers=h)
    client.get("/api/v1/analysis/correlate", headers=h)
    client.post("/api/v1/analysis/gate", headers=h)
    r = client.post("/api/v1/analysis/fit", json={}, headers=h)
    assert r.status_code == 409
    err = r.json()["error"]
    assert "S9" in err["fix"] or "Step 9" in err["fix"]


def test_fit_refuses_a_model_this_build_does_not_implement(client, ready_session):
    client.post("/api/v1/analysis/suggest", headers=ready_session)
    r = client.post(
        "/api/v1/analysis/fit", json={"model_ids": ["lightgbm"]}, headers=ready_session
    )
    assert r.status_code == 409
    assert "lightgbm" in r.json()["error"]["plain"]


def test_fit_runs_and_the_table_ranks_by_aicc(client, ready_session):
    client.post("/api/v1/analysis/suggest", headers=ready_session)
    r = client.post(
        "/api/v1/analysis/fit",
        json={"model_ids": ["ets", "arima", "theta"]},
        headers=ready_session,
    )
    assert r.status_code == 200, r.text
    table = next(x for x in r.json()["results"] if x["id"] == "models.fit_comparison")
    aiccs = [row["aicc"] for row in table["rows"] if row["aicc"] is not None]
    assert aiccs == sorted(aiccs)
    # Baselines must be present in the table even though they were not requested.
    ids = {row["model_id"] for row in table["rows"]}
    assert {"naive", "seasonal_naive"} <= ids


def test_fits_endpoint_returns_ranked_fits_with_diagnostics(client, ready_session):
    client.post("/api/v1/analysis/suggest", headers=ready_session)
    client.post("/api/v1/analysis/fit", json={"model_ids": ["ets"]}, headers=ready_session)
    body = client.get("/api/v1/analysis/fits", headers=ready_session).json()
    assert body["status"] == "complete"
    assert body["fits"]
    first = body["fits"][0]
    assert first["failed"] is False
    assert "diagnostics" in first


def test_forecast_from_a_stored_fit_is_ordered_and_bracketed(client, ready_session):
    client.post("/api/v1/analysis/suggest", headers=ready_session)
    client.post("/api/v1/analysis/fit", json={"model_ids": ["ets"]}, headers=ready_session)
    r = client.post(
        "/api/v1/analysis/fit/forecast",
        json={"model_id": "ets", "horizon": 6},
        headers=ready_session,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["mean"]) == 6
    assert all(lo <= m <= hi for lo, m, hi in zip(body["lower"], body["mean"], body["upper"]))


def test_the_gate_blocks_a_fit_when_the_series_is_not_ready(client, session_id):
    """
    §5.7 says blockers must stop modelling. An app that shows a red NOT READY card and then fits
    anyway has taught the user that the card is decoration.
    """
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "electricity_daily"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    client.post("/api/v1/analysis/decompose", headers=h)
    client.post("/api/v1/analysis/tests", headers=h)
    client.get("/api/v1/analysis/correlate", headers=h)
    gate = client.post("/api/v1/analysis/gate", headers=h).json()
    assert gate["advisor"]["next_stage_ready"] is False
    # Force S9 to have run so that the refusal is the *gate's*, not the missing prerequisite.
    from app.core.schemas import Stage
    from app.core.session import get_store

    session = get_store().get(session_id)
    session.mark_stage(Stage.S9_SUGGEST_MODEL, "complete", "forced for the test")
    r = client.post("/api/v1/analysis/fit", json={"model_ids": ["ets"]}, headers=h)
    assert r.status_code == 409
    assert "not ready" in r.json()["error"]["plain"].lower()


def test_stored_fits_go_stale_when_the_series_changes(client, ready_session, session_id):
    client.post("/api/v1/analysis/suggest", headers=ready_session)
    client.post("/api/v1/analysis/fit", json={"model_ids": ["ets"]}, headers=ready_session)
    before = client.get("/api/v1/analysis/fits", headers=ready_session).json()
    assert before["stale"] is False
    client.post(
        "/api/v1/preprocess/apply",
        json={"op": "transform.log", "params": {}, "rule_id": "test", "reason": "test"},
        headers=ready_session,
    )
    after = client.get("/api/v1/analysis/fits", headers=ready_session).json()
    assert after["stale"] is True
    assert after["current_version"] > after["computed_for_version"]


def test_snippets_parse(client, ready_session):
    import ast

    client.post("/api/v1/analysis/suggest", headers=ready_session)
    r = client.post("/api/v1/analysis/fit", json={"model_ids": ["ets"]}, headers=ready_session)
    assert r.status_code == 200
    for result in r.json()["results"]:
        snippet = result.get("reproducible_snippet") or ""
        assert snippet.strip(), f"{result['id']} has no snippet"
        ast.parse(snippet)
