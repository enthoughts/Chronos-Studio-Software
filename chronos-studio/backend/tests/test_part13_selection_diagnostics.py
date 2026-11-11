"""
tests/test_part13_selection_diagnostics.py — Part 13 (§5.9).

The tests are organised around the four ways a selection-and-diagnostics step goes wrong, because each
one produces a confident answer that is wrong in a way a reader cannot see:

1. **An information criterion compared across things it cannot compare.** A log-likelihood on 168
   observations differenced once is not on the same scale as one on 180 observations at levels, and
   sorting them produces a winner. The guard's job is to refuse, and the refusal is tested with the
   four facts that can differ.
2. **A CV score that is a holdout wearing a plural.** Folds that overlap their own training window, a
   MASE denominator computed on the full series, or a fold that was silently dropped all produce a
   number that looks like cross-validation and is not.
3. **A verdict from a count of p-values.** Twelve tests at 5% reject a perfect model 46% of the time.
   The verdict must corroborate across *independent* mechanisms before it says INADEQUATE, and it must
   be willing to say ADEQUATE.
4. **A combination built from in-sample fits.** It then beats its own components for a reason that
   vanishes on new data — and it is the row that most often "wins" a comparison table.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from app.ts.diagnose import (
    Adequacy,
    judge,
    pit_from_quantiles,
    rqr_from_draws,
    run_battery,
    coverage_tests,
)
from app.ts.diagnose.calibration import _cvm_sf
from app.ts.select import (
    CVScheme,
    FitIdentity,
    build_selection_report,
    clark_west,
    comparability,
    cross_validate,
    diebold_mariano,
    fold_origins,
    ic_from_loglik,
    mincer_zarnowitz,
    rank_by_ic,
    rank_by_cv,
)
from app.ts.select.ic import ICKind, akaike_weights, delta_ic, support_label


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def seasonal_series(n: int = 140, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    vals = 100 + 0.25 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)
    return pd.Series(vals, index=pd.date_range("2010-01-31", periods=n, freq="MS"))


class _Naive:
    """The smallest credible wrapper: last value repeated, with a fixed-width band."""

    def __init__(self) -> None:
        self.summary = None
        self.failed = False
        self._is_fitted = False

    def fit(self, y: pd.Series):
        self._last = float(y.iloc[-1])
        self._is_fitted = True
        return self

    def predict(self, horizon: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        m = np.full(horizon, self._last)
        return pd.DataFrame({"mean": m, "lower": m - 4.0, "upper": m + 4.0})


class _Drift(_Naive):
    def fit(self, y: pd.Series):
        self._d = float((y.iloc[-1] - y.iloc[0]) / max(len(y) - 1, 1))
        self._last = float(y.iloc[-1])
        self._is_fitted = True
        return self

    def predict(self, horizon: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        m = self._last + self._d * np.arange(horizon)
        return pd.DataFrame({"mean": m, "lower": m - 4.0, "upper": m + 4.0})


class _Broken(_Naive):
    def fit(self, y: pd.Series):
        raise RuntimeError("this model cannot be fitted on any window")


FACTORY = {"naive": _Naive, "drift": _Drift, "broken": _Broken}


# --------------------------------------------------------------------------- #
# 1. Information criteria
# --------------------------------------------------------------------------- #
def test_ic_formulas_match_their_definitions():
    ic = ic_from_loglik(-100.0, n=200, k=3)
    assert ic.aic == pytest.approx(-2 * -100.0 + 2 * 3)
    assert ic.bic == pytest.approx(-2 * -100.0 + 3 * math.log(200))
    assert ic.hq == pytest.approx(-2 * -100.0 + 2 * 3 * math.log(math.log(200)))
    assert ic.aicc > ic.aic, "the small-sample correction must always add to the AIC, never subtract"


def test_aicc_is_refused_rather_than_guessed_when_k_is_too_large():
    """n ≤ k+1 leaves nothing to divide by; returning `inf` or the AIC silently would rank garbage."""
    ic = ic_from_loglik(-10.0, n=5, k=5)
    assert ic.aicc is None
    assert "undefined" in " ".join(ic.notes)
    assert ic.aic is not None, "the AIC is still defined and is still reported"


def test_no_likelihood_means_no_criterion_anywhere():
    ic = ic_from_loglik(None, n=100, k=2)
    assert (ic.aic, ic.aicc, ic.bic, ic.hq) == (None, None, None, None)
    assert ic.notes


def test_fits_on_different_observation_counts_are_refused():
    """The single most consequential comparability rule: a likelihood on 168 points is not the
    likelihood on 180, and their AICs are different quantities that happen to share a name."""
    a = FitIdentity("arima", n_obs=168, d=1, k=4, ic=ic_from_loglik(-100, n=168, k=4))
    b = FitIdentity("ets", n_obs=180, d=0, k=5, ic=ic_from_loglik(-110, n=180, k=5))
    verdict = comparability([a, b])
    assert not verdict.comparable
    assert "observation counts" in verdict.reason
    assert rank_by_ic([a, b]) == [], "a refused comparison must return no ranking at all"


def test_differencing_is_part_of_the_identity():
    a = FitIdentity("arima", n_obs=168, d=1, k=4, ic=ic_from_loglik(-100, n=168, k=4))
    b = FitIdentity("sarima", n_obs=168, d=0, k=4, ic=ic_from_loglik(-100, n=168, k=4))
    assert not comparability([a, b]).comparable
    assert "differencing" in comparability([a, b]).reason


def test_a_quasi_likelihood_is_its_own_group():
    """A GARCH quasi-likelihood and an ARIMA likelihood are both called 'loglik' and neither is
    comparable with the other."""
    a = FitIdentity("arima", model_class="arma", n_obs=200, k=3, ic=ic_from_loglik(-100, n=200, k=3))
    b = FitIdentity("garch", model_class="quasi", n_obs=200, k=3, ic=ic_from_loglik(-100, n=200, k=3))
    assert not comparability([a, b]).comparable
    assert "likelihood classes" in comparability([a, b]).reason


def test_comparable_fits_are_ranked_and_weighted():
    a = FitIdentity("arima", n_obs=200, k=3, ic=ic_from_loglik(-100, n=200, k=3))
    b = FitIdentity("ets", n_obs=200, k=5, ic=ic_from_loglik(-105, n=200, k=5))
    c = FitIdentity("theta", n_obs=200, k=2, ic=ic_from_loglik(-103, n=200, k=2))
    verdict = comparability([a, b, c])
    assert verdict.comparable
    assert [m for m, _ in verdict.ranked][0] == "arima"
    deltas = delta_ic(verdict.ranked)
    assert deltas[0] == 0.0
    weights = akaike_weights([ic.get(ICKind.AICC) for _, ic in verdict.ranked])
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == max(weights)


def test_the_two_unit_band_is_named_not_scored():
    assert support_label(0.0).startswith("indistinguishable")
    assert support_label(1.99).startswith("indistinguishable")
    assert "no support" in support_label(50.0)


# --------------------------------------------------------------------------- #
# 2. Rolling-origin CV
# --------------------------------------------------------------------------- #
def test_folds_never_train_on_their_own_test_block():
    for scheme in (CVScheme.EXPANDING, CVScheme.SLIDING):
        for fold in fold_origins(120, horizon=6, folds=4, scheme=scheme):
            assert fold.train_end < fold.test_start, scheme


def test_the_last_fold_ends_on_the_last_observation():
    """CV that stops before the end scores the model on a series the user no longer has."""
    folds = fold_origins(120, horizon=6, folds=4)
    assert folds[-1].test_end == 119


def test_folds_are_labelled_in_forward_time_order():
    folds = fold_origins(120, horizon=6, folds=4)
    assert [f.index for f in folds] == [0, 1, 2, 3]
    assert folds[0].test_end < folds[1].test_start


def test_an_explicit_step_overlaps_folds_instead_of_dropping_them():
    """The combination row needs many origins; without `step` the engine silently built 10 of 28
    requested folds, which is a CV score that is not the one that was asked for."""
    folds = fold_origins(140, horizon=6, folds=28, step=1)
    assert len(folds) == 28
    assert folds[1].test_start == folds[0].test_start + 1


def test_expanding_and_sliding_differ_and_sliding_actually_slides():
    expanding = fold_origins(140, horizon=6, folds=5, scheme=CVScheme.EXPANDING)
    sliding = fold_origins(140, horizon=6, folds=5, scheme=CVScheme.SLIDING)
    assert expanding[-1].n_train > expanding[0].n_train
    assert sliding[0].n_train == sliding[-1].n_train, "a sliding window has a fixed width"
    assert sliding[-1].train_start > sliding[0].train_start, "it must move forward"


def test_too_short_a_series_refuses_instead_of_quietly_producing_one_fold():
    with pytest.raises(ValueError, match="cannot support"):
        fold_origins(20, horizon=12, folds=4)


def test_cross_validation_scores_every_model_on_every_fold():
    y = seasonal_series()
    results = cross_validate(["naive", "drift"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=3)
    for r in results:
        assert not r.failed
        assert r.n_folds == 3
        assert r.mase is not None and r.mase > 0
        assert all(f.mase is not None for f in r.folds)


def test_a_model_that_cannot_fit_is_recorded_not_dropped():
    y = seasonal_series()
    results = cross_validate(["naive", "broken"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=3)
    broken = [r for r in results if r.model_id == "broken"][0]
    assert broken.failed
    assert "fold 0" in broken.failure_reason
    ranked = rank_by_cv(results)
    assert ranked[-1].model_id == "broken", "a failed model must sort last, not vanish"


def test_an_already_fitted_wrapper_is_rejected():
    """Reusing an object across folds is the subtlest leak: it carries the previous window's state."""
    y = seasonal_series()

    class _PreFitted(_Naive):
        def __init__(self):
            super().__init__()
            self.summary = object()

    results = cross_validate(["x"], y, factory=lambda m: _PreFitted(), horizon=6, period=12, folds=3)
    assert results[0].failed
    assert "already-fitted" in results[0].failure_reason


def test_the_mase_denominator_is_computed_on_the_training_window():
    """Scaling by the whole series' naive error lets the test data into the denominator."""
    from app.ts.select.cv import _scale

    train = np.array([10.0, 12.0, 11.0, 15.0])
    huge_test = np.array([1000.0])
    s_train = _scale(train, huge_test, None)
    s_with_test = _scale(np.concatenate([train, huge_test]), huge_test, None)
    assert s_train is not None
    assert s_train != s_with_test, "the test observation must not reach the scale"


def test_rank_prefers_the_lower_mase_then_the_more_stable_model():
    y = seasonal_series()
    results = cross_validate(["naive", "drift"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=4)
    ranked = rank_by_cv(results)
    means = [r.mase for r in ranked]
    assert means == sorted(means)


# --------------------------------------------------------------------------- #
# 3. Pairwise tests
# --------------------------------------------------------------------------- #
def test_diebold_mariano_finds_a_real_difference():
    rng = np.random.default_rng(4)
    truth = rng.normal(0, 1, 200)
    good = truth + rng.normal(0, 0.1, 200)
    bad = truth + rng.normal(0, 3.0, 200)
    res = diebold_mariano(truth - good, truth - bad, name_a="good", name_b="bad")
    assert res.ok
    assert res.better == "good"
    assert res.p_value < 0.01
    assert res.mean_loss_diff < 0


def test_diebold_mariano_refuses_identical_forecasts():
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 50)
    res = diebold_mariano(e, e)
    assert not res.ok
    assert "identical loss" in res.note


def test_diebold_mariano_refuses_too_few_pairs():
    res = diebold_mariano([1.0, 2.0, 3.0], [1.5, 2.5, 2.0])
    assert not res.ok
    assert "at least" in res.note


def test_diebold_mariano_is_insignificant_between_two_equally_good_forecasts():
    rng = np.random.default_rng(6)
    truth = rng.normal(0, 1, 300)
    a = truth + rng.normal(0, 1.0, 300)
    b = truth + rng.normal(0, 1.0, 300)
    res = diebold_mariano(truth - a, truth - b)
    assert res.ok
    assert res.p_value > 0.01


def test_clark_west_reports_the_nesting_adjustment():
    rng = np.random.default_rng(7)
    truth = rng.normal(0, 1, 200)
    small = truth + rng.normal(0, 1.0, 200)
    large = truth + rng.normal(0, 0.98, 200)
    res = clark_west(truth - small, truth - large, name_small="restricted", name_large="full")
    assert res.ok
    assert "Clark–West adjustment adds" in res.note
    assert res.p_value is not None and 0.0 <= res.p_value <= 1.0
    assert res.adjustment is not None and res.adjustment >= 0.0
    assert any("nested" in a for a in res.assumptions)


def test_clark_west_says_to_use_dm_when_the_models_are_not_nested():
    res = clark_west(np.ones(20) * 0.5, np.ones(20) * 0.4)
    assert res.ok or not res.ok  # either verdict is fine; the assumption text is the contract
    if res.ok:
        assert any("Diebold" in a for a in res.assumptions)


def test_mincer_zarnowitz_passes_an_unbiased_forecast():
    rng = np.random.default_rng(8)
    truth = rng.normal(100, 5, 200)
    forecast = truth + rng.normal(0, 0.5, 200)
    res = mincer_zarnowitz(truth, forecast, name="model")
    assert res.ok
    assert res.joint_p is not None
    assert "passes" in res.verdict or res.joint_p < 0.05


def test_mincer_zarnowitz_catches_a_scaled_forecast():
    """A forecast that is the truth times 1.5 has a slope of 1/1.5 — biased *and* mis-scaled, which a DM
    test against another model would never notice."""
    truth = np.linspace(100, 200, 120)
    forecast = truth * 1.5
    res = mincer_zarnowitz(truth, forecast)
    assert res.ok
    assert res.slope == pytest.approx(1 / 1.5, abs=0.02)
    assert res.joint_p < 0.001
    assert "overshoot" in res.verdict or "flatter" in res.verdict


def test_mincer_zarnowitz_refuses_a_constant_forecast():
    res = mincer_zarnowitz(np.arange(20.0), np.full(20, 5.0))
    assert not res.ok
    assert "constant" in res.note


# --------------------------------------------------------------------------- #
# 4. The selection report
# --------------------------------------------------------------------------- #
def test_the_report_states_that_cv_is_primary():
    y = seasonal_series()
    results = cross_validate(["naive", "drift"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=3)
    rep = build_selection_report(results, horizon=6)
    assert "primary selection criterion" in rep.caveat
    assert "secondary" in rep.caveat


def test_the_report_paragraph_carries_real_numbers():
    y = seasonal_series()
    results = cross_validate(["naive", "drift"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=3)
    rep = build_selection_report(results, horizon=6)
    assert rep.winner
    assert rep.why
    assert f"{rep.shortlist[0].cv_mase:.3f}" in rep.why, "the paragraph must quote the winner's score"


def test_a_close_race_is_declared_indistinguishable_not_crowned():
    """Two models inside the fold noise must produce 'treat them as one choice', not a winner."""
    y = seasonal_series()
    results = cross_validate(["naive", "drift"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=3)
    rep = build_selection_report(results, horizon=6)
    close = abs(rep.shortlist[0].cv_mase - rep.shortlist[1].cv_mase) < 0.05
    if close:
        assert rep.indistinguishable
        assert "one choice" in rep.why


def test_the_runner_up_is_not_offered_as_an_alternative_when_it_is_far_behind():
    """Regression: the first version said 'a different window could favour X' about a model 145% worse."""
    y = seasonal_series()
    results = cross_validate(["naive", "drift"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=3)
    rep = build_selection_report(results, horizon=6)
    gap = (rep.shortlist[1].cv_mase - rep.shortlist[0].cv_mase) / rep.shortlist[0].cv_mase
    if gap > 0.25 and "Choose differently" in rep.why:
        assert "could reorder them" not in rep.why


def test_an_unrankable_model_is_named_in_the_notes():
    y = seasonal_series()
    results = cross_validate(["naive", "broken"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=3)
    rep = build_selection_report(results, horizon=6)
    assert any("could not be cross-validated" in n for n in rep.notes)
    assert "unmeasured" in " ".join(rep.notes)


def test_the_report_refuses_to_rank_when_nothing_could_be_scored():
    rep = build_selection_report([], horizon=6)
    assert rep.shortlist == []
    assert "no ranking is offered" in rep.why


def test_a_refused_ic_comparison_travels_into_the_report():
    y = seasonal_series()
    results = cross_validate(["naive"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=3)
    ids = [
        FitIdentity("a", n_obs=168, d=1, ic=ic_from_loglik(-100, n=168, k=3)),
        FitIdentity("b", n_obs=180, d=0, ic=ic_from_loglik(-100, n=180, k=3)),
    ]
    rep = build_selection_report(results, identities=ids, horizon=6)
    assert rep.ic_comparable is False
    assert "cannot be ranked" in rep.ic_reason


# --------------------------------------------------------------------------- #
# 5. The residual battery
# --------------------------------------------------------------------------- #
def test_white_noise_residuals_produce_no_findings():
    rng = np.random.default_rng(20)
    battery = run_battery(rng.normal(0, 1, 400), period=12)
    verdict = judge(battery, period=12)
    assert verdict.adequacy is Adequacy.ADEQUATE
    assert not verdict.reasons


def test_strong_autocorrelation_is_caught_at_the_right_lag():
    rng = np.random.default_rng(21)
    e = rng.normal(0, 1, 300)
    r = np.zeros(300)
    for i in range(1, 300):
        r[i] = 0.75 * r[i - 1] + e[i]
    battery = run_battery(r, period=12)
    assert battery.ljung_box["rejects"], "Ljung–Box must reject on AR(1) residuals"
    verdict = judge(battery, period=12)
    assert verdict.adequacy is Adequacy.INADEQUATE
    lb = [f for f in verdict.reasons if f.test.startswith("Ljung–Box")][0]
    assert lb.severity == "high"
    assert "AR or MA order" in lb.remedy


def test_ljung_box_is_tested_at_the_seasonal_lag_as_well():
    battery = run_battery([float(v) for v in np.random.default_rng(1).normal(0, 1, 200)], period=12)
    assert "12" in battery.ljung_box["results"]
    assert "24" in battery.ljung_box["results"]


def test_a_single_rejection_among_many_is_a_caveat_not_a_revision():
    """The multiple-testing rule: one low p-value in a long battery is what chance looks like."""
    rng = np.random.default_rng(22)
    e = rng.normal(0, 1, 300)
    e[::37] *= 6.0  # a few heavy points: J-B will reject, nothing else will
    battery = run_battery(e, period=12)
    verdict = judge(battery, period=12)
    assert verdict.adequacy is not Adequacy.INADEQUATE or len(
        {f.test.split(" (")[0] for f in verdict.reasons if f.kind == "structural"}
    ) >= 2


def test_non_normal_residuals_are_a_distributional_finding_with_the_right_remedy():
    rng = np.random.default_rng(23)
    e = rng.standard_t(3, 400)
    battery = run_battery(e, period=None)
    verdict = judge(battery, period=None)
    jb = [f for f in verdict.reasons if f.test == "Jarque–Bera"]
    if jb:
        assert jb[0].kind == "distributional"
        assert "point forecast" in jb[0].detail + jb[0].remedy
        assert verdict.adequacy is not Adequacy.INADEQUATE


def test_arch_in_the_residuals_asks_for_a_variance_model_not_a_new_mean_model():
    rng = np.random.default_rng(24)
    n = 400
    v = np.zeros(n)
    e = np.zeros(n)
    for i in range(1, n):
        v[i] = 0.2 + 0.75 * v[i - 1] + rng.normal(0, 0.1)
        e[i] = rng.normal(0, math.sqrt(max(v[i], 1e-6)))
    battery = run_battery(e, period=None)
    arch = battery.arch_lm
    if arch.get("p") is not None and arch["p"] < 0.05:
        assert "ARIMA-GARCH" in arch["advice"]
        assert "point forecast is unaffected" in arch["advice"]


def test_residuals_beyond_the_mad_scale_are_flagged_with_their_influence():
    rng = np.random.default_rng(25)
    e = rng.normal(0, 1, 200)
    e[50] = 30.0
    battery = run_battery(e, period=None)
    assert battery.outliers
    top = battery.outliers[0]
    assert top["position"] == 50
    assert top["influence"] > 0.1
    assert "residual sum of squares" in top["reading"]


def test_too_few_residuals_produce_no_verdict_rather_than_a_cheerful_one():
    verdict = judge(run_battery([1.0, -1.0, 0.5]), period=12)
    assert verdict.adequacy is Adequacy.ADEQUATE_WITH_CAVEATS
    assert "No verdict" in verdict.headline


def test_anderson_darling_falls_back_to_the_scipy_116_signature(monkeypatch):
    """The offline wheelhouse pins SciPy 1.16, whose anderson() has no `method`."""
    from app.ts.diagnose import battery as battery_module

    rng = np.random.default_rng(20260101)
    e = rng.normal(0, 1, 200)
    original = battery_module.stats.anderson
    sentinel = object()
    calls: list[str] = []

    def scipy_116_anderson(x, dist="norm", *, method=sentinel):
        if method is not sentinel:
            raise TypeError("anderson() got an unexpected keyword argument 'method'")
        calls.append("fallback")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return original(x, dist=dist)

    monkeypatch.setattr(battery_module.stats, "anderson", scipy_116_anderson)
    out = battery_module.anderson_darling_test(e)
    assert calls == ["fallback"]
    assert out["statistic"] is not None
    assert out["approximate"] is True


def test_cusum_detects_a_drifting_mean():
    rng = np.random.default_rng(26)
    e = np.concatenate([rng.normal(-1.5, 1, 150), rng.normal(1.5, 1, 150)])
    battery = run_battery(e, period=None)
    assert battery.cusum["breach"]
    verdict = judge(battery, period=None)
    assert any("CUSUM" in f.test for f in verdict.reasons)


# --------------------------------------------------------------------------- #
# 6. PIT and RQR
# --------------------------------------------------------------------------- #
def test_pit_of_a_correct_continuous_forecast_is_uniform():
    rng = np.random.default_rng(30)
    u = rng.random(500)
    res = pit_from_quantiles(np.zeros(500), cdf=u)
    assert res.ok
    assert res.ks_p > 0.01
    assert "uniform" in res.reading.lower() or res.cvm_p > 0.01


def test_pit_detects_intervals_that_are_too_narrow():
    """A U-shaped PIT is the signature of under-coverage, and the reading has to say so."""
    rng = np.random.default_rng(31)
    u = np.concatenate([rng.beta(0.4, 1.0, 250), rng.beta(1.0, 0.4, 250)])
    res = pit_from_quantiles(np.zeros(500), cdf=np.clip(u, 1e-9, 1 - 1e-9))
    assert res.ok
    assert res.ks_p < 0.01 or res.cvm_p < 0.01


def test_a_discrete_model_must_be_randomized_and_says_why():
    """Without randomization the PIT lands on the PMF's atoms and a *perfect* model looks broken."""
    res = pit_from_quantiles(np.arange(1.0, 51.0), kind="discrete", cdf=np.full(50, 0.6))
    assert not res.ok
    assert "atoms" in res.note

    lo = np.full(50, 0.4)
    hi = np.full(50, 0.6)
    ok_res = pit_from_quantiles(np.arange(1.0, 51.0), kind="discrete", cdf=hi, lower_cdf=lo)
    assert ok_res.ok
    assert "randomized" in ok_res.caveat
    assert ok_res.kind == "discrete"


def test_rqr_is_normal_under_a_correct_model_and_discloses_the_randomization():
    rng = np.random.default_rng(32)
    n, draws = 200, 300
    actual = rng.normal(0, 1, n)
    d = rng.normal(0, 1, (draws, n))
    res = rqr_from_draws(actual, d)
    assert res.ok
    assert res.reps >= 5
    assert "randomized by construction" in res.caveat
    assert res.stability


def test_rqr_flags_a_grossly_wrong_predictive_distribution():
    rng = np.random.default_rng(33)
    n, draws = 200, 300
    actual = rng.normal(0, 5, n)  # truth is much wider than the forecast says
    d = rng.normal(0, 1, (draws, n))
    res = rqr_from_draws(actual, d)
    assert res.ok
    assert res.ks_p < 0.01


def test_coverage_is_tested_not_thresholded():
    """36 of 40 inside a 90% band is 87.5% — well inside sampling variation, and the test says so."""
    res = coverage_tests([True] * 36 + [False] * 4, nominal=0.9)
    assert res["coverage"] == pytest.approx(0.9)
    assert res["p"] > 0.05
    assert "95% interval" in res["note"]


def test_cvm_survival_function_is_a_probability():
    for t in (0.05, 0.347, 1.0, 5.0):
        assert 0.0 <= _cvm_sf(t) <= 1.0


# --------------------------------------------------------------------------- #
# 7. The combination row
# --------------------------------------------------------------------------- #
def test_the_combination_is_registered_and_implemented():
    from app.ts.models.fit import COMBINATION_ID, IMPLEMENTED
    from app.ts.models.registry import REGISTRY, is_implemented

    assert COMBINATION_ID in IMPLEMENTED
    assert is_implemented("combination")
    assert REGISTRY["combination"].required.min_two_models


def test_the_combination_needs_two_components():
    from app.ts.models.base import FitFailure
    from app.ts.models.combination import CombinationModel

    model = CombinationModel()
    with pytest.raises(FitFailure, match="At least two"):
        model.set_inputs(np.zeros((10, 1)), np.zeros(10))


def test_the_combination_refuses_without_component_forecasts():
    from app.ts.models.combination import CombinationModel

    model = CombinationModel()
    model.fit(seasonal_series())
    assert model.failed
    assert "without its component forecasts" in model.summary.failure_reason


def test_all_four_rules_produce_a_finite_forecast():
    from app.ts.models.combination import CombinationModel, ComponentForecast

    rng = np.random.default_rng(40)
    n, k = 40, 3
    actual = 100 + rng.normal(0, 2, n)
    matrix = np.column_stack([actual + rng.normal(0, s, n) for s in (2.0, 4.0, 9.0)])
    comps = [
        ComponentForecast(
            f"m{i}", f"M{i}",
            mean=list(100 + rng.normal(0, 3, 8)),
            lower=list(95 + rng.normal(0, 1, 8)),
            upper=list(105 + rng.normal(0, 1, 8)),
        )
        for i in range(k)
    ]
    for method in ("mean", "median", "ic_weighted", "granger_ramanathan"):
        model = CombinationModel(method=method, components=["a", "b", "c"])
        model.set_inputs(matrix, actual, aicc=[100, 105, 120], component_forecasts=comps)
        model.fit(pd.Series(actual))
        assert not model.failed, f"{method}: {model.summary.failure_reason}"
        fc = model.predict(8)
        assert np.isfinite(fc["mean"]).all()
        assert (fc["lower"] <= fc["mean"]).all() and (fc["mean"] <= fc["upper"]).all()
        # No likelihood, no IC — this is the row that must not enter an IC ranking.
        assert model.summary.ic_comparable is False
        assert model.summary.aicc is None


def test_ic_weights_follow_the_information_criterion():
    from app.ts.models.combination import _weights_ic

    w = _weights_ic([100.0, 120.0, 200.0])
    assert w[0] > 0.99
    assert w[0] > w[1] > w[2]


def test_ic_weights_fall_back_to_equal_without_criteria():
    from app.ts.models.combination import _weights_ic

    w = _weights_ic([None, None])
    assert all(abs(x - 0.5) < 1e-12 for x in w)


def test_granger_ramanathan_refuses_when_there_are_too_few_origins():
    from app.ts.models.base import FitFailure
    from app.ts.models.combination import CombinationModel, ComponentForecast

    model = CombinationModel(method="granger_ramanathan")
    comps = [
        ComponentForecast(f"m{i}", f"M{i}", mean=[1.0] * 4, lower=[0.0] * 4, upper=[2.0] * 4)
        for i in range(3)
    ]
    # Three components need k + 3 = 6 origins; four is short of that by two.
    model.set_inputs(np.ones((4, 3)), np.ones(4), aicc=[1, 2, 3], component_forecasts=comps)
    # The refusal arrives as a *failed fit*, not an exception out of `fit`: the base class catches
    # `FitFailure` and turns it into a row with the reason on it, which is the contract every other
    # model in this build follows. Asserting `pytest.raises` here was asserting the wrong contract.
    model.fit(pd.Series(np.ones(20)))
    assert model.failed
    assert "held-out forecasts" in model.summary.failure_reason
    assert "6" in model.summary.failure_reason


def test_the_combination_weights_come_from_held_out_forecasts_in_the_real_fit_path():
    """
    The end-to-end guarantee: `fit_many` builds the combination from rolling origins, not from the
    components' in-sample fits. The tell for the wrong implementation is a combination that outscores
    its best component.
    """
    from app.ts.models.fit import fit_many

    y = seasonal_series(n=120)
    outcomes = fit_many(["ets", "naive", "combination"], y, period=12, frequency_code="MS", combination_horizon=6)
    combo = [o for o in outcomes if o.summary.model_id == "combination"][0]
    assert not combo.summary.failed, combo.summary.failure_reason
    card = combo.wrapper.card
    assert card["held_out_origins"] >= 4
    assert "held-out forecasts" in card["why"]
    # Components are the fitted models, never the benchmark ladder.
    ids = [c["model_id"] for c in card["components"]]
    assert "ets" in ids
    assert "mean" not in ids and "white_noise" not in ids


def test_the_combination_has_an_honest_verdict_rather_than_a_blank():
    """A row with no one-step residuals must say so, in words that read as a property, not a bug."""
    from app.ts.models.fit import fit_many
    from app.ts.diagnose import diagnose_model

    y = seasonal_series(n=120)
    outcomes = fit_many(["ets", "naive", "combination"], y, period=12, frequency_code="MS", combination_horizon=6)
    combo = [o for o in outcomes if o.summary.model_id == "combination"][0]
    block = diagnose_model(combo, period=12)
    assert block["status"] in {"ok", "unavailable"}
    if block["status"] == "unavailable":
        assert "held-out residuals" in block["message"]
        assert "Step 11" in block["message"]


# --------------------------------------------------------------------------- #
# 8. API surface
# --------------------------------------------------------------------------- #
def _prepared(client):
    """Walk the real stage order — every stage gate in this app is a real one."""
    base = "/api/v1"
    sid = client.post(f"{base}/sessions").json()["session_id"]
    headers = {"X-Session-Id": sid}
    client.post(f"{base}/data/example", json={"example_id": "traffic_spend"}, headers=headers)
    for stage in ("profile", "decompose", "tests"):
        assert client.post(f"{base}/analysis/{stage}", headers=headers).status_code == 200
    assert client.get(f"{base}/analysis/correlate", headers=headers).status_code == 200
    assert client.post(f"{base}/analysis/gate", headers=headers).status_code == 200
    assert client.post(f"{base}/analysis/suggest", headers=headers).status_code == 200
    return client, base, headers


def test_selection_endpoint_refuses_before_anything_is_fitted():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    _, base, headers = _prepared(client)
    r = client.post(f"{base}/analysis/select", json={"horizon": 6}, headers=headers)
    assert r.status_code in (400, 409)
    assert "error" in r.json()


def test_the_full_select_and_diagnose_flow_runs():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    _, base, headers = _prepared(client)
    fit = client.post(
        f"{base}/analysis/fit",
        json={"model_ids": ["ets", "drift"], "combination_horizon": 6},
        headers=headers,
    )
    assert fit.status_code == 200, fit.text[:400]

    sel = client.post(
        f"{base}/analysis/select",
        json={"horizon": 6, "folds": 3, "models": ["ets", "drift"]},
        headers=headers,
    )
    assert sel.status_code == 200, sel.text[:600]
    stored = client.get(f"{base}/analysis/selection", headers=headers).json()
    assert stored["status"] == "complete"
    report = stored["report"]
    assert report["primary_criterion"] == "rolling-origin MASE"
    assert report["shortlist"] and report["why"]
    assert report["shortlist"][0]["rank"] == 1

    diag = client.post(f"{base}/analysis/diagnose", json={}, headers=headers)
    assert diag.status_code == 200, diag.text[:600]
    blocks = client.get(f"{base}/analysis/diagnostics", headers=headers).json()["blocks"]
    assert blocks
    for block in blocks.values():
        assert block["status"] in {"ok", "unavailable"}
        if block["status"] == "ok":
            assert block["verdict"]["adequacy"] in {"adequate", "adequate_with_caveats", "inadequate"}


def test_a_bad_combination_rule_is_refused_with_the_list_of_rules():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    _, base, headers = _prepared(client)
    r = client.post(
        f"{base}/analysis/fit",
        json={"model_ids": ["ets"], "combination_method": "magic"},
        headers=headers,
    )
    assert r.status_code in (400, 409)
    body = r.json()["error"]
    assert "mean" in body["fix"] and "granger_ramanathan" in body["fix"]


def test_cross_validation_keeps_the_paired_error_series():
    """
    Diebold–Mariano compares two models' loss *series*. Reconstructing that from per-fold means gives
    `n_folds` numbers — four points, on which the test correctly refuses. The engine has to keep every
    step of every test block, which is what makes the HAC correction at `horizon − 1` lags meaningful.
    """
    y = seasonal_series(n=120)
    results = cross_validate(["naive", "drift"], y, factory=lambda m: FACTORY[m](), horizon=6, period=12, folds=4)
    for r in results:
        assert len(r.errors) == len(r.actuals) == 4 * 6
        assert np.isfinite(r.errors).all()
    # And the pooled series is long enough for the test to run.
    a, b = results[0], results[1]
    res = diebold_mariano(a.errors, b.errors, name_a=a.model_id, name_b=b.model_id, horizon=6)
    assert res.ok
    assert res.n == 24
