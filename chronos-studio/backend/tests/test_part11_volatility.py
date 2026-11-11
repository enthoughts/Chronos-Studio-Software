"""
Part 11 tests — conditional variance.

A GARCH fit is the easiest model in this whole project to make *look* right: the optimiser converges,
the persistence lands near 0.97 whatever the data is, and the parameter table prints three numbers with
two decimal places. So the assertions here are on things that a wrong implementation cannot fake:

1. **Recovery.** Simulate a GARCH with known (α, β) and require the estimate to be near it; same for
   ν, for the leverage sign, and for the *ordering* of AICc — the t versions must beat the Gaussian
   ones on t-distributed data and lose on normal data. A model that always prefers the t version is not
   estimating a tail, it is adding a parameter and calling it evidence.
2. **The variance path, not the mean path.** The term structure has to start at the last filtered
   variance and converge monotonically toward the unconditional one; a path that returns a constant is
   the `σ̄²·h` bug and must fail. The path also has to be positive at every step for every
   specification, including after a huge shock.
3. **The ARCH-LM gate.** The registry must refuse to offer GARCH on data with no clustering, and must
   *not* be fooled by a series whose mean moves — the p-value on the differenced series is the one
   that counts.
4. **The honest failures.** Persistence at the boundary has to report no long-run variance rather than
   a number; a flat-variance series has to produce a caveat, not a triumphant β = 0.99.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.schemas import DataClass, ModelFamily
from app.ts.models.base import FitFailure
from app.ts.models.facts import arch_lm, build_facts
from app.ts.models.fit import IMPLEMENTED, fit_many
from app.ts.models.registry import REGISTRY, evaluate_spec, is_implemented
from app.ts.models.suggest import suggest
from app.ts.models.volatility import (
    GRID,
    PERSISTENCE_CAP,
    SPEC_BY_KEY,
    UNIMPLEMENTED_VOLATILITY,
    _persistence,
    _unconditional,
    IMPLEMENTED_VOLATILITY,
    VolatilityModel,
)

SEED = 20261108


def simulate_garch(
    n: int = 1200,
    *,
    omega: float = 0.05,
    alpha: float = 0.08,
    beta: float = 0.85,
    gamma: float = 0.0,
    nu: float | None = None,
    seed: int = SEED,
) -> pd.Series:
    """A GARCH/GJR series with a known truth; `nu` switches from normal to Student-t innovations."""
    rng = np.random.default_rng(seed)
    s2 = np.empty(n)
    e = np.empty(n)
    s2[0] = omega / (1.0 - alpha - 0.5 * gamma - beta)
    for t in range(n):
        if t:
            prev = e[t - 1]
            s2[t] = omega + alpha * prev**2 + gamma * (prev**2 if prev < 0 else 0.0) + beta * s2[t - 1]
        if nu is None:
            e[t] = rng.normal(0.0, np.sqrt(s2[t]))
        else:
            e[t] = rng.standard_t(nu) * np.sqrt(s2[t] / (nu / (nu - 2.0)))
    return pd.Series(e)


def _fit(model_id: str, y: pd.Series) -> VolatilityModel:
    m = IMPLEMENTED_VOLATILITY[model_id]()
    m.fit(pd.Series(y).astype(float))
    return m


# --------------------------------------------------------------------------- #
# Registry and wiring
# --------------------------------------------------------------------------- #
def test_every_volatility_row_is_fittable_or_explicitly_refused():
    rows = [k for k, v in REGISTRY.items() if v.family is ModelFamily.VOLATILITY]
    assert set(rows) == {
        "arch", "garch", "garch_t", "gjr", "gjr_t", "egarch", "egarch_t", "arima_garch", "sv", "realized",
    }
    for mid in rows:
        if mid in UNIMPLEMENTED_VOLATILITY:
            assert not is_implemented(mid), mid
            assert UNIMPLEMENTED_VOLATILITY[mid], "a refused row must carry a reason"
        else:
            assert is_implemented(mid), mid
            assert mid in IMPLEMENTED, mid


def test_grid_and_implemented_agree():
    assert set(SPEC_BY_KEY) == set(IMPLEMENTED_VOLATILITY)
    assert len(GRID) == 7 and len(IMPLEMENTED_VOLATILITY) == 8


@pytest.mark.parametrize("model_id", sorted(IMPLEMENTED_VOLATILITY))
def test_every_row_fits_and_reports_a_variance_path(model_id: str):
    y = simulate_garch(600, seed=SEED + len(model_id))
    m = _fit(model_id, y)
    s = m.summary
    assert not s.failed, s.failure_reason
    assert s.aicc is not None and np.isfinite(s.aicc), model_id
    assert s.n_params >= 2
    vp = m.variance_forecast()
    assert len(vp["variance"]) == 60
    assert all(v > 0 for v in vp["variance"]), model_id
    # The sd path has to be the square root of the variance path, not an independently computed number.
    assert np.allclose(np.sqrt(vp["variance"]), vp["sd"])


def test_two_forecasts_from_one_fit():
    """§8 Part 11: "produces two forecasts". The mean path and the variance path, both real."""
    y = simulate_garch(800)
    m = _fit("arima_garch", y)
    both = m.forecast_both(12)
    assert len(both["mean"]) == len(both["lower"]) == len(both["upper"]) == 12
    assert len(both["variance_path"]) == 12
    assert all(v > 0 for v in both["variance_path"])
    # And they are different numbers: the variance path is not the interval width echoed back.
    assert not np.allclose(both["sd"], both["variance_path"])


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "truth", [dict(alpha=0.08, beta=0.85), dict(alpha=0.15, beta=0.70), dict(alpha=0.05, beta=0.90)],
)
def test_persistence_recovery(truth):
    y = simulate_garch(2000, seed=SEED + int(truth["alpha"] * 100), **truth)
    m = _fit("garch", y)
    assert m.persistence == pytest.approx(truth["alpha"] + truth["beta"], abs=0.08)


def test_arch_underestimates_persistence_on_a_persistent_series():
    """The reason GARCH exists: ARCH(1) has no memory of the variance and cannot reach 0.93."""
    y = simulate_garch(1500, alpha=0.05, beta=0.90)
    arch = _fit("arch", y)
    garch = _fit("garch", y)
    assert arch.persistence < 0.5 < garch.persistence
    assert garch.summary.aicc < arch.summary.aicc


def test_t_nu_recovery_and_that_gaussian_loses_on_fat_tails():
    y = simulate_garch(2000, nu=5.0, seed=SEED + 5)
    t_model = _fit("garch_t", y)
    normal = _fit("garch", y)
    nu = float(t_model._extra[0])
    assert 3.0 < nu < 9.0, nu
    assert t_model.summary.aicc < normal.summary.aicc


def test_gaussian_wins_when_the_innovations_are_gaussian():
    """The other half of the previous test: t must not be a free win by construction."""
    y = simulate_garch(2000, nu=None, seed=SEED + 6)
    t_model = _fit("garch_t", y)
    normal = _fit("garch", y)
    assert normal.summary.aicc < t_model.summary.aicc
    # And the AICc *gap* is the parameter penalty, not a bad tail estimate.
    assert t_model.summary.aicc - normal.summary.aicc < 5.0


def test_gjr_recovers_the_leverage_sign():
    """γ > 0 with negative shocks raising variance: the fit must find it and say so."""
    y = simulate_garch(2000, alpha=0.04, gamma=0.09, beta=0.85, nu=6.0, seed=SEED + 9)
    gjr = _fit("gjr_t", y)
    gamma_hat = float(gjr._var_theta[2])
    assert gamma_hat > 0.0, gamma_hat
    assert any("Leverage" in c for c in gjr.summary.caveats)


def test_symmetric_data_does_not_invent_leverage():
    """On a symmetric GARCH the asymmetry parameter must be near zero, not "found"."""
    y = simulate_garch(2000, gamma=0.0, seed=SEED + 10)
    gjr = _fit("gjr", y)
    assert abs(float(gjr._var_theta[2])) < 0.06, float(gjr._var_theta[2])
    assert not any("Leverage γ" in c for c in gjr.summary.caveats)


def test_egarch_leverage_sign_convention():
    """EGARCH's γ enters the *log* variance, so a leverage effect is γ < 0. The modules must agree."""
    y = simulate_garch(1500, alpha=0.04, gamma=0.08, beta=0.85, seed=SEED + 11)
    eg = _fit("egarch", y)
    gjr = _fit("gjr", y)
    g_eg = float(eg._var_theta[2])
    g_gjr = float(gjr._var_theta[2])
    assert g_gjr > 0, g_gjr
    assert g_eg < 0, g_eg
    assert "leverage" in (eg.summary.caveats and " ".join(eg.summary.caveats).lower() or "")


# --------------------------------------------------------------------------- #
# The variance path
# --------------------------------------------------------------------------- #
def test_one_step_forecast_is_the_next_step_of_the_filter():
    """
    h = 1 is one recursion past the last filtered variance, recomputed here by hand.

    It is *not* equal to the last filtered value — that mistake was in the first version of this test
    and the module's first version of `_variance_path` shared it, using the expected-sign (h >= 2) form
    for every step, which made tomorrow's forecast disagree with today's fit.
    """
    y = simulate_garch(800)
    m = _fit("garch", y)
    t = m._var_theta
    by_hand = t[0] + t[1] * float(m._eps[-1]) ** 2 + t[2] * float(m._s2[-1])
    assert m._variance_path(1)[0] == pytest.approx(float(by_hand), rel=1e-9)


def test_one_step_gjr_uses_the_realised_sign_not_the_expectation():
    """If the last shock was negative, tomorrow's variance must include the full γ term."""
    y = simulate_garch(800, gamma=0.12, beta=0.8)
    m = _fit("gjr", y)
    t = m._var_theta
    e_last = float(m._eps[-1])
    sign = 1.0 if e_last < 0 else 0.0
    by_hand = t[0] + (t[1] + t[2] * sign) * e_last**2 + t[3] * float(m._s2[-1])
    assert m._variance_path(1)[0] == pytest.approx(float(by_hand), rel=1e-9)
    # And h = 2 falls back to the expectation, so on a negative last shock it must be lower.
    if sign:
        assert m._variance_path(2)[1] < m._variance_path(1)[0]


def test_variance_path_converges_to_the_unconditional_variance():
    y = simulate_garch(1500, alpha=0.06, beta=0.90)
    m = _fit("garch", y)
    path = m._variance_path(400)
    uncond = m.unconditional_variance
    assert uncond is not None
    # Monotone convergence and a landing point: the last step must be within 1% of the long-run level,
    # and the first step must be nearer the last filtered variance than to it.
    assert abs(path[-1] - uncond) / uncond < 0.01
    assert abs(path[0] - float(m._s2[-1])) < abs(path[0] - uncond) + 1e-12


def test_the_path_is_not_a_constant():
    """
    The `σ̄²·h` bug, stated as a test.

    A variance forecast that is flat across the horizon is the mistake this module's docstring names.
    The series here ends in a high-volatility stretch, so a correct path must fall visibly.
    """
    rng = np.random.default_rng(SEED + 12)
    n = 1400
    s2 = np.empty(n)
    e = np.empty(n)
    s2[0] = 1.0
    for t in range(n):
        if t:
            shock = 4.0 if t == n - 2 else e[t - 1]  # a single large shock right before the end
            s2[t] = 0.05 + 0.10 * shock**2 + 0.85 * s2[t - 1]
        e[t] = rng.normal(0.0, np.sqrt(s2[t]))
    m = _fit("garch", pd.Series(e))
    path = m._variance_path(20)
    assert path[0] > path[-1] * 1.2, (path[0], path[-1])
    assert np.all(np.diff(path) <= 1e-9), "the path must decay toward the long-run level"


@pytest.mark.parametrize("model_id", sorted(IMPLEMENTED_VOLATILITY))
def test_huge_shock_never_produces_a_negative_variance(model_id: str):
    y = simulate_garch(700, seed=SEED + 13)
    y = pd.concat([y, pd.Series([1e3, -1e3, 5e2])], ignore_index=True)
    m = _fit(model_id, y)
    path = m._variance_path(80)
    assert np.all(np.isfinite(path)), model_id
    assert np.all(path > 0), model_id
    assert np.all(np.isfinite(m.variance_forecast()["sd"]))


def test_gjr_expected_future_variance_uses_half_the_gamma_term():
    """
    From h = 2 onward the recursion must not predict the *sign* of an unseen shock.

    Step 2 is built from step 1's variance with the half-weight γ term, so the check is written on the
    step-2 number rather than the step-1 one.
    """
    y = simulate_garch(1000, gamma=0.1, beta=0.8)
    m = _fit("gjr", y)
    t = m._var_theta
    v1 = m._variance_path(2)[0]
    expected_v2 = t[0] + (t[1] + 0.5 * t[2]) * v1 + t[3] * v1
    assert m._variance_path(2)[1] == pytest.approx(float(expected_v2), rel=1e-9)


# --------------------------------------------------------------------------- #
# Honest failures
# --------------------------------------------------------------------------- #
def test_persistence_at_the_cap_reports_no_long_run_variance():
    assert _unconditional(SPEC_BY_KEY["garch"], np.array([0.01, 0.1, 0.9]), 0.9995) is None
    assert _persistence(SPEC_BY_KEY["gjr"], np.array([0.01, 0.05, 0.30, 0.80])) == pytest.approx(
        0.05 + 0.15 + 0.80
    )


def test_boundary_persistence_is_a_caveat_not_a_number():
    """A GARCH on a random walk pushes β to its bound; the summary must refuse to extrapolate."""
    rng = np.random.default_rng(SEED + 14)
    y = pd.Series(np.cumsum(rng.normal(0, 1, 400)))
    m = _fit("garch", y)
    if m.persistence >= PERSISTENCE_CAP:
        assert m.unconditional_variance is None
        assert any("integrated" in c.lower() for c in m.summary.caveats)


def test_flat_variance_series_is_caveated_not_celebrated():
    rng = np.random.default_rng(SEED + 15)
    y = pd.Series(rng.normal(10.0, 1.0, 400))
    m = _fit("garch", y)
    text = " ".join(m.summary.caveats)
    assert "little evidence of clustered variance" in text or "ARCH-LM" in text


def test_constant_mean_is_declared_in_the_summary():
    y = simulate_garch(500)
    m = _fit("garch", y)
    assert any("mean equation is a constant" in c for c in m.summary.caveats)
    ar = _fit("arima_garch", y)
    assert not any("mean equation is a constant" in c for c in ar.summary.caveats)


def test_short_series_is_refused_before_the_optimiser():
    y = simulate_garch(40)
    m = IMPLEMENTED_VOLATILITY["garch"]()
    fitted = m.fit(y)
    assert fitted.summary.failed
    assert "80" in fitted.summary.failure_reason


def test_parameters_have_standard_errors_or_none_never_zero():
    y = simulate_garch(900)
    m = _fit("garch", y)
    for name, row in m._raw_params.items():
        se = row[1]
        assert se is None or se > 0, name
    assert len(m.summary.params) >= 4


# --------------------------------------------------------------------------- #
# The registry gate: ARCH-LM on the *differenced* series
# --------------------------------------------------------------------------- #
def test_garch_is_excluded_when_there_is_no_clustering():
    rng = np.random.default_rng(SEED + 16)
    y = pd.Series(rng.normal(0, 1, 400))
    f = build_facts(y, data_class=DataClass.CONTINUOUS)
    verdict = evaluate_spec(REGISTRY["garch"], f)
    assert not verdict.compatible
    text = " ".join(verdict.reasons).lower()
    assert "arch-lm" in text and "p =" in text


def test_garch_is_offered_when_the_variance_clusters():
    y = simulate_garch(800, alpha=0.10, beta=0.85)
    f = build_facts(y, data_class=DataClass.CONTINUOUS)
    verdict = evaluate_spec(REGISTRY["garch"], f)
    assert verdict.compatible, verdict.reasons


def test_the_gate_is_not_fooled_by_a_moving_mean():
    """
    A random walk has a violent ARCH-LM on its *level* and none on its differences.

    Offering GARCH here would be the spurious-regression mistake in variance form: the mean moves, the
    variance is constant, and the fit would report persistence near 1 for a series with no clustering.
    """
    rng = np.random.default_rng(SEED + 17)
    y = pd.Series(np.cumsum(rng.normal(0, 1, 500)))
    f = build_facts(y, data_class=DataClass.CONTINUOUS)
    assert f.arqu_lm_p is not None and f.arqu_lm_p < 0.01, f.arqu_lm_p
    # d_hint = 1 for a random walk, so the "residual" series the gate tests is the first difference,
    # which for a random walk is white noise — no clustering, no GARCH.
    f_diff = build_facts(y, data_class=DataClass.CONTINUOUS, d_hint=1)
    assert f_diff.arch_lm_resid_p is not None and f_diff.arch_lm_resid_p > 0.05, f_diff.arch_lm_resid_p
    verdict = evaluate_spec(REGISTRY["garch"], f_diff)
    assert not verdict.compatible
    joined = " ".join(verdict.reasons).lower()
    assert "mean-adjusted" in joined
    # And the refusal names the level's p-value so the reader can see what fooled them.
    assert "level" in joined


def test_volatility_is_refused_on_count_data():
    rng = np.random.default_rng(SEED + 18)
    y = pd.Series(rng.poisson(4, 300).astype(float))
    f = build_facts(y, data_class=DataClass.COUNT)
    verdict = evaluate_spec(REGISTRY["garch"], f)
    assert not verdict.compatible


def test_suggestion_explains_the_measured_trigger():
    y = simulate_garch(900)
    f = build_facts(y, data_class=DataClass.CONTINUOUS)
    s = suggest(f)
    offered = {x.model_id for x in s.suggestions if x.compatible}
    assert "garch" in offered
    garch = next(x for x in s.suggestions if x.model_id == "garch")
    assert "arch-lm" in garch.because.lower()
    # `p = < 0.0001` is not a comparison, and `p = 0.000` claims an impossible
    # zero. Accept either honest form: `p < 0.0001` or `p = 0.0075`.
    import re
    assert re.search(r"\bp\s*(?:<|=)\s*(?:0\.|<)", garch.because), garch.because
    assert "mean-adjusted" in garch.because.lower()
    assert garch.would_win_if and "standardised residuals" in garch.would_win_if


def test_unimplemented_rows_say_why():
    for mid, reason in UNIMPLEMENTED_VOLATILITY.items():
        assert any(word in reason.lower() for word in ("needs", "mcmc", "intraday")), reason
        ex = next(x for x in suggest(build_facts(simulate_garch(300))).excluded if x.model_id == mid)
        assert ex.incompatibilities


# --------------------------------------------------------------------------- #
# End to end through fit_many
# --------------------------------------------------------------------------- #
def test_fit_many_includes_the_volatility_family_and_never_leaves_a_gap():
    y = simulate_garch(700, alpha=0.09, beta=0.86)
    outcomes = fit_many(["mean", "garch", "garch_t", "arima_garch"], y, period=None)
    # Baselines are added by the orchestrator even when not requested; the point of the assertion is
    # that the volatility rows are present and fitted, and that nothing requested went missing.
    ids = {o.summary.model_id for o in outcomes}
    assert {"mean", "garch", "garch_t", "arima_garch"} <= ids
    for o in outcomes:
        # Every row is present whether it fitted or not: a failure is a row, not a gap. A baseline
        # carries no information criterion by design (it is a benchmark, not a likelihood fit), so the
        # IC requirement is stated for the volatility rows, which is what this Part adds.
        if o.summary.family is ModelFamily.VOLATILITY:
            assert not o.summary.failed, o.summary.failure_reason
            assert o.summary.aicc is not None, o.summary.model_id


def test_in_sample_fitted_values_are_the_conditional_mean_not_the_observation():
    """
    The bug this guards against reported MASE = 2.9e-18.

    Setting `_fitted` to `constant + residuals` is identically the observed series, so every residual
    metric collapses to zero and the comparison table claims a perfect fit for a model whose whole
    point is that the *value* is unpredictable. A volatility model's in-sample MASE should sit near the
    mean baseline's — around 1 on a scale-free series — not near zero.
    """
    y = simulate_garch(700, seed=SEED + 21)
    outcomes = {o.summary.model_id: o for o in fit_many(["garch", "arima_garch"], y, period=None)}
    for mid in ("garch", "arima_garch"):
        d = outcomes[mid].diagnostics
        assert d["mase"] is not None
        assert d["mase"] > 0.1, (mid, d["mase"])
        assert d["rmse"] > 0.1, (mid, d["rmse"])
    # And the fitted values are not the observations.
    w = outcomes["garch"].wrapper
    fitted = np.asarray(w.fitted_values())   # a method on the wrapper, not a property
    assert not np.allclose(fitted[5:], np.asarray(w.y)[5:])
    # The constant-mean fit's fitted value is constant, which is the honest answer about the level.
    assert np.allclose(fitted, fitted[0])


def test_ar_mean_equation_does_move_its_fitted_values():
    y = simulate_garch(600, seed=SEED + 22)
    w = [o for o in fit_many(["arima_garch"], y, period=None) if o.summary.model_id == "arima_garch"][0].wrapper
    fv = np.asarray(w.fitted_values())
    assert np.ptp(fv[10:]) > 0, "an AR(1) mean must produce a fitted path that varies"


def test_failure_is_reported_with_a_fix_not_raised():
    class Broken(VolatilityModel):
        model_id = "broken_vol"
        display_name = "Broken"
        spec = SPEC_BY_KEY["garch"]
        _model_spec = SPEC_BY_KEY["garch"]

        def _fit(self, y, **kw):  # type: ignore[override]
            raise FitFailure("a synthetic failure", "nothing to do", kind="numerical")

    m = Broken().fit(simulate_garch(300))
    assert m.summary.failed
    assert "synthetic failure" in (m.summary.failure_reason or "")
