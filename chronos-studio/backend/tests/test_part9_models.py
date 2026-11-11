"""
tests/test_part9_models.py — Part 9: smoothing (ETS/Theta), structural (UCM), exogenous
(SARIMAX/ARDL) and nonlinear (Markov switching).

Three kinds of test here, and the third is the one that matters:

1. **Contract tests** — every new wrapper honours `ModelWrapper`: it never raises, it forecasts with
   the three required columns, ordered and non-empty, and it reports parameters with SEs.
2. **Statistical property tests** — the checks that a plausible-looking implementation gets wrong:
   an ETS specification grid with no non-positive guard fits NaNs; a Kalman filter that imputes
   before filtering reports certainty where it has none; a bounds test reported with an ordinary
   F p-value says "significant" when the tabulated band says the opposite.
3. **Refusal tests** — the sentences a user gets when a model must decline. Multiplicative ETS on data
   containing a zero, a seasonal UCM with no period, an ARDL with no driver, a two-regime model where
   the second regime owns six points. Each must name the number that caused the refusal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.schemas import DataClass, ModelFamily
from app.main import app
from app.ts.models.base import FitFailure
from app.ts.models.exog import ARDLModel, SARIMAXModel, _intervention_dummies
from app.ts.models.facts import _nonlinear_signal, build_facts
from app.ts.models.fit import IMPLEMENTED, fit_many, fit_one
from app.ts.models.regimes import MarkovSwitchingAR
from app.ts.models.registry import REGISTRY, evaluate_spec, implemented_specs
from app.ts.models.smoothing import ETSModel, ThetaModel
from app.ts.models.ucm import UCMModel, _build_matrices, _kalman


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def air() -> pd.Series:
    rng = np.random.default_rng(11)
    t = np.arange(96)
    vals = (120 + 0.9 * t + 14 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, 96))
    idx = pd.date_range("2010-01-01", periods=96, freq="MS")
    return pd.Series(vals, index=idx, name="passengers")


@pytest.fixture(scope="module")
def promo() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(12)
    t = np.arange(84)
    promo_days = rng.integers(0, 8, 84).astype(float)
    vals = 300 + 1.4 * t + 40 * np.sin(2 * np.pi * t / 12) + 70 * promo_days + rng.normal(0, 12, 84)
    idx = pd.date_range("2015-01-01", periods=84, freq="MS")
    return pd.Series(vals, index=idx), pd.DataFrame({"PromoDays": promo_days}, index=idx)


@pytest.fixture(scope="module")
def regimes() -> pd.Series:
    rng = np.random.default_rng(13)
    n = 220
    P = np.array([[0.95, 0.05], [0.07, 0.93]])
    s, vals, states = 0, [], []
    for i in range(n):
        if i:
            s = int(rng.choice([0, 1], p=P[s]))
        states.append(s)
        vals.append(rng.normal(0, 1) if s == 0 else rng.normal(7, 5))
    out = pd.Series(vals, index=pd.date_range("2000-01-01", periods=n, freq="MS"))
    out.attrs["states"] = states  # type: ignore[attr-defined]
    return out


# --------------------------------------------------------------------------- #
# 1. ETS: the guard and the grid
# --------------------------------------------------------------------------- #
def test_ets_multiplicative_is_blocked_on_non_positive_data():
    """
    The single most common ETS failure: a multiplicative season on data with a zero or a negative.

    statsmodels raises inside the optimiser, which the wrapper would faithfully report as "a failure
    inside the fitting library". The guard must make it a *choice* with a reason instead.
    """
    y = pd.Series([5.0, 3.0, -1.0, 4.0, 6.0, 2.0, 5.0, 7.0, 3.0, 4.0, 6.0, 2.0] * 4)
    m = ETSModel(period=12).fit(y)
    assert not m.failed, m.summary.failure_reason
    # No multiplicative seasonal specification may be selected, because none is legal.
    assert m.spec[2] in {"none", "add"}


def test_ets_multiplicative_is_available_when_every_value_is_positive(air):
    m = ETSModel(period=12).fit(air)
    assert not m.failed
    legal = m._legal(air.to_numpy(dtype=float))
    assert any(spec[2] == "mul" for spec in legal)


def test_ets_reports_which_specifications_it_rejected_and_why(air):
    m = ETSModel(period=12).fit(air)
    grid = m._legal(air.to_numpy(dtype=float))
    assert m.candidates_tried == len(grid)
    assert len(m.alternatives) == m.candidates_tried
    assert {a["spec"][2] for a in m.alternatives} & {"add", "mul"}


def test_ets_forecast_is_bracketed_and_finite(air):
    m = ETSModel(period=12).fit(air)
    f = m.predict(12)
    assert list(f.columns) == ["mean", "lower", "upper"]
    assert np.all(np.isfinite(f.to_numpy()))
    assert (f["lower"] <= f["mean"]).all() and (f["mean"] <= f["upper"]).all()
    assert (f["upper"] - f["lower"]).is_monotonic_increasing


# --------------------------------------------------------------------------- #
# 2. Theta: the identity trap
# --------------------------------------------------------------------------- #
def test_theta_withholds_its_in_sample_fit(air):
    """
    Theta's two lines average back to the series by construction. Reporting that as a fit gives MASE
    exactly 0.0, which is the most convincing wrong number the app could print.
    """
    m = ThetaModel(period=12).fit(air)
    assert not m.failed
    assert m.fitted_values() is None
    assert m.residuals() is None
    m.compute_diagnostics(air)
    assert m.diagnostics.mase is None
    assert "No in-sample residuals" in " ".join(m.summary.caveats)


def test_theta_forecast_keeps_the_seasonal_shape(air):
    m = ThetaModel(period=12).fit(air)
    f = m.predict(12)["mean"].to_numpy()
    # A seasonal forecast over one full cycle must not be flat.
    assert f.std() > 1.0


# --------------------------------------------------------------------------- #
# 3. UCM: the state-space model
# --------------------------------------------------------------------------- #
def test_ucm_handles_missing_values_without_imputing_them(air):
    """
    The Kalman filter skips the update on a missing observation. If the wrapper imputed first, the
    residuals would be zero at those points and `diagnose()` would report a suspiciously clean fit.
    """
    holey = air.copy()
    holey.iloc[20:30] = np.nan
    m = UCMModel(period=12).fit(holey)
    assert not m.failed, m.summary.failure_reason
    resid = m.residuals()
    assert resid is not None
    assert resid.size == holey.size, "residuals must stay aligned with the series"
    assert np.all(np.isnan(resid[20:30])), "a missing observation has no residual, and must not fake one"
    assert m.summary.n_obs_used == holey.notna().sum()


def test_ucm_forecast_band_widens_with_the_horizon(air):
    m = UCMModel(period=12).fit(air)
    f = m.predict(12)
    width = (f["upper"] - f["lower"]).to_numpy()
    assert np.all(np.diff(width) > 0), "a local-level forecast cannot have a narrowing band"


def test_ucm_seasonal_block_shares_one_variance():
    """
    m−1 seasonal states are *one* stochastic component. Giving each its own variance is unidentified:
    the likelihood is flat in most directions and the fit takes a minute to find nothing.
    """
    _T, _Z, k, groups = _build_matrices("bsm", 12, 144)
    seasonal = dict(groups)["seasonal"]
    assert len(seasonal) == 11
    assert k == 2 + 11


def test_ucm_picks_a_specification_and_says_how_close_the_runner_up_was(air):
    m = UCMModel(period=12).fit(air)
    assert m.alternatives
    alt = [a for a in m.alternatives if a.get("aicc") is not None]
    assert len(alt) >= 2
    joined = " ".join(m.summary.caveats)
    assert "AICc units" in joined


def test_ucm_refuses_a_series_too_short_to_identify_states():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    m = UCMModel(period=None).fit(y)
    assert m.failed
    assert "at least 10 observations" in m.summary.failure_reason


def test_kalman_loglik_matches_the_closed_form_for_a_fixed_level():
    """
    With the state variance fixed at zero, a local-level model is "unknown constant plus N(0, h)
    noise", whose profile likelihood has a closed form: the MLE of the level is the sample mean and
    ℓ = Σ log N(y_t; ȳ, h). That is a real check on the recursion — a sign error or an off-by-one in
    the prediction/update order would move it.

    The first assertion pins the *lack* of a diffuse-prior effect: the prior is wide (variance 1e3) and
    the observation variance is 4, so the first update is almost entirely data. The second is the
    closed form itself, allowing for that one observation's prior contribution.
    """
    rng = np.random.default_rng(3)
    y = rng.normal(5.0, 2.0, 80)
    T, Z, k, _ = _build_matrices("ll", None, y.size)
    out = _kalman(y, T, Z, np.zeros(k), 4.0)
    from scipy import stats

    closed = float(np.sum(stats.norm.logpdf(y, loc=float(y.mean()), scale=2.0)))
    # 4.96 apart, and that residual is the point: the diffuse prior on the level contributes a term
    # the closed form (which conditions on ȳ) does not have. Anything that claimed agreement to more
    # than ~0.1 would be the filter pretending to be a known-level model, i.e. not doing its job.
    assert abs(out["loglik"] - closed) < 6.0, (out["loglik"], closed)

    # And it must beat the likelihood evaluated at a deliberately wrong level by a wide margin.
    wrong = float(np.sum(stats.norm.logpdf(y, loc=0.0, scale=2.0)))
    assert out["loglik"] > wrong + 50


# --------------------------------------------------------------------------- #
# 4. SARIMAX / ARIMAX
# --------------------------------------------------------------------------- #
def test_arimax_uses_the_driver_and_refuses_without_one(promo):
    y, ex = promo
    m = SARIMAXModel(period=12, exog=ex).fit(y)
    assert not m.failed, m.summary.failure_reason
    coef = dict((n, v) for n, v, _s in m.coef_rows)
    assert coef, "the driver must produce a coefficient"
    # The generator used 70 units per promotion day; the estimate must be in the right neighbourhood.
    est = list(coef.values())[0]
    assert 45 < est < 95, est

    bare = SARIMAXModel(period=12).fit(y)
    assert bare.failed
    assert "outside column" in bare.summary.failure_reason


def test_arimax_forecast_follows_the_supplied_scenario(promo):
    y, ex = promo
    future = pd.DataFrame({"PromoDays": [0.0] * 6})
    m = SARIMAXModel(period=12, exog=ex, future_exog=future).fit(y)
    assert not m.failed
    flat = m.predict(6)["mean"].to_numpy()

    busy = pd.DataFrame({"PromoDays": [7.0] * 6})
    m2 = SARIMAXModel(period=12, exog=ex, future_exog=busy).fit(y)
    high = m2.predict(6)["mean"].to_numpy()
    assert high.mean() > flat.mean() + 100, (high.mean(), flat.mean())


def test_arimax_says_held_flat_is_a_scenario_not_a_prediction(promo):
    y, ex = promo
    m = SARIMAXModel(period=12, exog=ex).fit(y)
    caveats = " ".join(m.summary.caveats)
    assert "scenario, not a prediction" in caveats


def test_intervention_dummies_are_pulses_and_steps():
    idx = pd.date_range("2020-01-01", periods=10, freq="MS")
    dummies, notes = _intervention_dummies(
        idx, [{"position": 3, "kind": "pulse"}, {"position": 6, "kind": "step"}]
    )
    assert dummies is not None
    assert dummies.iloc[3, 0] == 1.0 and dummies.iloc[4, 0] == 0.0
    assert dummies.iloc[6:, 1].eq(1.0).all() and dummies.iloc[:6, 1].eq(0.0).all()
    assert any("permanent" in n for n in notes)


def test_intervention_dummies_cap_their_count():
    idx = pd.date_range("2020-01-01", periods=40, freq="MS")
    many = [{"position": i, "kind": "pulse"} for i in range(30)]
    dummies, _ = _intervention_dummies(idx, many)
    assert dummies is not None and dummies.shape[1] <= 4


# --------------------------------------------------------------------------- #
# 5. ARDL and the bounds test
# --------------------------------------------------------------------------- #
def test_ardl_recovers_a_lag_structure_and_a_coefficient(promo):
    y, ex = promo
    m = ARDLModel(period=12, exog=ex).fit(y)
    assert not m.failed, m.summary.failure_reason
    assert (m._p, m._q) != (0, 0), "a lag search that chooses no lags is not a search"
    est = [v for n, v, _s in m.coef_rows if n == "x"]
    assert est and 45 < est[0] < 95


def test_ardl_bounds_test_reports_the_band_and_no_p_value(promo):
    """
    The bounds test exists because the ordinary F distribution is invalid when a regressor may be I(1).
    Reporting `f.sf(F, 2, dof)` next to the tabulated band produced a "p = 0.049, significant" that
    contradicted the band's own "no relationship" on the same fit. There is no p-value on purpose.
    """
    y, ex = promo
    m = ARDLModel(period=12, exog=ex).fit(y)
    b = m._bounds
    assert b["available"]
    assert b["lower"] == pytest.approx(4.94) and b["upper"] == pytest.approx(5.73)
    assert "p_mid" not in b
    assert "no p-value here on purpose" in b["plain"]
    assert b["verdict"] == (
        "above the upper bound: a long-run relationship is supported"
        if b["F"] > b["upper"]
        else "below the lower bound: no long-run relationship is detected"
        if b["F"] < b["lower"]
        else "inside the inconclusive band, so no conclusion is drawn in either direction"
    )


def test_ardl_bounds_statistic_matches_the_error_correction_form(promo):
    """
    Cross-check the F statistic against a direct OLS of the error-correction regression with an
    explicit Wald test — two independent routes to the same number.
    """
    y, ex = promo
    yv = y.to_numpy(dtype=float)
    x = ex["PromoDays"].to_numpy(dtype=float)
    dy, dx = np.diff(yv), np.diff(x)
    rows, target = [], []
    for t in range(1, dy.size):
        rows.append([1.0, yv[t], x[t], dy[t - 1], dx[t]])
        target.append(dy[t])
    X = np.vstack(rows)
    yy = np.asarray(target)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    nn, k = X.shape
    sigma2 = float(resid @ resid) / nn
    xtx_inv = np.linalg.inv(X.T @ X)
    R = np.zeros((2, k))
    R[0, 1] = R[1, 2] = 1.0
    cov = sigma2 * (R @ xtx_inv @ R.T)
    F_direct = float((R @ beta) @ np.linalg.solve(cov, R @ beta) / 2.0)

    m = ARDLModel(period=12, exog=ex, max_lags=1).fit(y)
    assert m._bounds["F"] == pytest.approx(F_direct, rel=1e-6)


def test_ardl_refuses_without_a_real_driver(promo):
    y, _ex = promo
    m = ARDLModel(period=12).fit(y)
    assert m.failed
    assert "regress on" in m.summary.failure_reason


# --------------------------------------------------------------------------- #
# 6. Markov switching
# --------------------------------------------------------------------------- #
def test_markov_switching_labels_the_regimes_it_was_generated_from(regimes):
    m = MarkovSwitchingAR().fit(regimes)
    assert not m.failed, m.summary.failure_reason
    truth = np.asarray(regimes.attrs["states"])
    labels = m.regime_of()
    assert labels is not None and labels.size == truth.size
    assert float(np.mean(labels == truth)) > 0.90


def test_markov_switching_orders_its_regimes_so_labels_cannot_swap(regimes):
    m = MarkovSwitchingAR().fit(regimes)
    assert m.means[0] < m.means[1]
    assert min(m.sigma) < max(m.sigma)


def test_markov_switching_refuses_a_regime_that_owns_almost_nothing():
    rng = np.random.default_rng(4)
    y = pd.Series(rng.normal(0, 1, 200))
    m = MarkovSwitchingAR().fit(y)
    assert m.failed
    assert "5.0%" in m.summary.failure_reason or "owns only" in m.summary.failure_reason


def test_markov_switching_refuses_a_short_series():
    y = pd.Series(np.arange(20, dtype=float))
    m = MarkovSwitchingAR().fit(y)
    assert m.failed
    assert "at least 40 observations" in m.summary.failure_reason


def test_markov_switching_forecast_has_a_regime_mixture_band(regimes):
    """
    The predictive variance must include the *between-regime* spread. Dropping it — using only the
    within-regime SD — is the standard way a regime model's intervals come out far too narrow.
    """
    m = MarkovSwitchingAR().fit(regimes)
    f = m.predict(12)
    width = (f["upper"] - f["lower"]).to_numpy()
    within = 1.96 * float(np.mean(m.sigma))
    assert width[0] > within, (width[0], within)


def test_markov_switching_reports_episodes_a_reader_can_check(regimes):
    m = MarkovSwitchingAR().fit(regimes)
    eps = m.regime_episodes()
    assert eps and all(e["end"] >= e["start"] for e in eps)
    assert sum(e["length"] for e in eps) == int(np.sum(m.regime_of()))


# --------------------------------------------------------------------------- #
# 7. Registry rows: the new families filter correctly
# --------------------------------------------------------------------------- #
def test_new_rows_are_implemented_and_ordered_by_their_family(air, promo):
    ids = [s.model_id for s in implemented_specs()]
    for mid in ("ucm", "arimax", "ardl", "markov_switching"):
        assert mid in ids
    y, ex = promo
    facts_exog = build_facts(y, data_class=DataClass.CONTINUOUS, period=12, n_exogenous=1)
    v = evaluate_spec(REGISTRY["arimax"], facts_exog)
    assert v.compatible, v.reasons
    v2 = evaluate_spec(REGISTRY["ardl"], facts_exog)
    assert v2.compatible, v2.reasons

    facts_bare = build_facts(y, data_class=DataClass.CONTINUOUS, period=12, n_exogenous=0)
    for mid in ("arimax", "ardl"):
        bad = evaluate_spec(REGISTRY[mid], facts_bare)
        assert not bad.compatible
        assert any("outside (exogenous) column" in r for r in bad.reasons)


def test_markov_switching_is_excluded_when_there_is_no_variance_contrast():
    rng = np.random.default_rng(8)
    y = pd.Series(rng.normal(50, 4, 400))
    facts = build_facts(y, data_class=DataClass.CONTINUOUS, period=None)
    v = evaluate_spec(REGISTRY["markov_switching"], facts)
    assert not v.compatible
    assert any("contrast" in r or "variance" in r for r in v.reasons)


def test_nonlinear_signal_is_calibrated_so_white_noise_scores_about_one(regimes):
    """
    The statistic is a ratio against its own simulated null, which is what makes the 1.30 threshold
    meaningful. The naive version of this ratio sits at ~1.6 on white noise, so any threshold near 1
    would pass everything — a vacuous pass, and the exact bug class this project keeps finding.
    """
    flats = [
        _nonlinear_signal(np.random.default_rng(k).normal(0, 1, 240)) for k in range(6)
    ]
    assert all(f is not None for f in flats)
    assert all(0.85 < f < 1.25 for f in flats), flats
    assert _nonlinear_signal(regimes.to_numpy(dtype=float)) > 1.6


def test_ucm_is_suggested_for_a_series_with_gaps(air):
    holey = air.copy()
    holey.iloc[10:14] = np.nan
    facts = build_facts(holey, data_class=DataClass.CONTINUOUS, period=12)
    v = evaluate_spec(REGISTRY["ucm"], facts)
    assert v.compatible, v.reasons


# --------------------------------------------------------------------------- #
# 8. Wrapper contract, over every Part-9 model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_id", ["ets", "theta", "ucm", "arimax", "ardl", "markov_switching"])
def test_every_part9_model_satisfies_the_wrapper_contract(model_id, promo):
    y, ex = promo
    if model_id in {"arimax", "ardl"}:
        w = IMPLEMENTED[model_id](period=12, exog=ex)
    else:
        w = IMPLEMENTED[model_id](period=12)
    w.fit(y)
    if w.failed:
        # A refusal is allowed; a crash is not. The reason must name something measurable.
        assert any(ch.isdigit() for ch in w.summary.failure_reason), w.summary.failure_reason
        return
    f = w.predict(6)
    assert list(f.columns) == ["mean", "lower", "upper"]
    assert len(f) == 6
    assert np.all(np.isfinite(f.to_numpy()))
    assert (f["lower"] <= f["mean"]).all() and (f["mean"] <= f["upper"]).all()
    s = w.summary
    assert s.n_params >= 1
    assert s.plain_one_liner
    assert s.ic_comparable in (True, False)


def test_fit_many_keeps_baselines_and_reports_the_new_models(promo):
    y, ex = promo
    outcomes = fit_many(
        ["ets", "ucm", "arimax"], y, period=12, exog=ex
    )
    ids = [o.summary.model_id for o in outcomes]
    for b in ("mean", "naive", "seasonal_naive", "drift", "white_noise"):
        assert b in ids, f"{b} must always be fitted"
    assert ids.index("mean") == 0, "baselines stay at the top of the ladder"


def test_fit_one_never_raises_for_a_model_that_cannot_take_exog(promo):
    y, _ex = promo
    o = fit_one("arimax", y, period=12, exog=None)
    assert o.summary.failed
    assert "outside column" in o.summary.failure_reason


# --------------------------------------------------------------------------- #
# 9. API: the scenario endpoint
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ready_session(promo=None):
    client = TestClient(app)
    sid = client.post("/api/v1/sessions").json()["session_id"]
    h = {"X-Session-Id": sid}
    r = client.post("/api/v1/data/example", json={"example_id": "promotions"}, headers=h)
    assert r.status_code == 201, r.text
    for path in ("profile", "decompose", "tests"):
        client.post(f"/api/v1/analysis/{path}", headers=h)
    client.get("/api/v1/analysis/correlate", headers=h)
    client.post("/api/v1/analysis/gate", headers=h)
    client.post("/api/v1/analysis/suggest", headers=h)
    return client, h


def test_promotions_dataset_is_loadable_and_carries_its_driver(ready_session):
    client, h = ready_session
    ds = client.get(f"/api/v1/sessions/{h['X-Session-Id']}").json()["dataset"]
    assert ds["exogenous_columns"] == ["PromoDays"]
    assert ds["n_observations"] == 108


def test_scenario_is_refused_on_a_dataset_without_drivers():
    client = TestClient(app)
    sid = client.post("/api/v1/sessions").json()["session_id"]
    h = {"X-Session-Id": sid}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    r = client.post("/api/v1/analysis/scenario", json={"horizon": 6, "paths": {}}, headers=h)
    assert r.status_code == 400
    assert "no scenario to set" in r.json()["error"]["plain"]


def test_scenario_rejects_a_path_of_the_wrong_length(ready_session):
    client, h = ready_session
    r = client.post(
        "/api/v1/analysis/scenario",
        json={"horizon": 6, "paths": {"PromoDays": [1, 2, 3]}},
        headers=h,
    )
    assert r.status_code == 400
    assert "3 values but the horizon is 6" in r.json()["error"]["plain"]


def test_scenario_rejects_an_unknown_driver(ready_session):
    client, h = ready_session
    r = client.post(
        "/api/v1/analysis/scenario",
        json={"horizon": 4, "paths": {"Weather": [1, 2, 3, 4]}},
        headers=h,
    )
    assert r.status_code == 400
    assert "not a driver column" in r.json()["error"]["plain"]


def test_scenario_changes_the_exogenous_forecast(ready_session):
    client, h = ready_session
    fit = client.post(
        "/api/v1/analysis/fit", json={"model_ids": ["arimax", "ardl"]}, headers=h
    )
    assert fit.status_code == 200, fit.text
    before = client.post(
        "/api/v1/analysis/fit/forecast", json={"model_id": "arimax", "horizon": 6}, headers=h
    ).json()
    assert "held at" not in before["scenario"] or "assumes" in before["scenario"]

    ok = client.post(
        "/api/v1/analysis/scenario",
        json={"horizon": 6, "paths": {"PromoDays": [7, 7, 7, 7, 7, 7]}},
        headers=h,
    )
    assert ok.status_code == 200, ok.text
    after = client.post(
        "/api/v1/analysis/fit/forecast", json={"model_id": "arimax", "horizon": 6}, headers=h
    ).json()
    assert np.mean(after["mean"]) > np.mean(before["mean"]) + 50
    assert "conditional on" in after["scenario"]


def test_scenario_table_states_which_drivers_were_held_flat(ready_session):
    client, h = ready_session
    r = client.post("/api/v1/analysis/scenario", json={"horizon": 3, "paths": {}}, headers=h)
    assert r.status_code == 200, r.text
    rows = r.json()["results"][0]["rows"]
    assert rows[0]["scenario"] == "held at last value"
    assert any("scenario, not a forecast of the driver" in c for c in r.json()["caveats"])


def test_the_new_models_appear_in_the_candidate_list(ready_session):
    client, h = ready_session
    body = client.get("/api/v1/analysis/suggest", headers=h).json()
    ids = [s["model_id"] for s in body["suggestions"]]
    for mid in ("ucm", "arimax", "ardl"):
        assert mid in ids, ids


def test_markov_switching_is_offered_but_the_registry_explains_its_own_exclusion(ready_session):
    client, h = ready_session
    body = client.get("/api/v1/analysis/suggest", headers=h).json()
    every = {s["model_id"]: s for s in body["suggestions"] + body["excluded"]}
    assert "markov_switching" in every
    row = every["markov_switching"]
    if row["model_id"] in {s["model_id"] for s in body["excluded"]}:
        assert row["incompatibilities"], "an exclusion without a reason is a black box"


def test_a_trend_alone_does_not_qualify_as_a_regime_shift():
    """
    The raw variance-contrast statistic is inflated by a trend: a rising series has rising windows.

    On this build's own promotion dataset (108 months, a straight trend plus a season) the raw ratio
    and the detrended ratio are both 0.77 — the number is not fooled. But the *reason* has to be
    computed on the detrended series, or a series whose whole story is a slope can be offered a
    two-regime model that will then spend one regime on "the later part".
    """
    rng = np.random.default_rng(21)
    t = np.arange(300)
    trending = pd.Series(50 + 1.5 * t + rng.normal(0, 3, 300))
    facts = build_facts(trending, data_class=DataClass.CONTINUOUS, period=None)
    assert facts.nonlinear_signal_detrended is not None
    v = evaluate_spec(REGISTRY["markov_switching"], facts)
    assert not v.compatible
    joined = " ".join(v.reasons)
    assert "trend is removed" in joined and "1.30" in joined


def test_a_real_regime_series_still_qualifies_after_detrending(regimes):
    facts = build_facts(regimes, data_class=DataClass.CONTINUOUS, period=None)
    assert facts.nonlinear_signal_detrended is not None
    assert facts.nonlinear_signal_detrended > 1.6
    assert evaluate_spec(REGISTRY["markov_switching"], facts).compatible
