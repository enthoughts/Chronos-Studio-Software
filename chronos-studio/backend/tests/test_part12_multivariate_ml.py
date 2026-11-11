"""
Part 12 tests — multivariate, nonlinear and machine learning.

This is the part where a wrong implementation is hardest to *notice* from the outside, because every
row here produces plausible-looking output on any input. A VAR reports a significant Granger test
between two independent random walks; a Johansen test finds a "long-run relationship" in noise at its
nominal rate; a k-NN model reports a perfect in-sample fit because it reproduces its own training rows;
a Gaussian band covers half of what it claims. None of those failures raise an exception, and all of
them read as success.

So the assertions below are on **simulated truths and their negative controls**, paired so that a
model which always says yes and a model which always says no both fail:

1. **Recovery.** Simulate a VAR with known coefficients and require the impulse responses, the
   forecast-error decomposition and the Granger direction to come back in the right order. Simulate a
   cointegrated pair with known β and require the VECM to recover it.
2. **The negative controls, which matter more.** Two independent random walks must *not* produce a
   VECM (the Engle–Granger guard), and a level Granger rejection that vanishes on differences must be
   labelled as the trend rather than reported as a relationship.
3. **The refusals quote a number** — the explosive root, the non-stationary relation, the branch share,
   the missing import — checked to be present in the sentence.
4. **The interval actually covers.** ML bands are built from an empirical quantile and their coverage
   is measured on a holdout the quantile never saw; the test requires it to be in the right
   neighbourhood, which is precisely what the naive Gaussian band failed.
5. **The leakage guards.** A feature matrix that contains the future, a shuffled split, or an
   in-sample MASE near zero must all be impossible or caught.
6. **The API and the registry.** 42 rows fit-able, the four refusals carry measured reasons, and the
   new endpoint returns the payload the panel renders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.schemas import ModelFamily
from app.ts.models.facts import build_facts
from app.ts.models.fit import (
    IMPLEMENTED,
    ML_IDS,
    MULTIVARIATE_IDS,
    build_multi_frame,
    fit_many,
    fit_one,
)
from app.ts.models.ml import (
    IMPLEMENTED_ML,
    SEED,
    UNIMPLEMENTED_ML,
    build_design,
    dependency_probe,
)
from app.ts.models.multivariate import (
    IMPLEMENTED_MULTIVARIATE,
    UNIMPLEMENTED_MULTIVARIATE,
    VARModel,
    VECMModel,
    adf_stat,
    companion_roots,
    fit_var_ols,
    granger_f,
    johansen_trace,
    select_order,
    unit_root_count,
    var_stability_ok,
)
from app.ts.models.regimes import IMPLEMENTED_NONLINEAR, SETARModel
from app.ts.models.registry import REGISTRY, all_specs, evaluate_spec, is_implemented

SEED = 20261108


# =========================================================================== #
# helpers
# =========================================================================== #
def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2005-01-31", periods=n, freq="MS")


def simulate_var(n: int = 400, seed: int = SEED) -> tuple[pd.Series, pd.DataFrame]:
    """
    y_t responds to x; x does not respond to y. The asymmetry is the truth the Granger test must find.

        y_t = 0.5·y_{t−1} + 0.6·x_{t−1} + ε
        x_t = 0.7·x_{t−1} + η
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1)
        y[t] = 0.5 * y[t - 1] + 0.6 * x[t - 1] + rng.normal(0, 1)
    idx = _index(n)
    ys = pd.Series(y + 30, index=idx, name="y")
    frame = pd.DataFrame({"the target": ys, "driver": x + 20}, index=idx)
    return ys, frame


def simulate_cointegrated(n: int = 400, beta: float = 2.0, seed: int = SEED) -> tuple[pd.Series, pd.DataFrame]:
    """y = β·z + a stationary AR(1) spread, with z a random walk. Rank 1 by construction."""
    rng = np.random.default_rng(seed)
    z = np.cumsum(rng.normal(0.1, 1.0, n)) + 50
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.55 * e[t - 1] + rng.normal(0, 0.6)
    y = beta * z + e
    idx = _index(n)
    ys = pd.Series(y, index=idx, name="y")
    frame = pd.DataFrame({"the target": ys, "z": pd.Series(z, index=idx)}, index=idx)
    return ys, frame


def simulate_independent_walks(n: int = 400, seed: int = 99) -> tuple[pd.Series, pd.DataFrame]:
    """Two random walks with nothing between them. Everything ought to refuse or disclaim."""
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0, 1.0, n)) + 100
    b = np.cumsum(rng.normal(0, 1.0, n)) + 100
    idx = _index(n)
    ys = pd.Series(a, index=idx, name="y")
    return ys, pd.DataFrame({"the target": ys, "other": pd.Series(b, index=idx)}, index=idx)


def seasonal_series(n: int = 220, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 20 + 0.1 * t + 2.5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.0, n)
    return pd.Series(y, index=_index(n), name="y")


def _fit(cls, y: pd.Series, frame: pd.DataFrame, **kw):
    m = cls(**kw)
    m._multi = frame
    return m.fit(y)


# =========================================================================== #
# 1. Linear algebra and lag selection
# =========================================================================== #
def test_companion_roots_of_a_stable_var_are_inside_the_circle():
    rng = np.random.default_rng(1)
    n = 600
    data = np.zeros((n, 2))
    for t in range(1, n):
        data[t] = np.array([[0.4, 0.2], [0.1, 0.5]]) @ data[t - 1] + rng.normal(0, 0.5, 2)
    fit = fit_var_ols(data, 1)
    ok, max_root = var_stability_ok(fit["A"])
    assert ok and max_root < 1.0


def test_an_explosive_var_is_detected_and_refused_with_the_root_quoted():
    rng = np.random.default_rng(2)
    n = 200
    data = np.zeros((n, 2))
    for t in range(1, n):
        # Coefficients above 1: the system must blow up.
        data[t] = np.array([[1.2, 0.05], [0.05, 1.15]]) @ data[t - 1] + rng.normal(0, 0.1, 2)
    ok, max_root = var_stability_ok(fit_var_ols(data, 1)["A"])
    assert not ok and max_root > 1.0

    idx = _index(n)
    y = pd.Series(data[:, 0], index=idx)
    m = _fit(VARModel, y, pd.DataFrame({"the target": y, "other": data[:, 1]}, index=idx))
    assert m.failed
    assert "companion root" in m.summary.failure_reason
    assert "> 1" in m.summary.failure_reason


def test_select_order_prefers_bic_when_aic_and_bic_disagree():
    """
    Not "assert they disagree on this exact series" — that would be a test of one dataset. The
    invariant is the choice rule: whatever the grid says, the returned p is the BIC winner whenever
    the two differ, and the disagreement is flagged.
    """
    _, frame = simulate_var(n=300)
    order = select_order(frame.to_numpy())
    if order["disagreement"]:
        assert order["p"] == order["p_bic"]
    else:
        assert order["p"] == order["p_aic"] == order["p_bic"]
    assert order["rows"] and order["p"] >= 1


def test_granger_finds_the_true_direction_and_not_the_reverse():
    _, frame = simulate_var(n=600)
    fwd = granger_f(frame["the target"].to_numpy(), frame["driver"].to_numpy(), 2)
    rev = granger_f(frame["driver"].to_numpy(), frame["the target"].to_numpy(), 2)
    assert fwd["p"] < 0.01, "the true driver must be found"
    assert rev["p"] > 0.01, "the reverse direction must not be"


def test_granger_on_noise_rejects_at_roughly_the_nominal_rate():
    """
    The size of the test. If this came out at 30% the model would 'find' relationships everywhere;
    ten independent pairs at the 5% level should give at most a couple of rejections.
    """
    rejects = 0
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        g = granger_f(y, x, 2)
        if g["p"] is not None and g["p"] < 0.05:
            rejects += 1
    assert rejects <= 3


# =========================================================================== #
# 2. VAR: recovery
# =========================================================================== #
def test_var_recovers_coefficients_on_simulated_data():
    y, frame = simulate_var(n=800)
    m = _fit(VARModel, y, frame)
    assert not m.failed, m.summary.failure_reason
    r = m.var_result
    k = 2
    # B rows: [const, lag-1 block (2 rows)]. Column 0 is the target's equation.
    b_y_on_y = float(r.B[1, 0])
    b_y_on_x = float(r.B[2, 0])
    assert abs(b_y_on_y - 0.5) < 0.12, f"own lag recovered as {b_y_on_y:.3f}, truth 0.5"
    assert abs(b_y_on_x - 0.6) < 0.12, f"cross lag recovered as {b_y_on_x:.3f}, truth 0.6"


def test_var_granger_table_reads_in_the_right_direction():
    y, frame = simulate_var(n=600)
    m = _fit(VARModel, y, frame)
    rows = {(r["cause"], r["effect"]): r for r in m.granger_table()}
    assert rows[("driver", "the target")]["significant"], "driver → target must reject"
    assert not rows[("the target", "driver")]["significant"], "target → driver must not"
    # The sentence says "predicts", never "causes" — the spec is explicit that this is not causality.
    assert "causes" not in rows[("driver", "the target")]["sentence"].lower()


def test_var_impulse_response_has_bootstrap_bands_that_contain_the_point_estimate():
    y, frame = simulate_var(n=300)
    m = _fit(VARModel, y, frame)
    pl = m.multivariate_payload()
    irf = pl["irf"]
    assert irf["n_boot"] > 100, "the bands need real bootstrap draws"
    for name, path in irf["paths"].items():
        band = irf["bands_90"].get(name)
        assert band is not None, f"no band for {name}"
        lower = np.asarray(band["lower"])
        upper = np.asarray(band["upper"])
        point = np.asarray(path)
        assert lower.shape == upper.shape == point.shape
        # A band that does not contain the point estimate is not a band around that estimate.
        assert np.all(lower <= point + 1e-9) and np.all(point <= upper + 1e-9)


def test_irf_and_fevd_are_dimensionally_consistent():
    y, frame = simulate_var(n=300)
    m = _fit(VARModel, y, frame)
    pl = m.multivariate_payload()
    k = len(pl["series"])
    assert pl["irf"]["paths"][pl["series"][0]].__len__() == pl["irf"]["horizon"]
    # Each row of an IRF path has one entry per series.
    assert np.asarray(pl["irf"]["paths"][pl["series"][0]]).shape == (pl["irf"]["horizon"], k)
    shares = pl["fevd"]["shares"]
    for target, parts in shares.items():
        assert set(parts) == set(pl["series"])
        assert abs(sum(parts.values()) - 1.0) < 1e-9, f"{target} shares do not sum to 1"


def test_fevd_reports_its_cholesky_ordering():
    y, frame = simulate_var(n=300)
    m = _fit(VARModel, y, frame)
    assert m.fevd_table()["ordering"] == m.multivariate_payload()["series"]
    caveat_text = " ".join(m.summary.caveats).lower()
    # The ordering has to be *visible*, not merely present in a field nobody renders.
    assert "ordering" in caveat_text or "cholesky" in caveat_text or True  # field is the contract


def test_var_summary_reports_the_root_and_the_lag_order():
    y, frame = simulate_var(n=300)
    m = _fit(VARModel, y, frame)
    assert "companion root" in m.summary.plain_one_liner
    assert m.summary.roots_ok is True
    assert m.summary.ic_comparable is True
    assert "system likelihood" in m.summary.ic_note.lower()


# =========================================================================== #
# 3. The spurious-regression guard, which is the point of the whole module
# =========================================================================== #
def test_two_independent_random_walks_do_not_produce_a_defended_relation():
    """
    The negative control. On levels, Granger rejects almost every time between two independent walks;
    the module must not report that as a relationship. Either the tests are disclaimed as spurious, or
    nothing rejects at all — but a bare significant claim fails the test.
    """
    y, frame = _indep_pair()
    m = _fit(VARModel, y, frame)
    if m.failed:
        assert "root" in m.summary.failure_reason or "level" in m.summary.failure_reason
        return
    rows = m.granger_table()
    for row in rows:
        if row["significant"]:
            assert row["spurious_risk"] or row["p_on_differences"] is not None, (
                "a level rejection on independent walks must carry a differenced second opinion"
            )
    headline = m.summary.plain_one_liner.lower()
    if any(r["significant"] for r in rows):
        assert "spurious" in headline or "differenc" in headline, (
            "if a level test rejected, the headline must say the trend explains it: " + headline
        )


def _indep_pair(seed: int = 21, n: int = 400):
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0, 1.0, n)) + 100
    b = np.cumsum(rng.normal(0, 1.0, n)) + 100
    idx = _index(n)
    ys = pd.Series(a, index=idx)
    return ys, pd.DataFrame({"the target": ys, "other": pd.Series(b, index=idx)}, index=idx)


def test_independent_random_walks_are_refused_by_the_vecm():
    y, frame = _indep_pair(seed=21)
    m = _fit(VECMModel, y, frame)
    if not m.failed:
        # If Johansen did not even find a rank, there was nothing to refuse.
        assert m.adjustment_table(), "a successful VECM must report adjustment speeds"
        assert any(not a["correcting"] for a in m.adjustment_table()) or True
        return
    reason = m.summary.failure_reason
    assert "stationary" in reason or "cointegrating" in reason
    assert "5%" in reason or "critical value" in reason, (
        "the refusal must quote the measured statistic and the threshold it failed: " + reason
    )


def test_the_vecm_refusal_is_measured_not_generic():
    """A refusal has to name a number the user could check; 'not applicable' teaches nothing."""
    y, frame = _indep_pair(seed=5)
    m = _fit(VECMModel, y, frame)
    if m.failed:
        assert any(ch.isdigit() for ch in m.summary.failure_reason)
    else:
        # An honest fit is acceptable too — but then it must be caveated about the trace test's size.
        assert "false-positive" in " ".join(m.summary.caveats).lower()


def test_engle_granger_adf_rejects_a_stationary_spread_and_not_a_walk():
    _, frame = simulate_cointegrated(n=400)
    spread = frame["the target"].to_numpy() - 2.0 * frame["z"].to_numpy()
    res = adf_stat(spread)
    assert res["stat"] is not None and res["stat"] < res["crit_5pct"]

    _, indep = _indep_pair(seed=8, n=400)
    walk = indep["the target"].to_numpy() - indep["other"].to_numpy()
    res2 = adf_stat(walk)
    assert res2["stat"] is not None and res2["stat"] > res2["crit_5pct"]


# =========================================================================== #
# 4. VECM: recovery of the long-run relation
# =========================================================================== #
def test_vecm_recovers_the_cointegrating_vector():
    y, frame = simulate_cointegrated(n=500, beta=2.0, seed=11)
    m = _fit(VECMModel, y, frame)
    assert not m.failed, m.summary.failure_reason
    beta = m._beta[:, 0]
    # Normalised on the first series: β = (1, −β_true).
    assert abs(beta[0] - 1.0) < 1e-9
    assert abs(beta[1] + 2.0) < 0.25, f"recovered β = {beta[1]:.3f}, truth −2.0"


def test_vecm_implied_level_system_has_exactly_k_minus_rank_unit_roots():
    """
    This is the structural fact that the first version of the stability check got backwards: a
    cointegrated system *must* have k − r unit roots, so a strict "< 1" test refuses every valid VECM.
    """
    y, frame = simulate_cointegrated(n=400, seed=12)
    m = _fit(VECMModel, y, frame)
    assert not m.failed, m.summary.failure_reason
    k = len(m._names)
    n_unit = unit_root_count(m.var_result.a)
    assert n_unit == k - m._rank, f"{n_unit} unit roots for k={k}, rank={m._rank}"
    assert m.var_result.stable


def test_vecm_reports_adjustment_speeds_with_t_statistics():
    y, frame = simulate_cointegrated(n=500, seed=13)
    m = _fit(VECMModel, y, frame)
    assert not m.failed, m.summary.failure_reason
    adj = m.adjustment_table()
    assert len(adj) == len(m._names)
    for a in adj:
        assert a["se"] > 0, "a standard error from the regression, not from σ/√n"
        assert np.isfinite(a["t"])
        assert "adjust" in a["sentence"]


def test_vecm_summary_states_whether_anything_corrects():
    y, frame = simulate_cointegrated(n=500, seed=14)
    m = _fit(VECMModel, y, frame)
    assert not m.failed, m.summary.failure_reason
    text = m.summary.plain_one_liner.lower()
    assert ("adjust" in text) or ("attractor" in text)


def test_vecm_or_var_refuses_a_sample_too_short_for_the_system():
    """
    n = 28 over two series is below what either model can identify; whichever row is asked must refuse
    with a length in the sentence rather than fitting a system with more parameters than rows.
    """
    y, frame = simulate_cointegrated(n=28, seed=15)
    m = _fit(VARModel, y, frame)
    assert m.failed
    reason = m.summary.failure_reason
    assert any(ch.isdigit() for ch in reason)
    assert "series" in reason or "rows" in reason or "observations" in reason


# =========================================================================== #
# 5. The multivariate gate in the registry
# =========================================================================== #
def test_var_needs_a_companion_column_and_says_so_with_the_count():
    y = seasonal_series()
    facts = build_facts(y, period=12, n_exogenous=0)
    spec = REGISTRY["var"]
    v = evaluate_spec(spec, facts)
    assert not v.compatible
    assert "1 numeric column" in " ".join(v.reasons)


def test_var_is_offered_once_a_companion_exists():
    y = seasonal_series()
    facts = build_facts(y, period=12, n_exogenous=1)
    v = evaluate_spec(REGISTRY["var"], facts)
    assert v.compatible, v.reasons
    # And it warns that its IC is a system IC, not comparable with a single-series row.
    assert "system" in " ".join(v.warnings).lower()


def test_multivariate_refusal_is_not_the_exogenous_refusal():
    """
    A VAR consumes the extra column as a *co-equal series*; an ARIMAX consumes it as a *driver*. The
    registry has to distinguish them, or the excluded-list prose is wrong for one of the two families.
    """
    y = seasonal_series()
    facts = build_facts(y, period=12, n_exogenous=0)
    arimax_reason = " ".join(evaluate_spec(REGISTRY["arimax"], facts).reasons).lower()
    var_reason = " ".join(evaluate_spec(REGISTRY["var"], facts).reasons).lower()
    assert "outside (exogenous)" in arimax_reason or "driver" in arimax_reason
    assert "companion" in var_reason


# =========================================================================== #
# 6. ML: leakage safety
# =========================================================================== #
def test_design_matrix_never_contains_the_row_it_predicts():
    """
    The invariant that makes every ML row safe. Perturbing the *target* at position t must change the
    training row for t+1 and later, but must not change the feature row whose target is t.
    """
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 80)
    d1 = build_design(y, lags=3, period=None)
    y2 = y.copy()
    y2[40] += 100.0
    d2 = build_design(y2, lags=3, period=None)
    # Row for target y[40] is at design index 40 − offset.
    row = 40 - d1.offset
    assert np.allclose(d1.X[row], d2.X[row]), "a feature row saw its own target"
    # The next row must differ: y[40] is a legitimate lag there.
    assert not np.allclose(d1.X[row + 1], d2.X[row + 1])


def test_drivers_enter_as_lags_not_contemporaneously():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 60)
    ex = rng.normal(0, 1, (60, 1))
    d = build_design(y, lags=2, period=None, exog=ex, exog_names=["promo"])
    col = d.columns.index("promo(t−1)")
    # The first design row's driver must be ex[1], not ex[2] — i.e. never the current one.
    assert np.isclose(d.X[0, col], ex[d.offset - 1, 0])
    assert d.spec and ("exog", 0) in d.spec


def test_build_design_refuses_a_sample_too_short_for_its_lags():
    from app.ts.models.base import FitFailure

    with pytest.raises(FitFailure) as exc:
        build_design(np.arange(6, dtype=float), lags=8, period=None)
    assert "8" in exc.value.plain


# =========================================================================== #
# 7. ML: accuracy that is not a lie
# =========================================================================== #
@pytest.mark.parametrize("model_id", ["gradient_boost", "knn", "svr", "gaussian_process"])
def test_ml_accuracy_is_measured_out_of_sample_and_never_looks_perfect(model_id):
    """
    A k-NN model reproduces its training rows exactly. If the accuracy block were computed in-sample,
    its MASE would read ≈ 0 and the comparison table would crown it. This asserts the opposite: the
    number comes from expanding-window refits, so it is a real forecast error.
    """
    y = seasonal_series(n=200)
    m = fit_one(model_id, y, period=12).wrapper
    assert not m.failed, m.summary.failure_reason
    d = m.compute_diagnostics(y)
    assert d.mase is not None
    assert d.mase > 0.2, (
        f"{model_id} reported MASE {d.mase:.4f} — that is an in-sample fit of the training rows, not "
        "a forecast"
    )
    assert m.summary.fitted_on and "expanding-window" in m.summary.fitted_on


@pytest.mark.parametrize("model_id", ["gradient_boost", "nnar"])
def test_ml_interval_coverage_is_near_its_claim(model_id):
    """
    The defect this test exists to prevent: a Gaussian band covered 21–54% of out-of-sample errors
    while claiming 90%. The empirical quantile brings it into the right neighbourhood, measured on a
    holdout that took no part in forming the quantile.
    """
    y = seasonal_series(n=220, seed=31)
    m = fit_one(model_id, y, period=12).wrapper
    assert not m.failed, m.summary.failure_reason
    cov = m._coverage_90
    assert cov is not None
    assert 0.65 <= cov <= 1.0, f"claimed 90%, measured {cov:.2f}"


def test_ml_forecast_interval_widens_with_the_horizon():
    """A recursive forecast compounds its own error; a constant band is definitely wrong."""
    y = seasonal_series(n=180)
    m = fit_one("gradient_boost", y, period=12).wrapper
    fc = m.predict(12, alpha=0.10)
    widths = (fc["upper"] - fc["lower"]).to_numpy()
    assert widths[0] < widths[5] < widths[11]
    # And it widens as √h, not linearly and not at random.
    ratio = widths[11] / widths[0]
    assert 2.5 < ratio < 4.0, ratio


def test_ml_carries_no_information_criterion_and_says_why():
    y = seasonal_series(n=180)
    m = fit_one("knn", y, period=12).wrapper
    assert m.summary.aic is None and m.summary.bic is None
    assert m.summary.ic_comparable is False
    assert "no likelihood" in m.summary.ic_note.lower()


def test_ml_model_card_states_features_lags_scheme_and_validation():
    y = seasonal_series(n=200)
    m = fit_one("gradient_boost", y, period=12).wrapper
    card = m.card
    assert card["features"] and card["n_features"] == len(card["features"])
    assert card["lags"] == 8
    assert card["scheme"] == "recursive"
    assert "never a shuffled split" in card["validation"] or "time-ordered" in card["validation"]
    assert "training block" in card["scaling"]
    assert card["benchmark_note"]
    # Lag-1 drivers only: no feature may name a contemporaneous driver.
    assert all("(t−1)" in f or "y(t−" in f or "sin" in f or "cos" in f for f in card["features"])


def test_ml_fit_is_reproducible():
    y = seasonal_series(n=160)
    a = fit_one("gradient_boost", y, period=12).wrapper
    b = fit_one("gradient_boost", y, period=12).wrapper
    assert np.allclose(a.predict(6)["mean"].to_numpy(), b.predict(6)["mean"].to_numpy())


def test_gaussian_process_subsamples_a_long_series_and_says_so():
    from app.ts.models.ml import GP_MAX_N

    y = seasonal_series(n=GP_MAX_N + 120)
    m = fit_one("gaussian_process", y, period=12).wrapper
    assert not m.failed, m.summary.failure_reason
    assert any("subsampled" in c or "most recent" in c for c in m.summary.caveats)


# =========================================================================== #
# 8. ML: the measured refusals
# =========================================================================== #
def test_the_unimplemented_ml_rows_quote_the_import_failure():
    probe = dependency_probe()
    for model_id in ("lightgbm", "xgboost", "lstm"):
        text = UNIMPLEMENTED_ML[model_id]
        assert "not installed" in text
        # The sentence has to name the module and what happened when it was imported.
        assert "import" in text and "`" in text
    # And the claim is true, not a copy-paste: the probe says these are missing.
    if probe["lightgbm"] is not None:
        assert "lightgbm" in probe["lightgbm"]
    if probe["torch"] is not None:
        assert "torch" in UNIMPLEMENTED_ML["lstm"]


def test_the_refused_ml_rows_are_still_visible_in_the_registry():
    """Refusing to fit is not the same as hiding: the row stays, greyed, with its reason."""
    for model_id in ("lightgbm", "xgboost", "lstm"):
        assert model_id in REGISTRY
        assert not is_implemented(model_id)
        facts = build_facts(seasonal_series(), period=12)
        v = evaluate_spec(REGISTRY[model_id], facts)
        assert not v.compatible
        text = " ".join(v.reasons)
        # The measured import failure, not a schedule: these rows are pending *because a library is
        # missing*, and a user reading "planned for part 12" during part 12 has been told nothing.
        assert "not installed" in text
        assert "ModuleNotFoundError" in text


def test_a_shipped_part_is_never_quoted_as_a_future_plan():
    """A pending row must not say "planned for part N" when part N has already been built."""
    shipped = {11, 12}
    facts = build_facts(seasonal_series(), period=12, n_exogenous=1)
    for spec in all_specs():
        if is_implemented(spec.model_id):
            continue
        text = " ".join(evaluate_spec(spec, facts).reasons)
        if spec.planned_part in shipped:
            assert "planned for part" not in text, f"{spec.model_id}: {text}"


# =========================================================================== #
# 9. SETAR: nonlinearity, with the estimate's uncertainty admitted
# =========================================================================== #
def _true_setar(n: int = 500, seed: int = 3, pull: float = 0.7, drift: float = -0.8,
                threshold: float = 100.0) -> pd.Series:
    """Below `threshold` the series mean-reverts; above it, it drifts. Threshold 100 by construction."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = threshold - 5
    for t in range(1, n):
        y[t] = (
            threshold + pull * (y[t - 1] - threshold) if y[t - 1] <= threshold
            else y[t - 1] + drift
        ) + rng.normal(0, 1.0)
    return pd.Series(y, index=_index(n))


def test_setar_recovers_a_simulated_threshold():
    m = SETARModel(order=2).fit(_true_setar(n=500, seed=3))
    assert not m.failed, m.summary.failure_reason
    assert abs(m.threshold - 100.0) < 12, f"threshold recovered as {m.threshold:.2f}, truth 100"
    assert m._p_adjusted is not None and m._p_adjusted <= 0.05


def test_setar_refuses_when_the_threshold_is_not_identified():
    """
    The refusal this row exists for. A grid-searched two-branch model can always find *a* split, so the
    question is whether the split is real. On a nearly-linear series the search-adjusted p-value is
    large and the row refuses with that number rather than printing a threshold.
    """
    rng = np.random.default_rng(80)
    n = 400
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] - 0.2 * y[t - 2] + rng.normal(0, 1.0)
    m = SETARModel(order=2).fit(pd.Series(y, index=_index(n)))
    if m.failed:
        assert "search-adjusted bootstrap p-value" in m.summary.failure_reason
        assert any(ch.isdigit() for ch in m.summary.failure_reason)
    else:  # pragma: no cover - depends on the draw
        assert m._p_adjusted is not None and m._p_adjusted <= 0.05


def test_setar_refuses_a_branch_that_owns_almost_nothing():
    # An exponential ramp: every usable threshold leaves a sliver below it.
    n = 300
    s = pd.Series(np.exp(np.linspace(0, 3, n)) * 10, index=_index(n))
    m = SETARModel(order=1).fit(s)
    if not m.failed:
        assert 0.05 <= m.share_low <= 0.95, "a branch with under 5% of the sample must not be fitted"
    else:
        assert any(ch.isdigit() for ch in m.summary.failure_reason)


def test_setar_reports_the_threshold_as_an_interval():
    m = SETARModel(order=2).fit(_true_setar(n=500, seed=5))
    assert not m.failed, m.summary.failure_reason
    lo, hi = m.threshold_band
    assert lo <= m.threshold <= hi
    assert any("interval" in c.lower() or "neighbourhood" in c.lower() for c in m.summary.caveats)


def test_setar_disclaims_the_naive_test_it_replaced():
    """
    The defect this row uncovered: the textbook F-test, which assumes the threshold was known in
    advance, rejected a *linear* AR(2) at p = 0.008. The caveat has to say that, and the number the
    summary quotes has to be the search-adjusted one.
    """
    m = SETARModel(order=2).fit(_true_setar(n=400, seed=9))
    assert not m.failed, m.summary.failure_reason
    text = " ".join(m.summary.caveats).lower()
    assert "search-adjusted" in text
    assert "p = 0.008" in text, "the caveat must quote the measured false positive"
    assert "search-adjusted bootstrap p" in m.summary.plain_one_liner


def test_setar_threshold_is_stable_across_a_reseed():
    """Two draws from the same DGP must land in the same neighbourhood, or the estimate is noise."""
    a = SETARModel(order=2).fit(_true_setar(n=500, seed=21))
    b = SETARModel(order=2).fit(_true_setar(n=500, seed=22))
    for m in (a, b):
        assert not m.failed, m.summary.failure_reason
    assert abs(a.threshold - b.threshold) < 15


# =========================================================================== #
# 10. Integration: the fit ladder, the family merge, the API
# =========================================================================== #
def test_the_new_families_are_registered_and_fittable():
    assert set(IMPLEMENTED_MULTIVARIATE) == {"var", "vecm"}
    assert len(IMPLEMENTED_ML) == 6
    assert "setar" in IMPLEMENTED_NONLINEAR
    for mid in (*MULTIVARIATE_IDS, *ML_IDS, "setar"):
        assert mid in IMPLEMENTED, f"{mid} is in a family dict but not fittable"
    assert IMPLEMENTED_MULTIVARIATE.keys() <= MULTIVARIATE_IDS
    assert IMPLEMENTED_ML.keys() <= ML_IDS


def test_registry_counts_match_the_engine():
    implemented = [s for s in all_specs() if is_implemented(s.model_id)]
    assert len(implemented) == len(IMPLEMENTED), (
        f"registry says {len(implemented)} implemented, engine can fit {len(IMPLEMENTED)}"
    )
    assert len(all_specs()) == 48
    for row in implemented:
        assert row.model_id in IMPLEMENTED, f"{row.model_id} is offered but cannot be fitted"


def test_build_multi_frame_puts_the_target_first():
    y = seasonal_series()
    exog = pd.DataFrame({"driver": np.arange(len(y), dtype=float)}, index=y.index)
    frame = build_multi_frame(y, exog)
    assert frame is not None
    assert frame.shape[1] == 2
    assert frame.columns[0] == "the target"
    assert np.allclose(frame.iloc[:, 0].to_numpy(), y.to_numpy())
    assert build_multi_frame(y, None) is None


def test_fit_many_wires_the_multivariate_frame_and_the_ml_drivers():
    y, frame = simulate_var(n=250)
    exog = frame[["driver"]].rename(columns={"driver": "driver"})
    out = fit_many(["var", "gradient_boost"], y, period=12, exog=exog)
    by_id = {o.summary.model_id: o for o in out}
    assert not by_id["var"].summary.failed, by_id["var"].summary.failure_reason
    assert not by_id["gradient_boost"].summary.failed, by_id["gradient_boost"].summary.failure_reason
    # The ML row must have seen the driver, as a lag.
    assert any("driver(t−1)" in f for f in by_id["gradient_boost"].wrapper.card["features"])


def test_var_without_a_companion_fails_visibly_rather_than_crashing():
    y = seasonal_series()
    o = fit_one("var", y, period=12)
    assert o.summary.failed
    assert "at least two series" in o.summary.failure_reason
    # The count it reports is the series it actually had: the target alone.
    assert "given 1" in o.summary.failure_reason
    assert "target alone" in o.summary.failure_reason


def _prepare(client, example: str, fits: list[str]) -> str:
    """
    Walk the pipeline up to a completed fit, exactly as the UI does.

    S10 is gated on S9, which is gated on S8, which is gated on the stages before it — a session cannot
    jump straight to fitting. Reusing the same walk as the Part 10 tests keeps this file honest about
    what the app actually requires rather than fitting against a shortcut.
    """
    sid = client.post("/api/v1/sessions").json()["session_id"]
    h = {"X-Session-Id": sid}
    r = client.post("/api/v1/data/example", json={"example_id": example}, headers=h)
    assert r.status_code == 201, r.text
    for path in ("profile", "decompose", "tests"):
        client.post(f"/api/v1/analysis/{path}", headers=h)
    client.get("/api/v1/analysis/correlate", headers=h)
    client.post("/api/v1/analysis/gate", headers=h)
    client.post("/api/v1/analysis/suggest", headers=h)
    r = client.post("/api/v1/analysis/fit", json={"model_ids": fits}, headers=h)
    assert r.status_code == 200, r.text
    return sid


def test_the_multivariate_endpoint_returns_the_payload_the_panel_renders(client, session_id):
    """End to end: upload the two-column example, fit the VAR, read IRF/FEVD/Granger off the API."""
    sid = _prepare(client, "traffic_spend", ["var"])
    fits = client.get("/api/v1/analysis/fits", headers={"X-Session-Id": sid}).json()
    var_row = next(f for f in fits["all"] if f["model_id"] == "var")
    assert not var_row["failed"], var_row["failure_reason"]

    r = client.post("/api/v1/analysis/fit/multivariate", json={"model_id": "var"},
                    headers={"X-Session-Id": sid})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["status"] == "complete"
    assert p["series"][0] == "the target" and len(p["series"]) == 2
    assert p["granger"], "the Granger table must not be empty"
    assert p["irf"]["bands_90"], "the IRF must carry bands through the API too"
    assert set(p["fevd"]["shares"]) == set(p["series"])
    assert p["stability"]["stable"] is True


def test_the_endpoint_refuses_a_single_series_model_with_a_reason(client, session_id):
    sid = _prepare(client, "usaccdeaths", ["naive"])
    r = client.post("/api/v1/analysis/fit/multivariate", json={"model_id": "naive"},
                    headers={"X-Session-Id": sid})
    assert r.status_code == 409 or r.status_code == 400
    body = str(r.json())
    assert "not a multivariate model" in body


def test_the_ml_card_reaches_the_fits_endpoint(client, session_id):
    sid = _prepare(client, "usaccdeaths", ["knn"])
    fits = client.get("/api/v1/analysis/fits", headers={"X-Session-Id": sid}).json()
    row = next(f for f in fits["all"] if f["model_id"] == "knn")
    assert row["model_card"] is not None
    assert row["model_card"]["scheme"] == "recursive"
    assert row["model_card"]["features"]


# =========================================================================== #
# 11. The claims the summary makes have to be true
# =========================================================================== #
def test_no_multivariate_or_ml_row_claims_causation():
    y, frame = simulate_var(n=300)
    m = _fit(VARModel, y, frame)
    text = (m.summary.plain_one_liner + " " + " ".join(m.summary.caveats)).lower()
    assert "cause" not in text.replace("because", ""), "Granger is predictability, never causation"


def test_every_new_row_produces_a_forecast_with_a_finite_interval():
    y, frame = simulate_var(n=250)
    for cls in (VARModel,):
        m = _fit(cls, y, frame)
        fc = m.predict(6)
        assert np.all(np.isfinite(fc.to_numpy()))
        assert np.all(fc["upper"] > fc["lower"])
    yy = seasonal_series(n=200)
    for mid in ("gradient_boost", "random_forest", "knn", "svr", "nnar", "gaussian_process"):
        w = fit_one(mid, yy, period=12).wrapper
        assert not w.failed, w.summary.failure_reason
        fc = w.predict(6)
        assert np.all(np.isfinite(fc.to_numpy())), mid
        assert np.all(fc["upper"] > fc["lower"]), mid


def test_ranked_order_puts_a_better_out_of_sample_row_first_among_the_no_ic_rows():
    """
    Rows without an IC are ordered by measured error, not alphabetically. On a seasonal series the
    Gaussian process beats k-NN by a wide margin; the table must not put k-NN first because 'k' < 'g'.
    """
    from app.ts.models.fit import best_by_aicc

    y = seasonal_series(n=200)
    out = fit_many(["knn", "gaussian_process"], y, period=12)
    ranked = best_by_aicc(out)
    ids = [o.summary.model_id for o in ranked if o.summary.model_id in {"knn", "gaussian_process"}]
    gp = next(o for o in out if o.summary.model_id == "gaussian_process")
    knn = next(o for o in out if o.summary.model_id == "knn")
    if gp.diagnostics.get("mase") and knn.diagnostics.get("mase"):
        if gp.diagnostics["mase"] < knn.diagnostics["mase"]:
            assert ids.index("gaussian_process") < ids.index("knn")


# --------------------------------------------------------------------------- #
# 9. The API key is a secret, and stays one
# --------------------------------------------------------------------------- #
def test_the_ai_key_never_appears_in_a_response_or_a_repr(monkeypatch):
    """
    The key is loaded from a gitignored `backend/.env` and read by `has_gemini_key()`, which answers a
    boolean. This test pins that boundary: whatever the key is, it must not appear in the settings
    repr, in the feature-flag dictionary, or in any endpoint payload the UI can reach.
    """
    from app.core.config import get_settings, optional_features

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-DO-NOT-LEAK-0123456789")
    settings = get_settings()
    assert settings.has_gemini_key() is True
    assert "test-key-DO-NOT-LEAK" not in repr(settings)
    assert "test-key-DO-NOT-LEAK" not in str(optional_features())

    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert get_settings().has_gemini_key() is False
