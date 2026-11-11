"""
Part 15 audit regressions — statistical correctness, leakage completeness and
honest reporting. These tests reproduce the defects found in the 2026-11 audit.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.metrics import _integrity, _verdict
from app.charts import evaluation as ev
from app.core.schemas import Stage
from app.ts.diagnose.calibration import pit_from_quantiles, rqr_from_draws
from app.ts.metrics.comparative import (
    directional_accuracy,
    dm_matrix,
    model_confidence_set,
    nested_tests,
)
from app.ts.metrics.discrete import (
    discrete_crps,
    discrete_log_score,
    discrete_scores,
    dunn_smyth_rqr,
)
from app.ts.metrics.point import point_metrics
from app.ts.metrics.probabilistic import (
    IntervalBand,
    coverage_curve,
    empirical_pit,
    pinball_from_bands,
    winkler_scores,
)
from app.ts.metrics.protocols import (
    _decisions,
    _metric_value,
    _prob_combined_p,
    _ranked_models,
    leakage_audit,
)
from app.ts.select.cv import CVFold, score_fold
from app.ts.select.tests_compare import _hac_variance, clark_west, diebold_mariano

BASE = "/api/v1"


# --------------------------------------------------------------------------- #
# 1. Point / fold scoring
# --------------------------------------------------------------------------- #
def test_pooled_mase_and_rmsse_use_each_folds_training_scale():
    actual = np.asarray([10.0, 20.0])
    forecast = np.asarray([8.0, 16.0])
    benchmark = np.asarray([9.0, 18.0])
    scales = [1.0, 10.0]
    errors = (actual - forecast).reshape(2, 1)
    pm = point_metrics(
        actual=actual, forecast=forecast, benchmark=benchmark, scales=scales,
        errors_by_horizon=errors, allow_mape=True,
    )
    # Old bug: MAE / mean(scales) = 3 / 5.5 = 0.545 and RMSE / 5.5 = 0.575.
    assert pm.mase == pytest.approx((2.0 / 1.0 + 4.0 / 10.0) / 2.0)
    assert pm.rmsse == pytest.approx(math.sqrt((4.0 + 0.16) / 2.0))
    assert pm.mase_by_horizon == [pytest.approx(1.2)]
    assert pm.scale == pytest.approx(5.5)  # descriptive only
    assert pm.n_scaled == 2
    assert any("own fold's" in n for n in pm.notes)


def test_score_fold_does_not_count_missing_bands_as_uncovered():
    fold = CVFold(index=0, train_start=0, train_end=1, test_start=2, test_end=3)
    fs = score_fold(
        fold, np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0]),
        y_train=np.asarray([0.0, 1.0, 2.0]),
        lower=np.asarray([-1.0, np.nan]), upper=np.asarray([1.0, np.nan]),
    )
    assert fs.coverage == pytest.approx(1.0)
    assert any("excluded" in n for n in fs.notes)


def test_point_metrics_pads_short_vectors_and_counts_every_excluded_row():
    pm = point_metrics(
        actual=[1.0, 2.0, 3.0, 4.0],
        forecast=[1.0, 2.0],
        benchmark=[1.0, 2.0, 3.0],
        scales=[1.0],
    )
    assert pm.n == 2
    assert pm.n_excluded == 2
    assert pm.n + pm.n_excluded == 4
    assert any("padded, not silently truncated" in w for w in pm.warnings)


# --------------------------------------------------------------------------- #
# 2. Forecast-comparison tests
# --------------------------------------------------------------------------- #
def test_clark_west_is_canonical_one_sided_and_uses_forecast_gap():
    rng = np.random.default_rng(20261108)
    es = rng.normal(0.0, 1.0, 200)
    el = rng.normal(0.0, 1.0, 200)
    expected_gap = el - es  # f_small − f_large
    res = clark_west(es, el, horizon=1, name_small="benchmark", name_large="model")
    assert res.ok
    assert res.adjusted_mean == pytest.approx(
        float(np.mean(es**2 - el**2 + expected_gap**2)), rel=1e-4
    )
    assert res.adjustment is not None and res.adjustment > 0
    assert 0.0 <= float(res.p_value) <= 1.0
    assert "adjustment adds" in res.note
    assert any("one-sided" in a for a in res.assumptions)
    # A supplied zero gap must change the statistic; benchmark_forecast is not ignored.
    zero_gap = clark_west(es, el, horizon=1, forecast_gap=np.zeros_like(es))
    assert zero_gap.ok
    assert not math.isclose(float(res.statistic), float(zero_gap.statistic), rel_tol=1e-6)


def test_dm_refuses_a_non_positive_hac_variance_instead_of_manufacturing_significance():
    d = np.asarray([(-1.0) ** i for i in range(40)])
    assert _hac_variance(d, 1) >= 0.0
    res = diebold_mariano(d, np.zeros_like(d), horizon=2)
    assert not res.ok
    assert "HAC long-run variance" in res.note


def test_nested_tests_uses_benchmark_forecast_in_the_adjustment():
    rng = np.random.default_rng(91)
    n = 80
    f = rng.normal(0, 1, n)
    me = rng.normal(0, 1, n)
    be = rng.normal(0, 1, n)
    y = f + me
    bf1 = rng.normal(0, 1, n)
    bf2 = bf1 + 0.5
    a = nested_tests(
        actual=y, forecast=f, benchmark_forecast=bf1, benchmark_errors=be,
        model_errors=me, horizon=2, model_name="ets",
    )
    b = nested_tests(
        actual=y, forecast=f, benchmark_forecast=bf2, benchmark_errors=be,
        model_errors=me, horizon=2, model_name="ets",
    )
    assert a["clark_west"]["ok"] and b["clark_west"]["ok"]
    assert not math.isclose(
        float(a["clark_west"]["statistic"]), float(b["clark_west"]["statistic"]), rel_tol=1e-6
    )
    assert "Canonical Clark" in a["clark_west"]["note"]


def test_dm_matrix_reports_holm_adjusted_family_wise_p_values():
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 60)
    errors = {
        "a": [base],
        "b": [base + rng.normal(0, 0.4, 60)],
        "c": [base + rng.normal(0, 0.8, 60)],
    }
    dm = dm_matrix(errors, horizon=2)
    valid = [r for r in dm["rows"] if r.get("p") is not None]
    assert valid
    for row in valid:
        assert row["p_holm"] is not None
        assert row["p_holm"] >= row["p"] - 1e-9
        assert row["p_holm"] <= 1.0
    assert "Holm" in dm["note"]


def test_mcs_bootstrap_p_is_a_valid_probability_and_not_decorative():
    rng = np.random.default_rng(19)
    n = 80
    good = rng.normal(0, 1, n)
    errors = {"good": [good], "okay": [good + 0.2], "bad": [good + 3.0]}
    mcs = model_confidence_set(errors)
    assert mcs["applicable"] is True
    assert 0.0 <= float(mcs["mcs_p"]) <= 1.0
    assert set(mcs["survivors"]) <= set(errors)
    assert "centred the same way" in mcs["note"]


def test_directional_ties_are_excluded_not_counted_as_down():
    baseline = np.zeros(20)
    actual = np.ones(20)
    forecast = np.concatenate([np.zeros(5), np.full(15, 2.0)])
    actual[:5] = 0.0
    hit = directional_accuracy(baseline, actual, forecast)
    assert hit.n_ties == 5
    assert hit.n == 15
    assert hit.rate == pytest.approx(1.0)
    assert any("tie(s) were excluded" in w for w in hit.warnings)


# --------------------------------------------------------------------------- #
# 3. Discrete / probabilistic calibration
# --------------------------------------------------------------------------- #
def _exact_crps(grid: np.ndarray, pmf: np.ndarray, y: float) -> float:
    g = np.asarray(grid, float)
    p = np.asarray(pmf, float)
    keep = np.isfinite(g) & np.isfinite(p)
    g, p = g[keep], np.maximum(p[keep], 0.0)
    p = p / p.sum()
    return float(np.dot(p, np.abs(g - y)) - 0.5 * np.sum(p[:, None] * p[None, :] * np.abs(g[:, None] - g[None, :])))


def test_discrete_crps_matches_energy_identity_for_irregular_and_outside_support():
    rng = np.random.default_rng(2026)
    for _ in range(25):
        k = int(rng.integers(1, 7))
        g = np.sort(rng.choice(np.arange(-3.0, 9.0, 0.5), size=k, replace=False))
        p = rng.random(k) + 0.05
        y = float(rng.uniform(-5.0, 12.0))
        assert discrete_crps(g, p, y) == pytest.approx(_exact_crps(g, p, y), abs=1e-9)
    assert discrete_crps([0.0, 2.0], [0.5, 0.5], 1.0) == pytest.approx(0.5)
    assert discrete_crps([0.0], [1.0], -1.0) == pytest.approx(1.0)


def test_out_of_support_pmf_refuses_the_mean_log_score():
    grids = [np.asarray([0.0, 1.0]) for _ in range(24)]
    pmfs = [np.asarray([0.5, 0.5]) for _ in range(24)]
    observed = np.asarray([i % 2 for i in range(24)], dtype=float)
    observed[3] = 7.0  # integer, outside the stored support
    assert math.isnan(discrete_log_score(grids[3], pmfs[3], 7.0))
    ds = discrete_scores(grids=grids, pmfs=pmfs, observed=observed, reps=3, seed=11)
    assert ds.n_out_of_support == 1
    assert ds.log_score is None
    assert any("outside the stored PMF support" in w for w in ds.warnings)


def test_dunn_smyth_rqr_does_not_fisher_combine_dependent_repetitions():
    grids = [np.asarray([0.0, 1.0]) for _ in range(24)]
    pmfs = [np.asarray([0.5, 0.5]) for _ in range(24)]
    observed = np.asarray([i % 2 for i in range(24)], dtype=float)
    rqr = dunn_smyth_rqr(grids, pmfs, observed, reps=5, seed=17)
    assert rqr.ok
    assert rqr.combined_p == rqr.ks_p
    assert len(rqr.per_rep_ks_p) == 5
    assert "NOT a Fisher" in rqr.note


def test_pit_and_rqr_use_one_inferential_draw_not_a_shrunk_mean():
    rng = np.random.default_rng(30)
    y = np.zeros(500)
    pit = pit_from_quantiles(y, lower_cdf=np.zeros(500), cdf=np.ones(500),
                            kind="discrete", reps=5, seed=17)
    assert pit.ok
    assert len(pit.values) == 500
    assert abs(float(np.mean(pit.values)) - 0.5) < 0.05
    assert pit.ks_p is not None and pit.ks_p > 0.01
    assert pit.cvm_p is not None and pit.cvm_p > 0.01
    assert len(pit.per_rep_ks_p) == 5 and pit.stability

    actual = rng.normal(0, 1, 200)
    draws = rng.normal(0, 1, (300, 200))
    rqr = rqr_from_draws(actual, draws)
    assert rqr.ok
    assert rqr.n == 200 and rqr.reps >= 5
    assert rqr.ks_p is not None and rqr.ks_p > 0.01
    assert "ONE seeded draw" in rqr.caveat


def test_empirical_pit_divides_by_finite_draws_only():
    rng = np.random.default_rng(44)
    paths = rng.normal(0, 1, (2000, 20))
    paths[1000:, :] = np.nan
    res = empirical_pit(np.zeros(20), paths, reps=3, seed=5)
    assert res.ok
    assert abs(float(np.mean(res.values)) - 0.5) < 0.05


def test_band_scores_pad_to_full_length_and_count_exclusions():
    y = np.zeros(12)
    short = IntervalBand(0.95, np.full(8, -1.96), np.full(8, 1.96),
                         "95% short band", "analytic")
    pin = pinball_from_bands(y, [short])
    win = winkler_scores(y, [short])
    cov = coverage_curve(y, [short])
    assert pin["n_expected"] == 12 and pin["n_excluded"] > 0 and pin["warnings"]
    assert win["n_expected"] == 12 and win["n_excluded"] > 0 and win["warnings"]
    assert cov["rows"][0]["n"] == 8 and cov["rows"][0]["n_excluded"] == 4
    assert all(len(v) == 12 for v in win["per_step"].values())


def test_calibration_chart_refuses_to_invent_a_missing_nominal_level():
    chart = ev.calibration_curve([{
        "nominal": None, "coverage_pct": None, "empirical": None,
        "p": "not-a-number", "mean_width": "also-not-a-number",
        "construction": "unknown",
    }])
    assert chart.data_behind[0]["p_display"] == "—"
    assert "Empirical coverage is unavailable" in chart.alt_text
    assert "p-values unavailable" in chart.alt_text
    assert "Widths unavailable" in chart.alt_text


# --------------------------------------------------------------------------- #
# 4. Protocol / reporting helpers
# --------------------------------------------------------------------------- #
def test_leakage_audit_refuses_an_incomplete_fit_check_list():
    audit = leakage_audit(
        raw=pd.Series(dtype=float), folds=[], contexts=[],
        spec_template=None, methods=[], period=None,
        fit_checks=[], expected_fits=2,
    )
    fit = next(c for c in audit.checks if c["name"] == "factory_received_training_window_only")
    assert not fit["ok"]
    assert "Only 0 of 2" in fit["detail"]


def test_future_driver_audit_recomputes_the_held_flat_frame():
    raw = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fold = CVFold(index=0, train_start=0, train_end=2, test_start=3, test_end=4)
    decisions = _decisions(
        raw.iloc[0:3], spec_template=None, methods=[], period=None,
    )
    train_e = pd.DataFrame({"driver": [10.0, 20.0], "other": [1.0, 2.0]})
    last = train_e.iloc[-1]
    bad_future = pd.DataFrame([{"driver": 999.0, "other": 2.0}])
    good_future = pd.DataFrame([last.to_dict()])

    def ctx(future):
        return SimpleNamespace(
            decisions=decisions, spec=None, exog_train=train_e,
            future_exog=future, exog_held_flat=True,
        )

    def check(future):
        audit = leakage_audit(
            raw=raw, folds=[fold], contexts=[ctx(future)],
            spec_template=None, methods=[], period=None,
            fit_checks=[], expected_fits=0,
        )
        return next(c for c in audit.checks if c["name"] == "future_drivers_held_flat")["ok"]

    assert check(bad_future) is False
    assert check(good_future) is True


def test_combined_rqr_p_is_not_a_fisher_chi_square():
    assert _prob_combined_p({"per_rep_ks_p": [0.001, 0.002], "ks_p": 0.3,
                             "combined_p": 0.3}) == pytest.approx(0.3)


def test_ranked_models_respects_higher_is_better():
    rows = [
        {"model": "a", "hit_rate": 0.9},
        {"model": "b", "hit_rate": 0.4},
        {"model": "c", "hit_rate": 0.7},
    ]
    assert _ranked_models(rows, "hit_rate") == ["a", "c", "b"]
    assert _ranked_models(rows, "mase") == ["a", "b", "c"]  # all missing -> stable alpha order


def test_coverage_metric_is_deviation_from_nominal_not_mean_coverage():
    prob = {"m": {"calibration": {"rows": [
        {"nominal": 0.5, "empirical": 0.95},
        {"nominal": 0.95, "empirical": 0.95},
    ]}}}
    value = _metric_value(
        "m", "coverage", point={}, prob=prob, discrete={}, variance={}, hit={}
    )
    assert value == pytest.approx(0.225)


def _verdict_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = {
        "inputs": {
            "n_pooled_test_observations": 12,
            "n_missing_test_observations": 1,
            "test_observations_used_for_fitting": 0,
            "test_observations_scored": 12,
            "benchmark_id": "seasonal_naive",
            "folds": 3,
            "evaluation_window": "Jan … Jun",
            "interval_construction": "analytic",
        },
        "point": {"ets": {"n": 11, "n_excluded": 1, "n_scaled": 11}},
        "leakage": {
            "ok": True,
            "checks": [
                {"name": "factory_received_training_window_only", "ok": True},
                {"name": "strict_train_test_order", "ok": True},
            ],
        },
        "coverage_construction": [{"coverage": 0.95, "construction": "analytic bands"}],
        "accuracy": {"n_scored": 12},
        "fold_failures": [],
        "warnings": [],
    }
    common = {
        "stage": Stage.S14_EVALUATE,
        "inputs": {},
        "citations": [],
        "caveats": ["A travelling caveat."],
        "warnings": [],
        "reproducible_snippet": "print('verified')",
    }
    adequacy = {"status": "evaluated", "adequacy": "adequate", "headline": "Residuals adequate."}
    return report, common, adequacy


def _verdict_value(result: Any) -> str:
    return str(getattr(result.verdict, "value", result.verdict))


def test_verdict_checks_safeguards_instead_of_asserting_they_passed():
    report, common, adequacy = _verdict_fixture()
    integrity = _integrity(report)
    assert integrity["rows_ok"] and integrity["out_of_sample"] and integrity["benchmark_ok"]
    ready = _verdict(headline="Audit", adequacy=adequacy, report=report, common=common)
    assert _verdict_value(ready) == "ready"
    assert all(c.passed for c in ready.checklist)

    broken_rows = deepcopy(report)
    broken_rows["point"]["ets"]["n_excluded"] = 0
    bad = _verdict(headline="Audit", adequacy=adequacy, report=broken_rows, common=common)
    assert _verdict_value(bad) == "not_ready"
    assert any(c.key == "dropped_rows" and not c.passed for c in bad.checklist)
    assert any("Row accounting" in b for b in bad.blockers)

    fitted_on_test = deepcopy(report)
    fitted_on_test["inputs"]["test_observations_used_for_fitting"] = 1
    assert _verdict_value(_verdict(
        headline="Audit", adequacy=adequacy, report=fitted_on_test, common=common,
    )) == "not_ready"

    leaked = deepcopy(report)
    leaked["leakage"]["ok"] = False
    assert _verdict_value(_verdict(
        headline="Audit", adequacy=adequacy, report=leaked, common=common,
    )) == "not_ready"

    unscaled = deepcopy(report)
    unscaled["point"]["ets"]["n_scaled"] = 0
    assert _verdict_value(_verdict(
        headline="Audit", adequacy=adequacy, report=unscaled, common=common,
    )) == "not_ready"

    uncovered = deepcopy(report)
    uncovered["coverage_construction"] = [{"coverage": 0.95, "construction": ""}]
    assert _verdict_value(_verdict(
        headline="Audit", adequacy=adequacy, report=uncovered, common=common,
    )) == "inconclusive"


# --------------------------------------------------------------------------- #
# 5. API / report contract
# --------------------------------------------------------------------------- #
def _prepared(client: TestClient) -> dict[str, str]:
    sid = client.post(f"{BASE}/sessions").json()["session_id"]
    headers = {"X-Session-Id": sid}
    assert client.post(f"{BASE}/data/example", json={"example_id": "traffic_spend"}, headers=headers).status_code == 201
    assert client.post(f"{BASE}/analysis/profile", headers=headers).status_code == 200
    for stage in ("decompose", "tests"):
        assert client.post(f"{BASE}/analysis/{stage}", headers=headers).status_code == 200
    assert client.get(f"{BASE}/analysis/correlate", headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/gate", headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/suggest", headers=headers).status_code == 200
    return headers


def _evaluate(client: TestClient, headers: dict[str, str], **body: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    r = client.post(f"{BASE}/analysis/evaluate", json={
        "horizon": 6, "folds": 3, "n_paths": 2000, "reps": 3, **body,
    }, headers=headers)
    assert r.status_code == 200, r.text[:800]
    read = client.get(f"{BASE}/analysis/evaluate", headers=headers).json()
    return r.json(), read["evaluation"]


@pytest.fixture(scope="module")
def audit_api(client):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        headers = _prepared(client)
        assert client.post(f"{BASE}/analysis/fit", json={"model_ids": ["ets", "drift"]}, headers=headers).status_code == 200
        assert client.post(f"{BASE}/analysis/select", json={"horizon": 6, "folds": 3, "models": ["ets", "drift"]}, headers=headers).status_code == 200
        assert client.post(f"{BASE}/analysis/diagnose", json={}, headers=headers).status_code == 200
        assert client.post(f"{BASE}/analysis/forecast", json={"horizon": 6}, headers=headers).status_code == 200
        default_response, default_report = _evaluate(client, headers)
        no_baseline_response, no_baseline_report = _evaluate(client, headers, include_baselines=False)
    return {
        "headers": headers,
        "default_response": default_response,
        "default_report": default_report,
        "no_baseline_response": no_baseline_response,
        "no_baseline_report": no_baseline_report,
    }


def test_evaluation_report_has_complete_leakage_checks_and_honest_primary(audit_api):
    rep = audit_api["default_report"]
    leak = rep["leakage"]
    assert leak["ok"] is True
    names = {c["name"] for c in leak["checks"]}
    assert {
        "one_training_context_per_fold", "fold_decisions_match_context",
        "training_only_transform_parameters", "factory_received_training_window_only",
    } <= names
    assert all(c["ok"] for c in leak["checks"])
    assert rep["inputs"]["interval_construction"]
    assert rep["inputs"]["include_baselines"] is True
    valid = [r for r in rep["comparative"]["dm"]["rows"] if r.get("p") is not None]
    assert valid and all(r.get("p_holm") is not None for r in valid)
    text = " ".join(str(r.get("plain", "")) for r in audit_api["default_response"]["results"])
    assert "Primary model" in text and "ranks first" in text
    assert "leads on" not in audit_api["default_response"]["advisor"]["headline"]
    assert "scores" in audit_api["default_response"]["advisor"]["headline"]
    verdict = next(
        r for r in audit_api["default_response"]["results"] if r.get("id") == "evaluate.verdict"
    )
    checks = {c["key"]: c["passed"] for c in verdict["checklist"]}
    assert all(checks.get(k) is True for k in (
        "out_of_sample", "leakage", "benchmark", "coverage", "dropped_rows",
    ))
    assert verdict["verdict"] in {"ready", "ready_with_caveats"}
    assert not verdict["blockers"]


def test_include_baselines_false_is_respected_without_losing_the_scaling_benchmark(audit_api):
    rep = audit_api["no_baseline_report"]
    benchmark = str(rep["inputs"].get("benchmark_id") or rep["inputs"].get("benchmark"))
    assert benchmark
    assert benchmark not in set(rep["models"])
    assert rep["inputs"]["benchmark_id"] == benchmark
    assert any("excluded from the comparison" in w for w in rep["warnings"])
    assert benchmark in set(audit_api["default_report"]["models"])
