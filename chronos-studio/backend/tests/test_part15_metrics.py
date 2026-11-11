"""
tests/test_part15_metrics.py — S14 evaluation metrics (§§5.10, 5.12, §8 row 15).

The file is built around the ways a metric panel can be wrong while looking
impressive:

1. **In-sample fit relabelled as accuracy.** Every score here comes from a
   rolling origin; the leakage audit poisons test values and proves that no
   training-only decision changed.
2. **A denominator that saw the future.** MASE/RMSSE use one training-window
   naive scale per fold. Perfect forecasts score 0 and the naive forecast 1.
3. **A continuous density score on count atoms.** Discrete PMF CRPS, PMF log
   score, randomized PIT and Dunn–Smyth RQR are separate code paths.
4. **A generic metric list.** The policy switches for intermittent, count,
   short, probabilistic and variance targets, and carries its reasons.
5. **A figure without its table.** All ten S14 builders are exercised from a
   real stored report.
6. **A backend report with no usable UI/API contract.** The API tests walk the
   actual stage order, inspect strict JSON and stored result cards.
"""

from __future__ import annotations

import ast
import json
import math
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.charts import evaluation as ev
from app.core.schemas import DataClass
from app.main import app
from app.ts.forecast.intervals import IntervalBand
from app.ts.metrics.comparative import directional_accuracy, dm_matrix, model_confidence_set
from app.ts.metrics.discrete import (
    discrete_crps,
    discrete_log_score,
    discrete_scores,
    dunn_smyth_rqr,
)
from app.ts.metrics.policy import choose_policy
from app.ts.metrics.point import mase_scale, point_metrics
from app.ts.metrics.probabilistic import (
    coverage_curve,
    crps_ensemble,
    log_score_normal,
    pinball_from_bands,
    winkler_scores,
)
from app.ts.metrics.variance import mse_variance, qlike, var_backtests
from app.ts.models.intermittent import CrostonModel

BASE = "/api/v1"


# --------------------------------------------------------------------------- #
# Stage-order helpers — the gates are real, so the tests walk them
# --------------------------------------------------------------------------- #
def _prepared(client: TestClient, example: str = "traffic_spend", *, transform: bool = False):
    sid = client.post(f"{BASE}/sessions").json()["session_id"]
    headers = {"X-Session-Id": sid}
    assert client.post(f"{BASE}/data/example", json={"example_id": example}, headers=headers).status_code == 201
    assert client.post(f"{BASE}/analysis/profile", headers=headers).status_code == 200
    if transform:
        applied = client.post(
            f"{BASE}/preprocess/apply",
            json={"op": "transform.log", "params": {"base": "e"}, "reason": "variance"},
            headers=headers,
        )
        assert applied.status_code == 200, applied.text[:400]
    for stage in ("decompose", "tests"):
        assert client.post(f"{BASE}/analysis/{stage}", headers=headers).status_code == 200
    assert client.get(f"{BASE}/analysis/correlate", headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/gate", headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/suggest", headers=headers).status_code == 200
    return sid, headers


def _fitted(client: TestClient, headers: dict[str, str], models: list[str], horizon: int = 6):
    fit = client.post(f"{BASE}/analysis/fit", json={"model_ids": models}, headers=headers)
    assert fit.status_code == 200, fit.text[:400]
    sel = client.post(
        f"{BASE}/analysis/select",
        json={"horizon": horizon, "folds": 3, "models": models},
        headers=headers,
    )
    assert sel.status_code == 200, sel.text[:400]
    diag = client.post(f"{BASE}/analysis/diagnose", json={}, headers=headers)
    assert diag.status_code == 200, diag.text[:400]


def _forecasted(client: TestClient, headers: dict[str, str], **body: Any):
    payload = {"horizon": 6, **body}
    r = client.post(f"{BASE}/analysis/forecast", json=payload, headers=headers)
    assert r.status_code == 200, r.text[:600]


def _evaluated(client: TestClient, headers: dict[str, str], **body: Any):
    payload = {"horizon": 6, "folds": 3, "n_paths": 2000, "reps": 3, **body}
    r = client.post(f"{BASE}/analysis/evaluate", json=payload, headers=headers)
    assert r.status_code == 200, r.text[:800]
    return r


@pytest.fixture(scope="module")
def continuous_run(client):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sid, headers = _prepared(client)
        _fitted(client, headers, ["ets", "drift"])
        _forecasted(client, headers)
        r = _evaluated(client, headers, coverages=[0.5, 0.8, 0.95])
        read = client.get(f"{BASE}/analysis/evaluate", headers=headers).json()
    return {
        "sid": sid, "headers": headers, "response": r.json(), "text": r.text,
        "report": read["evaluation"], "options": read["options"],
    }


@pytest.fixture(scope="module")
def branch_runs(client):
    out: dict[str, dict[str, Any]] = {}
    cases = [
        ("count", "outbreak_counts", ["poisson", "neg_binomial"]),
        ("intermittent", "spare_parts", ["croston", "tsb"]),
        ("variance", "volatile_returns", ["garch", "naive"]),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, example, models in cases:
            sid, headers = _prepared(client, example)
            _fitted(client, headers, models)
            _forecasted(client, headers)
            r = _evaluated(client, headers)
            read = client.get(f"{BASE}/analysis/evaluate", headers=headers).json()
            out[label] = {
                "sid": sid, "headers": headers, "response": r.json(),
                "report": read["evaluation"], "options": read["options"],
            }
    return out


# --------------------------------------------------------------------------- #
# 1. Automatic metric-selection policy
# --------------------------------------------------------------------------- #
def test_policy_uses_probabilistic_scores_when_a_distribution_is_available():
    p = choose_policy(data_class=DataClass.CONTINUOUS, n=180, period=12, horizon=6,
                      probabilistic=True)
    assert p.cluster == "probabilistic" and p.primary == "crps"
    assert {"crps", "pinball", "pit", "winkler", "coverage", "mase"} <= set(p.metrics)
    assert p.reasons and p.probabilistic


def test_policy_respects_short_series_intermittent_and_zero_heavy():
    short = choose_policy(data_class=DataClass.CONTINUOUS, n=35, period=12, horizon=6)
    assert short.cluster == "short_series" and short.primary == "mase"
    assert set(short.metrics) == {"mase", "mae", "rmse", "coverage"}

    intermittent = choose_policy(data_class=DataClass.INTERMITTENT, n=96, period=12,
                                 horizon=6, zero_share=0.72)
    assert intermittent.cluster == "intermittent_zero_heavy"
    assert intermittent.primary == "mase"
    assert {"mase", "rmsse", "crps"} <= set(intermittent.metrics)
    assert {"mape", "smape", "mpe"} <= set(intermittent.banned)

    zero_heavy = choose_policy(data_class=DataClass.CONTINUOUS, n=180, period=12,
                               horizon=6, zero_share=0.40)
    assert zero_heavy.cluster == "intermittent_zero_heavy"


def test_policy_reasons_notes_and_bans_are_always_string_sequences():
    cases = [
        dict(data_class=DataClass.CONTINUOUS, n=180, period=12, horizon=6, probabilistic=True),
        dict(data_class=DataClass.CONTINUOUS, n=35, period=12, horizon=6),
        dict(data_class=DataClass.INTERMITTENT, n=96, period=12, horizon=6, zero_share=0.7),
        dict(data_class=DataClass.COUNT, n=60, period=12, horizon=6, discrete=True),
        dict(data_class=DataClass.CONTINUOUS, n=741, horizon=6, target_kind="variance"),
        dict(data_class=DataClass.CONTINUOUS, n=180, period=12, horizon=6,
             probabilistic=False, short_series=False),
    ]
    for kwargs in cases:
        p = choose_policy(**kwargs)
        for field in (p.reasons, p.notes, p.banned):
            assert isinstance(field, tuple), (kwargs, type(field))
            assert all(isinstance(x, str) and len(x) > 1 for x in field)


def test_policy_switches_for_count_and_variance_targets():
    count = choose_policy(data_class=DataClass.COUNT, n=60, period=12, horizon=6,
                          discrete=True)
    assert count.cluster == "count_discrete" and count.primary == "discrete_crps"
    assert count.discrete and "mape" in count.banned
    variance = choose_policy(data_class=DataClass.CONTINUOUS, n=741, period=None,
                             horizon=6, target_kind="variance")
    assert variance.cluster == "volatility_target" and variance.primary == "qlike"
    assert {"qlike", "mse_variance", "kupiec", "christoffersen"} <= set(variance.metrics)
    assert variance.variance and "crps" in variance.banned


# --------------------------------------------------------------------------- #
# 2. Point metrics and their property tests
# --------------------------------------------------------------------------- #
def test_training_only_mase_scale_and_perfect_forecast_is_zero():
    assert mase_scale([0, 1, 2, 3, 4]) == pytest.approx(1.0)
    actual = np.asarray([5.0, 6.0])
    naive = np.asarray([4.0, 5.0])
    scales = [1.0, 1.0]
    errors = (actual - actual).reshape(2, 1)
    pm = point_metrics(actual=actual, forecast=actual, benchmark=naive,
                       scales=scales, errors_by_horizon=errors, allow_mape=False)
    assert pm.mae == 0 and pm.rmse == 0
    assert pm.mase == 0 and pm.rmsse == 0
    assert pm.mase_by_horizon == [0.0]
    assert pm.rel_mae == 0 and pm.rel_rmse == 0


def test_naive_forecast_mase_is_one():
    actual = np.asarray([5.0, 6.0])
    naive = np.asarray([4.0, 5.0])
    scales = [1.0, 1.0]
    errors = (actual - naive).reshape(2, 1)
    pm = point_metrics(actual=actual, forecast=naive, benchmark=naive,
                       scales=scales, errors_by_horizon=errors, allow_mape=False)
    assert pm.mase == pytest.approx(1.0)
    assert pm.rmsse == pytest.approx(1.0)
    assert pm.rel_mae == pytest.approx(1.0)
    assert pm.mase_by_horizon == [pytest.approx(1.0)]


def test_near_zero_actual_refuses_mape_without_dropping_rows():
    actual = np.asarray([0.0, 1.0, 2.0])
    forecast = np.asarray([0.0, 1.0, 2.0])
    pm = point_metrics(actual=actual, forecast=forecast, benchmark=forecast,
                       scales=[1.0], allow_mape=True)
    assert pm.n == 3 and pm.n_excluded == 0 and pm.n_near_zero == 1
    assert pm.mape is None and pm.mpe is None
    assert pm.smape == 0.0
    assert any("Rows were not dropped" in w for w in pm.warnings)
    assert any("0–200%" in n for n in pm.notes)


def test_policy_banned_percentage_metrics_are_all_absent():
    actual = np.asarray([1.0, 2.0, 3.0])
    pm = point_metrics(actual=actual, forecast=actual, benchmark=actual,
                       scales=[1.0], allow_mape=False)
    assert (pm.mpe, pm.mape, pm.smape, pm.gmrae, pm.mdrae) == (None, None, None, None, None)
    assert any("banned" in w.lower() for w in pm.warnings)
    assert pm.mae == 0 and pm.mase == 0


# --------------------------------------------------------------------------- #
# 3. Probabilistic, discrete and variance scores
# --------------------------------------------------------------------------- #
def test_crps_sorted_draw_identity_and_deterministic_ensemble():
    per, mean = crps_ensemble(np.zeros((20, 3)), np.zeros(3))
    assert np.allclose(per, 0.0) and mean == pytest.approx(0.0)
    paths = np.asarray([[0.0], [2.0]])
    per2, mean2 = crps_ensemble(paths, np.asarray([1.0]))
    assert per2[0] == pytest.approx(0.5) and mean2 == pytest.approx(0.5)


def test_log_score_states_the_density_caveat():
    out = log_score_normal([0.0], [0.0], [1.0])
    assert out["value"] == pytest.approx(0.5 * math.log(2 * math.pi))
    assert "DENSITY" in out["note"] and out["formula"] == "log score = -mean(log p(y))"


def test_pinball_winkler_and_coverage_are_finite_and_stated():
    # coverage_tests deliberately refuses a binomial p-value below n=10, so use
    # twelve paired observations rather than a two-row toy.
    zeros = np.zeros(12)
    band = IntervalBand(0.95, np.full(12, -1.96), np.full(12, 1.96),
                        "95% analytic test band", "analytic")
    pin = pinball_from_bands(zeros, [band])
    win = winkler_scores(zeros, [band])
    cov = coverage_curve(zeros, [band])
    assert pin["mean"] == pytest.approx(0.049, abs=1e-9)
    assert win["mean"] == pytest.approx(3.92)
    assert cov["rows"][0]["empirical"] == 1.0
    assert cov["rows"][0]["mean_width"] == pytest.approx(3.92)
    assert "construction" in cov["rows"][0] and "source" in cov["rows"][0]
    assert "Sharpness subject to calibration" in cov["philosophy"]


def test_discrete_crps_and_pmf_log_score_use_probability_not_density():
    assert discrete_crps([0.0], [1.0], 0.0) == 0.0
    assert discrete_crps([0.0], [1.0], 1.0) == 1.0
    score = discrete_log_score([0.0, 1.0], [0.25, 0.75], 1.0)
    assert score == pytest.approx(-math.log(0.75))
    assert math.isnan(discrete_log_score([0.0, 1.0], [0.25, 0.75], 1.5))


def test_randomized_pit_and_dunn_smyth_rqr_are_seeded_and_repeatable():
    n = 24
    grids = [np.asarray([0.0, 1.0]) for _ in range(n)]
    pmfs = [np.asarray([0.5, 0.5]) for _ in range(n)]
    observed = np.asarray([i % 2 for i in range(n)], dtype=float)
    ds = discrete_scores(grids=grids, pmfs=pmfs, observed=observed, reps=5, seed=17)
    assert ds.n == n and ds.crps is not None and ds.log_score is not None
    assert ds.pit.get("kind") == "discrete"
    assert all(0 <= float(v) <= 1 for v in ds.pit.get("values", []))
    assert ds.rqr.get("reps") == 5 and ds.rqr.get("combined_p") is not None
    assert any("randomized" in str(x).lower() for x in ds.notes + [ds.pit.get("caveat", "")])
    again = dunn_smyth_rqr(grids, pmfs, observed, reps=5, seed=17)
    assert again.as_dict()["values"] == ds.rqr["values"]


def test_variance_scores_and_low_power_warning():
    realized = np.ones(20)
    forecast = np.ones(20)
    assert qlike(realized, forecast) == pytest.approx(1.0)
    assert mse_variance(realized, forecast) == 0.0
    returns = np.sin(np.arange(20) / 3)
    bt = var_backtests(returns, np.ones(20) * 2)
    assert bt.n == 20 and bt.alpha == 0.05
    assert any("low power" in w.lower() for w in bt.warnings)


# --------------------------------------------------------------------------- #
# 4. Comparative tests
# --------------------------------------------------------------------------- #
def test_directional_accuracy_reports_exact_binomial_and_low_power():
    baseline = np.zeros(20)
    actual = np.ones(20)
    forecast = np.full(20, 2.0)
    hit = directional_accuracy(baseline, actual, forecast)
    assert hit.rate == 1.0 and hit.correct == 20 and hit.p is not None
    assert hit.p < 0.0001
    assert any("low power" in w.lower() for w in hit.warnings)
    confusion = hit.as_dict()["confusion"]
    assert confusion["predicted_up"]["actual_up"] == 20


def test_dm_matrix_uses_same_paired_errors_and_hac_lag():
    t = np.arange(40, dtype=float)
    e1 = np.sin(t / 4)
    e2 = e1 + 0.15 * np.cos(t / 2)
    dm = dm_matrix({"a": [e1[:20], e1[20:]], "b": [e2[:20], e2[20:]]}, horizon=2)
    assert dm["ids"] == ["a", "b"] and len(dm["matrix"]) == 2
    assert dm["matrix"][0][0] is None and dm["matrix"][1][1] is None
    assert dm["lags"] == 1 and "HAC lag" in dm["note"]
    assert len(dm["rows"]) == 2


def test_model_confidence_set_refuses_decorative_bootstrap():
    mcs = model_confidence_set({"a": [np.arange(10.0)], "b": [np.arange(10.0) + 1]})
    assert mcs["applicable"] is False and "30" in mcs["note"]


# --------------------------------------------------------------------------- #
# 5. Intermittent empirical integer PMF plumbing
# --------------------------------------------------------------------------- #
def test_croston_evaluation_pmf_is_integer_normalised_and_training_only():
    y = np.zeros(40, dtype=float)
    y[::5] = [3, 4, 5, 6, 7, 8, 9, 10]
    wrapper = CrostonModel().fit(pd.Series(y))
    assert wrapper.summary is not None and not wrapper.summary.failed
    pred = wrapper.evaluation_predictive(horizon=4, n_paths=4000, seed=91)
    assert pred.pmf.shape[0] == 4 and pred.grid.shape[0] == 4
    assert np.allclose(pred.pmf.sum(axis=1), 1.0)
    assert np.all(pred.grid == np.floor(pred.grid))
    assert not np.any(pred.truncated)
    assert pred.family == "empirical_integer_croston"
    payload = pred.as_dict(max_steps=4)
    assert payload["discrete"] is True and payload["support_kind"] == "integer"


def test_stochastic_integerisation_preserves_the_simulated_mean():
    y = np.zeros(40, dtype=float)
    y[::5] = [3, 4, 5, 6, 7, 8, 9, 10]
    wrapper = CrostonModel().fit(pd.Series(y))
    draws = wrapper._simulation_draws(4, n_paths=6000, seed=91)
    pred = wrapper.evaluation_predictive(horizon=4, n_paths=6000, seed=91)
    assert abs(float(np.mean(draws)) - float(np.mean(pred.mean))) < 0.12


# --------------------------------------------------------------------------- #
# 6. Ten S14 chart builders from a stored report
# --------------------------------------------------------------------------- #
def test_all_evaluation_chart_builders_ship_figures_alt_text_and_tables(continuous_run):
    rep = continuous_run["report"]
    horizon = int(rep["inputs"]["horizon"])
    n = int(rep["inputs"]["n"])
    series = pd.Series(
        100 + np.arange(n), dtype=float,
        index=pd.date_range("2020-01-01", periods=n, freq="MS"),
    )
    charts = [
        ev.metric_heatmap(rep["metric_rows"], policy=rep["policy"],
                          calibration_by_model=rep["probabilistic"]),
        ev.mase_by_horizon_chart(rep["mase_by_horizon"], horizon),
        ev.fold_boxplots(rep["fold_scores"], metric="mase"),
        ev.loss_curves_chart(rep["loss_curves"], horizon),
        ev.pit_histograms(rep["pit_histograms"]),
        ev.calibration_curve(rep["calibration"]["rows"]),
        ev.dm_heatmap(rep["comparative"]["dm"]),
        ev.confusion_heatmap(rep["comparative"]["hit_rate"]),
        ev.rqr_qq(rep["rqr_by_model"]),
        ev.split_chart(series, rep["folds"], n=n,
                       frequency=str(rep["inputs"]["frequency"]),
                       evaluation_window=str(rep["accuracy"]["evaluation_window"])),
    ]
    assert len(charts) == 10
    for chart in charts:
        assert chart.figure_json["data"], chart.alt_text
        assert chart.alt_text and isinstance(chart.data_behind, list)


# --------------------------------------------------------------------------- #
# 7. Protocol and leakage
# --------------------------------------------------------------------------- #
def test_rolling_origins_are_strictly_out_of_sample_and_count_missing_rows(continuous_run):
    rep = continuous_run["report"]
    folds = rep["folds"]
    assert len(folds) == int(rep["inputs"]["folds"])
    assert all(int(f["train_end"]) < int(f["test_start"]) for f in folds)
    assert all(f["zone"] == "out-of-sample only" for f in folds)
    assert all("decision_digest" in f and "benchmark" in f for f in folds)
    assert rep["inputs"]["test_observations_used_for_fitting"] == 0
    assert rep["inputs"]["test_observations_scored"] >= rep["inputs"]["n_unique_test_observations"]
    assert "n_missing_test_observations" in rep["inputs"]
    assert "NOT an accuracy score" in rep["inputs"]["s13_source"]


def test_leakage_audit_poisoned_test_invariance_is_proved(continuous_run):
    leak = continuous_run["report"]["leakage"]
    assert leak["ok"] is True
    names = {c["name"] for c in leak["checks"]}
    assert {"strict_train_test_order", "poisoned_test_invariance",
            "factory_received_training_window_only", "future_drivers_held_flat"} <= names
    assert all(c["ok"] for c in leak["checks"])
    assert leak["fold_digests"]
    for d in leak["fold_digests"]:
        assert d["decision_digest"] == d["poisoned_digest"]
        assert d["test_start"] > d["train_end"]
    assert any("Leakage is a defect" in n for n in leak["notes"])


def test_statsmodels_warnings_are_suppressed_inside_the_evaluation(continuous_run, client):
    headers = continuous_run["headers"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = client.post(
            f"{BASE}/analysis/evaluate",
            json={"horizon": 6, "folds": 3, "n_paths": 2000, "reps": 3},
            headers=headers,
        )
    assert r.status_code == 200, r.text[:400]
    bad = [
        w for w in caught
        if w.category.__name__ == "SingularMatrixWarning"
        or "adfuller" in str(w.message).lower()
        or "result_object" in str(w.message).lower()
    ]
    assert not bad, [str(w.message)[:120] for w in bad[:3]]


# --------------------------------------------------------------------------- #
# 8. API and result contract
# --------------------------------------------------------------------------- #
def test_readback_before_evaluation_and_stage_gate_are_explicit(client):
    sid, headers = _prepared(client)
    read = client.get(f"{BASE}/analysis/evaluate", headers=headers)
    assert read.status_code == 200 and read.json()["status"] == "not_started"
    assert read.json()["evaluation"] is None
    assert read.json()["options"]["policy"]["cluster"] in {
        "probabilistic", "short_series", "intermittent_zero_heavy"
    }
    assert read.json()["options"]["stage_dependencies"] == ["S13"]
    _fitted(client, headers, ["ets", "drift"])
    refused = client.post(f"{BASE}/analysis/evaluate", json={"horizon": 6, "folds": 3}, headers=headers)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "session.stage_order"
    assert refused.json()["error"]["context"]["missing"] == ["S13"]


def test_evaluation_response_is_strict_json_with_complete_result_cards(continuous_run):
    def reject(value: str):
        raise AssertionError(f"Non-finite JSON constant: {value}")

    body = json.loads(continuous_run["text"], parse_constant=reject)
    results = body["results"]
    assert len(results) >= 20 and body["stage"] == "S14"
    required = {"id", "stage", "title", "kind", "plain", "why", "method",
                "inputs", "citations", "caveats", "warnings"}
    figures: list[dict[str, Any]] = []
    for r in results:
        assert required <= set(r), r["id"]
        assert r["stage"] == "S14"
        assert r["plain"] and r["why"] and r["method"]
        # A stray string in a list unpack would turn one prose caveat into one
        # bullet per character; this guards the exact UI defect.
        assert all(isinstance(c, str) and len(c) > 1 for c in r["caveats"]), r["id"]
        assert all(isinstance(w, str) and len(w) > 1 for w in r["warnings"]), r["id"]
        if r.get("reproducible_snippet"):
            ast.parse(r["reproducible_snippet"])
        if r["kind"] == "figure":
            figures.append(r)
            assert r["caption"] and r["alt_text"]
            assert r["data_behind"]["kind"] == "table"
            assert r["data_behind"]["reproducible_snippet"]
    assert len(figures) == 10
    assert all(isinstance(c, str) and len(c) > 1 for c in body["caveats"])
    assert not re.search(r"\bp\s*=\s*(?:nan|None|null|<)", continuous_run["text"], flags=re.I)
    assert "p < 0.0001" in continuous_run["text"] or "p = 0." in continuous_run["text"]


def test_evaluation_marks_s14_complete_and_readback_does_not_rerun(continuous_run, client):
    sid, headers = continuous_run["sid"], continuous_run["headers"]
    state = client.get(f"{BASE}/sessions/{sid}", headers=headers).json()
    assert state["pipeline"]["stages"]["S14"]["status"] == "complete"
    before = client.get(f"{BASE}/analysis/evaluate", headers=headers).json()["evaluation"]
    assert before["status"] if "status" in before else True
    again = client.get(f"{BASE}/analysis/evaluate", headers=headers).json()["evaluation"]
    assert again["seconds"] == before["seconds"]
    assert again["accuracy"]["evaluation_window"] == before["accuracy"]["evaluation_window"]
    assert "evaluate" in {p.split("/")[-1] for p in app.openapi()["paths"]}


def test_api_request_validation_and_unknown_models_are_not_silent(continuous_run, client):
    headers = continuous_run["headers"]
    for payload, status in [
        ({"horizon": 0}, 422),
        ({"n_paths": 1000}, 422),
        ({"coverages": [0.55]}, 422),
        ({"scheme": "magic"}, 422),
    ]:
        r = client.post(f"{BASE}/analysis/evaluate", json=payload, headers=headers)
        assert r.status_code == status, (payload, r.text[:300])
    unknown = client.post(
        f"{BASE}/analysis/evaluate",
        json={"horizon": 6, "folds": 3, "models": ["not_a_model"], "n_paths": 2000, "reps": 3},
        headers=headers,
    )
    assert unknown.status_code == 200, unknown.text[:500]
    rep = client.get(f"{BASE}/analysis/evaluate", headers=headers).json()["evaluation"]
    assert "not_a_model" not in rep["models"]
    assert any("not_a_model" in w for w in rep["warnings"])


def test_transformed_series_requires_explicit_back_transform_for_evaluation(client):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, headers = _prepared(client, transform=True)
        _fitted(client, headers, ["ets", "drift"])
        _forecasted(client, headers, back_transform="naive")
        refused = client.post(
            f"{BASE}/analysis/evaluate",
            json={"horizon": 6, "folds": 3, "n_paths": 2000, "reps": 3},
            headers=headers,
        )
        assert refused.status_code == 400
        assert refused.json()["error"]["code"] == "evaluate.back_transform_required"
        ok = client.post(
            f"{BASE}/analysis/evaluate",
            json={"horizon": 6, "folds": 3, "n_paths": 2000, "reps": 3,
                  "back_transform": "naive"},
            headers=headers,
        )
        assert ok.status_code == 200, ok.text[:600]
        rep = client.get(f"{BASE}/analysis/evaluate", headers=headers).json()["evaluation"]
        assert rep["inputs"]["back_transform"] == "naive"
        assert rep["leakage"]["ok"] is True


def test_count_branch_scores_the_exact_discrete_pmf_and_bans_mape(branch_runs):
    rep = branch_runs["count"]["report"]
    assert rep["policy"]["cluster"] == "count_discrete"
    assert rep["policy"]["primary"] == "discrete_crps"
    assert "mape" in rep["policy"]["banned"]
    assert {"poisson", "neg_binomial"} <= set(rep["discrete"])
    for mid in ("poisson", "neg_binomial"):
        d = rep["discrete"][mid]
        assert d["crps"] is not None and d["log_score"] is not None
        assert not d.get("approximate", False)
        assert d["pit"]["kind"] == "discrete"
        assert rep["point"][mid]["mape"] is None and rep["point"][mid]["smape"] is None
        assert any("banned" in w.lower() for w in rep["point"][mid]["warnings"])
    assert any(r["id"] == "evaluate.discrete_scores" for r in branch_runs["count"]["response"]["results"])


def test_intermittent_branch_keeps_mase_policy_and_discloses_empirical_pmf(branch_runs):
    rep = branch_runs["intermittent"]["report"]
    assert rep["policy"]["cluster"] == "intermittent_zero_heavy"
    assert rep["policy"]["primary"] == "mase"
    assert {"croston", "tsb"} <= set(rep["discrete"])
    for mid in ("croston", "tsb"):
        d = rep["discrete"][mid]
        assert d.get("approximate") is True and "Empirical integer PMF" in d["construction"]
        assert any("empirical" in w.lower() for w in d["warnings"])
        assert len(rep["loss_curves"][mid]["discrete_crps"]) == int(rep["inputs"]["horizon"])
        assert "empirical integer PMF" in rep["loss_curves"][mid]["source"]
    assert any("no untransformed integer predictive PMF" in w for w in rep["warnings"])
    table = next(r for r in branch_runs["intermittent"]["response"]["results"]
                 if r["id"] == "evaluate.discrete_scores")
    assert any("empirical simulated integer PMF" in str(row.get("pmf_source")) for row in table["rows"])


def test_volatility_branch_scores_variance_and_var_backtests(branch_runs):
    rep = branch_runs["variance"]["report"]
    assert rep["policy"]["cluster"] == "volatility_target"
    assert rep["policy"]["primary"] == "qlike"
    assert "garch" in rep["variance"]
    d = rep["variance"]["garch"]
    assert d["qlike"] is not None and math.isfinite(float(d["qlike"]))
    assert {"kupiec_p", "christoffersen_p"} <= set(d["var_backtests"])
    assert any(r["id"] == "evaluate.variance_scores" for r in branch_runs["variance"]["response"]["results"])


def test_options_attach_policy_reasons_window_and_adequacy(continuous_run):
    opts = continuous_run["options"]
    rep = continuous_run["report"]
    assert opts["policy_reasons"] and opts["policy_notes"]
    assert opts["n"] == int(rep["inputs"]["n"])
    assert opts["frequency"] == rep["inputs"]["frequency"]
    assert opts["benchmark_id"] == rep["inputs"]["benchmark_id"]
    assert opts["stage_dependencies"] == ["S13"]
    assert opts["next_stage_requires"] == ["S14", "S12"]
    assert "adequacy" in opts and rep["adequacy"]
    assert "evaluation_window" in rep["inputs"]
    assert "coverage" in " ".join(opts["policy"]["metrics"])
