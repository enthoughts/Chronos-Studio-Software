"""
tests/test_part14_forecast.py — Part 14 (§5.11, §6.7–11).

The tests are organised around the five ways a forecast lies while looking correct:

1. **A band with no construction.** A 95% ribbon could be a normal formula, a PMF
   quantile or 2,000 simulated paths; without the label it is not checkable.
2. **A horizon that outruns the data.** H > n/5 must be refused until the user
   acknowledges it, and short seasonal history must be flagged.
3. **A silently exponentiated log forecast.** Back-transformation must be an
   explicit choice, and bias adjustment must change the mean without touching the
   monotone quantiles.
4. **Simulation that leaks the test period.** The innovation pool is in-sample
   residuals from the final fit only; the bundle says so with a zero.
5. **A forecast without its S12 adequacy verdict.** Every result produced here
   carries the verdict, the coverage/construction, and the warnings.

API tests walk the real stage order, because every gate in this app is a real one.
"""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from app.charts.primitives import fan_chart
from app.main import app
from app.ts.forecast.backtransform import (
    TransformSpec,
    apply_back_transform,
    inverse_boxcox,
    inverse_yeojohnson,
)
from app.ts.forecast.combine import Component, combine_components, ic_weights
from app.ts.forecast.core import build_bundle, single_core
from app.ts.forecast.horizons import COVERAGES, advise_horizon
from app.ts.forecast.intervals import (
    MIN_PATHS,
    IntervalBand,
    analytic_bands,
    make_draws,
    simulate_bands,
)
from app.ts.forecast.reconcile import reconcile
from app.ts.models.predictive import poisson_pmf, summarise


# --------------------------------------------------------------------------- #
# Small fakes
# --------------------------------------------------------------------------- #
class _FakeWrapper:
    """A fitted model with a known, alpha-dependent symmetric interval."""

    def __init__(self, y: pd.Series, sd: float = 2.0) -> None:
        self.y = pd.Series(y).astype(float)
        self.sd = float(sd)
        self._resid = np.asarray([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25] * 4)

    def predict(self, horizon: int = 1, alpha: float = 0.05) -> pd.DataFrame:
        z = float(stats.norm.ppf(1 - alpha / 2))
        half = self.sd * z * np.sqrt(np.arange(1, horizon + 1))
        mean = np.full(horizon, float(self.y.iloc[-1]))
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=np.arange(1, horizon + 1))

    def residuals(self) -> np.ndarray:
        return self._resid


def _draws(seed: int = 11, paths: int = MIN_PATHS, horizon: int = 6, n_resid: int = 32):
    return make_draws(seed, paths, horizon, n_resid)


def _bundle_kwargs(horizon: int = 6) -> dict[str, Any]:
    return dict(
        model_id="fake",
        display_name="Fake model",
        horizon=horizon,
        index=[str(i) for i in range(1, horizon + 1)],
        coverages=list(COVERAGES),
        interval_method="auto",
        non_gaussian=False,
        seed=11,
        n_paths=MIN_PATHS,
        transform_spec=None,
        back_transform=None,
        residuals=np.asarray([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25] * 4),
        adequacy={
            "status": "evaluated", "adequacy": "adequate", "headline": "Residuals adequate.",
            "reasons": [], "remedies": [], "model_id": "fake", "source": "S12",
        },
        horizon_advice=advise_horizon(100, horizon, period=12).as_dict(),
        scenario={"note": "No drivers.", "held_flat": [], "comparison": None},
        warnings=[],
        caveats=[],
        inputs={"n_train": 100, "frequency": "MS", "period": 12},
    )


# --------------------------------------------------------------------------- #
# 1. Horizon advisor
# --------------------------------------------------------------------------- #
def test_horizon_advisor_refuses_a_long_horizon_until_acknowledged():
    advice = advise_horizon(100, 30, period=12)
    assert advice.long_horizon and advice.requires_acknowledgement and not advice.allowed
    assert advice.recommended_max == 20
    assert "100" in advice.warnings[0] and "30" in advice.warnings[0]

    accepted = advise_horizon(100, 30, period=12, acknowledged=True)
    assert accepted.allowed and accepted.acknowledged
    assert "acknowledgement" in accepted.advice.lower()

    short = advise_horizon(100, 12, period=12)
    assert short.allowed and not short.long_horizon and not short.season_short


def test_horizon_advisor_flags_less_than_two_seasons():
    a = advise_horizon(20, 3, period=12)
    assert a.season_short and not a.long_horizon
    assert "two full" in " ".join(a.warnings)
    assert not advise_horizon(24, 3, period=12).season_short


# --------------------------------------------------------------------------- #
# 2. Intervals
# --------------------------------------------------------------------------- #
def test_analytic_bands_carry_coverage_and_construction_and_are_nested():
    w = _FakeWrapper(pd.Series(np.arange(30.0)))
    point, bands = analytic_bands(w.predict, 6, COVERAGES)
    assert point.size == 6
    assert [b.coverage for b in bands] == list(COVERAGES)
    for b in bands:
        assert f"{b.coverage:.0%}" in b.construction
        assert "predict" in b.construction
        assert bool(b.lower.size) and np.all(b.lower <= b.upper)
    for inner, outer in zip(bands, bands[1:]):
        assert np.all(inner.lower >= outer.lower - 1e-9)
        assert np.all(inner.upper <= outer.upper + 1e-9)


def test_simulation_is_seeded_reproducible_and_never_below_2000_paths():
    point = np.full(6, 10.0)
    kwargs = dict(
        point=point,
        lower95=point - 4.0, upper95=point + 4.0,
        residuals=np.random.default_rng(3).normal(size=40),
        coverages=COVERAGES,
    )
    a = simulate_bands(**kwargs, draws=_draws(17))
    b = simulate_bands(**kwargs, draws=_draws(17))
    c = simulate_bands(**kwargs, draws=_draws(18))
    assert a.info["n_paths"] >= MIN_PATHS
    assert np.allclose(a.paths, b.paths), "the same seed must reproduce every path"
    assert not np.allclose(a.paths, c.paths), "a different seed must change the draws"
    # A residual bootstrap over a small pool can land on the same empirical quantile
    # under two seeds; the *paths* are the reproducible object, not just the band.
    assert "Monte Carlo" in a.bands[0].construction
    assert "50%" in a.bands[0].construction
    assert "seed=17" in a.bands[0].construction
    assert a.info["seed"] == 17
    with pytest.raises(ValueError, match="at least 2,000"):
        make_draws(1, 1000, 6, 10)


def test_discrete_intervals_are_integer_pmf_quantiles():
    lam = 4.3
    g, p, t = poisson_pmf(lam)
    pred = summarise([(g, p, t)] * 5, alpha=0.05, family="poisson")
    sim = simulate_bands(
        point=np.full(5, lam), lower95=pred.lower, upper95=pred.upper,
        residuals=None, draws=_draws(5, horizon=5), coverages=COVERAGES,
        discrete_predictive=pred,
    )
    assert sim.info["distribution"] == "discrete_pmf"
    for band in sim.bands:
        assert np.all(np.isclose(band.lower, np.round(band.lower)))
        assert np.all(np.isclose(band.upper, np.round(band.upper)))
        assert np.all(band.lower <= band.upper)
        assert "50%" not in band.construction or "PMF" in band.construction
    for inner, outer in zip(sim.bands, sim.bands[1:]):
        assert np.all(inner.lower >= outer.lower)
        assert np.all(inner.upper <= outer.upper)


def test_fan_chart_draws_one_band_per_coverage_and_ships_the_numbers():
    s = pd.Series(np.arange(20.0), index=pd.date_range("2020-01-01", periods=20, freq="MS"))
    bands = [
        {"coverage": c, "lower": np.arange(4.0) - c, "upper": np.arange(4.0) + c,
         "construction": f"{c:.0%} test band", "source": "analytic"}
        for c in COVERAGES
    ]
    chart = fan_chart(s, index=[f"2021-{i:02d}" for i in range(1, 5)],
                      point=np.arange(4.0) + 20, bands=bands, title="Fan")
    # observed + 2 traces per band + point = 12
    assert len(chart.figure_json["data"]) == 1 + 2 * len(bands) + 1
    assert len(chart.data_behind) == 4
    assert "point" in chart.data_behind[0] and "lower_95pct" in chart.data_behind[0]
    assert "95%" in chart.alt_text


# --------------------------------------------------------------------------- #
# 3. Back-transformation
# --------------------------------------------------------------------------- #
def _bands_for_transform() -> list[IntervalBand]:
    return [
        IntervalBand(c, np.full(3, 2.0 - z), np.full(3, 2.0 + z),
                     f"{c:.0%} analytic test band", "analytic", 1 - c)
        for c, z in zip(COVERAGES, [0.674, 0.954, 1.282, 1.645, 1.96])
    ]


def test_three_back_transform_methods_change_the_mean_but_not_the_quantiles():
    spec = TransformSpec("log", {"base": "e", "offset": 0.0})
    r = np.asarray([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25])
    point = np.full(3, 2.0)
    paths = np.full((20, 3), 2.0)
    results = {}
    for method in ("naive", "smearing", "lognormal"):
        p, bands, new_paths, info = apply_back_transform(
            point=point, bands=_bands_for_transform(), paths=paths,
            spec=spec, method=method, residuals=r,
        )
        assert info.applied and info.as_dict()["silent_exponentiation"] is False
        assert info.scale == "original"
        results[method] = (p, bands, new_paths, info)

    naive = results["naive"][0]
    assert np.allclose(naive, math.exp(2.0))
    assert np.allclose(results["smearing"][0], math.exp(2.0) * np.mean(np.exp(r)))
    var = (2 * 1.96 / (2 * float(stats.norm.ppf(0.975)))) ** 2
    assert np.allclose(results["lognormal"][0], math.exp(2.0 + var / 2.0))
    assert results["smearing"][3].factor is not None
    # Quantiles are exact monotone transforms: identical under all three methods.
    for a, b in [("naive", "smearing"), ("naive", "lognormal")]:
        for x, y in zip(results[a][1], results[b][1]):
            assert np.allclose(x.lower, y.lower) and np.allclose(x.upper, y.upper)
    assert np.allclose(results["naive"][2], math.exp(2.0))


def test_back_transform_is_never_silent_and_lognormal_needs_a_log_scale():
    spec = TransformSpec("log", {"base": "e"})
    with pytest.raises(ValueError, match="choose one of"):
        apply_back_transform(point=np.zeros(2), bands=_bands_for_transform(), paths=None,
                             spec=spec, method=None, residuals=np.ones(10))
    with pytest.raises(ValueError, match="log-scale"):
        apply_back_transform(point=np.zeros(2), bands=_bands_for_transform(), paths=None,
                             spec=TransformSpec("sqrt", {}), method="lognormal",
                             residuals=np.ones(10))
    with pytest.raises(ValueError, match="at least eight"):
        apply_back_transform(point=np.zeros(2), bands=_bands_for_transform(), paths=None,
                             spec=spec, method="smearing", residuals=np.ones(3))


def test_boxcox_and_yeojohnson_inverses_round_trip():
    for lam in (-1.0, 0.0, 0.4, 1.0, 2.0):
        x = np.asarray([-2.0, -0.5, 0.3, 2.0, 5.0])
        assert np.allclose(inverse_yeojohnson(stats.yeojohnson(x, lam), lam), x, atol=1e-10)
    for lam in (0.0, 0.3, 1.5):
        x = np.asarray([0.5, 1.0, 2.0, 10.0])
        assert np.allclose(inverse_boxcox(stats.boxcox(x, lam), lam), x, atol=1e-10)


# --------------------------------------------------------------------------- #
# 4. Reconciliation and combination
# --------------------------------------------------------------------------- #
def test_reconciliation_is_coherent_and_bottom_up_keeps_the_disaggregate():
    B = np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    a = np.asarray([32.0, 33.0, 34.0])
    for method in ("bottom_up", "top_down", "ols"):
        rec = reconcile(bottom_names=["North", "South"], base_bottom=B, base_total=a, method=method)
        assert rec.coherence_error < 1e-9
        assert np.allclose(rec.reconciled_bottom.sum(axis=0), rec.reconciled_total)
        assert rec.base_coherence_error == 2.0
        assert "95%" in rec.as_dict()["interval_construction"]
    bu = reconcile(bottom_names=["N", "S"], base_bottom=B, base_total=a, method="bottom_up")
    assert np.allclose(bu.reconciled_bottom, B)
    td = reconcile(bottom_names=["N", "S"], base_bottom=B, base_total=a, method="top_down")
    assert np.allclose(td.reconciled_total, a)
    ols = reconcile(bottom_names=["N", "S"], base_bottom=B, base_total=a, method="ols")
    assert np.all(ols.reconciled_total >= np.minimum(bu.reconciled_total, td.reconciled_total) - 1e-9)
    assert np.all(ols.reconciled_total <= np.maximum(bu.reconciled_total, td.reconciled_total) + 1e-9)


def test_reconciliation_refuses_a_single_bottom_series():
    with pytest.raises(ValueError, match="at least two"):
        reconcile(bottom_names=["only"], base_bottom=[[1.0, 2.0]], base_total=[1.0, 2.0])


def _component(idx: int, point: np.ndarray, aicc: float | None, comparable: bool = True) -> Component:
    d = _draws(21, horizon=point.size)
    w = _FakeWrapper(pd.Series(np.arange(30.0)))
    _, bands = analytic_bands(lambda h, alpha: pd.DataFrame({
        "mean": point, "lower": point - 2.0 * (1 - alpha), "upper": point + 2.0 * (1 - alpha)
    }), point.size, COVERAGES)
    sim = simulate_bands(
        point=point, lower95=bands[-1].lower, upper95=bands[-1].upper,
        residuals=w.residuals(), draws=d, coverages=COVERAGES,
    )
    return Component(f"m{idx}", f"Model {idx}", point, bands, sim.paths, aicc, comparable)


def test_combination_mean_median_and_ic_weights():
    p1, p2 = np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 4.0, 5.0])
    c1, c2 = _component(1, p1, 100.0), _component(2, p2, 104.0)
    mean = combine_components([c1, c2], "mean")
    assert np.allclose(mean.point, (p1 + p2) / 2)
    assert sum(mean.info["weights"].values()) == pytest.approx(1.0)
    assert len(mean.simulation_bands) == len(COVERAGES)
    med = combine_components([c1, c2], "median")
    assert np.allclose(med.point, np.median(np.vstack([p1, p2]), axis=0))
    ic = combine_components([c1, c2], "ic_weighted")
    w, note, _ = ic_weights([c1, c2])
    assert w[0] > w[1] > 0 and "IN-SAMPLE" in note
    assert ic.point[0] < mean.point[0]
    with pytest.raises(ValueError, match="comparable"):
        combine_components([c1, _component(3, p2, None, False)], "ic_weighted")


# --------------------------------------------------------------------------- #
# 5. Bundle-level safeguards
# --------------------------------------------------------------------------- #
def test_bundle_states_every_construction_and_uses_no_test_observations():
    train = pd.Series(np.arange(100.0), index=pd.date_range("2010-01-31", periods=100, freq="MS"))
    wrapper = _FakeWrapper(train)
    d = _draws(31, horizon=6)
    core, warns = single_core(wrapper, horizon=6, coverages=COVERAGES, draws=d)
    bundle = build_bundle(core, **_bundle_kwargs())
    payload = bundle.as_dict()
    assert payload["intervals_stated"] is True
    assert payload["test_observations_used"] == 0
    assert payload["inputs"]["test_observations_used"] == 0
    assert len(payload["bands"]) == len(COVERAGES)
    assert all(b["construction"] and b["coverage"] for b in payload["bands"])
    assert payload["adequacy"]["adequacy"] == "adequate"
    assert payload["transform"]["silent_exponentiation"] is False
    assert len(wrapper.y) == len(train), "the fit's own series must not be extended by the forecast"
    assert np.allclose(payload["point"], wrapper.predict(6, 0.05)["mean"].to_numpy())
    assert not warns


def test_simulation_defaults_for_non_gaussian_models():
    wrapper = _FakeWrapper(pd.Series(np.arange(40.0)))
    d = _draws(41, horizon=5)
    core, _ = single_core(wrapper, horizon=5, coverages=COVERAGES, draws=d,
                          non_gaussian=True)
    kwargs = _bundle_kwargs(horizon=5)
    kwargs["non_gaussian"] = True
    bundle = build_bundle(core, **kwargs)
    assert bundle.interval_method == "simulation"
    assert bundle.simulation is not None and bundle.simulation["n_paths"] >= MIN_PATHS


# --------------------------------------------------------------------------- #
# 6. API surface
# --------------------------------------------------------------------------- #
def _prepared(client: TestClient, example: str = "traffic_spend") -> tuple[str, dict[str, str]]:
    base = "/api/v1"
    sid = client.post(f"{base}/sessions").json()["session_id"]
    headers = {"X-Session-Id": sid}
    assert client.post(f"{base}/data/example", json={"example_id": example}, headers=headers).status_code == 201
    for stage in ("profile", "decompose", "tests"):
        assert client.post(f"{base}/analysis/{stage}", headers=headers).status_code == 200
    assert client.get(f"{base}/analysis/correlate", headers=headers).status_code == 200
    assert client.post(f"{base}/analysis/gate", headers=headers).status_code == 200
    assert client.post(f"{base}/analysis/suggest", headers=headers).status_code == 200
    return sid, headers


def _fitted(client: TestClient, headers: dict[str, str], models: list[str], horizon: int = 6):
    base = "/api/v1"
    fit = client.post(f"{base}/analysis/fit", json={"model_ids": models}, headers=headers)
    assert fit.status_code == 200, fit.text[:400]
    sel = client.post(f"{base}/analysis/select",
                      json={"horizon": horizon, "folds": 3, "models": models}, headers=headers)
    assert sel.status_code == 200, sel.text[:400]
    diag = client.post(f"{base}/analysis/diagnose", json={}, headers=headers)
    assert diag.status_code == 200, diag.text[:400]


def test_forecast_readback_before_anything_is_run():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        r = c.get("/api/v1/analysis/forecast", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "not_started" and body["forecast"] is None
        assert body["options"]["coverages"] == list(COVERAGES)
        assert body["options"]["min_paths"] == MIN_PATHS
        assert body["options"]["transform_required"] is False


def test_forecast_is_refused_before_selection():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        assert c.post("/api/v1/analysis/fit", json={"model_ids": ["ets", "drift"]}, headers=headers).status_code == 200
        r = c.post("/api/v1/analysis/forecast", json={"horizon": 6}, headers=headers)
        assert r.status_code in (400, 409)
        assert r.json()["error"]["code"] == "session.stage_order"


def test_full_forecast_flow_ships_bands_verdict_and_resultbase_cards():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "drift"])
        r = c.post("/api/v1/analysis/forecast", json={"horizon": 6}, headers=headers)
        assert r.status_code == 200, r.text[:800]
        body = r.json()
        ids = {x["id"] for x in body["results"]}
        assert {"forecast.adequacy", "forecast.horizon", "fig.forecast.fan",
                "forecast.points", "forecast.intervals", "forecast.construction"} <= ids
        for res in body["results"]:
            assert res["plain"] and res["why"] and res["method"]
            assert res["stage"] == "S13"
            assert res["caveats"], "the adequacy verdict must travel with every result"
            assert "test_observations_used" in res["inputs"]
            snippet = res.get("reproducible_snippet") or ""
            if snippet:
                ast.parse(snippet)
        fan = next(x for x in body["results"] if x["id"] == "fig.forecast.fan")
        assert fan["figure_json"]["data"] and fan["alt_text"]
        assert fan["data_behind"]["rows"], "a figure must ship its numbers"

        stored = c.get("/api/v1/analysis/forecast", headers=headers).json()
        assert stored["status"] == "complete"
        f = stored["forecast"]
        assert len(f["bands"]) == 5
        assert f["intervals_stated"] is True
        assert f["test_observations_used"] == 0
        assert f["simulation"]["n_paths"] >= MIN_PATHS
        assert f["seed"] == f["simulation"]["draws"]["seed"]
        assert f["adequacy"]["adequacy"] in {"adequate", "adequate_with_caveats", "inadequate"}
        assert f["horizon_advice"]["n"] == f["inputs"]["n_train"]
        assert f["transform"]["applied"] is False
        assert all(b["construction"] and b["coverage_pct"] for b in f["bands"])
        assert len(f["point"]) == 6 and len(f["index"]) == 6


def test_long_horizon_is_refused_until_acknowledged():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "drift"])
        r = c.post("/api/v1/analysis/forecast", json={"horizon": 60}, headers=headers)
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "forecast.horizon_acknowledgement"
        assert "one fifth" in err["plain"] and err["context"]["recommended_max"] < 60
        advice = c.get("/api/v1/analysis/forecast/horizon?horizon=60", headers=headers).json()
        assert advice["requires_acknowledgement"] and not advice["allowed"]
        ok = c.post("/api/v1/analysis/forecast",
                    json={"horizon": 60, "acknowledge_long_horizon": True}, headers=headers)
        assert ok.status_code == 200, ok.text[:600]
        f = c.get("/api/v1/analysis/forecast", headers=headers).json()["forecast"]
        assert f["horizon"] == 60 and f["horizon_advice"]["acknowledged"] is True
        assert any("one fifth" in w for w in f["warnings"])


def test_simulation_request_and_the_2000_path_floor():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "drift"])
        bad = c.post("/api/v1/analysis/forecast", json={"horizon": 6, "n_paths": 1000}, headers=headers)
        assert bad.status_code == 422
        ok = c.post("/api/v1/analysis/forecast",
                    json={"horizon": 6, "interval_method": "simulation", "seed": 4242}, headers=headers)
        assert ok.status_code == 200, ok.text[:600]
        f = c.get("/api/v1/analysis/forecast", headers=headers).json()["forecast"]
        assert f["interval_method"] == "simulation"
        assert all(b["source"] == "simulation" for b in f["bands"])
        assert f["simulation"]["draws"]["seed"] == 4242
        assert "Monte Carlo" in f["bands"][0]["construction"]


def test_transformed_series_requires_an_explicit_bias_adjustment_choice():
    with TestClient(app) as c:
        base = "/api/v1"
        sid = c.post(f"{base}/sessions").json()["session_id"]
        headers = {"X-Session-Id": sid}
        assert c.post(f"{base}/data/example", json={"example_id": "traffic_spend"}, headers=headers).status_code == 201
        assert c.post(f"{base}/analysis/profile", headers=headers).status_code == 200
        applied = c.post(f"{base}/preprocess/apply",
                         json={"op": "transform.log", "params": {"base": "e"}, "reason": "variance"},
                         headers=headers)
        assert applied.status_code == 200, applied.text[:400]
        for stage in ("decompose", "tests"):
            assert c.post(f"{base}/analysis/{stage}", headers=headers).status_code == 200
        assert c.get(f"{base}/analysis/correlate", headers=headers).status_code == 200
        assert c.post(f"{base}/analysis/gate", headers=headers).status_code == 200
        assert c.post(f"{base}/analysis/suggest", headers=headers).status_code == 200
        _fitted(c, headers, ["ets", "drift"])
        opts = c.get(f"{base}/analysis/forecast", headers=headers).json()["options"]
        assert opts["transform_required"] is True
        refused = c.post(f"{base}/analysis/forecast", json={"horizon": 6}, headers=headers)
        assert refused.status_code == 400
        assert refused.json()["error"]["code"] == "forecast.back_transform_required"

        bands_by_method = {}
        for method in ("naive", "smearing", "lognormal"):
            r = c.post(f"{base}/analysis/forecast",
                       json={"horizon": 6, "back_transform": method}, headers=headers)
            assert r.status_code == 200, r.text[:600]
            f = c.get(f"{base}/analysis/forecast", headers=headers).json()["forecast"]
            assert f["transform"]["applied"] is True and f["transform"]["method"] == method
            assert f["transform"]["silent_exponentiation"] is False
            assert f["transform"]["formula"]
            bands_by_method[method] = [(b["lower"], b["upper"]) for b in f["bands"]]
        assert bands_by_method["naive"] == bands_by_method["smearing"] == bands_by_method["lognormal"], (
            "bias adjustment changes the mean, never the monotone interval quantiles"
        )


def test_combination_mean_is_the_backend_average_of_its_components():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "drift"])
        singles = {}
        for mid in ("ets", "drift"):
            r = c.post("/api/v1/analysis/forecast",
                       json={"horizon": 6, "model_id": mid}, headers=headers)
            assert r.status_code == 200, r.text[:600]
            singles[mid] = np.asarray(c.get("/api/v1/analysis/forecast", headers=headers).json()["forecast"]["point"])
        r = c.post("/api/v1/analysis/forecast",
                   json={"horizon": 6, "combination": "mean", "models": ["ets", "drift"]},
                   headers=headers)
        assert r.status_code == 200, r.text[:600]
        f = c.get("/api/v1/analysis/forecast", headers=headers).json()["forecast"]
        assert f["combination"]["method"] == "mean"
        assert set(f["combination"]["component_ids"]) == {"ets", "drift"}
        assert sum(f["combination"]["weights"].values()) == pytest.approx(1.0)
        assert np.allclose(f["point"], (singles["ets"] + singles["drift"]) / 2)


def test_ic_weighted_combination_refuses_incomparable_criteria_with_a_usable_fix():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "drift"])
        r = c.post("/api/v1/analysis/forecast",
                   json={"horizon": 6, "combination": "ic_weighted", "models": ["ets", "drift"]},
                   headers=headers)
        # `drift` is a baseline with no likelihood, so this is a legitimate refusal.
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "forecast.combination_refused"
        assert "mean" in r.json()["error"]["fix"]


def test_scenario_paths_change_the_exogenous_forecast_and_are_disclosed():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "arimax"], horizon=6)
        sc = c.post("/api/v1/analysis/scenario",
                    json={"horizon": 6,
                          "paths": {"MarketingSpend": [300, 305, 310, 315, 320, 325]}},
                    headers=headers)
        assert sc.status_code == 200, sc.text[:400]
        r = c.post("/api/v1/analysis/forecast",
                   json={"horizon": 6, "model_id": "arimax"}, headers=headers)
        assert r.status_code == 200, r.text[:800]
        assert "forecast.scenario" in {x["id"] for x in r.json()["results"]}
        f = c.get("/api/v1/analysis/forecast", headers=headers).json()["forecast"]
        assert "MarketingSpend" in f["scenario"]["drivers"]
        assert f["scenario"]["comparison"] is not None
        assert f["scenario"]["comparison"]["changed"] is True
        assert f["scenario"]["comparison"]["max_abs_change"] > 0


def test_count_forecast_defaults_to_simulation_and_ships_a_pmf():
    with TestClient(app) as c:
        _, headers = _prepared(c, "outbreak_counts")
        _fitted(c, headers, ["poisson", "neg_binomial"])
        r = c.post("/api/v1/analysis/forecast", json={"horizon": 6}, headers=headers)
        assert r.status_code == 200, r.text[:800]
        assert "forecast.discrete" in {x["id"] for x in r.json()["results"]}
        f = c.get("/api/v1/analysis/forecast", headers=headers).json()["forecast"]
        assert f["interval_method"] == "simulation"
        assert f["discrete"] is not None and f["simulation"]["distribution"] == "discrete_pmf"
        for b in f["bands"]:
            assert all(float(x).is_integer() for x in b["lower"] + b["upper"])


def test_inadequate_s12_verdict_travels_with_every_forecast_result():
    from app.core.session import get_store

    with TestClient(app) as c:
        sid, headers = _prepared(c)
        _fitted(c, headers, ["ets", "drift"])
        sess = get_store().get(sid)
        winner = c.get("/api/v1/analysis/selection", headers=headers).json()["report"]["winner"]
        sess.objects["diagnostics"] = {
            "version": sess.state.series_version,
            "blocks": {
                winner: {
                    "status": "ok",
                    "verdict": {
                        "adequacy": "inadequate",
                        "headline": "Ljung–Box and ARCH-LM both reject; revise the mean and variance.",
                        "reasons": [], "remedies": ["Add a seasonal term."],
                    },
                }
            },
        }
        r = c.post("/api/v1/analysis/forecast", json={"horizon": 6}, headers=headers)
        assert r.status_code == 200, r.text[:600]
        f = c.get("/api/v1/analysis/forecast", headers=headers).json()["forecast"]
        assert f["adequacy"]["adequacy"] == "inadequate"
        assert any("INADEQUATE" in w or "inadequate" in w for w in f["caveats"])
        for res in r.json()["results"]:
            assert any("residual" in c.lower() or "Ljung" in c for c in res["caveats"])
        verdict = next(x for x in r.json()["results"] if x["id"] == "forecast.adequacy")
        assert verdict["verdict"] == "ready_with_caveats"
        assert "INADEQUATE" in verdict["headline"]


def test_hierarchy_endpoint_is_coherent_and_readable():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "drift"])
        bottom = {
            "North": [10 + i * 0.4 + (i % 12) for i in range(40)],
            "South": [20 + i * 0.2 - (i % 12) * 0.3 for i in range(40)],
        }
        r = c.post("/api/v1/analysis/forecast/hierarchy",
                   json={"bottom": bottom, "model_id": "ets", "horizon": 4, "method": "ols"},
                   headers=headers)
        assert r.status_code == 200, r.text[:800]
        assert {"forecast.hierarchy.total", "forecast.hierarchy.bottom",
                "fig.forecast.hierarchy"} <= {x["id"] for x in r.json()["results"]}
        rec = c.get("/api/v1/analysis/forecast/hierarchy", headers=headers).json()["reconciliation"]
        assert rec["coherence_error"] < 1e-8
        bottoms = np.asarray(rec["reconciled_bottom"])
        assert np.allclose(bottoms.sum(axis=0), np.asarray(rec["reconciled_total"]))
        assert "95%" in rec["interval_construction"]


def test_hierarchy_refuses_a_non_univariate_model():
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "arimax"])
        r = c.post("/api/v1/analysis/forecast/hierarchy",
                   json={"bottom": {"A": list(range(20)), "B": list(range(20))},
                         "model_id": "arimax", "horizon": 3},
                   headers=headers)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "forecast.hierarchy_model_unsupported"


def test_a_changed_series_refuses_a_stale_forecast():
    from app.core.session import get_store

    with TestClient(app) as c:
        sid, headers = _prepared(c)
        _fitted(c, headers, ["ets", "drift"])
        sess = get_store().get(sid)
        sess.state.series_version += 1
        r = c.post("/api/v1/analysis/forecast", json={"horizon": 6}, headers=headers)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "forecast.stale_fits"


def test_resultbase_caveats_are_unique_and_warnings_are_not_a_second_copy():
    """Regressions on the running UI: duplicate React keys and two identical callouts."""
    from app.api.forecast import _common
    from app.core.schemas import Stage

    adequacy = {
        "headline": "The structural failures do not corroborate each other.",
        "adequacy": "adequate_with_caveats",
        "source": "S12 /analysis/diagnose",
    }
    long = "A 60-step horizon exceeds one fifth of the 180 training observations."
    warnings = [
        "No back-transformation was needed: the fitted series is on the original scale.",
        "No back-transformation was needed: the fitted series is on the original scale.",
        f"Residual adequacy verdict for this fit: adequate_with_caveats. {adequacy['headline']}",
        long,
    ]
    common = _common(Stage.S13_FORECAST, adequacy, warnings)
    assert len(common["caveats"]) == len(set(common["caveats"]))
    assert common["caveats"].count(warnings[0]) == 1
    assert not any(w.startswith("Residual adequacy verdict for this fit:") for w in common["caveats"])
    assert common["warnings"] == [long], "only a decision-changing warning is repeated in red"

    from pathlib import Path
    card = (Path(__file__).resolve().parents[2] / "frontend/src/components/ResultCard.tsx").read_text()
    assert "key={c}" not in card and "key={w}" not in card
    assert 'key={`caveat-${i}`}' in card
    assert 'key={`warning-${i}`}' in card
    assert 'key={`travelling-${i}`}' in card


def test_future_index_labels_do_not_print_a_midnight_timestamp():
    from app.api.forecast import _index_labels
    from app.core.session import get_store

    with TestClient(app) as c:
        sid, headers = _prepared(c)
        labels = _index_labels(get_store().get(sid), 6)
        assert len(labels) == 6
        assert all("00:00:00" not in label for label in labels)
        assert labels[0].startswith("20")


def test_hierarchy_without_a_model_id_defaults_to_the_step11_winner():
    """The UI must not forward the exogenous forecast model into a standalone refit."""
    with TestClient(app) as c:
        _, headers = _prepared(c)
        _fitted(c, headers, ["ets", "arimax"])
        bottom = {
            "North": [10 + i * 0.4 + (i % 12) for i in range(40)],
            "South": [20 + i * 0.2 - (i % 12) * 0.3 for i in range(40)],
        }
        auto = c.post(
            "/api/v1/analysis/forecast/hierarchy",
            json={"bottom": bottom, "horizon": 4, "method": "ols"},
            headers=headers,
        )
        assert auto.status_code == 200, auto.text[:800]
        stored = c.get("/api/v1/analysis/forecast/hierarchy", headers=headers).json()
        assert stored["request"]["model_id"] is None
        default_total = np.asarray(stored["reconciliation"]["reconciled_total"], dtype=float)

        explicit = c.post(
            "/api/v1/analysis/forecast/hierarchy",
            json={"bottom": bottom, "model_id": "ets", "horizon": 4, "method": "ols"},
            headers=headers,
        )
        assert explicit.status_code == 200, explicit.text[:800]
        ets_total = np.asarray(
            c.get("/api/v1/analysis/forecast/hierarchy", headers=headers).json()
            ["reconciliation"]["reconciled_total"], dtype=float,
        )
        assert np.allclose(default_total, ets_total), "the omitted id resolved to Step 11's winner"
