"""
tests/test_part4_decompose.py — Part 4 verification.

Decomposition is the step where a picture can look convincing while being wrong, so the tests are
written as the specific misreadings:

  * a strong seasonal pattern must be *found* (F_S high) and a noisy non-seasonal one must not be
    dressed up as one — the seasonal rule must not fire on white noise alone;
  * the multiplicative guardrail must refuse on non-positive data, not silently switch to additive;
  * SHARES must sum to 1 and must be computed so that a multiplicative decomposition does not
    report "100% trend, 0% everything else" (which is what comparing levels around 1 does);
  * the trend test must run on the deseasonalised series — testing a smoothed component gives
    p < 0.0001 for a trend in white noise;
  * the periodogram must be taken after detrending, or the "strongest cycle" is always the whole
    sample;
  * a periodogram peak at a non-calendar lag must be called a cycle and must not be given a
    seasonal term;
  * a remainder that is still autocorrelated must be reported as "not noise", even when it holds
    a tiny share of the variance;
  * every method must be runnable and return components that add back up to the observation.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from app.advisor.rules_decompose import RULES, UNCONDITIONAL, DecompContext, rank
from app.core.schemas import Stage
from app.ts.decompose.components import (
    cox_stuart,
    cycle_finding,
    fisher_g,
    mann_kendall,
    remainder_finding,
    scorecard,
    seasonal_finding,
    sens_slope,
    trend_finding,
)
from app.ts.decompose.methods import (
    check_model,
    classical,
    decompose,
    hp_filter,
    mstl,
    stl,
)


# --------------------------------------------------------------------------- #
# Fixtures built in-test, so a broken example file cannot hide a broken test
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def seasonal_series() -> pd.Series:
    """Trend + a fixed 12-period seasonal shape + small noise: the textbook additive case."""
    rng = np.random.default_rng(5)
    t = np.arange(144)
    idx = pd.date_range("2000-01-01", periods=144, freq="MS")
    y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.standard_normal(144)
    return pd.Series(y, index=idx, name="seasonal")


@pytest.fixture(scope="module")
def noisy_series() -> pd.Series:
    rng = np.random.default_rng(6)
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    return pd.Series(rng.standard_normal(180), index=idx, name="noise")


@pytest.fixture(scope="module")
def seasonal_decomp(seasonal_series) -> "object":
    return stl(seasonal_series, 12)


# --------------------------------------------------------------------------- #
# Methods: do they actually decompose?
# --------------------------------------------------------------------------- #
def test_stl_components_add_back_up(seasonal_series, seasonal_decomp):
    """
    An additive decomposition is an identity: trend + seasonal + remainder = observed. If that
    does not hold, the picture is not a decomposition of the series.
    """
    total = (seasonal_decomp.trend + seasonal_decomp.seasonal + seasonal_decomp.remainder)
    np.testing.assert_allclose(total.to_numpy(dtype=float),
                              seasonal_series.to_numpy(dtype=float), rtol=1e-8, atol=1e-8)


def test_shares_are_a_partition(seasonal_decomp):
    shares = seasonal_decomp.shares
    assert set(shares) == {"trend", "seasonal", "remainder"}
    assert abs(sum(shares.values()) - 1.0) < 1e-8
    assert all(v >= -1e-12 for v in shares.values())


def test_multiplicative_shares_do_not_collapse(seasonal_series):
    """
    The parts of a multiplicative decomposition are ratios around 1, so comparing their raw
    variances scores the trend as 'the level' and everything else as zero. The shares are put on
    the same footing as the additive case; this test pins that.
    """
    dec = stl(seasonal_series, 12, "multiplicative")
    assert abs(sum(dec.shares.values()) - 1.0) < 1e-6
    assert dec.shares["seasonal"] > 0.02, dec.shares
    assert abs(float(dec.seasonal.mean()) - 1.0) < 0.15


def test_every_method_runs_and_keeps_the_index(seasonal_series):
    for fn in (lambda: stl(seasonal_series, 12),
               lambda: classical(seasonal_series, 12),
               lambda: mstl(seasonal_series, [12]),
               lambda: hp_filter(seasonal_series)):
        dec = fn()
        assert len(dec.trend) == len(seasonal_series)
        assert list(dec.trend.index) == list(seasonal_series.index)
        assert np.isfinite(np.asarray(dec.trend, dtype=float)).any()


def test_classical_loses_the_ends_and_says_so(seasonal_series):
    dec = classical(seasonal_series, 12)
    missing = int(dec.trend.isna().sum())
    assert missing > 0
    assert any("moving average leaves the first and last" in c for c in dec.caveats)


def test_stl_requires_two_cycles():
    short = pd.Series(np.arange(10, dtype=float),
                      index=pd.date_range("2000-01-01", periods=10, freq="MS"))
    with pytest.raises(ValueError, match="two full cycles"):
        stl(short, 12)


# --------------------------------------------------------------------------- #
# The multiplicative guardrail
# --------------------------------------------------------------------------- #
def test_multiplicative_is_refused_on_non_positive_data():
    """The §5.4 guardrail: refuse with a reason, never silently switch to additive."""
    y = pd.Series([1.0, 2.0, 0.0, 4.0] * 12,
                  index=pd.date_range("2000-01-01", periods=48, freq="MS"))
    with pytest.raises(ValueError, match="positive"):
        check_model("multiplicative", y)
    with pytest.raises(ValueError, match="positive"):
        stl(y, 12, "multiplicative")
    # additive is fine on the same data
    assert stl(y, 12, "additive").shares


def test_multiplicative_is_allowed_on_positive_data(seasonal_series):
    assert (seasonal_series > 0).all()
    check_model("multiplicative", seasonal_series)


# --------------------------------------------------------------------------- #
# Trend: tested on the right series
# --------------------------------------------------------------------------- #
def test_mann_kendall_and_sens_slope_on_a_known_trend():
    y = pd.Series(np.r_[np.arange(50, dtype=float), np.arange(50, dtype=float)[::-1] + 50])
    tau, p = mann_kendall(pd.Series(np.arange(100, dtype=float)))
    assert tau == pytest.approx(1.0)
    assert p < 1e-10
    slope, lo, hi = sens_slope(pd.Series(np.arange(100, dtype=float)))
    assert slope == pytest.approx(1.0)
    assert lo <= slope <= hi
    assert np.isfinite(tau)


def test_cox_stuart_detects_a_drift():
    y = pd.Series(np.arange(60, dtype=float))
    z, p = cox_stuart(y)
    assert p < 0.05
    flat = pd.Series(np.tile([0.0, 1.0], 30))
    assert cox_stuart(flat)[1] > 0.5


def test_trend_test_is_run_on_the_deseasonalised_series(noisy_series):
    """
    The trap this pins: running Mann–Kendall on the *smoothed trend component* gives p < 0.0001
    for a trend in pure white noise, because a loess line is massively autocorrelated by
    construction. On the deseasonalised series it must be non-significant.
    """
    dec = stl(noisy_series, 12)
    finding = trend_finding(dec)
    assert finding.p_value is not None and finding.p_value > 0.05
    assert "seasonally adjusted" in finding.plain
    assert any("independent observations" in c for c in finding.caveats)


def test_trend_direction_words_match_the_numbers(seasonal_series):
    finding = trend_finding(stl(seasonal_series, 12))
    assert "upward" in finding.plain
    assert "+" in finding.plain.split("over the window")[1][:40]


def test_trend_finding_survives_a_tiny_series():
    tiny = pd.Series(np.arange(5, dtype=float),
                     index=pd.date_range("2000-01-01", periods=5, freq="MS"))
    finding = trend_finding(stl(tiny, 2) if False else _forced_tiny(tiny))
    assert finding.statistic is None
    assert "not enough observations" in finding.plain.lower()


def _forced_tiny(series: pd.Series):
    """A Decomposition-shaped object with a 5-point trend, to exercise the guard branch."""
    from app.ts.decompose.methods import Decomposition

    return Decomposition(
        method="stl", model="additive", label="stub", observed=series,
        trend=series, seasonal=pd.Series(0.0, index=series.index),
        remainder=pd.Series(0.0, index=series.index),
        shares={"trend": 1.0, "seasonal": 0.0, "remainder": 0.0},
    )


# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #
def test_seasonality_is_found_and_quantified(seasonal_series, seasonal_decomp):
    finding = seasonal_finding(seasonal_decomp, seasonal_series, 12)
    assert finding.band == "strong"
    assert finding.statistic is not None and finding.statistic > 0.64
    assert finding.table and len(finding.table) == 12
    assert "peak" in finding.plain or "high point" in finding.plain
    # amplitude is reported against the mean, not in a vacuum
    assert "%" in finding.plain


def test_white_noise_is_not_called_seasonal(noisy_series):
    dec = stl(noisy_series, 12)
    finding = seasonal_finding(dec, noisy_series, 12)
    assert finding.band in {"weak", "none"}
    assert (finding.statistic or 0) < 0.64


def test_peak_position_is_correct(seasonal_series, seasonal_decomp):
    """sin peaks at position 4 of a 12-cycle starting at t=0 (quarter of the way through)."""
    finding = seasonal_finding(seasonal_decomp, seasonal_series, 12)
    effects = {r["position"]: r["effect"] for r in finding.table}
    peak = max(effects, key=lambda k: effects[k])
    assert 3 <= peak <= 5


def test_no_period_means_no_seasonal_component(seasonal_series, seasonal_decomp):
    finding = seasonal_finding(seasonal_decomp, seasonal_series, None)
    assert finding.statistic is None
    assert "No seasonal period was supplied" in finding.plain


# --------------------------------------------------------------------------- #
# Cycle vs seasonality
# --------------------------------------------------------------------------- #
def test_fisher_g_is_not_called_on_a_pure_trend(seasonal_series, seasonal_decomp):
    """
    A periodogram of a trending series puts all its power at the lowest frequency, so the
    'strongest cycle' is always 'the whole sample'. The finding must be computed on the detrended
    series and must not call the trend a cycle.
    """
    finding = cycle_finding(seasonal_series, 12, seasonal_decomp.trend)
    assert finding.table
    top_period = finding.table[0]["period"]
    assert top_period is not None and top_period < len(seasonal_series) * 0.8


def test_fisher_g_rejects_on_a_planted_cycle():
    rng = np.random.default_rng(9)
    t = np.arange(200)
    y = pd.Series(np.sin(2 * np.pi * t / 25) + 0.3 * rng.standard_normal(200))
    g, p, period = fisher_g(y)
    assert p < 0.05
    assert 20 <= period <= 30


def test_fisher_g_calibration_is_stated():
    """It over-rejects at a nominal 5%, so a 1% bar is used and the caveat says so."""
    finding = cycle_finding(pd.Series(np.random.default_rng(2).standard_normal(144)), None)
    assert any("over-rejects" in c for c in finding.caveats)


def test_a_non_seasonal_cycle_is_named_a_cycle_not_a_season():
    rng = np.random.default_rng(11)
    t = np.arange(240)
    y = pd.Series(np.sin(2 * np.pi * t / 40) + 0.25 * rng.standard_normal(240),
                  index=pd.date_range("2000-01-01", periods=240, freq="MS"))
    dec = stl(y, 12)
    finding = cycle_finding(y, 12, dec.trend)
    if finding.band != "none":
        assert "cycle" in finding.plain.lower()
        assert "NOT the seasonal period" in finding.plain or "not the seasonal period" in finding.plain


# --------------------------------------------------------------------------- #
# Remainder
# --------------------------------------------------------------------------- #
def test_remainder_of_white_noise_is_called_noise(noisy_series):
    dec = stl(noisy_series, 12)
    finding = remainder_finding(dec)
    assert "no autocorrelation left" in finding.plain


def test_autocorrelated_remainder_is_reported_as_structure():
    """
    A series whose remainder is an AR(1) with phi = 0.8: the decomposition will not remove it, and
    the app must say the remainder still has structure rather than calling it noise.
    """
    rng = np.random.default_rng(13)
    n = 180
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.standard_normal()
    t = np.arange(n)
    y = pd.Series(50 + 0.1 * t + e, index=pd.date_range("2000-01-01", periods=n, freq="MS"))
    dec = stl(y, 12)
    finding = remainder_finding(dec)
    assert finding.p_value is not None and finding.p_value < 0.05
    assert any("has not extracted all the predictable structure" in c for c in finding.caveats)


def test_constant_remainder_is_a_warning_not_a_triumph():
    """A deterministic series leaves a constant remainder; the honest response is suspicion."""
    t = np.arange(120, dtype=float)
    y = pd.Series(10 + 0.2 * t, index=pd.date_range("2000-01-01", periods=120, freq="MS"))
    dec = stl(y, 12)
    finding = remainder_finding(dec)
    assert "warning, not a triumph" in finding.plain


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
def test_scorecard_sentence_handles_remainder_dominance(noisy_series):
    dec = stl(noisy_series, 12)
    sc = scorecard(dec, noisy_series, 12)
    assert "remainder dominates" in sc["plain"]
    assert len(sc["rows"]) == 3


def test_scorecard_names_the_main_component(seasonal_series):
    dec = stl(seasonal_series, 12)
    sc = scorecard(dec, seasonal_series, 12)
    top = max(sc["rows"], key=lambda r: r["share"])["component"]
    assert top.lower() in sc["plain"].lower()
    assert "%" in sc["plain"]


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def test_strong_seasonality_forces_a_seasonal_term():
    ctx = DecompContext(period=12, seasonal_strength=0.9, seasonal_amplitude_pct=40.0, n=144)
    hits = {r.id: r for r in rank(ctx)}
    assert "dec.seasonal.needs_season" in hits
    assert "0.90" in hits["dec.seasonal.needs_season"].because(ctx)


def test_weak_seasonality_makes_the_term_optional():
    ctx = DecompContext(period=12, seasonal_strength=0.1, n=144)
    ids = {r.id for r in rank(ctx)}
    assert "dec.seasonal.weak_omit" in ids
    assert "dec.seasonal.needs_season" not in ids


def test_small_remainder_with_autocorrelation_is_not_reassuring():
    """
    The most common misreading in this step: "the remainder is tiny, so the series is solved".
    The rule must say the opposite, and it must not print the share as 0%.
    """
    ctx = DecompContext(remainder_share=0.003, remainder_lb_p=0.004, n=144)
    rule = {r.id: r for r in rank(ctx)}["dec.remainder.not_noise"]
    text = rule.because(ctx)
    assert "0.3%" in text
    assert "did not remove the predictability" in text


def test_multiplicative_blocker_is_a_blocker():
    ctx = DecompContext(multiplicative_blocked=True)
    blockers = [r for r in rank(ctx) if r.kind == "blocker"]
    assert blockers and blockers[0].id == "dec.mult.not_possible"
    assert "not a preference" in blockers[0].because(ctx)


def test_arch_rule_points_at_the_variance():
    ctx = DecompContext(remainder_arch_p=0.001)
    rule = {r.id: r for r in rank(ctx)}["dec.remainder.arch"]
    assert "variance" in rule.recommend.lower()
    assert "0.0010" in rule.because(ctx)


def test_cycle_rule_does_not_turn_a_cycle_into_a_season():
    ctx = DecompContext(period=12, cycle_p=0.001, cycle_period=41.0, cycle_on_calendar=False)
    rule = {r.id: r for r in rank(ctx)}["dec.cycle.not_season"]
    assert "41" in rule.because(ctx)
    assert "no fixed position" in rule.because(ctx).lower()
    # ... and it does not fire when the peak IS the seasonal period
    on = DecompContext(period=12, cycle_p=0.001, cycle_period=12.0, cycle_on_calendar=True)
    assert "dec.cycle.not_season" not in {r.id for r in rank(on)}


def test_every_rule_because_has_a_number_and_no_thousands_separator():
    ctx = DecompContext(
        period=12, seasonal_strength=0.8, seasonal_amplitude_pct=35.0, seasonal_growing=True,
        trend_strength=0.9, trend_p=0.001, trend_slope=0.5, trend_share=0.7,
        remainder_share=0.003, remainder_lb_p=0.004, remainder_arch_p=0.01, remainder_jb_p=0.02,
        cycle_p=0.001, cycle_period=41.0, n=1440, nested_seasonality=True,
        multiplicative_blocked=True, n_nonpositive=3,
        methods_compared=3, trend_spread=21.1,
    )
    for r in RULES:
        if not r.applies(ctx):
            continue
        text = r.because(ctx)
        assert any(ch.isdigit() for ch in text), r.id
        assert ",000" not in text, r.id


def test_unconditional_statements_are_stated():
    assert any("description, not a model" in s for s in UNCONDITIONAL)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@pytest.fixture()
def loaded(client, session_id):
    h = {"X-Session-Id": session_id}
    assert client.post("/api/v1/data/example", json={"example_id": "airpassengers"},
                       headers=h).status_code == 201
    assert client.post("/api/v1/analysis/profile", headers=h).status_code == 200
    return h


def test_api_requires_data(client, session_id):
    r = client.post("/api/v1/analysis/decompose", json={"method": "stl"},
                    headers={"X-Session-Id": session_id})
    assert r.status_code >= 400
    assert "no data" in r.json()["error"]["plain"].lower()


def test_api_method_menu_reports_availability_and_reasons(client, loaded):
    r = client.get("/api/v1/analysis/decompose/methods", headers=loaded)
    assert r.status_code == 200
    body = r.json()
    assert body["period"] == 12
    methods = {m["method"]: m for m in body["methods"]}
    assert methods["stl"]["available"] is True
    assert "why_not" in methods["stl"]
    models = {m["model"]: m for m in body["models"]}
    assert models["multiplicative"]["available"] is True


def test_api_run_returns_charts_tables_and_advice(client, loaded):
    r = client.post("/api/v1/analysis/decompose", json={"method": "stl"}, headers=loaded)
    assert r.status_code == 200, r.text
    body = r.json()
    ids = {x["id"] for x in body["results"]}
    assert {"narrative.decomp_scorecard", "table.decomp_scorecard", "fig.decomposition",
            "fig.periodogram"} <= ids
    assert body["advisor"] is not None
    assert body["advisor"]["recommendations"]
    for rec in body["advisor"]["recommendations"] + body["advisor"]["blockers"]:
        assert rec["stage"] == "S4"
        assert any(ch.isdigit() for ch in rec["because"])
    stages = {x["stage"]: x for x in body["pipeline"]["stages"]}
    assert stages["S4"]["status"] == "complete"


def test_api_multiplicative_on_positive_data_works(client, loaded):
    r = client.post("/api/v1/analysis/decompose",
                    json={"method": "stl", "model": "multiplicative"}, headers=loaded)
    assert r.status_code == 200, r.text


def test_api_multiplicative_guardrail_names_the_offending_values(client, session_id):
    """
    A series that dips negative must be refused with the count of offending values — never
    silently switched to additive.

    This test also pins a loader bug it found: a near-constant series used to crash `/data/paste`
    with a 500, because `dispersion_ratio` is undefined when the mean is not positive and the
    schema demanded a number.
    """
    h = {"X-Session-Id": session_id}
    rng = np.random.default_rng(1)
    values = np.abs(rng.standard_normal(180)) + 1.0
    values[10] = -5.0
    series = pd.Series(values, index=pd.date_range("2000-01-01", periods=180, freq="MS"))
    text = "date,value\n" + "\n".join(f"{t.date()},{v:.6f}" for t, v in series.items())
    assert client.post("/api/v1/data/paste", json={"text": text}, headers=h).status_code == 201
    assert client.post("/api/v1/analysis/profile", headers=h).status_code == 200

    r = client.post("/api/v1/analysis/decompose",
                    json={"method": "stl", "model": "multiplicative"}, headers=h)
    assert r.status_code == 400, r.text
    body = r.json()["error"]
    assert body["code"] == "decompose.multiplicative_undefined"
    assert "positive" in body["plain"]
    assert "1 of 180" in body["plain"]

    # ... and additive still works on the same series.
    ok = client.post("/api/v1/analysis/decompose", json={"method": "stl"}, headers=h)
    assert ok.status_code == 200, ok.text


def test_api_compare_overlays_methods_and_reports_disagreement(client, loaded):
    r = client.post("/api/v1/analysis/decompose/compare",
                    json={"methods": ["stl", "classical", "hp"]}, headers=loaded)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["ran"]) == 3
    assert body["trend_spread"] is not None and body["trend_spread"] >= 0
    assert body["figure"]["data"]


def test_api_read_back_after_run(client, loaded):
    client.post("/api/v1/analysis/decompose", json={"method": "stl"}, headers=loaded)
    r = client.get("/api/v1/analysis/decompose", headers=loaded)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert body["scorecard"]["rows"]
    assert len(body["findings"]) == 4


def test_api_read_back_before_run_is_empty_not_an_error(client, session_id):
    r = client.get("/api/v1/analysis/decompose", headers={"X-Session-Id": session_id})
    assert r.status_code == 200
    assert r.json()["status"] == "not_started"


def test_rerunning_does_not_duplicate_results(client, loaded):
    first = client.post("/api/v1/analysis/decompose", json={"method": "stl"},
                        headers=loaded).json()["results"]
    second = client.post("/api/v1/analysis/decompose", json={"method": "stl"},
                         headers=loaded).json()["results"]
    assert {r["id"] for r in first} == {r["id"] for r in second}
    assert len(first) == len(second)


def test_every_reproducible_snippet_parses(client, loaded):
    body = client.post("/api/v1/analysis/decompose", json={"method": "stl"},
                       headers=loaded).json()
    for result in body["results"]:
        snippet = result.get("reproducible_snippet", "")
        assert snippet, result["id"]
        ast.parse(snippet)


def test_scorecard_does_not_give_the_remainder_a_strength_band(seasonal_series):
    """
    F_T/F_S say how strong a *signal* is against the remainder; printing "strong" beside the
    remainder would read as the opposite of the truth (a strong remainder is bad news).
    """
    dec = decompose(seasonal_series, "stl", 12)
    rows = {r["component"]: r for r in scorecard(dec, seasonal_series, 12)["rows"]}
    assert rows["Remainder"]["reading"] == "not applicable"
    assert rows["Trend"]["reading"] in {"none", "weak", "moderate", "strong"}


def test_compare_reports_undefined_strengths_as_null_not_zero(client, session_id):
    """HP and band-pass have no remainder, so F_T/F_S are undefined — a 0.000 would look like a result."""
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    r = client.post("/api/v1/analysis/decompose/compare",
                    json={"methods": ["stl", "hp"], "model": "additive"}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    by = {row["method"].split(" (")[0]: row for row in body["rows"]}
    assert by["Hodrick–Prescott"]["F_T"] is None and by["Hodrick–Prescott"]["F_S"] is None
    assert by["STL"]["F_T"] == pytest.approx(0.996, abs=0.02)
    assert any("no separate remainder" in c for c in body["notes"])


def test_trend_band_can_never_disagree_with_its_own_drift_test(noisy_series):
    """
    F_T divides by the remainder, so a small remainder can make pure noise look like a "strong"
    trend. The band is capped by Mann–Kendall: no significant drift means at most "weak", and the
    caveat explains why F_T was high anyway.
    """
    # A positive re-scaling so the multiplicative model is legal; the point is a series whose
    # only structure is noise.
    positive = noisy_series - noisy_series.min() + 1.0
    dec = decompose(positive, "stl", 12, model="multiplicative")
    f = trend_finding(dec)
    if f.p_value is not None and f.p_value >= 0.05:
        assert f.band in {"none", "weak"}, f"p={f.p_value} but band={f.band}"
        if dec.trend_strength and dec.trend_strength >= 0.30:
            assert any("F_T" in c and "not" in c for c in f.caveats)
    else:
        assert f.band in {"weak", "moderate", "strong"}


def test_trend_band_on_a_real_trend_is_strong(seasonal_series):
    f = trend_finding(decompose(seasonal_series, "stl", 12))
    assert f.p_value < 0.05
    assert f.band == "strong"


def test_disagreeing_drift_tests_are_reported_as_disagreement(noisy_series):
    """When Cox–Stuart and Mann–Kendall split, the caveat must say so — not that they 'agree'."""
    f = trend_finding(decompose(noisy_series, "stl", 12))
    text = " ".join(f.caveats)
    assert "agrees there is drift" not in text


def test_scorecard_headline_is_the_whole_first_sentence(noisy_series):
    """
    The headline used to be `plain.split(".")[0] + "."`, which truncated "This series is 2.9%
    trend..." to "This series is 2." A headline is built as its own field, never sliced.
    """
    sc = scorecard(decompose(noisy_series, "stl", 12), noisy_series, 12)
    assert sc["headline"].startswith("This series is ")
    assert "% trend" in sc["headline"] and sc["headline"].endswith(".")
    assert sc["headline"] == sc["plain"].split("  ")[0] or len(sc["headline"]) > 30


def test_api_headline_keeps_the_decimal_percentages(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "white_noise"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    r = client.post("/api/v1/analysis/decompose", json={"method": "stl"}, headers=h)
    assert r.status_code == 200, r.text
    headline = r.json()["advisor"]["headline"]
    assert "% trend" in headline
    assert headline.strip() != "This series is 2."
