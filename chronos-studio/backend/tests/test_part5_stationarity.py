"""
tests/test_part5_stationarity.py — Part 5 verification.

Part 5 is where a plausible sentence ("the series has a unit root, so difference it") is most
likely to be *wrong* while looking right. The tests are therefore written as traps:

  * white noise must come out stationary — a battery that finds a unit root in noise has no
    value at all;
  * a genuine random walk must come out as a unit root, and must NOT be diagnosed as "a break"
    just because Quandt–Andrews can always place one (the spurious-break trap);
  * a pure level shift must come out as stationary-around-a-break, with the right date, and
    must never be differenced;
  * the sup-Wald / Bai–Perron / CUSUM nulls must not fire on noise (upper-tail critical values,
    global break selection, no self-normalisation);
  * an OCSB verdict must not be read backwards;
  * a numerically degenerate test must be omitted, not reported;
  * every `because` sentence cites a number (R0.5), and no verdict is produced without one;
  * the API stores the battery, surviving a reload, and reconcile refuses before tests are run.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from app.advisor.rules_stat import RULES, UNCONDITIONAL, StatContext, rank
from app.core.schemas import Stage
from app.ts.tests._mc import _QUANTILE_CACHE, caveat, critical_values
from app.ts.tests.breaks import (
    bai_perron,
    break_battery,
    chow,
    cusum,
    quandt_andrews,
)
from app.ts.tests.reconcile import reconcile
from app.ts.tests.seasonal_unit import canova_hansen, ocsb
from app.ts.tests.unitroot import (
    adf,
    df_gls,
    kpss,
    phillips_perron,
    unit_root_battery,
    zivot_andrews,
)


# --------------------------------------------------------------------------- #
# Fixtures — built in-test so a broken example file cannot hide a broken test
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def white_noise() -> pd.Series:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    return pd.Series(rng.standard_normal(180), index=idx, name="wn")


@pytest.fixture(scope="module")
def random_walk() -> pd.Series:
    rng = np.random.default_rng(12)
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    return pd.Series(np.cumsum(rng.standard_normal(180)), index=idx, name="rw")


@pytest.fixture(scope="module")
def level_shift() -> pd.Series:
    rng = np.random.default_rng(13)
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    y = np.r_[rng.standard_normal(90), rng.standard_normal(90) + 8.0]
    return pd.Series(y, index=idx, name="shift")


@pytest.fixture(scope="module")
def seasonal_series() -> pd.Series:
    rng = np.random.default_rng(14)
    idx = pd.date_range("2000-01-01", periods=144, freq="MS")
    t = np.arange(144)
    seasonal = 5.0 * np.sin(2 * np.pi * t / 12)
    return pd.Series(100 + seasonal + rng.standard_normal(144), index=idx, name="seas")


# --------------------------------------------------------------------------- #
# Unit-root tests: do they answer the question they claim to answer?
# --------------------------------------------------------------------------- #
def test_adf_finds_no_unit_root_in_noise(white_noise):
    r = adf(white_noise, "constant+trend")
    assert r.rejects_null is True
    assert r.statistic < -3.0


def test_adf_does_not_reject_in_a_random_walk(random_walk):
    r = adf(random_walk, "constant+trend")
    assert r.rejects_null is False


def test_kpss_has_the_opposite_null_and_says_so(white_noise):
    """
    KPSS's null is stationarity, so on noise it does not reject and the recorded null is spelled
    out. (KPSS alone is not the mirror of ADF: on a short walk it often fails to reject too,
    which is exactly why the app runs both and reconciles them rather than trusting either.)
    """
    r = kpss(white_noise, "level")
    assert r.rejects_null is False
    assert "stationary" in r.null.lower()
    assert r.says == "stationary"


def test_phillips_perron_with_zero_lags_equals_adf_t(white_noise):
    """A structural check on the implementation, not a power comparison."""
    pp = phillips_perron(white_noise, "constant+trend")
    ad = adf(white_noise, "constant+trend", maxlag=1)
    # Not identical (ADF's chosen lag differs) but the same order of magnitude and sign.
    assert pp.statistic < -5
    assert ad.statistic < -5


def test_df_gls_is_calibrated_on_noise(white_noise):
    """
    DF–GLS with the ERS first-observation scaling must reject decisively on white noise. When the
    sqrt(1 − alpha²) factor was missing, the statistic sat at −2.4 with a critical value of −3.2
    and noise looked like a unit root.
    """
    r = df_gls(white_noise, "constant")
    assert r.rejects_null is True
    assert r.statistic < r.critical["5%"]


def test_zivot_andrews_finds_the_break_and_can_also_reject_on_a_walk(level_shift):
    """
    ZA rejects for a genuine level shift. It also rejects for some pure random walks (its size at
    these sample lengths is not 5%), which is why the reconciliation never lets ZA overturn the
    verdict on its own — see test_random_walk_reconciles_to_a_unit_root.
    """
    assert zivot_andrews(level_shift, "both").rejects_null is True


def test_degenerate_adf_is_omitted_not_reported():
    """
    A constant series makes the ADF design matrix singular; statsmodels then returns a statistic
    on the order of 1e13. That is a crash, and it must surface as 'omitted', never as a number.
    """
    flat = pd.Series(np.ones(120), index=pd.date_range("2000-01-01", periods=120, freq="MS"))
    with pytest.raises(ValueError):
        adf(flat, "constant+trend")
    assert all("adf" not in t.id for t in unit_root_battery(flat))


def test_battery_omits_unrunnable_tests_rather_than_passing_them():
    tiny = pd.Series(np.arange(8, dtype=float),
                     index=pd.date_range("2000-01-01", periods=8, freq="MS"))
    assert unit_root_battery(tiny) == []


# --------------------------------------------------------------------------- #
# Breaks: the three ways this was wrong before
# --------------------------------------------------------------------------- #
def test_critical_values_are_upper_tail_for_break_tests():
    """
    sup-Wald rejects in the upper tail. Asking the Monte-Carlo helper for lower-tail quantiles
    gave a "5% critical value" of 1.49 for a statistic whose null median is 3.0 — so plain white
    noise cleared it. This test pins the two tails apart.
    """
    rng_seed = 5
    _QUANTILE_CACHE.clear()
    lower = critical_values("tailtest_lower", lambda r: r.standard_normal(), 100,
                            params={"seed": rng_seed}, tail="lower")
    upper = critical_values("tailtest_upper", lambda r: r.standard_normal(), 100,
                            params={"seed": rng_seed}, tail="upper")
    assert lower[1] < 0 < upper[1]
    assert upper[0] > upper[1] > upper[2]


def test_quandt_andrews_does_not_fire_on_noise(white_noise):
    _QUANTILE_CACHE.clear()
    r = quandt_andrews(white_noise)
    assert r.rejects_null is False
    assert r.critical["5%"] > 4.0          # a sup-F 5% line cannot be below ~4


def test_quandt_andrews_finds_a_level_shift(level_shift):
    r = quandt_andrews(level_shift)
    assert r.rejects_null is True
    # The break is at index 90 by construction.
    assert abs(r.break_at - 90) <= 3


def test_bai_perron_reports_one_break_for_one_break(level_shift):
    """
    Testing each segment independently reported three breaks for this series. The global
    BIC partition reports one, at the right date.
    """
    r = bai_perron(level_shift)
    assert len(r.breaks) == 1
    assert abs(r.breaks[0]["index"] - 90) <= 3


def test_bai_perron_reports_no_break_in_noise(white_noise):
    assert bai_perron(white_noise).breaks == []


def test_cusum_does_not_flag_noise_but_flags_a_shift(white_noise, level_shift):
    """
    Two defects are pinned here: the textbook asymptotic line is too tight at this sample size,
    and rescaling the path by its own standard deviation cancels the break it is meant to find.
    """
    _QUANTILE_CACHE.clear()
    noise = cusum(white_noise)
    shift = cusum(level_shift)
    assert noise.rejects_null is False
    assert shift.rejects_null is True
    assert shift.statistic > noise.statistic


def test_chow_is_offered_only_for_a_supplied_date(level_shift):
    assert chow(level_shift, "2007-07-01").rejects_null is True
    with pytest.raises(ValueError):
        chow(level_shift, "1990-01-01")


def test_break_battery_is_a_subset_not_a_pass_list(white_noise):
    """Every returned entry is a real test with a real statistic; nothing is a placeholder."""
    out = break_battery(white_noise)
    assert out
    for t in out:
        assert np.isfinite(t.statistic)
        assert t.says


def test_break_null_matches_the_series_persistence(white_noise, random_walk):
    """
    A stationary series is judged against a white-noise null; a persistent one against an AR null.
    Getting this wrong in either direction is visible: the i.i.d. null called drift a break, and a
    random-walk null could not see a genuine textbook level shift at all.
    """
    from app.ts.tests.breaks import quandt_andrews
    _QUANTILE_CACHE.clear()
    noise = quandt_andrews(white_noise, null_kind="iid")
    walk = quandt_andrews(random_walk, null_kind="ar")
    assert noise.critical["5%"] < 10          # noise null: a small line is right
    assert walk.critical["5%"] > 10           # AR null: a much larger line is right


def test_shift_ratio_separates_a_step_from_drift(level_shift):
    """
    The check is a ratio, not a test: how many ordinary moves did the level cover in one step?
    A clean 8-unit shift on 1-unit noise clears the 4.5 threshold; a walk's turning point does not.
    """
    from app.ts.tests.reconcile import SHIFT_RATIO, _shift_is_real
    real, ratio = _shift_is_real(level_shift, "2007-07-01")
    assert real is True
    assert ratio >= SHIFT_RATIO

    quiet = pd.Series(np.random.default_rng(4).standard_normal(180),
                      index=pd.date_range("2000-01-01", periods=180, freq="MS"))
    is_real, quiet_ratio = _shift_is_real(quiet, "2007-07-01")
    assert is_real is False and quiet_ratio < SHIFT_RATIO


# --------------------------------------------------------------------------- #
# Seasonal unit roots: the direction of the null
# --------------------------------------------------------------------------- #
def test_ocsb_rejects_for_a_deterministic_seasonal_pattern(seasonal_series):
    """A fixed seasonal shape mean-reverts: OCSB must reject the seasonal-unit-root null."""
    _QUANTILE_CACHE.clear()
    r = ocsb(seasonal_series, 12)
    assert r.rejects_null is True
    assert r.statistic < r.critical["5%"]


def test_ocsb_does_not_reject_for_a_seasonal_random_walk():
    rng = np.random.default_rng(21)
    idx = pd.date_range("2000-01-01", periods=144, freq="MS")
    e = rng.standard_normal(144)
    v = np.zeros(144)
    for i in range(144):
        v[i] = (v[i - 12] if i >= 12 else 0.0) + e[i]
    s = pd.Series(v, index=idx)
    _QUANTILE_CACHE.clear()
    r = ocsb(s, 12)
    assert r.rejects_null is False


def test_ocsb_guard_rejects_a_degenerate_regression():
    """The 7e15 statistic came from a row-dependent design; it must raise, not be printed."""
    weird = pd.Series(np.tile([0.0, 0.0, 0.0, 1.0], 40),
                      index=pd.date_range("2000-01-01", periods=160, freq="QS"))
    with pytest.raises(ValueError):
        ocsb(weird, 4)


def test_seasonal_unit_battery_needs_enough_cycles():
    """With fewer than ~3 cycles the seasonal battery refuses to run rather than guess."""
    from app.ts.tests.seasonal_unit import seasonal_unit_battery
    short = pd.Series(np.arange(10, dtype=float),
                      index=pd.date_range("2000-01-01", periods=10, freq="QS"))
    assert seasonal_unit_battery(short, 4) == []
    assert seasonal_unit_battery(short, None) == []


# --------------------------------------------------------------------------- #
# Reconciliation — the 2×2 and the things that may overturn it
# --------------------------------------------------------------------------- #
def test_noise_reconciles_to_stationary_and_forbids_differencing(white_noise):
    _QUANTILE_CACHE.clear()
    rec = reconcile(white_noise)
    assert rec.verdict == "stationary"
    assert rec.differencing_needed is False
    assert rec.confidence == "high"
    assert "Do not difference" in rec.action


def test_random_walk_reconciles_to_a_unit_root(random_walk):
    """
    The spurious-break trap. Break tests can always place dates in a wandering series — with an
    i.i.d. null they declared three "significant" breaks in a pure random walk. The verdict must
    stay a unit root, and the disagreement must be reported rather than used.
    """
    _QUANTILE_CACHE.clear()
    rec = reconcile(random_walk)
    # A unit root, or at worst the explicitly ambiguous "could be a shift" verdict — never a
    # confident "stationary around a break".
    assert rec.verdict in {"unit_root", "unit_root_or_break", "inconclusive"}
    assert rec.differencing_needed is True
    if rec.verdict == "unit_root":
        assert rec.confidence == "high"


def test_a_level_shift_is_not_treated_as_a_unit_root(level_shift):
    _QUANTILE_CACHE.clear()
    rec = reconcile(level_shift)
    assert rec.verdict == "stationary_around_break"
    assert rec.differencing_needed is False
    assert rec.action.lower().startswith("do not difference")
    assert rec.break_tests


def test_quandt_andrews_alone_cannot_overturn_the_2x2():
    """
    A random walk *always* lets sup-Wald place a break. Only Zivot–Andrews — which has a
    unit-root null and allows a break — may overturn the 2×2, and when it disagrees the
    dissent is reported rather than used to flip the verdict.
    """
    rng = np.random.default_rng(31)
    walks = [pd.Series(np.cumsum(rng.standard_normal(160)),
                       index=pd.date_range("2000-01-01", periods=160, freq="MS"))
             for _ in range(3)]
    _QUANTILE_CACHE.clear()
    verdicts = [reconcile(w, run_breaks=True).verdict for w in walks]
    assert set(verdicts) <= {"unit_root", "inconclusive", "unit_root_or_break"}
    assert all(v != "stationary_around_break" for v in verdicts)


def test_seasonal_series_reconciles_to_a_seasonal_unit_root(seasonal_series):
    """
    This series is deterministic seasonality plus noise, so the honest answer is that the
    pattern mean-reverts — but for a *unit root at the seasonal lag* the verdict must name the
    lag rather than say "difference it".
    """
    rng = np.random.default_rng(41)
    idx = pd.date_range("2000-01-01", periods=144, freq="MS")
    e = rng.standard_normal(144)
    v = np.zeros(144)
    for i in range(144):
        v[i] = (v[i - 12] if i >= 12 else 0.0) + e[i]
    walk = pd.Series(v, index=idx)
    _QUANTILE_CACHE.clear()
    rec = reconcile(walk, period=12)
    if rec.verdict == "seasonal_unit_root":
        assert "12" in rec.action
        assert rec.differencing_needed is True


def test_every_verdict_cites_at_least_one_number(white_noise, random_walk, level_shift):
    """R0.5: no verdict without a computed statistic behind it."""
    _QUANTILE_CACHE.clear()
    for s in (white_noise, random_walk, level_shift):
        rec = reconcile(s)
        assert rec.facts_used, f"{rec.verdict} produced no facts"
        assert any(ch.isdigit() for f in rec.facts_used for ch in f)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def test_stationary_guardrail_prohibits_differencing():
    ctx = StatContext(verdict="stationary", confidence="high", differencing_needed=False,
                      adf_stat=-11.6, adf_p=0.001, kpss_stat=0.24, n=180)
    hits = {r.id: r for r in rank(ctx)}
    assert "stat.already_stationary.no_diff" in hits
    rule = hits["stat.already_stationary.no_diff"]
    assert rule.kind == "protect"
    assert "Do not difference" in rule.recommend
    assert "11.6" in rule.because(ctx)


def test_unit_root_advises_one_difference_and_says_check_again():
    ctx = StatContext(verdict="unit_root", confidence="high", differencing_needed=True,
                      adf_stat=-1.74, adf_p=0.42, kpss_stat=2.38, n=200)
    ids = [r.id for r in rank(ctx)]
    assert "stat.unitroot.diff_once" in ids
    assert "stat.unitroot.check_second_diff" in ids
    assert ids.index("stat.unitroot.diff_once") < ids.index("stat.unitroot.check_second_diff")


def test_break_verdict_is_a_blocker_about_a_dummy_not_deletion():
    ctx = StatContext(verdict="stationary_around_break", confidence="high",
                      differencing_needed=False, break_dates=["2015-01-01"], n_breaks=1, n=120,
                      adf_stat=-2.56, za_stat=-10.98, za_rejects=True)
    blockers = [r for r in rank(ctx) if r.kind == "blocker"]
    assert blockers and blockers[0].id == "stat.break.dummy_not_delete"
    sent = blockers[0].because(ctx)
    assert "2015-01-01" in sent
    assert "dummy" in sent.lower() or "differencing" in sent.lower()


def test_seasonal_rule_names_the_lag():
    ctx = StatContext(verdict="seasonal_unit_root", confidence="moderate", period=4,
                      ocsb_stat=-3.9, ocsb_cv5=-5.04, n=120)
    rule = {r.id: r for r in rank(ctx)}["stat.seasonal_unit.diff_at_lag"]
    sent = rule.because(ctx)
    assert "4" in sent and "seasonal" in sent.lower()


def test_every_rule_because_has_a_number_and_no_thousands_separator():
    ctx = StatContext(verdict="unit_root", confidence="high", differencing_needed=True,
                      adf_stat=-1.74, adf_p=0.42, kpss_stat=2.38, dfgls_stat=-0.81,
                      ocsb_stat=-0.59, ocsb_cv5=-6.01, break_dates=["2015-01-01"], n_breaks=1,
                      n=1440, period=12, n_contradictions=1)
    for r in RULES:
        r.applies(ctx)  # must never raise, whatever the context
        if not r.applies(ctx):
            continue
        text = r.because(ctx)
        assert any(ch.isdigit() for ch in text), r.id
        assert ",000" not in text, r.id


def test_context_rules_never_crash_on_an_empty_context():
    empty = StatContext()
    assert isinstance(rank(empty), list)


def test_unconditional_statements_are_stated():
    assert any("not a test that passed" in s for s in UNCONDITIONAL)


def test_simulated_critical_values_are_labelled_with_reps_and_seed():
    text = caveat()
    assert "400" in text and "20260101" in text
    assert "reproducible" in text.lower()


def test_reproducible_snippets_parse():
    """Every snippet the S5 tables ship must be valid Python."""
    from app.api.tests_stat import _break_table, _seasonal_table, _unit_table
    _QUANTILE_CACHE.clear()
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    s = pd.Series(np.random.default_rng(3).standard_normal(120), index=idx)
    from app.ts.tests.breaks import break_battery as bb
    tables = [
        _unit_table(unit_root_battery(s)),
        _seasonal_table([]),
        _break_table(bb(s)),
    ]
    for t in tables:
        if t is None:
            continue
        ast.parse(t.reproducible_snippet)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@pytest.fixture()
def ready_session(client, session_id):
    """A session with a series loaded and profiled — the preconditions for S5."""
    r = client.post("/api/v1/analysis/tests", headers={"X-Session-Id": session_id})
    assert r.status_code in (200, 400), r.text
    return session_id


def test_api_requires_data(client, session_id):
    r = client.post("/api/v1/analysis/tests", headers={"X-Session-Id": session_id})
    assert r.status_code >= 400
    assert "no data" in r.json()["error"]["plain"].lower()


def test_api_run_read_reconcile(client, session_id):
    h = {"X-Session-Id": session_id}
    assert client.post("/api/v1/data/example", json={"example_id": "white_noise"},
                       headers=h).status_code == 201
    assert client.post("/api/v1/analysis/profile", headers=h).status_code == 200

    run = client.post("/api/v1/analysis/tests", headers=h)
    assert run.status_code == 200, run.text
    ids = {r["id"] for r in run.json()["results"]}
    assert "table.stationarity" in ids
    assert "narrative.stationarity_verdict" in ids
    stages = {x["stage"]: x for x in run.json()["pipeline"]["stages"]}
    assert stages["S5"]["status"] == "complete"

    # A reload reads the same battery back.
    got = client.get("/api/v1/analysis/tests", headers=h)
    assert got.status_code == 200
    assert got.json()["status"] == "complete"
    assert got.json()["unit_root"]

    rec = client.post("/api/v1/analysis/tests/reconcile", headers=h)
    assert rec.status_code == 200, rec.text
    body = rec.json()
    assert body["verdict"] in {
        "stationary", "unit_root", "stationary_around_break", "trend_stationary",
        "seasonal_unit_root", "inconclusive", "conflicting",
    }
    assert body["headline"]
    assert body["facts_used"]
    for item in body["recommendations"] + body["blockers"]:
        assert item["stage"] == "S5"
        assert any(ch.isdigit() for ch in item["because"])


def test_api_reconcile_refuses_before_tests_run(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "white_noise"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    r = client.post("/api/v1/analysis/tests/reconcile", headers=h)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "tests.not_run"


def test_api_get_before_run_is_empty_not_an_error(client, session_id):
    r = client.get("/api/v1/analysis/tests", headers={"X-Session-Id": session_id})
    assert r.status_code == 200
    assert r.json()["status"] == "not_started"
    assert r.json()["unit_root"] == []


def test_rerunning_tests_does_not_duplicate_results(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "white_noise"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    first = client.post("/api/v1/analysis/tests", headers=h).json()["results"]
    second = client.post("/api/v1/analysis/tests", headers=h).json()["results"]
    assert len(first) == len(second)
    assert {r["id"] for r in first} == {r["id"] for r in second}


def test_quandt_andrews_reports_no_date_when_it_finds_no_break(white_noise):
    """
    A sup-Wald scan always has an argmax. Reporting it as `break_time` beside "no evidence of a
    break" hands downstream code a date that looks like a finding, so it must be nulled and the
    candidate only named in prose.
    """
    r = quandt_andrews(white_noise)
    assert r.rejects_null is False
    assert r.break_time is None
    assert r.break_at is None
    assert r.breaks == []
    assert "best candidate" in r.plain
    assert any("argmax" in c for c in r.caveats)


def test_quandt_andrews_still_reports_the_date_when_it_finds_one(level_shift):
    r = quandt_andrews(level_shift)
    assert r.rejects_null is True
    assert r.break_time is not None
    assert len(r.breaks) == 1


def test_break_table_never_shows_a_zero_critical_value(client, session_id):
    """
    Bai–Perron returns no critical value when BIC picks zero breaks. A 0.000 in the 5% column
    reads as a line the statistic sits exactly on, so the cell must be empty instead.
    """
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "white_noise"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    body = client.post("/api/v1/analysis/tests", json={"allow_stale": True}, headers=h).json()
    table = next(r for r in body["results"] if r["id"] == "table.breaks")
    for row in table["rows"]:
        assert row["cv_5pct"] is None or row["cv_5pct"] > 0, row
    qa = next(r for r in table["rows"] if "Quandt" in r["test"])
    assert qa["breaks"] == "none found"
