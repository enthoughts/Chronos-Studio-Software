"""
tests/test_part7_gate.py — S8: the model-readiness gate (§5.7).

The gate is the one stage whose output is a *verdict*, so its tests are mostly about integrity
rather than arithmetic:

* a check that could not run must never pass;
* "unknown" must never be reported as "fine";
* the verdict must be traceable to the checklist above it;
* a verdict about an old version of the series must be visibly stale;
* every travelling caveat must stand alone in a report, with no checklist nearby.

The one place a threshold genuinely matters — the seasonal-cycle count — is tested with the
numbers that decide it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ts.gate.readiness import (
    CHECKS,
    HORIZON_RATIO,
    MIN_CYCLES,
    MIN_N,
    GateFacts,
    assess,
)


# --------------------------------------------------------------------------- #
# Fixtures: the four canonical outcomes
# --------------------------------------------------------------------------- #
@pytest.fixture
def clean() -> GateFacts:
    """A series where every check can be evaluated and every one passes."""
    return GateFacts(
        n=240, n_original=240, n_missing=0, longest_gap=0, period=12,
        horizon=24, stationarity_verdict="stationary", stationarity_confidence="high",
        n_unit_root_tests=4, n_agreeing=3, n_contradictions=0,
        breaks_tested=True, profile_ran=True,
        differencing_needed=False, n_outliers=0, variance_ratio=0.12,
        variance_growth_pct=8.0, signal_to_noise=0.65, seasonal_strength=0.15,
        remainder_share=0.25, correlogram_pattern="white_noise", ljung_box_p=0.41,
        seasonality_handled=False, n_transformations=0,
    )


# --------------------------------------------------------------------------- #
# 1. The four verdicts
# --------------------------------------------------------------------------- #
def test_all_checks_passing_gives_ready(clean):
    g = assess(clean)
    assert g.verdict == "ready", [c.key for c in g.caveats] + [c.key for c in g.unknown]
    assert g.passed == len(g.checklist)
    assert "all" in g.headline.lower() and "pass" in g.headline.lower()


def test_the_ready_headline_does_not_count_not_applicable_as_a_pass():
    """A headline of "all 10 checks pass" above a checklist showing nine passes and one "n/a" is
    the kind of small lie that makes every other number on the card suspect."""
    g = assess(GateFacts(
        n=240, period=12, profile_ran=True, breaks_tested=True, horizon=None,
        stationarity_verdict="stationary", n_agreeing=3,
        variance_ratio=0.1, variance_growth_pct=5, seasonal_strength=0.1,
        signal_to_noise=0.5, correlogram_pattern="white_noise", ljung_box_p=0.4,
    ))
    assert g.verdict == "ready"
    assert g.passed == 9 and len(g.checklist) == 10
    assert "all 9 checks pass" in g.headline
    assert "not applicable" in g.headline


def test_a_blocker_gives_not_ready_and_outranks_everything(clean):
    f = GateFacts(**{**clean.as_dict(), "n": 12})
    g = assess(f)
    assert g.verdict == "not_ready"
    assert [c.key for c in g.blockers] == ["length"]
    assert "12 observations" in g.blockers[0].value


def test_a_caveat_alone_gives_ready_with_caveats(clean):
    f = GateFacts(**{**clean.as_dict(), "n_outliers": 4, "outliers_addressed": False})
    g = assess(f)
    assert g.verdict == "ready_with_caveats"
    assert [c.key for c in g.caveats] == ["outliers"]
    assert g.blockers == []


def test_an_unassessed_check_gives_inconclusive_not_ready(clean):
    f = GateFacts(**{**clean.as_dict(), "stationarity_verdict": None})
    g = assess(f)
    assert g.verdict == "inconclusive"
    assert [c.key for c in g.unknown] == ["stationarity"]


def test_unknowns_outrank_caveats(clean):
    """
    The ordering that matters: if stationarity has not been tested, "ready with caveats" is a
    false assurance — the caveat list is only complete for the checks that ran, and the most
    consequential one is simply missing.
    """
    f = GateFacts(**{**clean.as_dict(), "n_outliers": 4, "stationarity_verdict": None})
    g = assess(f)
    assert g.verdict == "inconclusive"
    assert [c.key for c in g.caveats] == ["outliers"], "the caveat is still reported"
    assert "outlier" in g.plain, "and still appears in the prose"


def test_a_blocker_outranks_unknowns(clean):
    f = GateFacts(**{**clean.as_dict(), "n": 5, "stationarity_verdict": None})
    assert assess(f).verdict == "not_ready"


# --------------------------------------------------------------------------- #
# 2. Integrity: nothing passes without a number
# --------------------------------------------------------------------------- #
def test_every_check_returns_a_criterion_with_a_value(clean):
    for check in CHECKS:
        c = check(clean)
        assert c.key and c.label
        assert c.value, f"{c.key} passed with no number behind it"
        assert c.detail
        assert c.state in {"pass", "fail", "unknown", "not_applicable"}


def test_no_check_can_pass_with_an_empty_measurement():
    """
    An all-defaults fact set must produce exactly one pass: the length check, which needs nothing
    but `n`. Gaps, outliers and breaks all read "nothing found" from an unrun stage, so they are
    unknown until the stage that looks has run.
    """
    g = assess(GateFacts(n=200))
    passing = [c.key for c in g.checklist if c.state == "pass"]
    # These four are decided by what the series *is* rather than by a statistical measurement:
    # its length, whether it has gaps, whether a break was tested for and found, and whether any
    # outlier was flagged. Everything that needs a measurement must be unknown.
    assert set(passing) <= {"length", "gaps", "breaks", "outliers"}, passing
    # Without a period there is nothing to handle, so `seasonality` is "not applicable" rather
    # than unknown — the distinction this module exists to preserve.
    for key in ("gaps", "stationarity", "breaks", "outliers", "variance", "signal", "correlogram"):
        assert key in {c.key for c in g.unknown}, f"{key} claimed a pass with no measurement"
    seasonality = next(c for c in g.checklist if c.key == "seasonality")
    assert seasonality.state == "not_applicable"


def test_unmeasured_seasonal_strength_is_not_a_pass():
    """
    Regression test for a real bug: with a period detected but no F_S, the seasonality check
    reported "pass (weak)" — a tick with no number behind it, which is exactly what §5.7 forbids.
    """
    g = assess(GateFacts(n=200, period=12))
    seasonality = next(c for c in g.checklist if c.key == "seasonality")
    assert seasonality.state == "unknown"
    assert "not measured" in seasonality.value
    assert "Step 4" in seasonality.why_unknown


def test_unknown_checks_name_the_stage_that_would_fix_them():
    g = assess(GateFacts(n=200, period=12))
    assert g.unknown
    for c in g.unknown:
        assert c.jump_to_stage, f"{c.key} is unknown but does not say where to fix it"
        assert c.why_unknown, f"{c.key} is unknown but does not say why"


def test_an_unset_horizon_is_not_applicable_rather_than_unknown():
    """
    An unset horizon cannot be too long. Treating it as "unknown" would make every series
    inconclusive merely for not having planned its forecast yet — and would train users to ignore
    the word.
    """
    g = assess(GateFacts(
        n=240, period=12, stationarity_verdict="stationary", n_agreeing=2,
        breaks_tested=True, profile_ran=True,
        variance_ratio=0.1, variance_growth_pct=5, seasonal_strength=0.1,
        signal_to_noise=0.5, correlogram_pattern="white_noise", ljung_box_p=0.4,
    ))
    horizon = next(c for c in g.checklist if c.key == "horizon")
    assert horizon.state == "not_applicable"
    assert horizon.passed is None
    assert g.verdict == "ready"


# --------------------------------------------------------------------------- #
# 3. The specific thresholds
# --------------------------------------------------------------------------- #
def test_the_length_floor_is_enforced_with_its_number():
    g = assess(GateFacts(n=MIN_N - 1))
    length = next(c for c in g.checklist if c.key == "length")
    assert length.state == "fail" and length.severity == "blocker"
    assert str(MIN_N) in length.detail


def test_too_few_seasonal_cycles_blocks_a_seasonal_model():
    """
    18 monthly observations is 1.5 cycles. A seasonal term estimated from that is mostly
    extrapolation — the model cannot tell a stable shape from a coincidence.
    """
    g = assess(GateFacts(n=18, period=12))
    length = next(c for c in g.checklist if c.key == "length")
    assert length.state == "fail"
    assert "1.5 seasonal cycles" in length.value
    assert g.verdict == "not_ready"


def test_enough_cycles_passes_and_reports_the_count():
    g = assess(GateFacts(n=144, period=12))
    length = next(c for c in g.checklist if c.key == "length")
    assert length.state == "pass"
    assert "12.0 complete cycles" in length.detail


def test_min_cycles_boundary_is_inclusive(clean):
    n_at_limit = int(MIN_CYCLES * 12)
    g = assess(GateFacts(**{**clean.as_dict(), "n": n_at_limit, "period": 12}))
    length = next(c for c in g.checklist if c.key == "length")
    assert length.state == "pass", length.detail


def test_a_horizon_beyond_n_over_five_is_a_caveat(clean):
    f = GateFacts(**{**clean.as_dict(), "n": 120, "horizon": 40})
    g = assess(f)
    horizon = next(c for c in g.checklist if c.key == "horizon")
    assert horizon.state == "fail"
    assert horizon.severity == "caveat"
    assert f"{HORIZON_RATIO:.0f}×" in horizon.detail


def test_a_horizon_inside_the_sample_passes(clean):
    f = GateFacts(**{**clean.as_dict(), "n": 240, "horizon": 12})
    horizon = next(c for c in assess(f).checklist if c.key == "horizon")
    assert horizon.state == "pass"
    assert "20.0×" in horizon.value


def test_a_long_gap_is_a_blocker_but_a_short_one_is_a_caveat():
    short = assess(GateFacts(n=200, profile_ran=True, n_missing=2, longest_gap=1))
    assert next(c for c in short.checklist if c.key == "gaps").severity == "caveat"

    long = assess(GateFacts(n=200, profile_ran=True, n_missing=40, longest_gap=30))
    gap = next(c for c in long.checklist if c.key == "gaps")
    assert gap.severity == "blocker"
    assert long.verdict == "not_ready"


def test_untested_breaks_are_not_a_pass():
    """The one check that could pass by finding nothing. \"No break found\" and \"no test ran\"
    are the same observation until you record that a test ran — so it is recorded."""
    g = assess(GateFacts(n=240, n_missing=0, period=12, breaks_tested=False))
    brk = next(c for c in g.checklist if c.key == "breaks")
    assert brk.state == "unknown"
    assert brk.passed is None
    assert brk.jump_to_stage == "S5"
    assert "break" in brk.why_unknown.lower()


def test_unprofiled_series_cannot_pass_its_gaps_or_outliers():
    """The same vacuous-pass trap as `breaks`, for the two counts Step 1 supplies. A series nobody
    looked at has no *known* gaps and no *known* outliers — which is not the same as none."""
    g = assess(GateFacts(n=240, profile_ran=False, period=12, n_missing=0, n_outliers=0))
    for key in ("gaps", "outliers"):
        c = next(x for x in g.checklist if x.key == key)
        assert c.state == "unknown", key
        assert c.passed is None
        assert c.jump_to_stage == "S1"
    assert g.verdict != "ready"


def test_tested_breaks_with_none_found_do_pass():
    g = assess(GateFacts(n=240, n_missing=0, period=12, breaks_tested=True, profile_ran=True, break_count=0))
    brk = next(c for c in g.checklist if c.key == "breaks")
    assert brk.state == "pass"
    assert "no break detected" in brk.value


def test_growing_variance_is_flagged_with_the_growth_figure():
    g = assess(GateFacts(n=200, variance_growth_pct=242.0, variance_ratio=0.95))
    var = next(c for c in g.checklist if c.key == "variance")
    assert var.state == "fail"
    assert "242" in var.value


def test_shrinking_variance_is_described_without_a_double_negative():
    g = assess(GateFacts(n=200, profile_ran=True, variance_ratio=0.1, variance_growth_pct=-19.0))
    v = next(c for c in g.checklist if c.key == "variance")
    assert v.state == "pass"
    assert "shrank 19%" in v.value
    assert "-19" not in v.value


def test_stable_variance_passes(clean):
    var = next(c for c in assess(clean).checklist if c.key == "variance")
    assert var.state == "pass"


def test_strong_unhandled_seasonality_blocks(clean):
    f = GateFacts(**{**clean.as_dict(), "seasonal_strength": 0.85, "seasonality_handled": False})
    g = assess(f)
    season = next(c for c in g.checklist if c.key == "seasonality")
    assert season.state == "fail" and season.severity == "blocker"
    assert g.verdict == "not_ready"


def test_strong_handled_seasonality_passes(clean):
    f = GateFacts(**{**clean.as_dict(), "seasonal_strength": 0.85, "seasonality_handled": True})
    season = next(c for c in assess(f).checklist if c.key == "seasonality")
    assert season.state == "pass"


# --------------------------------------------------------------------------- #
# 4. The stationarity criterion — the one most likely to lie
# --------------------------------------------------------------------------- #
def test_a_single_test_is_not_a_verdict(clean):
    """
    ADF and KPSS have opposite nulls; agreement between them is what makes a verdict worth
    quoting. One test is a hypothesis.
    """
    f = GateFacts(**{**clean.as_dict(), "n_agreeing": 1})
    g = assess(f)
    stat = next(c for c in g.checklist if c.key == "stationarity")
    assert stat.state == "fail"
    assert "single test" in stat.value


def test_a_verdict_with_no_supporting_test_blocks():
    """A verdict the battery does not support is not a weak finding, it is a missing one — so it
    blocks rather than travels as a caveat."""
    g = assess(GateFacts(
        n=240, period=7, profile_ran=True, breaks_tested=True,
        stationarity_verdict="seasonal_unit_root", n_agreeing=0, n_unit_root_tests=4,
        variance_ratio=0.1, variance_growth_pct=5, seasonal_strength=0.1,
        signal_to_noise=0.5, correlogram_pattern="seasonal", ljung_box_p=0.001,
    ))
    st = next(c for c in g.checklist if c.key == "stationarity")
    assert st.severity == "blocker"
    assert "no supporting test" in st.value
    assert g.verdict == "not_ready"


def test_conflicting_tests_block(clean):
    f = GateFacts(**{
        **clean.as_dict(), "stationarity_verdict": "conflicting",
        "n_agreeing": 1, "n_contradictions": 3,
    })
    g = assess(f)
    stat = next(c for c in g.checklist if c.key == "stationarity")
    assert stat.severity == "blocker"
    assert g.verdict == "not_ready"


def test_a_seasonal_verdict_does_not_overstate_agreement(clean):
    """
    Regression test: the gate reported "6 tests agree" on a *seasonal* unit root, when four of
    those six had never addressed the seasonal question. The value must name which tests agreed
    with what.
    """
    f = GateFacts(**{
        **clean.as_dict(), "stationarity_verdict": "seasonal_unit_root",
        "n_agreeing": 2, "n_regular_supporting": 4, "n_regular_tests": 8,
        "differencing_needed": True,
    })
    stat = next(c for c in assess(f).checklist if c.key == "stationarity")
    assert "2 seasonal witnesses" in stat.value
    assert "4 of 8 regular tests" in stat.value
    assert "6 tests agree" not in stat.value


def test_an_inconclusive_verdict_is_a_caveat_not_a_blocker(clean):
    f = GateFacts(**{**clean.as_dict(), "stationarity_verdict": "inconclusive"})
    g = assess(f)
    stat = next(c for c in g.checklist if c.key == "stationarity")
    assert stat.severity == "caveat"
    assert g.verdict == "ready_with_caveats"


# --------------------------------------------------------------------------- #
# 5. Travelling caveats
# --------------------------------------------------------------------------- #
def test_travelling_caveats_stand_alone(clean):
    """
    These will be attached to a forecast, a metrics table and a methods paragraph — three places
    with no checklist in sight. Each one has to make sense by itself.
    """
    f = GateFacts(**{
        **clean.as_dict(), "n_outliers": 3, "n_missing": 4,
        "differencing_needed": True, "break_count": 1, "break_dates": ["2015-01-01"],
        "horizon": 60,
    })
    g = assess(f)
    assert g.caveats_travelling
    for cav in g.caveats_travelling:
        assert cav.endswith(".")
        assert len(cav) > 30, f"too terse to stand alone: {cav!r}"
        assert " above" not in cav.lower() and "this check" not in cav.lower().replace(
            "the checks above", "")


def test_the_horizon_caveat_is_not_printed_twice():
    """One fact, one line. The horizon appears as a checklist-driven caveat and as its own
    consequence sentence; only the consequence sentence belongs in the travelling list."""
    g = assess(GateFacts(
        n=144, period=12, profile_ran=True, breaks_tested=True,
        horizon=60, stationarity_verdict="stationary", n_agreeing=2,
        variance_ratio=0.1, variance_growth_pct=5, seasonal_strength=0.1,
        signal_to_noise=0.5, correlogram_pattern="white_noise", ljung_box_p=0.4,
    ))
    hits = [c for c in g.caveats_travelling if "horizon" in c.lower()]
    assert len(hits) == 1, hits
    assert "long relative to the 144 observations" in hits[0]
    assert len(g.caveats_travelling) == len(set(g.caveats_travelling))


def test_differencing_shows_up_in_the_travelling_caveats(clean):
    f = GateFacts(**{**clean.as_dict(), "differencing_needed": True})
    joined = " ".join(assess(f).caveats_travelling)
    assert "integrated back" in joined


def test_a_clean_series_still_reports_its_stationarity(clean):
    """Even a spotless verdict must be quotable — 'never a bare stationary: yes' cuts both ways."""
    joined = " ".join(assess(clean).caveats_travelling)
    assert "stationarity verdict" in joined.lower()


# --------------------------------------------------------------------------- #
# 6. Traceability
# --------------------------------------------------------------------------- #
def test_the_verdict_is_traceable_to_the_checklist():
    """
    Every verdict headline must be derivable from the lines above it. This is the property that
    stops the summary drifting from the table.
    """
    cases = [
        GateFacts(n=5),
        GateFacts(n=240, period=12, stationarity_verdict="stationary", n_agreeing=2),
        GateFacts(n=240, period=12, stationarity_verdict="stationary", n_agreeing=2,
                  variance_ratio=0.1, variance_growth_pct=5, seasonal_strength=0.1,
                  signal_to_noise=0.5, correlogram_pattern="white_noise", ljung_box_p=0.4,
                  n_outliers=2),
        GateFacts(n=240, period=12, stationarity_verdict="stationary", n_agreeing=2,
                  variance_ratio=0.1, variance_growth_pct=5, seasonal_strength=0.1,
                  signal_to_noise=0.5, correlogram_pattern="white_noise", ljung_box_p=0.4),
    ]
    for f in cases:
        g = assess(f)
        assert g.verdict in {"ready", "ready_with_caveats", "not_ready", "inconclusive"}
        if g.verdict == "not_ready":
            assert str(len(g.blockers)) in g.headline
        if g.verdict == "ready_with_caveats":
            assert str(len(g.caveats)) in g.headline
        if g.verdict == "inconclusive":
            assert str(len(g.unknown)) in g.headline
        assert g.passed + len(g.blockers) + len(g.caveats) + len(g.unknown) + \
            sum(1 for c in g.checklist if c.state == "not_applicable") == len(g.checklist)


def test_assess_is_pure(clean):
    """Two calls with the same facts must give the same verdict, in the same order."""
    a, b = assess(clean), assess(clean)
    assert a.verdict == b.verdict
    assert [c.key for c in a.checklist] == [c.key for c in b.checklist]
    assert a.plain == b.plain


# --------------------------------------------------------------------------- #
# 7. The API
# --------------------------------------------------------------------------- #
def _run_pipeline(client, session_id: str, example: str = "airpassengers") -> dict[str, str]:
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": example}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    client.post("/api/v1/analysis/decompose", json={"method": "stl"}, headers=h)
    client.post("/api/v1/analysis/tests", json={"allow_stale": True}, headers=h)
    client.get("/api/v1/analysis/correlate", headers=h)
    return h


def test_gate_read_before_running_says_not_started(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    r = client.get("/api/v1/analysis/gate", headers=h)
    assert r.status_code == 200
    assert r.json()["status"] == "not_started"


def test_gate_produces_a_verdict_result_and_a_checklist_table(client, session_id):
    h = _run_pipeline(client, session_id)
    r = client.post("/api/v1/analysis/gate", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    ids = {x["id"] for x in body["results"]}
    assert {"verdict.readiness", "table.gate_checklist", "narrative.readiness"} <= ids

    verdict = next(x for x in body["results"] if x["id"] == "verdict.readiness")
    assert verdict["verdict"] in {"ready", "ready_with_caveats", "not_ready", "inconclusive"}
    assert verdict["headline"]
    assert len(verdict["checklist"]) == len(CHECKS)
    for item in verdict["checklist"]:
        assert item["value"], f"{item['key']} has no number"
        assert item["label"]


def test_a_break_verdict_does_not_crash_the_gate(client, session_id):
    """Regression: the break-based verdicts credit the detected breaks to the agreeing-test count,
    and the break dates were read *after* that count — `UnboundLocalError` on precisely the series
    (a level shift) the breaks check exists for."""
    h = {"X-Session-Id": session_id}
    r = client.post("/api/v1/data/example", json={"example_id": "structural_break"}, headers=h)
    assert r.status_code == 201, r.text
    for ep in ("/api/v1/analysis/profile", "/api/v1/analysis/decompose",
               "/api/v1/analysis/tests"):
        assert client.post(ep, headers=h).status_code == 200, ep
    assert client.get("/api/v1/analysis/correlate", headers=h).status_code == 200
    r = client.post("/api/v1/analysis/gate", headers=h)
    assert r.status_code == 200, r.text
    g = client.get("/api/v1/analysis/gate", headers=h).json()
    assert g["verdict"] in {"ready", "ready_with_caveats", "not_ready", "inconclusive"}
    assert g["status"] == "complete"


def test_the_gate_never_claims_stationarity_without_step_5(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    body = client.post("/api/v1/analysis/gate", headers=h).json()
    verdict = next(x for x in body["results"] if x["id"] == "verdict.readiness")
    stat = next(i for i in verdict["checklist"] if i["key"] == "stationarity")
    assert stat["passed"] is None
    assert "Step 5" in stat["detail"]


def test_the_gate_flags_airpassengers_variance_growth(client, session_id):
    """
    Airline passengers' spread grows ~242% across the window. Reading the EDA's `variance_ratio`
    (a growth multiple of 3.4) as if it were the level-to-spread correlation (0.95) would have
    called this "stable" — the mapping between the two fact dictionaries is the thing under test.
    """
    h = _run_pipeline(client, session_id)
    body = client.post("/api/v1/analysis/gate", headers=h).json()
    verdict = next(x for x in body["results"] if x["id"] == "verdict.readiness")
    var = next(i for i in verdict["checklist"] if i["key"] == "variance")
    assert var["passed"] is False, var
    assert "242" in var["value"]


def test_setting_a_long_horizon_downgrades_the_verdict(client, session_id):
    h = _run_pipeline(client, session_id)
    before = client.post("/api/v1/analysis/gate", headers=h).json()
    v_before = next(x for x in before["results"] if x["id"] == "verdict.readiness")

    r = client.post("/api/v1/analysis/gate/horizon", json={"horizon": 120}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["horizon_check"]["state"] == "fail"
    assert body["ratio"] < HORIZON_RATIO

    after = client.get("/api/v1/analysis/gate", headers=h).json()
    horizon = next(i for i in after["checklist"] if i["key"] == "horizon")
    assert horizon["state"] == "fail"
    assert any("horizon" in c.lower() for c in after["caveats_travelling"])
    assert v_before["verdict"] in {"ready", "ready_with_caveats", "inconclusive"}


def test_a_good_horizon_passes(client, session_id):
    h = _run_pipeline(client, session_id)
    r = client.post("/api/v1/analysis/gate/horizon", json={"horizon": 12}, headers=h)
    assert r.json()["horizon_check"]["state"] == "pass"
    assert r.json()["ratio"] >= HORIZON_RATIO


def test_the_stored_verdict_goes_stale_when_the_series_changes(client, session_id):
    """
    A verdict about a different version of the series is worse than no verdict: it looks current
    and is not.
    """
    h = _run_pipeline(client, session_id)
    client.post("/api/v1/analysis/gate", headers=h)
    assert client.get("/api/v1/analysis/gate", headers=h).json()["stale"] is False

    client.post("/api/v1/analysis/next/apply",
                json={"op": "difference", "order": 1, "rule_id": "next.diff.first"}, headers=h)
    after = client.get("/api/v1/analysis/gate", headers=h).json()
    assert after["stale"] is True
    assert after["current_version"] != after["computed_for_version"]


def test_travelling_caveats_reach_the_session_caveats(client, session_id):
    """
    The report reads `session.state.caveats`. A caveat that lives only in this stage's output
    would never reach it.
    """
    h = _run_pipeline(client, session_id)
    body = client.post("/api/v1/analysis/gate", headers=h).json()
    verdict = next(x for x in body["results"] if x["id"] == "verdict.readiness")
    travelling = verdict["caveats_travelling"]
    assert travelling
    facts = client.get("/api/v1/analysis/facts", headers=h).json()
    session_caveats = " ".join(facts.get("caveats", []))
    assert any(c[:40] in session_caveats for c in travelling)


def test_setting_the_horizon_updates_the_session_caveats(client, session_id):
    """A horizon set at the gate must reach the report — and a superseded horizon must not linger
    there, or the report keeps warning about a plan the user has already abandoned."""
    h = _run_pipeline(client, session_id)
    client.post("/api/v1/analysis/gate", headers=h)
    before = len(client.get("/api/v1/analysis/facts", headers=h).json()["caveats"])

    client.post("/api/v1/analysis/gate/horizon", json={"horizon": 60}, headers=h)
    long_cavs = client.get("/api/v1/analysis/facts", headers=h).json()["caveats"]
    assert any("horizon of 60" in c for c in long_cavs), long_cavs
    assert len(long_cavs) == before + 1

    client.post("/api/v1/analysis/gate/horizon", json={"horizon": 12}, headers=h)
    back = client.get("/api/v1/analysis/facts", headers=h).json()["caveats"]
    assert not any("horizon of 60" in c for c in back), back
    assert len(back) == before, (len(back), before)


def test_repeated_horizon_changes_do_not_accumulate(client, session_id):
    h = _run_pipeline(client, session_id)
    client.post("/api/v1/analysis/gate", headers=h)
    base = len(client.get("/api/v1/analysis/facts", headers=h).json()["caveats"])
    for H in (60, 12, 60, 12, 60, 12):
        client.post("/api/v1/analysis/gate/horizon", json={"horizon": H}, headers=h)
    after = client.get("/api/v1/analysis/facts", headers=h).json()["caveats"]
    assert len(after) == base, after


def test_the_gate_has_no_hidden_defaults(client, session_id):
    """
    A gate run on a bare series must not invent variance or seasonality numbers. Every criterion
    that needs a measurement must be unknown, not guessed.
    """
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    body = client.post("/api/v1/analysis/gate", headers=h).json()
    verdict = next(x for x in body["results"] if x["id"] == "verdict.readiness")
    by_key = {i["key"]: i for i in verdict["checklist"]}
    assert by_key["stationarity"]["passed"] is None
    assert by_key["correlogram"]["passed"] is None
    assert by_key["signal"]["passed"] is None
    # Seasonality IS measurable from the S1 profile, so it may pass — but only with a number.
    assert by_key["seasonality"]["value"]


def test_snippets_parse(client, session_id):
    import ast
    h = _run_pipeline(client, session_id)
    body = client.post("/api/v1/analysis/gate", headers=h).json()
    snippets = [r["reproducible_snippet"] for r in body["results"]
                if r.get("reproducible_snippet")]
    assert snippets
    for s in snippets:
        ast.parse(s)


def test_the_advisor_carries_blockers_and_caveats(client, session_id):
    h = _run_pipeline(client, session_id)
    body = client.post("/api/v1/analysis/gate", headers=h).json()
    advisor = body["advisor"]
    assert advisor is not None
    assert advisor["stage"] == "S8"
    assert advisor["headline"]
    for rule in advisor["recommendations"] + advisor["blockers"]:
        assert rule["because"]
        assert rule["id"].startswith("gate.")
    assert advisor["next_stage_ready"] == (
        advisor["next_stage_ready"] and not advisor["blockers"]
    )
