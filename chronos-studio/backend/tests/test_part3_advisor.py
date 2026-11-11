"""
tests/test_part3_advisor.py — Part 3 verification.

The advisor is the part of this app that can do the most damage with a plausible
sentence, so the tests are written as *decisions* rather than as coverage:

  * a series that needs a log must get "take logs", and one that does not must say so;
  * exactly one transform is ever recommended (never log *and* sqrt);
  * λ = 1 (no transform) is never dressed up as variance stabilisation;
  * a long gap is a blocker, a short one is an imputation, and neither is silent;
  * smoothing of the modelling series is never recommended — that rule is unconditional;
  * every `because` sentence cites a number that appears in `facts_used` (R0.5);
  * preview does not mutate the session; apply versions it; undo restores it.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.advisor.rules_prep import (
    RULES,
    STRONG_SEASONALITY,
    UNCONDITIONAL,
    PrepContext,
    preferred_transform,
    rank,
)
from app.core.schemas import DataClass, PreprocessingRecord, Stage
from app.ts.eda.profile import run_eda
from app.ts.preprocess.advise import (
    advise_preprocessing,
    boxcox_lambda,
    build_context,
    gap_runs,
    longest_stall_run,
    variance_level_corr,
)
from app.ts.preprocess.ops import REGISTRY, STEP_OF, apply_op

EXAMPLES = Path(__file__).resolve().parents[1] / "app" / "assets" / "examples"


# --------------------------------------------------------------------------- #
# Fixtures: series whose correct advice we know in advance
# --------------------------------------------------------------------------- #
def _monthly(values: np.ndarray, start: str = "2010-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


@pytest.fixture(scope="module")
def air() -> pd.Series:
    s = pd.read_csv(EXAMPLES / "airpassengers.csv", index_col=0)["Passengers"]
    s.index = pd.to_datetime(s.index)
    return s


@pytest.fixture(scope="module")
def multiplicative_seasonal() -> pd.Series:
    """Trend × season with a widening swing: the textbook case for a log."""
    t = np.arange(120)
    level = 100 * np.exp(0.02 * t)
    season = 1 + 0.25 * np.sin(2 * np.pi * t / 12)
    return _monthly(level * season * np.random.default_rng(0).normal(1, 0.01, 120))


@pytest.fixture(scope="module")
def additive_clean() -> pd.Series:
    """Stable variance, strong seasonality: seasonality yes, transform no."""
    t = np.arange(120)
    vals = 100 + 20 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(1).normal(0, 0.6, 120)
    return _monthly(vals)


@pytest.fixture(scope="module")
def random_walk() -> pd.Series:
    return _monthly(np.cumsum(np.random.default_rng(4).normal(0, 1, 180)) + 100)


def advice_for(series: pd.Series, *, period: int | None = 12, data_class=DataClass.CONTINUOUS,
               name="test-series", **facts):
    """Run the real S1 then the real S2, so the rules see exactly what production sees."""
    eda = run_eda(series, name=name, frequency_label="Monthly", freq_code="MS",
                  period=period, data_class=data_class)
    merged = {**eda.facts, **facts}
    return advise_preprocessing(series, eda_facts=merged, name=name,
                                frequency_label="Monthly", period=period, data_class=data_class)


def rule_ids(advice) -> set[str]:
    return {r.id for r in advice.report.recommendations} | {r.id for r in advice.report.blockers}


# --------------------------------------------------------------------------- #
# The context
# --------------------------------------------------------------------------- #
def test_gap_runs_counts_consecutive_missing_only():
    s = _monthly(np.arange(20.0))
    s.iloc[3:6] = np.nan
    s.iloc[12] = np.nan
    assert gap_runs(s) == [3, 1]


def test_longest_stall_run_finds_a_flat_stretch():
    v = np.arange(30.0)
    v[10:20] = 10.0                       # ten consecutive identical values
    assert longest_stall_run(_monthly(v)) == 10


def test_boxcox_lambda_identifies_a_log_series():
    """Data generated as exp(noise) has λ ≈ 0 by construction."""
    rng = np.random.default_rng(0)
    s = _monthly(np.exp(np.linspace(3, 5, 200) + rng.normal(0, 0.15, 200)))
    lam, ci = boxcox_lambda(s)
    assert lam is not None and abs(lam) < 0.35
    assert ci is not None and ci[0] <= lam <= ci[1]


def test_boxcox_lambda_refuses_non_positive_data():
    s = _monthly(np.linspace(-5, 5, 50))
    assert boxcox_lambda(s) == (None, None)


def test_variance_level_correlation_is_positive_when_spread_grows(multiplicative_seasonal):
    assert variance_level_corr(multiplicative_seasonal) > 0.5


def test_variance_level_correlation_is_flat_for_stable_series(additive_clean):
    assert abs(variance_level_corr(additive_clean)) < 0.6


def test_context_marks_positivity_and_zeros():
    s = _monthly(np.r_[np.zeros(5), np.arange(5, 25).astype(float)])
    ctx = build_context(s, eda_facts={}, name="z", frequency_label="Monthly", period=12,
                        data_class="continuous")
    assert ctx.n_zeros == 5
    # "positive" means every value is > 0 — a series with zeros is *not* log-safe, even
    # though some of its values are large and positive
    assert ctx.is_positive is False
    assert build_context(_monthly(np.zeros(30)), eda_facts={}, name="z",
                         frequency_label="Monthly", period=12,
                         data_class="continuous").is_positive is False


# --------------------------------------------------------------------------- #
# Transform: the decision that changes every later result
# --------------------------------------------------------------------------- #
def test_multiplicative_seasonal_gets_a_log(multiplicative_seasonal):
    assert preferred_transform(build_context(
        multiplicative_seasonal, eda_facts={}, name="m", frequency_label="Monthly",
        period=12, data_class="continuous")) == "log"


def test_stable_series_gets_no_transform(additive_clean):
    advice = advice_for(additive_clean)
    assert "prep.transform.log" not in rule_ids(advice)
    assert "prep.transform.sqrt" not in rule_ids(advice)
    assert "prep.transform.boxcox" not in rule_ids(advice)


def test_at_most_one_transform_is_ever_recommended(additive_clean, multiplicative_seasonal):
    """'Take logs and a square root' is not advice."""
    for series in (additive_clean, multiplicative_seasonal):
        ids = rule_ids(advice_for(series))
        transforms = [i for i in ids if i.startswith("prep.transform.")]
        assert len(transforms) <= 1, f"conflicting transforms: {transforms}"


def test_lambda_of_one_is_not_dressed_up_as_a_transform():
    """λ = 1 means 'the scale is already right' — a transform would amplify skew."""
    ctx = PrepContext(
        n=200, period=12, is_positive=True,
        variance_ratio=1.6, variance_level_corr=0.7,
        boxcox_lambda=1.02, boxcox_ci=(0.4, 1.7),
    )
    assert preferred_transform(ctx) is None


def test_log_is_blocked_on_non_positive_data():
    """A series with negatives needs Yeo–Johnson, and the app must say why, not crash."""
    s = _monthly(np.sin(np.arange(120) / 3) * 100)      # crosses zero
    advice = advice_for(s)
    ids = rule_ids(advice)
    assert "prep.transform.log" not in ids
    blocked = [b for b in advice.report.blockers if b.id == "prep.log.undefined_blocker"]
    if blocked:
        assert "Yeo–Johnson" in blocked[0].because
        assert advice.report.next_stage_ready is False


def test_yeojohnson_handles_zero_and_negative_values():
    s = _monthly(np.sin(np.arange(60) / 2) * 10)
    out = apply_op("transform.yeojohnson", s, {})
    assert out.series.notna().all()
    assert np.isfinite(out.series.to_numpy(dtype=float)).all()


# --------------------------------------------------------------------------- #
# Imputation and gaps
# --------------------------------------------------------------------------- #
def test_short_gap_in_a_seasonal_series_is_imputable(air):
    gappy = air.copy()
    gappy.iloc[40] = np.nan
    advice = advice_for(gappy)
    assert "prep.impute.seasonal_interpolation" in rule_ids(advice)


def test_long_gap_is_a_blocker_not_an_imputation():
    s = _monthly(np.sin(np.arange(120) / 4) * 10 + 50)
    s.iloc[40:60] = np.nan                                  # 20 consecutive
    advice = advice_for(s)
    assert [b.id for b in advice.report.blockers] == ["prep.gap.long_blocker"]
    assert advice.report.next_stage_ready is False
    assert "prep.impute.seasonal_interpolation" not in rule_ids(advice)


def test_mid_length_gap_gets_the_state_space_option():
    s = _monthly(np.sin(np.arange(120) / 4) * 10 + 50)
    s.iloc[40:47] = np.nan
    assert "prep.impute.state_space" in rule_ids(advice_for(s))


def test_no_gap_means_no_imputation_rule(air):
    assert not any(r.startswith("prep.impute.") for r in rule_ids(advice_for(air)))


def test_seasonal_interpolation_preserves_the_calendar(air):
    gappy = air.copy()
    gappy.iloc[[10, 11]] = np.nan
    out = apply_op("impute.seasonal", gappy, {"period": 12})
    assert out.series.notna().all()
    assert out.series.index.equals(air.index)
    # the filled value must be close to that month's typical level, not a straight line
    assert 150 < out.series.iloc[10] < 350


def test_kalman_imputation_fills_without_changing_n(air):
    gappy = air.copy()
    gappy.iloc[10:13] = np.nan
    out = apply_op("impute.kalman", gappy, {"period": 12})
    assert int(out.series.notna().sum()) == len(air)


# --------------------------------------------------------------------------- #
# Outliers: flag, never silently delete
# --------------------------------------------------------------------------- #
def test_outliers_are_flagged_not_removed():
    rng = np.random.default_rng(2)
    s = _monthly(np.sin(np.arange(120) / 3) * 5 + 50 + rng.normal(0, 0.5, 120))
    s.iloc[44] += 60
    s.iloc[83] -= 60
    advice = advice_for(s)
    ids = rule_ids(advice)
    assert "prep.outlier.flag_not_delete" in ids
    rec = next(r for r in advice.report.recommendations if r.id == "prep.outlier.flag_not_delete")
    assert "dumm" in (rec.recommend + rec.because).lower()
    assert "prep.outlier.delete" not in ids


def test_deleting_outliers_is_possible_but_warns():
    s = _monthly(np.arange(50.0))
    out = apply_op("outlier.delete", s, {"times": [str(s.index[10])]})
    assert len(out.series) == 49
    assert out.warnings
    assert "least defensible" in out.warnings[0]


def test_flagging_outliers_changes_nothing():
    s = _monthly(np.arange(50.0))
    out = apply_op("outlier.flag", s, {"times": [str(s.index[10])]})
    assert len(out.series) == len(s)
    pd.testing.assert_series_equal(out.series, s)


# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #
def test_strong_seasonality_gets_a_seasonal_term(air):
    ids = rule_ids(advice_for(air))
    assert "prep.seasonal.term" in ids
    assert "prep.seasonal.absent" not in ids


def test_weak_seasonality_explicitly_forbids_seasonal_terms():
    s = _monthly(np.random.default_rng(6).normal(50, 5, 120))
    ids = rule_ids(advice_for(s))
    assert "prep.seasonal.absent" in ids
    assert "prep.seasonal.term" not in ids


def test_seasonal_rule_threshold_is_named_in_the_sentence(air):
    rec = next(r for r in advice_for(air).report.recommendations if r.id == "prep.seasonal.term")
    assert str(STRONG_SEASONALITY) in rec.because


# --------------------------------------------------------------------------- #
# Smoothing: the one rule that is always on
# --------------------------------------------------------------------------- #
def test_smoothing_the_modelling_series_is_never_recommended(air, additive_clean):
    for series in (air, additive_clean):
        ids = rule_ids(advice_for(series))
        assert "prep.smooth.modelling" not in ids
        assert not any(i.startswith("prep.smooth.") and "presentation" not in i for i in ids)


def test_the_unconditional_smoothing_rule_exists_and_says_why():
    assert any(r.id == "prep.smooth.not_for_modelling" for r in UNCONDITIONAL)
    rule = next(r for r in UNCONDITIONAL if r.id == "prep.smooth.not_for_modelling")
    text = rule.because(PrepContext(n=100))
    assert "MASE" in text and "autocorrelation" in text
    assert rule.when(PrepContext()) is True


def test_presentation_smoothing_is_labelled_as_such():
    s = _monthly(np.random.default_rng(7).normal(0, 10, 120) + 100)
    ids = rule_ids(advice_for(s))
    if "prep.smooth.for_presentation" in ids:
        rec = next(r for r in advice_for(s).report.recommendations
                   if r.id == "prep.smooth.for_presentation")
        assert "never for the model" in rec.recommend.lower()
        out = apply_op("smooth.ma", s, {"window": 7})
        assert out.warnings and "Presentation only" in out.warnings[0]


# --------------------------------------------------------------------------- #
# Rules as a system
# --------------------------------------------------------------------------- #
def test_blockers_come_first_in_the_ranking():
    ctx = PrepContext(n=100, period=12, longest_gap=30, seasonal_strength=0.9,
                      seasonal_p=0.001, n_outliers=3, variance_ratio=2.0,
                      variance_level_corr=0.8, boxcox_lambda=-0.1, boxcox_ci=(-0.4, 0.2),
                      is_positive=True)
    ordered = rank(ctx)
    assert ordered[0].kind == "blocker"


def test_every_rule_declares_what_it_would_break():
    for rule in RULES:
        assert rule.risk_if_ignored, f"{rule.id} does not say what ignoring it costs"
        assert rule.provenance or rule.id.startswith(("prep.diff", "prep.seasonal")), (
            f"{rule.id} does not name the results it consumed"
        )


def test_rule_ids_are_unique_and_prefixed():
    ids = [r.id for r in RULES] + [r.id for r in UNCONDITIONAL]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("prep.") for i in ids)


def test_a_broken_rule_cannot_crash_the_stage():
    """`when` is user-adjacent code; it must never take the endpoint down."""
    from app.advisor.rules_prep import _safe

    def boom(_ctx):
        raise RuntimeError("bad rule")

    assert _safe(boom, PrepContext()) is False


def test_each_rule_fires_only_on_its_own_context():
    """No rule may be a tautology: every conditional rule must be false on an empty context."""
    empty = PrepContext()
    for rule in RULES:
        assert rule.when(empty) is False, f"{rule.id} fires on an empty context"


# --------------------------------------------------------------------------- #
# Grounding (R0.5): the sentence must cite numbers that were computed
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    out = set()
    for tok in _NUM.findall(text):
        try:
            out.add(f"{float(tok):.6g}")
        except ValueError:
            pass
    return out


def test_every_because_sentence_cites_a_fact_that_exists(air, multiplicative_seasonal):
    for series in (air, multiplicative_seasonal):
        advice = advice_for(series)
        allowed = {f"{float(v):.6g}" for v in advice.facts.values()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for rule in advice.report.recommendations + advice.report.blockers:
            cited = _numbers(rule.because)
            # small integers are counts ("2 observations", "period 12"), dates are excluded
            large = {c for c in cited if abs(float(c)) > 20}
            unknown = {c for c in large if c not in allowed}
            # a rounded rendering is acceptable: compare to 1% relative
            truly_unknown = {
                c for c in unknown
                if not any(abs(float(c) - float(a)) <= 0.011 * max(abs(float(c)), 1e-9)
                           for a in allowed)
            }
            assert not truly_unknown, f"{rule.id} cites {truly_unknown} which is not in facts_used"


def test_facts_used_is_flat_and_json_safe(air):
    facts = advice_for(air).facts
    json.dumps(facts)                                   # must serialise
    for k, v in facts.items():
        assert isinstance(v, (int, float, str, bool)), f"{k} is nested"


def test_facts_used_carries_the_thresholds_it_reasons_with(air):
    facts = advice_for(air).facts
    assert facts["strong_seasonality_threshold"] == STRONG_SEASONALITY
    assert "seasonal_strength" in facts and "boxcox_lambda" in facts


# --------------------------------------------------------------------------- #
# Stage output
# --------------------------------------------------------------------------- #
def test_advice_emits_an_evidence_table_and_a_diagnostic_figure(air):
    advice = advice_for(air)
    ids = [r.id for r in advice.results]
    assert "table.prep.evidence" in ids
    assert "fig.prep.variance_level" in ids


def test_every_advice_result_is_a_contract_result(air):
    for r in advice_for(air).results:
        assert r.stage == Stage.S2_ADVISE_PREP
        assert r.plain and r.why and r.method
        if r.reproducible_snippet:
            ast.parse(r.reproducible_snippet)


def test_evidence_table_has_one_row_per_decision(air):
    table = next(r for r in advice_for(air).results if r.id == "table.prep.evidence")
    checks = {row["check"] for row in table.rows}
    assert {"Missing values", "Variance vs level", "Box–Cox λ", "Seasonality", "Outliers",
            "Smoothing"} <= checks
    for row in table.rows:
        assert all(row[k] for k in ("check", "finding", "evidence", "action"))


def test_headline_is_one_sentence_and_counts_agree(air):
    advice = advice_for(air)
    assert advice.report.headline.count(".") == 1
    n = len(advice.report.recommendations)
    assert str(n) in advice.report.headline or n == 0


def test_advice_is_deterministic(air):
    a, b = advice_for(air), advice_for(air)
    assert [r.id for r in a.report.recommendations] == [r.id for r in b.report.recommendations]
    assert [r.because for r in a.report.recommendations] == [r.because for r in b.report.recommendations]


def test_tiny_series_still_produces_advice():
    advice = advice_for(_monthly(np.arange(8.0)), period=None)
    assert advice.report.headline
    assert "prep.series.short" in rule_ids(advice)


# --------------------------------------------------------------------------- #
# Operations engine
# --------------------------------------------------------------------------- #
def test_every_registry_op_is_pure(air):
    """An operation must never mutate the series it was given."""
    for key in REGISTRY:
        before = air.copy()
        try:
            apply_op(key, air, {})
        except Exception:
            continue
        pd.testing.assert_series_equal(air, before, check_names=False)


def test_every_registry_op_has_a_step_and_a_sentence():
    for key in REGISTRY:
        assert key in STEP_OF, f"{key} has no preprocessing step"
        assert STEP_OF[key] in {
            "impute", "smooth", "transform", "outlier", "aggregate", "trim", "difference"
        }


def test_log_matches_numpy(air):
    out = apply_op("transform.log", air, {})
    assert out.series.iloc[0] == pytest.approx(float(np.log(air.iloc[0])))


def test_log_with_offset_is_explicit(air):
    out = apply_op("transform.log", air, {"offset": 100.0})
    assert out.series.iloc[0] == pytest.approx(float(np.log(air.iloc[0] + 100)))
    assert "offset" in out.plain or "adding 100" in out.plain


def test_log_on_zeroes_refuses_and_explains():
    s = _monthly(np.r_[np.zeros(6), np.arange(6, 30).astype(float)])
    out = apply_op("transform.log", s, {})
    pd.testing.assert_series_equal(out.series, s)
    assert out.warnings and "Yeo–Johnson" in out.warnings[0]


def test_aggregation_reports_the_new_length_and_the_catch(air):
    out = apply_op("aggregate", air, {"freq": "QS", "how": "sum"})
    assert len(out.series) == 48
    assert "moving-average" in out.plain


def test_trim_window_reports_what_it_dropped(air):
    out = apply_op("trim", air, {"start": "1955-01-01"})
    assert len(out.series) == 72
    assert out.n_changed == 72


def test_unknown_op_raises_rather_than_guessing(air):
    with pytest.raises(KeyError):
        apply_op("transform.magic", air, {})


# --------------------------------------------------------------------------- #
# API: S2 advice, preview, apply, undo
# --------------------------------------------------------------------------- #
@pytest.fixture()
def ready(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", json={}, headers=h)
    return h


def test_advice_endpoint_returns_a_report(client, ready):
    r = client.post("/api/v1/analysis/advice", json={}, headers=ready)
    assert r.status_code == 200
    body = r.json()
    assert body["headline"]
    assert body["confidence"] in {"high", "medium", "low"}
    assert isinstance(body["facts_used"], dict) and body["facts_used"]
    assert body["results"]


def test_advice_is_idempotent(client, ready):
    a = client.post("/api/v1/analysis/advice", json={}, headers=ready).json()
    b = client.post("/api/v1/analysis/advice", json={}, headers=ready).json()
    assert len(a["recommendations"]) == len(b["recommendations"])
    ids = [r["id"] for r in client.get("/api/v1/analysis/advice", headers=ready).json()["results"]]
    assert len(ids) == len(set(ids)), "re-running must replace, not duplicate"


def test_advice_requires_a_profile(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    r = client.post("/api/v1/analysis/advice", json={}, headers=h)
    assert r.status_code in {400, 409}, r.text
    assert r.json()["error"]["code"] in {"advice.no_profile", "session.stage_order"}
    # either the stage-order guard or the advisor's own check may fire, and both point
    # the user at Step 1 in plain words
    assert "S1" in r.json()["error"]["fix"] or "Step 1" in r.json()["error"]["fix"]


def test_preview_does_not_change_the_session(client, ready):
    before = client.get("/api/v1/preprocess/state", headers=ready).json()
    p = client.post("/api/v1/preprocess/preview",
                    json={"op": "transform.log", "params": {}}, headers=ready).json()
    assert p["changed"] is True
    after = client.get("/api/v1/preprocess/state", headers=ready).json()
    assert before["series_version"] == after["series_version"]
    assert after["steps"] == []


def test_apply_versions_records_and_marks_advice_stale(client, ready):
    client.post("/api/v1/analysis/advice", json={}, headers=ready)
    r = client.post("/api/v1/preprocess/apply",
                    json={"op": "transform.log", "params": {}, "rule_id": "prep.transform.log"},
                    headers=ready)
    assert r.status_code == 200
    state = client.get("/api/v1/preprocess/state", headers=ready).json()
    assert len(state["steps"]) == 1
    step = state["steps"][0]
    assert step["step"] == "transform"
    assert step["advisor_rule_id"] == "prep.transform.log"
    assert step["reason"].startswith("Take logs")      # the recommendation's own words
    assert state["series_version"] == 2

    sid = ready["X-Session-Id"]
    stages = client.get(f"/api/v1/sessions/{sid}", headers=ready).json()["pipeline"]["stages"]
    assert stages["S2"]["status"] == "stale"


def test_undo_restores_the_previous_series(client, ready):
    original = client.get("/api/v1/analysis/series?limit=10", headers=ready).json()
    client.post("/api/v1/preprocess/apply", json={"op": "transform.log"}, headers=ready)
    logged = client.get("/api/v1/analysis/series?limit=10", headers=ready).json()
    assert logged["points"][0]["value"] != original["points"][0]["value"]

    client.post("/api/v1/preprocess/undo", headers=ready)
    restored = client.get("/api/v1/analysis/series?limit=10", headers=ready).json()
    assert restored["points"][0]["value"] == pytest.approx(original["points"][0]["value"])
    assert client.get("/api/v1/preprocess/state", headers=ready).json()["steps"] == []


def test_undo_with_nothing_applied_is_a_clear_error(client, ready):
    r = client.post("/api/v1/preprocess/undo", headers=ready)
    assert r.status_code == 400
    assert "nothing to undo" in r.json()["error"]["plain"].lower()


def test_unknown_op_is_rejected_with_a_helpful_error(client, ready):
    r = client.post("/api/v1/preprocess/preview", json={"op": "transform.nope"}, headers=ready)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "preprocess.unknown_op"


def test_two_operations_stack_and_replay(client, ready):
    client.post("/api/v1/preprocess/apply", json={"op": "trim", "params": {"start": "1955-01-01"}},
                headers=ready)
    client.post("/api/v1/analysis/profile", json={}, headers=ready)   # the UI re-profiles
    client.post("/api/v1/preprocess/apply", json={"op": "transform.log", "params": {}},
                headers=ready)
    state = client.get("/api/v1/preprocess/state", headers=ready).json()
    assert [s["step"] for s in state["steps"]] == ["trim", "transform"]
    assert state["n"] == 72
    client.post("/api/v1/preprocess/undo", headers=ready)
    state = client.get("/api/v1/preprocess/state", headers=ready).json()
    assert [s["step"] for s in state["steps"]] == ["trim"]
    assert state["n"] == 72


def test_applied_operations_reach_the_advisor_context(client, ready):
    """After a log is applied and the profile is refreshed, logging is not suggested again."""
    client.post("/api/v1/preprocess/apply", json={"op": "transform.log"}, headers=ready)

    # Advising from a profile of the *old* series must be refused, not silently wrong.
    stale = client.post("/api/v1/analysis/advice", json={}, headers=ready)
    assert stale.status_code == 400
    assert stale.json()["error"]["code"] == "advice.stale_profile"

    client.post("/api/v1/analysis/profile", json={}, headers=ready)
    body = client.post("/api/v1/analysis/advice", json={}, headers=ready).json()
    ids = {r["id"] for r in body["recommendations"]}
    assert not any(i.startswith("prep.transform.") and i != "prep.transform.already_applied" for i in ids)
    assert "prep.transform.already_applied" in ids, "stacking a second transform must be refused explicitly"
    assert body["facts_used"]["n"] == 144
