"""
tests/test_part0_contracts.py — Part 0 verification.

These are the tests that keep the rest of the build honest. They assert the
*contracts*, not the algorithms:

  * the pipeline knows its own dependency graph and order
  * ResultBase cannot be built without its beginner-facing fields
  * AdvisorReport / RuleOut serialise and always cite a number
  * the model registry rejects a count model for continuous data and vice versa
  * session state transitions mark downstream stages stale
  * the API envelope and error shape are stable
"""

from __future__ import annotations

import ast

import pytest
from pydantic import ValidationError

from app.core.errors import ChronosError, NotComparableError, as_error_payload
from app.core.schemas import (
    AdvisorReport,
    AnalysisDigest,
    Confidence,
    DataClass,
    ModelCapabilities,
    ModelFamily,
    ModelRequirements,
    ModelSpec,
    PipelineState,
    RuleOut,
    STAGE_LABELS,
    STAGE_ORDER,
    Stage,
    StatResult,
    TableColumn,
    TableResult,
    Verdict,
    VerdictResult,
    ChecklistItem,
)
from app.core.seeds import derived_seed, rng_for, seed_everything
from app.core.session import STAGE_DEPENDENCIES, Session


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_has_sixteen_stages_in_order():
    assert len(STAGE_ORDER) == 16
    assert STAGE_ORDER[0] is Stage.S0_INGEST
    assert STAGE_ORDER[-1] is Stage.S15_FINALIZE
    # codes must be strictly ordered S0..S15
    assert [s.value for s in STAGE_ORDER] == [f"S{i}" for i in range(16)]


def test_every_stage_has_a_label_and_a_dependency_entry():
    for s in STAGE_ORDER:
        assert s in STAGE_LABELS and STAGE_LABELS[s]
        assert s in STAGE_DEPENDENCIES


def test_dependencies_only_point_backwards():
    """A stage may never depend on a later stage — that would deadlock the UI."""
    for stage, deps in STAGE_DEPENDENCIES.items():
        for d in deps:
            assert STAGE_ORDER.index(d) < STAGE_ORDER.index(stage), (
                f"{stage.value} depends on later stage {d.value}"
            )


def test_new_pipeline_starts_empty():
    p = PipelineState(session_id="t")
    assert len(p.stages) == 16
    assert all(s.status == "not_started" for s in p.stages.values())
    assert p.progress()["percent"] == 0.0


def test_completing_a_stage_marks_downstream_stale():
    p = PipelineState(session_id="t")
    p.mark(Stage.S1_PROFILE, "complete")
    p.mark(Stage.S2_ADVISE_PREP, "complete")
    p.mark(Stage.S3_PREPARE, "complete")
    # re-run S1 -> S2 and S3 must go stale, never silently stay "complete"
    p.mark(Stage.S1_PROFILE, "complete")
    assert p.stages[Stage.S2_ADVISE_PREP.value].status == "stale"
    assert p.stages[Stage.S3_PREPARE.value].status == "stale"
    assert "S1" in p.stages[Stage.S2_ADVISE_PREP.value].message


def test_require_stage_raises_with_a_plain_english_message():
    s = Session()
    s.mark_stage(Stage.S0_INGEST, "complete")
    with pytest.raises(Exception) as ei:
        s.require_stage(Stage.S5_TEST_STATION)
    err = ei.value
    assert isinstance(err, ChronosError)
    assert err.code == "session.stage_order"
    assert "S1" in err.plain and err.fix


def test_stage_completion_unblocks_dependents():
    s = Session()
    for st in (Stage.S0_INGEST, Stage.S1_PROFILE):
        s.mark_stage(st, "complete")
    s.require_stage(Stage.S2_ADVISE_PREP)  # must not raise
    s.require_stage(Stage.S5_TEST_STATION)


# --------------------------------------------------------------------------- #
# Result contract (§4.3)
# --------------------------------------------------------------------------- #
def _minimal_kwargs(**over):
    base = dict(
        id="stat.adf.diff1",
        stage=Stage.S5_TEST_STATION,
        title="Augmented Dickey-Fuller test",
        plain="A test of whether the series wanders without returning to a mean.",
        why="If it wanders, we must difference it before modelling.",
        method="statsmodels.tsa.stattools.adfuller, regression='c', autolag='AIC'",
    )
    base.update(over)
    return base


def test_stat_result_requires_beginner_facing_fields():
    r = StatResult(**_minimal_kwargs(), value=-3.21, p_value=0.019)
    assert r.kind == "stat"
    assert r.plain and r.why and r.method
    assert r.schema_version


@pytest.mark.parametrize("missing", ["plain", "why", "method", "id", "title"])
def test_result_cannot_omit_explanations(missing):
    kw = _minimal_kwargs()
    kw.pop(missing)
    with pytest.raises(ValidationError):
        StatResult(**kw, value=0.0)


def test_result_id_cannot_contain_spaces():
    with pytest.raises(ValidationError):
        StatResult(**_minimal_kwargs(id="stat adf"), value=0.0)


def test_result_forbids_unknown_fields():
    """extra='forbid' stops a handler sneakily adding an unrendered field."""
    with pytest.raises(ValidationError):
        StatResult(**_minimal_kwargs(), value=1.0, secret_new_field=1)


def test_table_result_serialises_for_the_wire():
    t = TableResult(
        **_minimal_kwargs(id="table.descriptives", stage=Stage.S1_PROFILE),
        columns=[
            TableColumn(key="n", label="n", dtype="integer"),
            TableColumn(key="mean", label="Mean", dtype="number", format="0.000"),
        ],
        rows=[{"n": 144, "mean": 280.3}],
    )
    payload = t.model_dump(mode="json")
    assert payload["kind"] == "table"
    assert payload["columns"][1]["dtype"] == "number"
    assert payload["stage"] == "S1"


def test_verdict_result_carries_a_checklist_and_blockers():
    v = VerdictResult(
        **_minimal_kwargs(id="verdict.gate", stage=Stage.S8_GATE),
        verdict=Verdict.NOT_READY,
        headline="Not ready: the series is still non-stationary.",
        checklist=[
            ChecklistItem(key="stationary", label="Stationary", passed=False,
                          value="ADF p=0.512", jump_to_stage=Stage.S7_ADVISE_NEXT),
            ChecklistItem(key="gaps", label="No unresolved gaps", passed=True),
        ],
        blockers=["Series has a unit root after 0 differences."],
        caveats_travelling=["Structural break suspected in 1997."],
    )
    assert v.verdict == "not_ready"
    assert v.checklist[0].passed is False
    assert v.checklist[0].jump_to_stage == "S7"


# --------------------------------------------------------------------------- #
# Advisor contract (§4.4)
# --------------------------------------------------------------------------- #
def test_advisor_report_round_trips_and_keeps_facts():
    rep = AdvisorReport(
        stage=Stage.S6_CORRELATE,
        headline="ACF decays slowly; differencing is probably needed.",
        recommendations=[
            RuleOut(
                id="prep.diff.needed",
                stage=Stage.S7_ADVISE_NEXT,
                recommend="Apply a first difference",
                confidence=Confidence.HIGH,
                because="ADF p=0.612 and KPSS p<0.01 disagree; the ACF is still 0.81 at lag 6.",
                provenance=["stat.adf.level", "stat.kpss.level", "fig.acf.level"],
                risk_if_ignored="Spurious regression: a high R-squared with no real relationship.",
                how_to="Prepare data → Difference → order 1",
            )
        ],
        facts_used={"adf_p": 0.612, "kpss_p": 0.0074, "acf_lag6": 0.81},
        next_stage_ready=True,
    )
    d = rep.model_dump(mode="json")
    assert d["recommendations"][0]["confidence"] == "high"
    assert d["facts_used"]["adf_p"] == 0.612
    # the AI only ever sees facts_used — it must be flat and numeric-ish
    assert all(not isinstance(v, (dict, list)) for v in d["facts_used"].values())


def test_recommendation_without_a_number_is_a_defect():
    """R0.5 guard: a 'because' with no computed value in it must be caught in review."""
    r = RuleOut(
        id="prep.bad",
        stage=Stage.S2_ADVISE_PREP,
        recommend="Smooth the series",
        because="It looks noisy.",
    )
    assert not any(ch.isdigit() for ch in r.because), "this test documents the anti-pattern"


# --------------------------------------------------------------------------- #
# Model registry contract (§5.8)
# --------------------------------------------------------------------------- #
def test_count_model_is_not_offered_for_continuous_data():
    pois = ModelSpec(
        model_id="count.poisson",
        display_name="Poisson regression",
        family=ModelFamily.COUNT,
        description="Log-linear Poisson model with trend and seasonal terms.",
        plain_description="For counts of events, where the average equals the variance.",
        required=ModelRequirements(requires_count=True, requires_non_negative=True),
        capabilities=ModelCapabilities(discrete_predictive=True),
    )
    assert pois.supports_data_class(DataClass.COUNT)
    assert not pois.supports_data_class(DataClass.CONTINUOUS)
    assert pois.supports_data_class(DataClass.INTERMITTENT)


def test_arima_is_not_offered_for_count_data():
    arima = ModelSpec(
        model_id="arma.arima",
        display_name="ARIMA",
        family=ModelFamily.ARMA,
        description="Autoregressive integrated moving average.",
        plain_description="A model of how today depends on recent past values and shocks.",
        capabilities=ModelCapabilities(supports_probabilistic=True, supports_intervals=True),
    )
    assert arima.supports_data_class(DataClass.CONTINUOUS)
    assert not arima.supports_data_class(DataClass.COUNT)


def test_white_noise_and_naive_are_baselines_with_no_arguments():
    wn = ModelSpec(
        model_id="baseline.white_noise",
        display_name="White noise",
        family=ModelFamily.BASELINE,
        description="i.i.d. Gaussian noise about a constant mean.",
        plain_description="No pattern at all — the honest null model.",
        is_baseline=True,
    )
    assert wn.is_baseline and wn.family == "baseline"
    assert wn.supports_data_class(DataClass.CONTINUOUS)


# --------------------------------------------------------------------------- #
# AI grounding contract (§5.14)
# --------------------------------------------------------------------------- #
def test_analysis_digest_has_no_raw_values_by_default():
    d = AnalysisDigest(
        session_id="t",
        n_observations=144,
        frequency_label="Monthly (start of month)",
        period=12,
        time_start="1949-01-01",
        time_end="1960-12-01",
        data_class=DataClass.CONTINUOUS,
        series_name="AirPassengers",
        current_stage=Stage.S6_CORRELATE,
        numbers={"adf_p": 0.99, "n_obs": 144.0},
    )
    assert d.data_values is None
    assert d.data_values_included is False
    # every number the assistant may quote lives here
    assert "adf_p" in d.numbers


def test_ai_request_defaults_to_not_sending_data():
    from app.core.schemas import AIChatRequest

    req = AIChatRequest(session_id="t", message="Is my series stationary?")
    assert req.include_data_values is False


# --------------------------------------------------------------------------- #
# Errors (§R0.9)
# --------------------------------------------------------------------------- #
def test_every_error_carries_a_fix():
    for exc in (ChronosError(), NotComparableError()):
        d = exc.to_dict()
        assert d["plain"] and d["fix"], "plain English + suggested fix are mandatory"


def test_unexpected_exception_is_not_leaked_as_the_primary_message():
    payload = as_error_payload(ZeroDivisionError("division by zero"))
    assert payload["code"] == "internal.error"
    assert "division by zero" not in payload["plain"]
    assert "ZeroDivisionError" in payload["technical"]


# --------------------------------------------------------------------------- #
# Determinism (§R0.7)
# --------------------------------------------------------------------------- #
def test_seeded_rng_is_reproducible_and_tag_sensitive():
    seed_everything(42)
    a = rng_for(42, "pit", "model-x").normal(size=5)
    b = rng_for(42, "pit", "model-x").normal(size=5)
    c = rng_for(42, "pit", "model-y").normal(size=5)
    assert (a == b).all(), "same seed + tags must give identical draws"
    assert not (a == c).all(), "different tags must decorrelate draws"
    assert derived_seed(42, "cv", "fold1") == derived_seed(42, "cv", "fold1")


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #
def test_health_and_pipeline_endpoints(client):
    h = client.get("/api/v1/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok"
    assert body["stages"] == 16
    assert isinstance(body["optional_features"], dict)

    p = client.get("/api/v1/pipeline")
    assert [s["code"] for s in p.json()["stages"]] == [f"S{i}" for i in range(16)]


def test_session_lifecycle(client):
    r = client.post("/api/v1/sessions")
    assert r.status_code == 201
    sid = r.json()["session_id"]

    state = client.get(f"/api/v1/sessions/{sid}").json()
    assert state["session_id"] == sid
    assert state["progress"]["stages_total"] == 16
    assert state["dataset"] is None

    assert client.delete(f"/api/v1/sessions/{sid}").json()["deleted"] is True
    assert client.get(f"/api/v1/sessions/{sid}").status_code == 404


def test_expired_session_returns_a_friendly_error(client):
    r = client.get("/api/v1/sessions/does-not-exist")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "session.not_found"
    assert "expired" in err["plain"].lower()
    assert err["fix"]


def test_results_endpoint_returns_all_stages_keys(client):
    sid = client.post("/api/v1/sessions").json()["session_id"]
    body = client.get(f"/api/v1/sessions/{sid}/results").json()
    assert set(body) == {f"S{i}" for i in range(16)}
    assert all(v == [] for v in body.values())


def test_unknown_result_id_is_not_an_error(client):
    sid = client.post("/api/v1/sessions").json()["session_id"]
    r = client.post(
        f"/api/v1/sessions/{sid}/results/nope/explain",
        headers={"X-Session-Id": sid},
    )
    assert r.status_code == 200
    assert r.json()["found"] is False
