"""
core/schemas.py — the API's single source of truth.

Everything the frontend renders is one of the Result types below, and every
Result carries `id / stage / plain / why / technical / method / inputs /
citations / caveats / warnings / reproducible_snippet`. The frontend is not
allowed to invent a number; it renders a payload that arrived from here.

This file also declares:
  * the pipeline state machine (S0..S15)          — §4.2
  * the Advisor pattern (Rule / AdvisorReport)     — §4.4
  * the model registry contract (ModelSpec)        — §5.8
  * the AI grounding contract (AnalysisDigest)     — §5.14

Nothing here computes anything. If you are tempted to add computation to this
file, it belongs in `app/ts/`.
"""

from __future__ import annotations

import enum
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.1.0"


# =========================================================================== #
# 1. Enumerations
# =========================================================================== #
class Stage(str, enum.Enum):
    """The pipeline. Order matters: `PipelineState` advances in this order."""

    S0_INGEST = "S0"
    S1_PROFILE = "S1"
    S2_ADVISE_PREP = "S2"
    S3_PREPARE = "S3"
    S4_DECOMPOSE = "S4"
    S5_TEST_STATION = "S5"
    S6_CORRELATE = "S6"
    S7_ADVISE_NEXT = "S7"
    S8_GATE = "S8"
    S9_SUGGEST_MODEL = "S9"
    S10_FIT = "S10"
    S11_SELECT = "S11"
    S12_DIAGNOSE = "S12"
    S13_FORECAST = "S13"
    S14_EVALUATE = "S14"
    S15_FINALIZE = "S15"


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)
STAGE_LABELS: dict[Stage, str] = {
    Stage.S0_INGEST: "Load data",
    Stage.S1_PROFILE: "Explore (EDA)",
    Stage.S2_ADVISE_PREP: "Preprocessing advice",
    Stage.S3_PREPARE: "Prepare data",
    Stage.S4_DECOMPOSE: "Decompose",
    Stage.S5_TEST_STATION: "Test stationarity",
    Stage.S6_CORRELATE: "ACF / PACF",
    Stage.S7_ADVISE_NEXT: "What next",
    Stage.S8_GATE: "Ready to model?",
    Stage.S9_SUGGEST_MODEL: "Suggest models",
    Stage.S10_FIT: "Fit models",
    Stage.S11_SELECT: "Select model",
    Stage.S12_DIAGNOSE: "Diagnostics",
    Stage.S13_FORECAST: "Forecast",
    Stage.S14_EVALUATE: "Evaluate",
    Stage.S15_FINALIZE: "Ready to write up?",
}

StageStatus = Literal["not_started", "running", "complete", "stale", "blocked", "failed"]


class DataClass(str, enum.Enum):
    """
    Drives the entire downstream branch (§5.8). Detected in Part 1's quality
    report and gate-kept everywhere after: a count series must never be sent
    down the ETS/ARIMA path without the app saying why.
    """

    CONTINUOUS = "continuous"
    COUNT = "count"
    INTERMITTENT = "intermittent"
    BINARY_OR_PROPORTION = "binary_or_proportion"
    EVENT_TIMESTAMP = "event_timestamp"


class Confidence(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Verdict(str, enum.Enum):
    READY = "ready"
    READY_WITH_CAVEATS = "ready_with_caveats"
    NOT_READY = "not_ready"
    INCONCLUSIVE = "inconclusive"


class ModelFamily(str, enum.Enum):
    BASELINE = "baseline"          # white noise, naive, seasonal naive, mean, drift
    ARMA = "arma"                  # AR, MA, ARMA, ARIMA, SARIMA, ARFIMA
    SMOOTHING = "smoothing"        # ETS / Holt-Winters / Theta
    STRUCTURAL = "structural"      # UCM, BSM, TBATS
    EXOGENOUS = "exogenous"        # ARIMAX, SARIMAX, ARDL, dynamic regression
    COUNT = "count"                # Poisson, NB, zero-inflated, INAR, INGARCH
    INTERMITTENT = "intermittent"  # Croston, SBA, SY, TSB, ADIDA
    POINT_PROCESS = "point_process"  # Hawkes, renewal
    VOLATILITY = "volatility"      # ARCH, GARCH, EGARCH, GJR, SV
    MULTIVARIATE = "multivariate"  # VAR, VECM
    NONLINEAR = "nonlinear"        # SETAR, Markov-switching, NNAR
    ML = "ml"                      # LightGBM, RF, kNN, SVR, LSTM
    COMBINATION = "combination"    # ensemble of the above


# =========================================================================== #
# 2. Result base and derived payloads  (§4.3)
# =========================================================================== #
class ResultBase(BaseModel):
    """
    Every visible output in the app extends this. `reproducible_snippet` must be
    real, runnable Python — a golden test asserts it is non-empty and that it
    round-trips through `ast.parse`.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    id: str = Field(..., description="Stable id, e.g. 'stat.adf.diff1'")
    stage: Stage
    title: str
    plain: str = Field(..., description="One-sentence beginner meaning")
    why: str = Field(..., description="Why the user should care")
    technical: str = Field("", description="Formal definition; LaTeX allowed")
    method: str = Field(..., description="Exact procedure + library + settings")
    inputs: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reproducible_snippet: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = SCHEMA_VERSION

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not v or " " in v:
            raise ValueError("result id must be non-empty and contain no spaces")
        return v


class TableColumn(BaseModel):
    key: str
    label: str
    dtype: Literal["string", "number", "integer", "date", "bool"] = "string"
    format: str | None = Field(None, description="e.g. '0.000' for display, 'p' for p-values")
    align: Literal["left", "right", "center"] = "left"


class TableResult(ResultBase):
    kind: Literal["table"] = "table"
    columns: list[TableColumn]
    rows: list[dict[str, Any]]
    caption: str = ""
    footnote: str = ""


class FigureResult(ResultBase):
    """
    `figure_json` is a Plotly figure dict produced by the single styling module
    in `app/charts/`. No handler may build its own figure.
    """

    kind: Literal["figure"] = "figure"
    figure_json: dict[str, Any]
    caption: str = ""
    alt_text: str = Field("", description="Accessible summary incl. the key numbers")
    data_behind: TableResult | None = None


class StatResult(ResultBase):
    kind: Literal["stat"] = "stat"
    value: float
    p_value: float | None = None
    statistic_name: str = ""
    df: float | None = None
    effect_size: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    interpretation: str = ""
    band: Literal["none", "weak", "moderate", "strong", "inconclusive"] = "none"


class NarrativeResult(ResultBase):
    kind: Literal["narrative"] = "narrative"
    paragraphs: list[str]
    audience: Literal["beginner", "peer", "reviewer"] = "beginner"
    ai_drafted: bool = Field(False, description="True only for AI-written prose")
    numbers_cited: dict[str, float] = Field(default_factory=dict)


class VerdictResult(ResultBase):
    kind: Literal["verdict"] = "verdict"
    verdict: Verdict
    headline: str
    checklist: list["ChecklistItem"] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    caveats_travelling: list[str] = Field(default_factory=list)


class ChecklistItem(BaseModel):
    key: str
    label: str
    passed: bool | None = Field(None, description="None = not applicable")
    value: str = ""
    detail: str = ""
    jump_to_stage: Stage | None = None


AnyResult = TableResult | FigureResult | StatResult | NarrativeResult | VerdictResult


# =========================================================================== #
# 3. The Advisor pattern  (§4.4)
# =========================================================================== #
class RuleOut(BaseModel):
    """
    The serializable face of a Rule. `because` MUST interpolate real computed
    values; a test asserts it contains at least one digit.
    """

    id: str
    stage: Stage
    recommend: str
    confidence: Confidence = Confidence.MEDIUM
    because: str
    provenance: list[str] = Field(
        default_factory=list, description="Result ids this recommendation consumed"
    )
    alternative: str | None = None
    risk_if_ignored: str = ""
    how_to: str = Field("", description="Exact click path in the UI")


class AdvisorReport(BaseModel):
    """
    Returned by every stage transition. This object — and only this object —
    is what the AI assistant is allowed to reason over.
    """

    stage: Stage
    headline: str
    recommendations: list[RuleOut] = Field(default_factory=list)
    blockers: list[RuleOut] = Field(default_factory=list)
    facts_used: dict[str, float | int | str | bool] = Field(default_factory=dict)
    confidence: Confidence = Confidence.MEDIUM
    next_stage_ready: bool = False
    narrative: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =========================================================================== #
# 4. Dataset & ingestion contracts  (Part 1)
# =========================================================================== #
class FrequencySpec(BaseModel):
    """A calendar, spelled out so the UI can show two candidate calendars as pictures."""

    code: str = Field(..., description="pandas offset alias: 'MS', 'QS', 'D', 'h', ...")
    label: str = Field(..., description="'Monthly (start of month)'")
    period: int | None = Field(None, description="Seasonal period m; None if non-seasonal")
    offset_step_ns: int | None = Field(None, description="Nominal spacing in nanoseconds")
    confidence: Confidence = Confidence.MEDIUM
    evidence: str = ""


class ParseIssue(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"]
    plain: str
    fix: str = ""
    rows_affected: int | None = None


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    n_non_null: int
    n_missing: int
    pct_missing: float
    n_unique: int
    is_numeric: bool
    is_datetime_like: bool
    sample_values: list[str] = Field(default_factory=list)
    role_guess: Literal["time", "target", "exogenous", "ignore", "unknown"] = "unknown"
    role_confidence: Confidence = Confidence.LOW


class MissingPattern(BaseModel):
    index: str
    n_missing: int
    pct_missing: float
    longest_run: int
    longest_run_start: str | None = None
    gaps_over_two: int = 0
    plain: str = ""


class DataQualityReport(BaseModel):
    """
    Produced once at ingest and consumed by the preprocessing advisor (S2) and
    the readiness gate (S8). Nothing here is a recommendation yet.
    """

    n_rows_raw: int
    n_rows_used: int
    n_columns: int
    dropped_rows: int = 0
    dropped_reason: str = ""

    time_column: str | None = None
    target_column: str | None = None
    exogenous_columns: list[str] = Field(default_factory=list)

    time_start: str | None = None
    time_end: str | None = None
    frequency: FrequencySpec | None = None
    n_observations: int = 0
    n_expected_observations: int | None = None
    n_gaps: int = 0
    is_regular: bool = True

    data_class: DataClass = DataClass.CONTINUOUS
    #: The evidence for the data-class verdict. Values are numbers, strings or booleans — and
    #: `None` where a statistic is undefined (a dispersion ratio needs a positive mean). Allowing
    #: None here is deliberate: a near-constant series would otherwise crash the *loader*, which
    #: turns "this series has no dispersion" into a 500 error on the first screen.
    data_class_evidence: dict[str, float | str | bool | None] = Field(default_factory=dict)

    n_missing_total: int = 0
    missing_by_column: list[MissingPattern] = Field(default_factory=list)
    n_zeros: int = 0
    pct_zeros: float = 0.0
    n_negatives: int = 0
    n_constant_runs: int = 0
    n_duplicate_timestamps: int = 0
    n_outlier_candidates: int = 0

    issues: list[ParseIssue] = Field(default_factory=list)
    plain_summary: str = ""


class SessionData(BaseModel):
    """The dataset as the rest of the app sees it. Never a raw DataFrame on the wire."""

    dataset_id: str
    name: str
    source: Literal["upload", "example", "url", "paste"] = "upload"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    time_column: str
    target_column: str
    exogenous_columns: list[str] = Field(default_factory=list)

    frequency: FrequencySpec
    n_observations: int
    time_start: str
    time_end: str
    data_class: DataClass = DataClass.CONTINUOUS

    quality: DataQualityReport
    content_hash: str = Field(..., description="sha256 of the values array, for reproducibility")
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)

    @staticmethod
    def hash_values(values: list[float]) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(values, separators=(",", ":")).encode())
        return h.hexdigest()[:16]


# =========================================================================== #
# 5. Session & pipeline state
# =========================================================================== #
class StageState(BaseModel):
    stage: Stage
    status: StageStatus = "not_started"
    completed_at: datetime | None = None
    duration_ms: int | None = None
    n_results: int = 0
    message: str = ""
    blockers: list[str] = Field(default_factory=list)


class PreprocessingRecord(BaseModel):
    """Every transformation is versioned and reversible — nothing is destructive."""

    version: int
    step: Literal[
        "impute", "smooth", "transform", "outlier", "aggregate", "trim", "difference"
    ]
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    advisor_rule_id: str | None = None
    applied_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reversible: bool = True


class PipelineState(BaseModel):
    """
    The spine of the app. Re-running a stage marks every downstream stage
    `stale` — visibly greyed out, never silently reused.
    """

    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    current_stage: Stage = Stage.S0_INGEST
    stages: dict[str, StageState] = Field(default_factory=dict)

    dataset: SessionData | None = None
    preprocessing: list[PreprocessingRecord] = Field(default_factory=list)
    series_version: int = 0

    result_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)

    seed: int = 20260101
    library_versions: dict[str, str] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.stages:
            self.stages = {
                s.value: StageState(stage=s) for s in STAGE_ORDER
            }

    def mark(self, stage: Stage, status: StageStatus, message: str = "") -> None:
        st = self.stages.setdefault(stage.value, StageState(stage=stage))
        st.status = status
        st.message = message
        if status == "complete":
            st.completed_at = datetime.now(timezone.utc)
        if status in {"complete", "running"}:
            # Anything downstream is now stale.
            idx = STAGE_ORDER.index(stage)
            for later in STAGE_ORDER[idx + 1:]:
                ls = self.stages.get(later.value)
                if ls and ls.status == "complete":
                    ls.status = "stale"
                    ls.message = f"Needs re-running because {stage.value} changed."

    def progress(self) -> dict[str, Any]:
        done = sum(1 for s in self.stages.values() if s.status == "complete")
        return {
            "session_id": self.session_id,
            "current_stage": self.current_stage.value,
            "stages_complete": done,
            "stages_total": len(STAGE_ORDER),
            "percent": round(100 * done / len(STAGE_ORDER), 1),
            "stages": list(self.stages.values()),
        }


# =========================================================================== #
# 6. Model registry contract  (§5.8)
# =========================================================================== #
class ModelRequirements(BaseModel):
    """What a model needs from the data. The suggestion engine filters on these."""

    min_n: int = 10
    requires_positive: bool = False
    requires_count: bool = False
    requires_non_negative: bool = False
    allows_missing: bool = False
    requires_regular: bool = True
    requires_stationarity: bool = False
    max_pct_zeros: float = 1.0
    min_zeros_for_support: float = 0.0
    requires_exogenous: bool = Field(
        False,
        description="True → the model is undefined without at least one outside column. Distinct from "
        "`capabilities.supports_exogenous`, which only says extra columns are allowed.",
    )
    min_two_models: bool = Field(
        False,
        description="True → the row is fitted *from* other fits rather than from the series, so it "
        "needs at least two of them. Forecast combination is the only such row.",
    )


class ModelCapabilities(BaseModel):
    supports_exogenous: bool = False
    supports_seasonality: bool = False
    supports_probabilistic: bool = False
    supports_intervals: bool = False
    supports_missing_values: bool = False
    supports_multistep_recursive: bool = True
    discrete_predictive: bool = Field(
        False, description="True → CRPS/PIT must use the PMF, not a density"
    )


class ModelSpec(BaseModel):
    """
    One row of the registry. Adding a model to Chronos Studio should mean adding
    a row here plus one wrapper file — never editing the suggestion engine.
    """

    model_id: str
    display_name: str
    family: ModelFamily
    description: str
    plain_description: str
    required: ModelRequirements = Field(default_factory=ModelRequirements)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    min_zeros_for_support: float = 0.0

    is_baseline: bool = False
    is_default_candidate: bool = True
    feature_flag: str | None = Field(None, description="Env/config flag that enables it")
    planned_part: int | None = Field(
        None,
        description=(
            "Build part that adds the fitting code, for rows this build cannot fit yet. The UI quotes "
            "it so the excluded list says *when*, not just *no* — 'arrives in a later part' is a "
            "sentence a user can plan around."
        ),
    )
    order_params: list[str] = Field(default_factory=list, description="e.g. ['p','d','q']")

    triggered_when: str = Field("", description="Plain-English trigger for the advisor")
    assumption_checks: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    def supports_data_class(self, dc: DataClass) -> bool:
        if self.required.requires_count:
            return dc in (DataClass.COUNT, DataClass.INTERMITTENT)
        if dc is DataClass.COUNT:
            return self.family in (
                ModelFamily.COUNT, ModelFamily.BASELINE, ModelFamily.INTERMITTENT
            )
        if dc is DataClass.INTERMITTENT:
            return self.family in (ModelFamily.INTERMITTENT, ModelFamily.BASELINE)
        if dc is DataClass.EVENT_TIMESTAMP:
            return self.family is ModelFamily.POINT_PROCESS
        return self.family not in (
            ModelFamily.COUNT, ModelFamily.INTERMITTENT, ModelFamily.POINT_PROCESS
        )


class ModelSuggestion(BaseModel):
    """
    One row of the S9 candidate list: a model, why it is offered *here*, and what would have to be
    true for it to win.

    `assumption_warnings` is separate from `incompatibilities` on purpose. An incompatibility means
    the model cannot be fitted to this data at all (a log transform on a series with zeros); an
    assumption warning means it can be fitted but the data visibly violates one of its premises.
    Collapsing the two either hides a warning or bans a usable model.
    """

    model_id: str
    display_name: str
    family: ModelFamily
    rank: int = 0
    is_baseline: bool = False
    recommended: bool = False
    #: Carried from the registry so the UI can group "advanced" models separately instead of mixing
    #: an INGARCH in with a mean forecast as if they were the same kind of choice.
    advanced: bool = False

    because: str = Field(..., description="Why it is suggested for *this* series")
    would_win_if: str = Field(..., description="What would have to be true for it to win")
    plain_description: str = ""

    compatible: bool = True
    incompatibilities: list[str] = Field(default_factory=list)
    assumption_warnings: list[str] = Field(default_factory=list)
    order_hint: str | None = None

    #: Rough cost signal so "fit everything" is never a surprise: seconds on this machine class.
    expected_seconds: float | None = None


class ModelSuggestionSet(BaseModel):
    stage: Stage = Stage.S9_SUGGEST_MODEL
    data_class: DataClass = DataClass.CONTINUOUS
    headline: str = ""
    plain: str = ""
    suggestions: list[ModelSuggestion] = Field(default_factory=list)
    excluded: list[ModelSuggestion] = Field(default_factory=list)
    default_fit_ids: list[str] = Field(default_factory=list)
    facts_used: dict[str, Any] = Field(default_factory=dict)
    n_registry: int = 0
    caveats: list[str] = Field(default_factory=list)


class ParamRow(BaseModel):
    """One estimated parameter with its standard error, and whether it hit a boundary."""

    name: str
    value: float | None = None
    std_err: float | None = None
    z: float | None = None
    p_value: float | None = None
    at_boundary: bool = False
    plain: str = ""


class FitSummary(BaseModel):
    """Serializable summary of a fitted model. The object itself stays server-side."""

    model_id: str
    display_name: str
    fit_id: str
    converged: bool = True
    n_params: int = 0
    loglik: float | None = None
    aic: float | None = None
    aicc: float | None = None
    bic: float | None = None
    hqic: float | None = None
    n_effective: int = 0
    data_hash: str = Field("", description="IC comparability guard (§5.9)")
    transform_stack: list[str] = Field(default_factory=list)
    converged_warnings: list[str] = Field(default_factory=list)
    failed: bool = False
    failure_reason: str = ""
    plain_one_liner: str = ""

    # --- Part 8 additions ------------------------------------------------- #
    family: ModelFamily = ModelFamily.BASELINE
    params: list[ParamRow] = Field(default_factory=list)
    fitted_on: str = Field("levels", description="levels | diff | log-diff | seasonal-diff | ...")
    #: The stationarity/invertibility check for ARMA-family fits; None where it does not apply.
    roots_ok: bool | None = None
    roots_note: str = ""
    n_obs_used: int = 0
    #: Caveats that must travel with anything this model produces.
    caveats: list[str] = Field(default_factory=list)
    #: True when the fit is a reference that is never expected to win (white noise, mean).
    is_baseline: bool = False


# =========================================================================== #
# 7. AI grounding contract  (§5.14)
# =========================================================================== #
class AnalysisDigest(BaseModel):
    """
    The ONLY thing the Gemini assistant receives by default. No raw values.
    Every number the assistant is allowed to quote must appear in `numbers`.
    """

    session_id: str
    schema_version: str = SCHEMA_VERSION

    n_observations: int
    frequency_label: str
    period: int | None
    time_start: str
    time_end: str
    data_class: DataClass
    series_name: str

    current_stage: Stage
    stage_verdicts: dict[str, str] = Field(default_factory=dict)
    advisor_reports: list[AdvisorReport] = Field(default_factory=list)
    top_results: list[dict[str, Any]] = Field(
        default_factory=list, description="id, title, plain, headline number"
    )
    numbers: dict[str, float] = Field(
        default_factory=dict, description="Flat map used by the grounding validator"
    )
    warnings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    panel_context: str = ""

    data_values_included: bool = False
    data_values: list[float] | None = None


class AIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AIChatRequest(BaseModel):
    session_id: str
    message: str
    panel_context: str = ""
    audience: Literal["beginner", "peer", "reviewer"] = "beginner"
    include_data_values: bool = False
    max_tokens: int = 1200


class AIChatResponse(BaseModel):
    reply: str
    grounded: bool
    ungrounded_numbers: list[str] = Field(default_factory=list)
    result_ids_cited: list[str] = Field(default_factory=list)
    model_used: str = ""
    offline_mode: bool = False
    usage: dict[str, int] = Field(default_factory=dict)


# =========================================================================== #
# 8. Envelope types
# =========================================================================== #
class AnalysisResponse(BaseModel):
    """Uniform envelope for every analysis endpoint."""

    stage: Stage
    advisor: AdvisorReport | None = None
    results: list[AnyResult] = Field(default_factory=list)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    duration_ms: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str = SCHEMA_VERSION
    stages: int = len(STAGE_ORDER)
    ai_configured: bool = False
    optional_features: dict[str, bool] = Field(default_factory=dict)


__all__ = [
    "SCHEMA_VERSION",
    "Stage",
    "STAGE_ORDER",
    "STAGE_LABELS",
    "StageStatus",
    "DataClass",
    "Confidence",
    "Verdict",
    "ModelFamily",
    "ResultBase",
    "TableColumn",
    "TableResult",
    "FigureResult",
    "StatResult",
    "NarrativeResult",
    "VerdictResult",
    "ChecklistItem",
    "AnyResult",
    "RuleOut",
    "AdvisorReport",
    "FrequencySpec",
    "ParseIssue",
    "ColumnProfile",
    "MissingPattern",
    "DataQualityReport",
    "SessionData",
    "StageState",
    "PreprocessingRecord",
    "PipelineState",
    "ModelRequirements",
    "ModelCapabilities",
    "ModelSpec",
    "FitSummary",
    "AnalysisDigest",
    "AIChatMessage",
    "AIChatRequest",
    "AIChatResponse",
    "AnalysisResponse",
    "HealthResponse",
]
