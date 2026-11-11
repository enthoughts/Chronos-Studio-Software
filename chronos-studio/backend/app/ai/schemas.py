"""
ai/schemas.py — the Part 17 AI contracts, including the AnalysisDigest.

The Part 0 `app.core.schemas.AnalysisDigest` is the stable wire contract. Part 17
extends it here with the compact, previewable fields the assistant actually uses.
Nothing in this module knows about a DataFrame, a fitted object or an API key.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.schemas import SCHEMA_VERSION, AnalysisDigest as CoreAnalysisDigest

AIActionName = Literal[
    "ask", "explain", "interpret_plot", "what_next", "write_up",
    "simplify", "audience", "eli15",
]
AIAudience = Literal["beginner", "peer", "reviewer"]
AIMode = Literal["offline", "gemini", "degraded"]
AIRequestMode = Literal["auto", "offline", "gemini"]
AIModelChoice = Literal["flash", "pro"]

DIGEST_SCHEMA_VERSION = SCHEMA_VERSION
PRIVACY_NOTE = (
    "By default the assistant receives only computed summaries, not data values. "
    "The optional key travels in a request header, is never logged or stored "
    "server-side, and the entire session is deleted with the session TTL or "
    "Delete session."
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AIFigureBrief(_Strict):
    """Compact figure-data payload: trace names and shape, never pixels/values."""
    result_id: str
    title: str = ""
    caption: str = ""
    alt_text: str = ""
    n_traces: int = 0
    n_points: int = 0
    trace_names: list[str] = Field(default_factory=list)
    x_title: str = ""
    y_title: str = ""
    sample_values: list[float] = Field(default_factory=list)
    values_truncated: bool = False


class AIResultBrief(_Strict):
    id: str
    title: str
    stage: str
    kind: str
    plain: str = ""
    why: str = ""
    technical_label: str = ""
    numbers: dict[str, float] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    alt_text: str = ""
    caption: str = ""
    data_behind_rows: int = 0
    figure: AIFigureBrief | None = None


class AIRuleBrief(_Strict):
    """One already-ranked advisor rule. 'What next' may only quote these."""
    id: str
    stage: str
    recommend: str
    because: str = ""
    how_to: str = ""
    provenance: list[str] = Field(default_factory=list)
    confidence: str = ""
    kind: str = ""
    rank: int | None = None


class AnalysisDigest(CoreAnalysisDigest):
    """
    The ONLY statistical payload the assistant may see (§5.14).

    The base model from Part 0 stays importable and unchanged for existing
    clients. This subclass adds the Part 17 preview/facts/rule/figure fields;
    `data_values` remains `None` unless the user explicitly opts in.
    """
    generated_at: str = ""
    panel_id: str = "S15"
    facts_used: dict[str, float | int | str | bool] = Field(default_factory=dict)
    result_ids: list[str] = Field(default_factory=list)
    advisor_rules: list[AIRuleBrief] = Field(default_factory=list)
    figure_payloads: list[AIFigureBrief] = Field(default_factory=list)
    top_results: list[AIResultBrief] = Field(default_factory=list)
    privacy_note: str = PRIVACY_NOTE
    values_truncated: bool = False

    @property
    def focus_result(self) -> AIResultBrief | None:
        return self.top_results[0] if self.top_results else None


class AIAction(_Strict):
    """A clickable, user-confirmed app action — never an automatic mutation."""
    label: str
    stage: str
    kind: Literal["open_panel", "explain_result"] = "open_panel"
    rule_id: str = ""
    result_id: str | None = None
    description: str = ""


class AIGroundingReport(_Strict):
    grounded: bool
    citations: list[str] = Field(default_factory=list)
    ungrounded_numbers: list[str] = Field(default_factory=list)
    missing_citations: bool = False
    fallback_used: bool = False


class AIUsage(_Strict):
    requests: int = 0
    remaining: int = 60
    max_requests: int = 60
    prompt_tokens: int = 0
    response_tokens: int = 0
    estimated_cost_usd: float = 0.0
    queued: int = 0
    degraded: bool = False
    rate_limited: bool = False


class AIHistoryMessage(_Strict):
    role: Literal["user", "assistant"]
    content: str
    action: AIActionName = "ask"
    panel_id: str = ""
    citations: list[str] = Field(default_factory=list)
    grounded: bool = True
    ungrounded_numbers: list[str] = Field(default_factory=list)
    mode: AIMode = "offline"
    prompt_tokens: int = 0
    response_tokens: int = 0
    timestamp: str = ""


class AIChatRequest(_Strict):
    question: str = Field(..., min_length=1, max_length=4000)
    action: AIActionName = "ask"
    result_id: str | None = None
    panel_id: str = "S15"
    audience: AIAudience = "beginner"
    eli15: bool = False
    include_values: bool = False
    mode: AIRequestMode = "auto"
    model: AIModelChoice | None = None
    image_data_url: str | None = Field(default=None, max_length=6_000_000)


class AIPreviewRequest(_Strict):
    question: str = ""
    action: AIActionName = "ask"
    result_id: str | None = None
    panel_id: str = "S15"
    audience: AIAudience = "beginner"
    eli15: bool = False
    include_values: bool = False
    model: AIModelChoice | None = None
    mode: AIRequestMode = "auto"
    image_data_url: str | None = Field(default=None, max_length=6_000_000)
    acknowledge: bool = False


class AIPayloadPreview(_Strict):
    """Exactly the non-secret payload that a live request would send."""
    mode: AIMode
    model: str | None
    system_prompt: str
    contents: list[dict[str, str]]
    digest: AnalysisDigest
    question: str
    panel_id: str
    estimated_prompt_tokens: int
    api_key_in_payload: bool = False
    data_values_included: bool = False
    image_attached: bool = False
    image_bytes: int = 0
    image_mime: str | None = None
    privacy_note: str = PRIVACY_NOTE
    preview_acknowledged: bool = False


class AIConfigResponse(_Strict):
    offline_ready: bool = True
    sdk_available: bool = False
    env_key_present: bool = False
    live_ready: bool = False
    default_model: str
    pro_model: str
    temperature: float
    max_requests_per_session: int
    max_requests_per_minute: int
    min_interval_seconds: float
    max_values: int
    allow_data_values_by_default: bool
    input_usd_per_million: float
    output_usd_per_million: float
    privacy_note: str = PRIVACY_NOTE


class AIHistoryResponse(_Strict):
    messages: list[AIHistoryMessage] = Field(default_factory=list)
    usage: AIUsage = Field(default_factory=AIUsage)
    preview_acknowledged: bool = False
    session_id: str = ""


class AIStreamMeta(_Strict):
    mode: AIMode
    model: str | None = None
    action: AIActionName
    panel_id: str
    citations: list[str] = Field(default_factory=list)
    usage: AIUsage = Field(default_factory=AIUsage)


class AIStreamChunk(_Strict):
    text: str


class AIStreamDone(_Strict):
    text: str
    grounding: AIGroundingReport
    citations: list[str] = Field(default_factory=list)
    actions: list[AIAction] = Field(default_factory=list)
    usage: AIUsage
    mode: AIMode
    ai_drafted: bool = False
    draft_label: str = ""


class AIStreamError(_Strict):
    code: str
    plain: str
    fix: str


__all__ = [
    "AIActionName", "AIAudience", "AIMode", "AIRequestMode", "AIModelChoice",
    "DIGEST_SCHEMA_VERSION", "PRIVACY_NOTE", "AIFigureBrief", "AIResultBrief",
    "AIRuleBrief", "AnalysisDigest", "AIAction", "AIGroundingReport", "AIUsage",
    "AIHistoryMessage", "AIChatRequest", "AIPreviewRequest", "AIPayloadPreview",
    "AIConfigResponse", "AIHistoryResponse", "AIStreamMeta", "AIStreamChunk",
    "AIStreamDone", "AIStreamError",
]
