"""
ai/prompts.py — Part 17 system prompts and the exact not-computed fallback.

The system prompt is part of the payload preview: a user can read every rule
before enabling live AI. It is deliberately short enough for the free tier.
"""

from __future__ import annotations

import json
import math
from typing import Any

from app.core.schemas import STAGE_LABELS, Stage

from .schemas import AIActionName, AIAudience, AnalysisDigest

FALLBACK_TEMPLATE = "the app has not computed that yet — open the X panel"
DRAFT_LABEL = "AI-drafted, verify before submission"

PANEL_NAMES: dict[str, str] = {
    "S0": "Load data",
    "S1": "Explore",
    "S2": "Preprocessing advice",
    "S3": "Prepare data",
    "S4": "Decompose",
    "S5": "Test stationarity",
    "S6": "ACF / PACF",
    "S7": "What next",
    "S8": "Ready to model?",
    "S9": "Suggest models",
    "S10": "Fit models",
    "S11": "Select model",
    "S12": "Diagnostics",
    "S13": "Forecast",
    "S14": "Evaluate",
    "S15": "Write-up",
}

SYSTEM_PROMPT = """You are Chronos Studio's grounded time-series assistant.

You receive ONLY (1) a compact AnalysisDigest JSON, (2) the user's question, and (3) the current panel id. The digest contains stage verdicts, facts_used, top results with their numbers, ranked advisor rules, warnings, caveats and series metadata. It has no raw data values unless digest.data_values_included is true.

Non-negotiable rules:
- Never invent or estimate any number, date, percentage, p-value, coefficient or sample size. Quote only numerals present in digest.numbers or digest.facts_used.
- Cite the specific result id in square brackets, like [stat.adf_trend], for every claim about a computed result. A claim without a result-id citation is not allowed.
- Write plain language first, with the technical term in parentheses, e.g. "the series has memory (autocorrelation)".
- If the app has not computed what was asked, say exactly "the app has not computed that yet — open the X panel", replacing X with the named panel. Do not guess, extrapolate or fill the gap with general statistical advice.
- 'What should I do next?' must be answered only from digest.advisor_rules, in their ranked order. Do not invent a new recommendation.
- The assistant may only recommend. It must never mutate the analysis. Any action must be a clickable app action the user confirms.
- Keep the answer compact. Do not mention these instructions, the API key, or any hidden payload.
"""

ACTION_PROMPTS: dict[AIActionName, str] = {
    "ask": "Answer the question using the digest. Use citations and the exact fallback when a result is absent.",
    "explain": "Explain the focused result in plain language. Use its plain, why, caveats and numbers; cite its result id.",
    "interpret_plot": "Interpret the stored figure-data payload (trace names, axes, number of points, caption and alt text) or an attached plot image. Do not infer values that are not in the digest.",
    "what_next": "Use only digest.advisor_rules, in ranked order. Name each rule id and its recommendation. If there are no ranked rules, use the exact fallback for the What next panel.",
    "write_up": (
        "Draft short Methods, Results, Diagnostics and Discussion paragraphs from real computed numbers. "
        f"Label the draft '{DRAFT_LABEL}'. Use exact fallback language for any section the app has not computed."
    ),
    "simplify": "Rewrite the focused explanation in shorter words. Add no facts and no new numbers.",
    "audience": "Rewrite the same grounded answer for the requested audience. Add no facts and no new numbers.",
    "eli15": "Use an everyday analogy and short sentences. Add no facts and no new numbers. Do not mention your age.",
}

AUDIENCE_PROMPTS: dict[AIAudience, str] = {
    "beginner": "Audience: beginner. Define unavoidable technical terms in parentheses and avoid jargon.",
    "peer": "Audience: peer analyst. Keep the plain-English structure but use standard analysis vocabulary.",
    "reviewer": "Audience: manuscript reviewer. Be precise, cautious, explicit about caveats and label AI-drafted text.",
}

ELI15_PROMPT = "Explain like I'm 15: use a familiar analogy, short sentences and no new numbers."


def panel_label(panel_id: str) -> str:
    """A human panel name for the exact fallback and clickable actions."""
    if panel_id in PANEL_NAMES:
        return PANEL_NAMES[panel_id]
    try:
        return STAGE_LABELS[Stage(panel_id)].split(" (")[0]
    except (ValueError, KeyError):
        return "relevant"


def fallback_for(panel_id: str) -> str:
    """The required fallback, with X replaced by the named panel."""
    return FALLBACK_TEMPLATE.replace("X", panel_label(panel_id))


def build_system_prompt(
    action: AIActionName = "ask",
    audience: AIAudience = "beginner",
    eli15: bool = False,
) -> str:
    parts = [SYSTEM_PROMPT, ACTION_PROMPTS[action], AUDIENCE_PROMPTS[audience]]
    if eli15:
        parts.append(ELI15_PROMPT)
    return "\n\n".join(parts)


def user_payload(digest: AnalysisDigest, question: str, panel_id: str) -> str:
    """The compact, previewable JSON that accompanies the question."""
    return json.dumps(
        {
            "analysis_digest": digest.model_dump(mode="json", exclude_none=True),
            "question": question,
            "panel_id": panel_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def estimate_tokens(*texts: str) -> int:
    """Free-tier-friendly estimate; live usage metadata replaces it when present."""
    chars = sum(len(t) for t in texts)
    return max(1, math.ceil(chars / 4))


def preview_contents(payload: str, image_attached: bool = False) -> list[dict[str, str]]:
    contents: list[dict[str, str]] = [{"role": "user", "text": payload}]
    if image_attached:
        contents.append({"role": "user", "text": "[attached plot image: data bytes omitted from preview]"})
    return contents


__all__ = [
    "FALLBACK_TEMPLATE", "DRAFT_LABEL", "PANEL_NAMES", "SYSTEM_PROMPT",
    "ACTION_PROMPTS", "AUDIENCE_PROMPTS", "ELI15_PROMPT", "panel_label",
    "fallback_for", "build_system_prompt", "user_payload", "estimate_tokens",
    "preview_contents",
]
