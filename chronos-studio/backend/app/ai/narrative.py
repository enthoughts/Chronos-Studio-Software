"""
ai/narrative.py — deterministic, offline answers from the AnalysisDigest.

No key, SDK or network call is required. Every sentence is built from stored
results; every numeral is passed through the grounding sanitizer before it can
reach the UI. This is the exact generator used by "No API key? Use offline mode".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .grounding import canonical, citations_in, sanitize_numbers, validate_grounding
from .prompts import DRAFT_LABEL, fallback_for, panel_label
from .schemas import (
    AIAction,
    AIActionName,
    AIAudience,
    AIResultBrief,
    AnalysisDigest,
)

KIND_TECHNICAL = {
    "table": "results table",
    "figure": "figure-data payload",
    "stat": "statistical test",
    "narrative": "computed narrative",
    "verdict": "readiness verdict",
}

AUDIENCE_LEAD: dict[AIAudience, str] = {
    "beginner": "In plain language (beginner audience):",
    "peer": "For a peer analyst (audience rewrite):",
    "reviewer": "For a manuscript reviewer (audience rewrite):",
}


@dataclass(frozen=True)
class OfflineAnswer:
    text: str
    citations: list[str] = field(default_factory=list)
    actions: list[AIAction] = field(default_factory=list)
    ai_drafted: bool = False
    draft_label: str = ""


_LABEL_OVERRIDES = {
    "observations n": "observations",
    "n": "sample size",
    "n breaks": "break count",
    "p value": "p-value",
    "df": "degrees of freedom",
    "sd": "standard deviation",
    "cv": "coefficient of variation",
}


def _clean_label(key: str) -> str:
    label = key.split(".")[-1].replace("_", " ")
    label = re.sub(r"\d+", "", label)
    label = " ".join(label.split())
    return _LABEL_OVERRIDES.get(label, label)


def _number_label(key: str) -> str:
    label = _clean_label(key)
    if label in {"value", "r value", "v", ""} and "." in key:
        label = _clean_label(key.rsplit(".", 1)[0])
    return label


_STOPWORDS = frozenset({
    "about", "could", "explain", "figure", "from", "given", "please", "result",
    "results", "show", "that", "their", "there", "these", "this", "what",
    "which", "with", "would", "your",
})

_KEYWORD_HINTS: dict[str, tuple[str, ...]] = {
    "stationarity": ("unit root", "stationar", "adf", "kpss", "differenc"),
    "descriptive": ("level", "spread", "shape", "summary", "descriptive"),
    "diagnostic": ("residual", "ljung", "white", "adequacy", "diagnos"),
    "forecast": ("forecast", "horizon", "prediction interval"),
    "accuracy": ("mase", "mape", "rmse", "holdout", "evaluation"),
    "correlation": ("acf", "pacf", "lag", "autocorrelation"),
    "readiness": ("ready", "gate", "checklist", "blocker", "caveat"),
    "seasonal": ("seasonal", "seasonality", "month", "period"),
    "outlier": ("outlier", "anomal", "extreme"),
}


def _clean_prose(text: str, *, limit: int = 240) -> str:
    """Keep complete sentences and remove hard truncation fragments."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{2,}", ".", text)
    if len(text) <= limit:
        # A short field such as a rule recommendation may legitimately have no
        # terminal punctuation. Do not invent a truncation ellipsis for it.
        return text.rstrip(" ,;:-")
    cut = text[:limit]
    boundaries = [cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("; ")]
    boundary = max(boundaries)
    if boundary >= 60:
        return cut[: boundary + 1].strip()
    clause = max(cut.rfind(", "), cut.rfind(": "), cut.rfind(" — "))
    if clause >= 60:
        return cut[:clause].rstrip(" ,;:-") + "…"
    return cut.rstrip(" ,;:-") + "…"


def _first_sentence(text: str, *, limit: int = 220) -> str:
    """Use the first complete claim from a possibly truncated result field."""
    clean = _clean_prose(text, limit=limit)
    match = re.search(r"[.!?](?:\s|$)", clean)
    if match and match.end() >= 45:
        return clean[: match.end()].strip()
    return clean


def _sentence(text: str, *, limit: int = 220) -> str:
    """A complete sentence with terminal punctuation, or an empty string."""
    clean = _first_sentence(text, limit=limit)
    if not clean or clean == "…":
        return ""
    return clean if clean.endswith((".", "!", "?", "…")) else clean + "."


def _fmt_number(value: float) -> str:
    # Use the validator's canonical representation. Rounding here would create
    # a numeral that is absent from the digest and get replaced by the sanitizer.
    return canonical(str(value)) or str(value)


def _meaningful_number_key(key: str) -> bool:
    lower = key.lower()
    return not any(marker in lower for marker in (
        ".text_", "text_", ".cav", ".warn", ".caption", ".alt", ".check_",
    ))


def _numbers_sentence(brief: AIResultBrief, limit: int = 3) -> str:
    seen_values: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for key, value in brief.numbers.items():
        if not _meaningful_number_key(key):
            continue
        label = _number_label(key)
        number = _fmt_number(value)
        if not number or number in seen_values:
            continue
        seen_values.add(number)
        if label in {"value", "r value", "v"}:
            label = ""
        pairs.append((label, number))
        if len(pairs) >= limit:
            break
    if not pairs:
        return ""
    if all(not label for label, _ in pairs):
        rendered = ", ".join(number for _, number in pairs)
    else:
        rendered = ", ".join(
            f"{label} {number}" if label else number for label, number in pairs
        )
    return " The reported numbers are " + rendered + "."


def _focus(digest: AnalysisDigest, result_id: str | None) -> AIResultBrief | None:
    # Only an explicit result id is authoritative. Otherwise let the question
    # choose the most relevant brief; a bare keyword-less question falls back
    # to the digest's first result through _keyword_focus().
    if result_id:
        for brief in digest.top_results:
            if brief.id == result_id:
                return brief
    return None


def _keyword_focus(digest: AnalysisDigest, question: str) -> AIResultBrief | None:
    words = [
        w for w in re.findall(r"[a-z]+", question.lower())
        if len(w) >= 4 and w not in _STOPWORDS
    ]
    if not words:
        return digest.focus_result
    best: AIResultBrief | None = None
    best_score = 0
    for brief in digest.top_results:
        haystack = " ".join((
            brief.id, brief.title, brief.stage, brief.plain, brief.why,
            brief.technical_label,
        )).lower()
        score = 0
        for word in words:
            hints = _KEYWORD_HINTS.get(word)
            if hints:
                if word in haystack:
                    score += 3
                elif any(hint in haystack for hint in hints):
                    score += 2
                continue
            stem = word[:5] if len(word) > 5 else word
            if word in haystack or stem in haystack:
                score += 2
        if score > best_score:
            best, best_score = brief, score
    return best or digest.focus_result


def _cite(brief: AIResultBrief | None) -> str:
    return f"[{brief.id}]" if brief is not None else ""


def _technical(brief: AIResultBrief) -> str:
    known = {
        "models.excluded": "model-registry eligibility rules",
        "models.suggestions": "ranked candidate-model rules",
        "table.descriptives": "descriptive statistics",
        "narrative.readiness": "readiness narrative",
        "table.gate_checklist": "readiness checklist",
        "verdict.readiness": "readiness verdict",
        "narrative.stationarity_verdict": "ADF and KPSS unit-root tests",
    }
    if brief.id in known:
        return known[brief.id]
    return KIND_TECHNICAL.get(brief.kind, "computed result")


def _caveat_line(brief: AIResultBrief) -> str:
    note = _sentence((brief.caveats or brief.warnings or [""])[0])
    return f" Caveat: {note}" if note else ""


def _plain_answer(
    brief: AIResultBrief,
    *,
    audience: AIAudience,
    eli15: bool,
    lead: str | None = None,
) -> str:
    intro = lead or AUDIENCE_LEAD[audience]
    if eli15:
        intro = f"{AUDIENCE_LEAD[audience]} Think of the series like a repeating playlist (an everyday analogy):"
    plain = _sentence(brief.plain).rstrip(".!?")
    why = _sentence(brief.why, limit=180)
    why_line = f" Why it matters: {why}" if why else ""
    return (
        f"{intro} {plain} ({_technical(brief)})."
        f"{why_line}{_numbers_sentence(brief)}{_caveat_line(brief)} {_cite(brief)}"
    )


def _rule_actions(digest: AnalysisDigest) -> list[AIAction]:
    actions: list[AIAction] = []
    for rule in digest.advisor_rules[:3]:
        stage = rule.stage if re.fullmatch(r"S\d{1,2}", rule.stage) else "S7"
        short_rule = rule.id.rsplit(".", 1)[-1].replace("_", " ")
        actions.append(AIAction(
            label=f"{panel_label(stage)} · {short_rule}",
            stage=stage,
            kind="open_panel",
            rule_id=rule.id,
            description=rule.recommend,
        ))
    return actions


def _rule_citations(digest: AnalysisDigest) -> list[str]:
    cites: list[str] = []
    for rule in digest.advisor_rules[:3]:
        for rid in rule.provenance:
            if rid in digest.result_ids and rid not in cites:
                cites.append(rid)
    if not cites and digest.result_ids:
        cites.append(digest.result_ids[0])
    return cites


def _what_next(digest: AnalysisDigest) -> tuple[str, list[str], list[AIAction]]:
    if not digest.advisor_rules:
        return fallback_for("S7"), [], []
    bullets: list[str] = []
    for rule in digest.advisor_rules[:3]:
        recommend = _sentence(rule.recommend).rstrip(".!?") or "Open the suggested panel"
        because = f" Because: {_sentence(rule.because)}" if rule.because else ""
        how_to = f" How to: {_sentence(rule.how_to)}" if rule.how_to else ""
        bullets.append(
            f"• {recommend} ({rule.id}, ranked advisor rule)."
            f"{because}{how_to}"
        )
    cites = _rule_citations(digest)
    cite_text = " " + " ".join(f"[{rid}]" for rid in cites[:3]) if cites else ""
    text = (
        "The app's ranked advisor rules (rule engine) say, in order:\n"
        + "\n".join(bullets)
        + cite_text
        + " These are recommendations only; open the suggested panel yourself to confirm an action."
    )
    return text, cites, _rule_actions(digest)


def _figure(digest: AnalysisDigest, result_id: str | None, image_attached: bool) -> AIResultBrief | None:
    if result_id:
        for brief in digest.top_results:
            if brief.id == result_id and brief.figure is not None:
                return brief
    for brief in digest.top_results:
        if brief.figure is not None:
            return brief
    return None


def _interpret_plot(
    digest: AnalysisDigest,
    result_id: str | None,
    image_attached: bool,
) -> str:
    brief = _figure(digest, result_id, image_attached)
    if brief is None or brief.figure is None:
        return fallback_for("S1")
    fig = brief.figure
    names = ", ".join(fig.trace_names) if fig.trace_names else "the stored traces"
    axes = " / ".join(x for x in (fig.x_title, fig.y_title) if x) or "the plotted axes"
    image_note = (
        " An image was attached; offline mode used the stored figure-data payload rather than pixel data."
        if image_attached else ""
    )
    caption = f" Caption: {_sentence(brief.caption)}" if brief.caption else ""
    alt = f" Accessible summary: {_sentence(brief.alt_text)}" if brief.alt_text else ""
    return (
        f"This plot shows {names} against {axes} (figure-data payload). "
        f"The stored payload has {fig.n_traces} named traces and {fig.n_points} points; "
        f"the data-behind table has {brief.data_behind_rows} rows."
        f"{caption}{alt}{image_note} "
        f"Interpret only what is in the payload: {_sentence(brief.plain)} {_cite(brief)}"
    )


def _section_result(digest: AnalysisDigest, keywords: tuple[str, ...]) -> AIResultBrief | None:
    for brief in digest.top_results:
        haystack = f"{brief.title} {brief.plain} {brief.stage}".lower()
        if any(word in haystack for word in keywords):
            return brief
    return None


def _write_up(digest: AnalysisDigest) -> tuple[str, list[str], bool]:
    methods = (
        f"Methods. The analysis used {digest.series_name}, with "
        f"{digest.n_observations} observations at {digest.frequency_label} frequency "
        f"from {digest.time_start or 'the start'} to {digest.time_end or 'the end'} "
        f"(time-series analysis). The assistant used only computed summaries."
    )
    results_brief = digest.focus_result
    results = (
        f"Results. {_sentence(results_brief.plain).rstrip('.!?')} ({_technical(results_brief)})."
        f"{_numbers_sentence(results_brief)} {_cite(results_brief)}"
        if results_brief else fallback_for("S1")
    )
    diagnostics = _section_result(digest, ("diagnos", "residual", "adequacy", "ljung", "white"))
    if diagnostics is None:
        diagnostics_text = fallback_for("S12")
    else:
        diagnostics_text = (
            f"Diagnostics. {_sentence(diagnostics.plain).rstrip('.!?')} (diagnostic check)."
            f"{_numbers_sentence(diagnostics, limit=2)} {_cite(diagnostics)}"
        )
    discussion_brief = _section_result(digest, ("caveat", "warning", "blocker", "verdict")) or results_brief
    if discussion_brief is None:
        discussion = fallback_for("S15")
    else:
        caveat = _sentence((digest.caveats or digest.warnings or [""])[0])
        caveat_text = f" Main caveat: {caveat}" if caveat else ""
        discussion = (
            f"Discussion. The computed evidence supports a cautious reading of "
            f"{discussion_brief.title} (interpretation).{caveat_text} {_cite(discussion_brief)}"
        )
    cites = citations_in(
        f"{methods} {results} {diagnostics_text} {discussion}", digest
    )
    if not cites and digest.result_ids:
        rid = digest.result_ids[0]
        results += f" [{rid}]"
        cites = [rid]
    extra = [rid for rid in digest.result_ids[:5] if rid not in cites]
    grounded_line = (
        " Grounded in " + ", ".join(f"[{rid}]" for rid in extra) + "."
        if extra else ""
    )
    text = (
        f"{DRAFT_LABEL}\n\n"
        f"{methods}\n\n{results}\n\n{diagnostics_text}\n\n{discussion}"
        f"{grounded_line}"
    )
    cites = citations_in(text, digest)
    return text, cites, True


def _ask(
    digest: AnalysisDigest,
    request: Any,
) -> str:
    question = str(getattr(request, "question", "") or "")
    lower = question.lower()
    if any(term in lower for term in ("what next", "next step", "what should i do")):
        text, _, _ = _what_next(digest)
        return text
    if any(term in lower for term in ("plot", "figure", "chart", "graph")):
        return _interpret_plot(digest, getattr(request, "result_id", None), False)
    if any(term in lower for term in ("write up", "methods", "discussion", "manuscript")):
        text, _, _ = _write_up(digest)
        return text
    if digest.data_values is None and any(term in lower for term in ("raw value", "data value", "actual value")):
        return (
            "Raw data values were not sent (privacy default). Turn on the explicit "
            "Send data values toggle in the assistant, review the payload preview, "
            f"and ask again. {fallback_for('S0')}"
        )
    brief = _focus(digest, getattr(request, "result_id", None)) or _keyword_focus(digest, question)
    if brief is None:
        return fallback_for(str(getattr(request, "panel_id", "S15")))
    return _plain_answer(
        brief,
        audience=getattr(request, "audience", "beginner"),
        eli15=bool(getattr(request, "eli15", False)),
        lead="From the computed digest (grounded answer),",
    )


def generate_offline(
    digest: AnalysisDigest,
    request: Any,
    *,
    image_attached: bool = False,
) -> OfflineAnswer:
    """Return a deterministic, grounded answer for any Part 17 action."""
    action: AIActionName = getattr(request, "action", "ask")
    audience: AIAudience = getattr(request, "audience", "beginner")
    eli15 = bool(getattr(request, "eli15", False))
    result_id = getattr(request, "result_id", None)
    focus = _focus(digest, result_id) or _keyword_focus(digest, str(getattr(request, "question", "")))
    citations: list[str] = []
    actions: list[AIAction] = []
    ai_drafted = False
    draft_label = ""

    if action == "ask":
        text = _ask(digest, request)
    elif action in {"explain", "simplify", "audience", "eli15"}:
        if focus is None:
            text = fallback_for(str(getattr(request, "panel_id", "S15")))
        else:
            text = _plain_answer(
                focus,
                audience=audience,
                eli15=eli15 or action == "eli15",
                lead=(
                    "In shorter words (plain-language simplification),"
                    if action in {"simplify", "eli15"} else None
                ),
            )
    elif action == "interpret_plot":
        text = _interpret_plot(digest, result_id, image_attached)
    elif action == "what_next":
        text, citations, actions = _what_next(digest)
    elif action == "write_up":
        text, citations, ai_drafted = _write_up(digest)
        draft_label = DRAFT_LABEL if ai_drafted else ""
    else:  # pragma: no cover - Literal exhaustiveness guard
        text = fallback_for(str(getattr(request, "panel_id", "S15")))

    text = "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())
    text = sanitize_numbers(text, digest)
    report = validate_grounding(text, digest)
    if report.ungrounded_numbers:
        text = sanitize_numbers(text, digest)
    citations = citations_in(text, digest) or citations
    if not citations:
        # A fallback sentence is allowed without a citation; every other
        # computed claim must still carry a real result id.
        if not report.fallback_used and digest.result_ids:
            rid = digest.result_ids[0]
            text += f" [{rid}]"
            citations = [rid]
    return OfflineAnswer(
        text=text,
        citations=citations,
        actions=actions,
        ai_drafted=ai_drafted,
        draft_label=draft_label,
    )


__all__ = ["OfflineAnswer", "generate_offline"]
