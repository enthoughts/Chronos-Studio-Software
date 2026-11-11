"""
ai/grounding.py — post-hoc number-grounding and result-id citation checks.

Every numeral in an assistant reply must be traceable to `AnalysisDigest.numbers`
or `AnalysisDigest.facts_used`. Numbers embedded in identifiers, dates, model
names, versions and list markers are not factual claims and are masked first.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from app.core.config import get_settings

from .prompts import FALLBACK_TEMPLATE
from .schemas import AIGroundingReport, AnalysisDigest

NUM_RE = re.compile(
    r"(?<![\w.])(?:[-−–+]\s*)?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?:%|pp)?"
)
DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?\b"
)
VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+\b")
STAGE_RE = re.compile(r"\bS\d{1,2}\b")
PART_RE = re.compile(r"\bPart\s+\d+\b")
MODEL_RE = re.compile(r"\bgemini-[\w.\-]+\b", re.IGNORECASE)
LIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
CITATION_RE = re.compile(r"\[([^\]]+)\]")


def canonical(token: str) -> str | None:
    """Canonical decimal string: '1,234.50%' → '1234.5', '1E+2' → '100'."""
    s = token.strip().replace("−", "-").replace("–", "-")
    s = s.removesuffix("%").removesuffix("pp").replace(",", "").replace(" ", "")
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    # 'f' keeps large/small values out of scientific notation.
    out = format(d.normalize(), "f")
    return out or "0"


def _mask_identifiers(text: str, digest: AnalysisDigest | None = None) -> str:
    masked = text
    if digest is not None:
        for rid in sorted(digest.result_ids, key=len, reverse=True):
            if rid:
                masked = masked.replace(rid, " " * len(rid))
        masked = masked.replace(digest.session_id, " " * len(digest.session_id))
    settings = get_settings()
    for name in (settings.gemini_model, settings.gemini_model_pro):
        masked = MODEL_RE.sub(" ", masked) if name in masked else masked
    masked = CITATION_RE.sub(lambda m: " " * len(m.group(0)), masked)
    masked = DATE_RE.sub(lambda m: " " * len(m.group(0)), masked)
    masked = VERSION_RE.sub(lambda m: " " * len(m.group(0)), masked)
    masked = PART_RE.sub(lambda m: " " * len(m.group(0)), masked)
    masked = STAGE_RE.sub(lambda m: " " * len(m.group(0)), masked)
    masked = LIST_RE.sub(lambda m: " " * len(m.group(0)), masked)
    # A model name that is not one of the configured two still contains a
    # version-like numeral. Mask any remaining `gemini-2.5-...` token.
    masked = MODEL_RE.sub(" ", masked)
    return masked


def numerals(text: str, digest: AnalysisDigest | None = None) -> list[str]:
    """Factual numeral tokens, canonicalized, with identifiers/dates removed."""
    masked = _mask_identifiers(text, digest)
    out: list[str] = []
    for match in NUM_RE.finditer(masked):
        c = canonical(match.group(0))
        if c is not None:
            out.append(c)
    return out


def _numbers_from_value(value: Any) -> Iterable[str]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float, Decimal)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(f):
            c = canonical(str(f))
            if c is not None:
                yield c
    elif isinstance(value, str):
        yield from numerals(value)


def allowed_numbers(digest: AnalysisDigest) -> set[str]:
    """Everything the validator is allowed to see in a grounded reply."""
    allowed: set[str] = set()
    for value in digest.numbers.values():
        allowed.update(_numbers_from_value(value))
    for value in digest.facts_used.values():
        allowed.update(_numbers_from_value(value))
    for value in (
        digest.n_observations,
        digest.period,
        len(digest.result_ids),
        len(digest.advisor_rules),
        len(digest.figure_payloads),
    ):
        allowed.update(_numbers_from_value(value))
    for value in digest.data_values or []:
        allowed.update(_numbers_from_value(value))
    for value in digest.stage_verdicts.values():
        allowed.update(_numbers_from_value(value))
    for text in (*digest.warnings, *digest.caveats):
        allowed.update(numerals(text, digest))
    for brief in digest.top_results:
        for value in brief.numbers.values():
            allowed.update(_numbers_from_value(value))
        for value in (
            brief.plain, brief.why, brief.technical_label, brief.alt_text,
            brief.caption, *brief.caveats, *brief.warnings,
        ):
            allowed.update(numerals(value, digest))
        allowed.update(_numbers_from_value(brief.data_behind_rows))
        if brief.figure is not None:
            f = brief.figure
            for value in (f.n_traces, f.n_points, *f.sample_values):
                allowed.update(_numbers_from_value(value))
            for value in (f.caption, f.alt_text, *f.trace_names, f.x_title, f.y_title):
                allowed.update(numerals(value, digest))
    for rule in digest.advisor_rules:
        for value in (
            rule.recommend, rule.because, rule.how_to, rule.confidence,
            *rule.provenance, rule.rank,
        ):
            allowed.update(_numbers_from_value(value))
    return allowed


def citations_in(text: str, digest: AnalysisDigest) -> list[str]:
    """Only known result ids count; `[1]` is not a specific result-id citation."""
    found: list[str] = []
    for rid in digest.result_ids:
        if rid and (f"[{rid}]" in text or re.search(rf"(?<![\w.]){re.escape(rid)}(?![\w.])", text)):
            if rid not in found:
                found.append(rid)
    return found


def validate_grounding(
    text: str,
    digest: AnalysisDigest,
    *,
    require_citations: bool = True,
) -> AIGroundingReport:
    """Flag numerals absent from the digest and missing result-id citations."""
    allowed = allowed_numbers(digest)
    ungrounded = sorted({n for n in numerals(text, digest) if n not in allowed})
    citations = citations_in(text, digest)
    fallback_used = FALLBACK_TEMPLATE.split(" — ")[0] in text
    missing = require_citations and not citations and not fallback_used
    return AIGroundingReport(
        grounded=not ungrounded and not missing,
        citations=citations,
        ungrounded_numbers=ungrounded,
        missing_citations=missing,
        fallback_used=fallback_used,
    )


def sanitize_numbers(text: str, digest: AnalysisDigest) -> str:
    """Replace ungrounded numerals with a non-numeric phrase before streaming."""
    allowed = allowed_numbers(digest)
    masked = _mask_identifiers(text, digest)
    spans: list[tuple[int, int]] = []
    for match in NUM_RE.finditer(masked):
        c = canonical(match.group(0))
        if c is not None and c not in allowed:
            spans.append(match.span())
    if not spans:
        return text
    out: list[str] = []
    pos = 0
    for start, end in spans:
        out.append(text[pos:start])
        out.append("the reported value")
        pos = end
    out.append(text[pos:])
    return "".join(out)


__all__ = [
    "NUM_RE", "canonical", "numerals", "allowed_numbers", "citations_in",
    "validate_grounding", "sanitize_numbers",
]
