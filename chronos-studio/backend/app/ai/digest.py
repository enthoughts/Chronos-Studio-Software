"""
ai/digest.py — build the compact, previewable AnalysisDigest from a Session.

Design rule: no raw series values, no fitted objects, no opaque DataFrames and
no API keys. Only computed result summaries, advisor facts/rules and series
metadata. `include_values=True` is an explicit, previewable user opt-in.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel

from app.core.schemas import (
    SCHEMA_VERSION,
    DataClass,
    FigureResult,
    NarrativeResult,
    STAGE_ORDER,
    Stage,
    StatResult,
    TableResult,
    VerdictResult,
)

from .grounding import numerals
from .prompts import panel_label
from .schemas import (
    AIFigureBrief,
    AIRuleBrief,
    AIResultBrief,
    AnalysisDigest,
    PRIVACY_NOTE,
)

MAX_RESULTS = 6
MAX_FACTS = 40
MAX_NUMBERS = 160
MAX_BRIEF_NUMBERS = 8
MAX_RESULT_IDS = 30
MAX_RULES = 3
MAX_FIGURES = 3
MAX_WARNINGS = 4
MAX_CAVEATS = 6
SAFE_FACT_KEYS = ("eda_facts", "prep_facts", "gate", "next_advice")
BANNED_FACT_WORDS = (
    "key", "secret", "token", "password", "raw", "values", "series",
    "dataframe", "frame", "fit", "model", "object", "preview_rows",
)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clean_key(key: Any) -> str:
    return str(key).strip().replace(" ", "_")[:60]


def _label_key(value: Any, fallback: str) -> str:
    label = _clean_text(value, 40)
    key = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return key or fallback


def _clean_text(value: Any, limit: int = 180) -> str:
    return " ".join(str(value or "").split())[:limit]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")  # type: ignore[no-any-return]
        except Exception:
            pass
    if hasattr(value, "as_dict"):
        try:
            return dict(value.as_dict())  # type: ignore[no-any-return]
        except Exception:
            pass
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        try:
            return asdict(value)
        except Exception:
            pass
    return {k: v for k, v in vars(value).items() if not k.startswith("_")} if hasattr(value, "__dict__") else {}


def _stage_code(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "S15")


def _primitive(value: Any) -> float | int | str | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    f = _finite_float(value)
    if f is not None:
        return f if not float(f).is_integer() else int(f)
    if isinstance(value, str):
        return _clean_text(value, 100)
    return None


def _flatten_facts(obj: Any, prefix: str, out: dict[str, float | int | str | bool], depth: int = 0) -> None:
    if len(out) >= MAX_FACTS or depth > 2:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = _clean_key(key)
            if any(word in key_s.lower() for word in BANNED_FACT_WORDS):
                continue
            path = f"{prefix}.{key_s}" if prefix else key_s
            p = _primitive(value)
            if p is not None:
                out[path] = p
            elif isinstance(value, dict):
                _flatten_facts(value, path, out, depth + 1)
            if len(out) >= MAX_FACTS:
                return


def _numbers_from_text(text: str, numbers: dict[str, float], prefix: str) -> None:
    for i, token in enumerate(numerals(text)):
        if len(numbers) >= MAX_NUMBERS:
            return
        f = _finite_float(token)
        if f is not None:
            numbers[f"{prefix}.text_{i}"] = f


def _add_number(numbers: dict[str, float], key: str, value: Any) -> None:
    if len(numbers) >= MAX_NUMBERS:
        return
    f = _finite_float(value)
    if f is not None:
        numbers[_clean_key(key)] = f


def _series_stats(session: Any) -> dict[str, float | int]:
    series = session.objects.get("series")
    out: dict[str, float | int] = {}
    if series is None:
        return out
    try:
        arr = np.asarray(series.to_numpy(dtype=float), dtype=float)
        finite = arr[np.isfinite(arr)]
        n_missing = int(np.count_nonzero(~np.isfinite(arr)))
    except Exception:
        return out
    out["n_missing"] = n_missing
    if finite.size:
        out["series_min"] = float(np.min(finite))
        out["series_max"] = float(np.max(finite))
        out["series_mean"] = float(np.mean(finite))
        out["series_std"] = float(np.std(finite, ddof=0))
    return out


def _series_values(session: Any, max_values: int) -> tuple[list[float], bool]:
    series = session.objects.get("series")
    if series is None:
        return [], False
    try:
        arr = np.asarray(series.to_numpy(dtype=float), dtype=float)
        finite = arr[np.isfinite(arr)]
    except Exception:
        return [], False
    values = [float(v) for v in finite[:max_values]]
    return values, int(finite.size) > len(values)


def _trace_names(fig: dict[str, Any], limit: int = 8) -> list[str]:
    names: list[str] = []
    for trace in (fig.get("data") or [])[:12]:
        if not isinstance(trace, dict):
            continue
        name = trace.get("name") or trace.get("type") or "trace"
        names.append(_clean_text(name, 60))
        if len(names) >= limit:
            break
    return names


def _trace_point_count(fig: dict[str, Any]) -> int:
    total = 0
    for trace in (fig.get("data") or [])[:20]:
        if not isinstance(trace, dict):
            continue
        for key in ("x", "y", "z", "values", "open", "high", "low", "close"):
            v = trace.get(key)
            if isinstance(v, (list, tuple, np.ndarray)):
                total = max(total, len(v))
    return total


def _axis_title(layout: dict[str, Any], axis: str) -> str:
    spec = layout.get(axis) or {}
    title = spec.get("title") if isinstance(spec, dict) else None
    if isinstance(title, dict):
        return _clean_text(title.get("text"), 80)
    return _clean_text(title, 80)


def _sample_trace_values(fig: dict[str, Any], limit: int) -> list[float]:
    out: list[float] = []
    for trace in (fig.get("data") or [])[:20]:
        if not isinstance(trace, dict):
            continue
        for key in ("y", "x", "values"):
            v = trace.get(key)
            if not isinstance(v, (list, tuple, np.ndarray)):
                continue
            for item in v[: max(0, limit - len(out))]:
                f = _finite_float(item)
                if f is not None:
                    out.append(f)
                if len(out) >= limit:
                    return out
    return out


def _figure_brief(
    result: FigureResult,
    *,
    include_values: bool,
    max_values: int,
) -> AIFigureBrief:
    fig = result.figure_json or {}
    layout = fig.get("layout") if isinstance(fig.get("layout"), dict) else {}
    layout = layout or {}
    behind = result.data_behind
    values = _sample_trace_values(fig, min(8, max_values)) if include_values else []
    return AIFigureBrief(
        result_id=result.id,
        title=_clean_text(result.title, 120),
        caption=_clean_text(result.caption, 140),
        alt_text=_clean_text(result.alt_text, 160),
        n_traces=len(fig.get("data") or []),
        n_points=_trace_point_count(fig),
        trace_names=_trace_names(fig),
        x_title=_axis_title(layout, "xaxis"),
        y_title=_axis_title(layout, "yaxis"),
        sample_values=values,
        values_truncated=bool(values) and include_values,
    )


def _technical_label(result: Any) -> str:
    raw = getattr(result, "technical", "") or getattr(result, "method", "")
    return _clean_text(raw, 100)


def _result_brief(
    result: Any,
    *,
    include_values: bool,
    max_values: int,
) -> tuple[AIResultBrief, AIFigureBrief | None]:
    numbers: dict[str, float] = {}
    figure: AIFigureBrief | None = None
    caption = _clean_text(getattr(result, "caption", ""), 140)
    alt_text = _clean_text(getattr(result, "alt_text", ""), 160)
    behind_rows = 0

    if isinstance(result, StatResult):
        for key in ("value", "p_value", "df", "effect_size", "ci_low", "ci_high"):
            _add_number(numbers, f"{result.id}.{key}", getattr(result, key, None))
        _numbers_from_text(result.interpretation or "", numbers, result.id)
    elif isinstance(result, TableResult):
        behind_rows = len(result.rows)
        n_added = 0
        for row_i, row in enumerate(result.rows[:6]):
            row_label = _label_key(
                row.get("quantity") or row.get("label") or row.get("name"),
                f"r{row_i}",
            )
            for col in result.columns[:8]:
                if col.dtype in {"number", "integer"} and n_added < MAX_BRIEF_NUMBERS:
                    f = _finite_float(row.get(col.key))
                    if f is not None:
                        key = f"{result.id}.{row_label}.{col.key}"
                        if key in numbers:
                            key = f"{result.id}.r{row_i}.{row_label}.{col.key}"
                        numbers[key] = f
                        n_added += 1
        _numbers_from_text(result.footnote or "", numbers, result.id)
    elif isinstance(result, FigureResult):
        figure = _figure_brief(result, include_values=include_values, max_values=max_values)
        behind_rows = len(result.data_behind.rows) if result.data_behind else 0
        for key, value in enumerate(figure.sample_values):
            _add_number(numbers, f"{result.id}.sample_{key}", value)
        _add_number(numbers, f"{result.id}.n_points", figure.n_points)
        _add_number(numbers, f"{result.id}.n_traces", figure.n_traces)
    elif isinstance(result, NarrativeResult):
        for key, value in result.numbers_cited.items():
            _add_number(numbers, f"{result.id}.{key}", value)
        for i, para in enumerate(result.paragraphs[:4]):
            _numbers_from_text(para, numbers, f"{result.id}.p{i}")
    elif isinstance(result, VerdictResult):
        for i, item in enumerate(result.checklist[:8]):
            f = _finite_float(item.value)
            if f is not None:
                numbers[f"{result.id}.check_{i}"] = f
        _numbers_from_text(result.headline, numbers, result.id)

    for field in ("title", "plain", "why", "technical", "method"):
        _numbers_from_text(_clean_text(getattr(result, field, ""), 180), numbers, result.id)
    for i, cav in enumerate(result.caveats[:4]):
        _numbers_from_text(_clean_text(cav, 140), numbers, f"{result.id}.cav{i}")
    for i, warn in enumerate(result.warnings[:4]):
        _numbers_from_text(_clean_text(warn, 140), numbers, f"{result.id}.warn{i}")
    _numbers_from_text(caption, numbers, f"{result.id}.caption")
    _numbers_from_text(alt_text, numbers, f"{result.id}.alt")
    numbers = dict(list(numbers.items())[:MAX_BRIEF_NUMBERS])

    brief = AIResultBrief(
        id=result.id,
        title=_clean_text(result.title, 120),
        stage=_stage_code(result.stage),
        kind=result.kind,
        plain=_clean_text(result.plain, 180),
        why=_clean_text(result.why, 180),
        technical_label=_technical_label(result),
        numbers=numbers,
        caveats=[_clean_text(c, 120) for c in result.caveats[:3]],
        warnings=[_clean_text(w, 120) for w in result.warnings[:3]],
        alt_text=alt_text,
        caption=caption,
        data_behind_rows=behind_rows,
        figure=figure,
    )
    return brief, figure


def _rule_brief(rule: dict[str, Any], stage: str, rank: int | None, kind: str) -> AIRuleBrief | None:
    rid = _clean_text(rule.get("id") or rule.get("rule_id") or "", 80)
    recommend = _clean_text(rule.get("recommend") or rule.get("label") or "", 160)
    if not rid or not recommend:
        return None
    return AIRuleBrief(
        id=rid,
        stage=_stage_code(rule.get("stage") or stage),
        recommend=recommend,
        because=_clean_text(rule.get("because") or rule.get("risk_if_ignored") or "", 180),
        how_to=_clean_text(rule.get("how_to") or "", 140),
        provenance=[_clean_text(p, 80) for p in (rule.get("provenance") or [])[:4]],
        confidence=_clean_text(rule.get("confidence") or "", 30),
        kind=_clean_text(kind, 30),
        rank=rank,
    )


def _advisor_rules(session: Any) -> list[AIRuleBrief]:
    out: list[AIRuleBrief] = []
    seen: set[str] = set()

    def add(rule: dict[str, Any], stage: str, rank: int | None, kind: str) -> None:
        brief = _rule_brief(rule, stage, rank, kind)
        if brief is not None and brief.id not in seen:
            seen.add(brief.id)
            out.append(brief)

    nxt = session.objects.get("next_advice") or {}
    ranked = nxt.get("ranked") if isinstance(nxt, dict) else []
    for i, rule in enumerate(ranked or []):
        if isinstance(rule, dict):
            add(rule, "S7", i, _clean_text(rule.get("kind") or "recommendation", 40))

    for stage, report in (session.advice or {}).items():
        rep = _as_dict(report)
        code = _stage_code(stage)
        for i, rule in enumerate((rep.get("recommendations") or [])[:3]):
            if isinstance(rule, dict):
                add(rule, code, i, "recommendation")
        for i, rule in enumerate((rep.get("blockers") or [])[:2]):
            if isinstance(rule, dict):
                add(rule, code, i, "blocker")
    return out[:MAX_RULES]


def _collect_facts(session: Any) -> dict[str, float | int | str | bool]:
    facts: dict[str, float | int | str | bool] = {}
    for key in SAFE_FACT_KEYS:
        obj = session.objects.get(key)
        if isinstance(obj, dict) and isinstance(obj.get("facts"), dict):
            _flatten_facts(obj["facts"], key, facts)
        elif isinstance(obj, dict):
            _flatten_facts(obj, key, facts)
        if len(facts) >= MAX_FACTS:
            break
    for key in ("forecast_horizon", "decomp_trend_spread"):
        p = _primitive(session.objects.get(key))
        if p is not None:
            facts[key] = p
    for stage, report in (session.advice or {}).items():
        rep = _as_dict(report)
        facts_src = rep.get("facts_used")
        if isinstance(facts_src, dict):
            _flatten_facts(facts_src, f"advice.{_stage_code(stage)}", facts)
        if len(facts) >= MAX_FACTS:
            break
    quality = session.state.dataset.quality if session.state.dataset else None
    if quality is not None:
        q = _as_dict(quality)
        for key in (
            "n_rows_raw", "n_rows_used", "n_gaps", "n_missing_total", "n_zeros",
            "pct_zeros", "n_negatives", "n_constant_runs", "n_duplicate_timestamps",
            "n_outlier_candidates", "n_expected_observations",
        ):
            p = _primitive(q.get(key))
            if p is not None:
                facts[f"quality.{key}"] = p
        evidence = q.get("data_class_evidence")
        if isinstance(evidence, dict):
            _flatten_facts(evidence, "quality.data_class", facts)
    return dict(list(facts.items())[:MAX_FACTS])


def _stage_verdicts(session: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for stage in STAGE_ORDER:
        st = session.state.stages.get(stage.value)
        if st is None:
            continue
        out[stage.value] = _clean_text(st.status, 30)
    for result in session.all_results():
        if isinstance(result, VerdictResult):
            out[f"{_stage_code(result.stage)}.verdict"] = _clean_text(
                getattr(result.verdict, "value", result.verdict), 60
            )
    return out


def _ordered_results(session: Any, focus_result_id: str | None, prefer_kind: str | None) -> list[Any]:
    all_results = list(session.all_results())
    focus = session.get_result(focus_result_id) if focus_result_id else None
    rank = {stage.value: i for i, stage in enumerate(STAGE_ORDER)}
    rest = [r for r in all_results if r is not focus]
    rest.sort(key=lambda r: (-rank.get(_stage_code(r.stage), -1), r.id))
    if prefer_kind:
        preferred = [r for r in rest if r.kind == prefer_kind]
        others = [r for r in rest if r.kind != prefer_kind]
        rest = preferred + others
    return ([focus] if focus is not None else []) + rest


def build_digest(
    session: Any,
    *,
    panel_id: str = "S15",
    focus_result_id: str | None = None,
    include_values: bool = False,
    eli15: bool = False,
    prefer_kind: str | None = None,
    max_results: int = MAX_RESULTS,
    max_values: int = 40,
) -> AnalysisDigest:
    """Build the only statistical payload the assistant is allowed to see."""
    dataset = session.state.dataset
    try:
        current_stage = Stage(panel_id)
    except ValueError:
        current_stage = session.state.current_stage
    if dataset is not None:
        try:
            data_class = DataClass(dataset.data_class)
        except ValueError:
            data_class = DataClass.CONTINUOUS
        n_observations = int(dataset.n_observations or 0)
        frequency_label = _clean_text(dataset.frequency.label, 100)
        period = dataset.frequency.period
        time_start = _clean_text(dataset.time_start, 100)
        time_end = _clean_text(dataset.time_end, 100)
        series_name = _clean_text(dataset.name or dataset.target_column, 100)
    else:
        data_class = DataClass.CONTINUOUS
        n_observations = 0
        frequency_label = "Unknown"
        period = None
        time_start = ""
        time_end = ""
        series_name = "No data loaded"

    facts = _collect_facts(session)
    numbers: dict[str, float] = {
        "zero": 0.0,
        "one": 1.0,
        "confidence_level_percent": 95.0,
        "conventional_alpha": 0.05,
        "n_observations": float(n_observations),
    }
    if period is not None:
        numbers["seasonal_period"] = float(period)
    for key, value in _series_stats(session).items():
        p = _primitive(value)
        if p is not None:
            facts[f"series.{key}"] = p
        _add_number(numbers, f"series.{key}", value)
    if eli15:
        facts["eli15_target_age"] = 15
        numbers["eli15_target_age"] = 15.0
    facts = dict(list(facts.items())[:MAX_FACTS])

    # Explicit result numbers come first: they are the numbers the assistant is
    # most likely to quote, and they must survive the compactness cap.
    briefs: list[AIResultBrief] = []
    figure_payloads: list[AIFigureBrief] = []
    results = _ordered_results(session, focus_result_id, prefer_kind)
    for result in results[:max_results]:
        brief, figure = _result_brief(
            result, include_values=include_values, max_values=max_values
        )
        briefs.append(brief)
        if figure is not None and len(figure_payloads) < MAX_FIGURES:
            figure_payloads.append(figure)
        for key, value in brief.numbers.items():
            _add_number(numbers, key, value)

    all_ids = [r.id for r in results]
    warnings: list[str] = []
    caveats: list[str] = list(session.state.caveats or [])
    for result in results[:max_results]:
        caveats.extend(result.caveats)
        warnings.extend(result.warnings)

    # Numbers embedded in visible labels and figure metadata must be grounded
    # too; otherwise a harmless year range or trace suffix becomes a false flag.
    for j, figure in enumerate(figure_payloads):
        for i, text in enumerate((
            figure.title, figure.caption, figure.alt_text, figure.x_title,
            figure.y_title, *figure.trace_names,
        )):
            _numbers_from_text(_clean_text(text, 180), numbers, f"fig{j}.t{i}")
        for i, value in enumerate(figure.sample_values):
            _add_number(numbers, f"{figure.result_id}.sample_{i}", value)
    for i, text in enumerate((series_name, frequency_label, time_start, time_end)):
        _numbers_from_text(_clean_text(text, 100), numbers, f"series.t{i}")

    data_values: list[float] | None = None
    values_truncated = False
    if include_values:
        data_values, values_truncated = _series_values(session, max_values)
        for i, value in enumerate(data_values):
            _add_number(numbers, f"data_value.{i}", value)

    for key, value in facts.items():
        _add_number(numbers, f"fact.{key}", value)
        if isinstance(value, str):
            _numbers_from_text(value, numbers, f"fact.{key}")
    verdicts = _stage_verdicts(session)
    for key, value in verdicts.items():
        _numbers_from_text(value, numbers, f"stage.{key}")
    for i, text in enumerate((*warnings, *caveats)):
        _numbers_from_text(_clean_text(text, 140), numbers, f"note.{i}")

    rules = _advisor_rules(session)
    for i, rule in enumerate(rules):
        for j, text in enumerate((
            rule.recommend, rule.because, rule.how_to, rule.confidence,
            *rule.provenance,
        )):
            _numbers_from_text(_clean_text(text, 180), numbers, f"rule{i}.t{j}")
        if rule.rank is not None:
            _add_number(numbers, f"rule{i}.rank", rule.rank)

    return AnalysisDigest(
        session_id=session.id,
        schema_version=SCHEMA_VERSION,
        n_observations=n_observations,
        frequency_label=frequency_label,
        period=period,
        time_start=time_start,
        time_end=time_end,
        data_class=data_class,
        series_name=series_name,
        current_stage=current_stage,
        stage_verdicts=verdicts,
        advisor_reports=[],
        top_results=briefs,
        numbers=dict(list(numbers.items())[:MAX_NUMBERS]),
        warnings=[_clean_text(w, 140) for w in dict.fromkeys(warnings)][:MAX_WARNINGS],
        caveats=[_clean_text(c, 140) for c in dict.fromkeys(caveats)][:MAX_CAVEATS],
        panel_context=panel_label(panel_id),
        data_values_included=bool(data_values),
        data_values=data_values,
        generated_at=datetime.now(timezone.utc).isoformat(),
        panel_id=panel_id,
        facts_used=facts,
        result_ids=all_ids[:MAX_RESULT_IDS],
        advisor_rules=rules,
        figure_payloads=figure_payloads,
        privacy_note=PRIVACY_NOTE,
        values_truncated=values_truncated,
    )


__all__ = [
    "build_digest", "MAX_RESULTS", "MAX_FACTS", "MAX_NUMBERS",
    "MAX_BRIEF_NUMBERS", "MAX_RESULT_IDS", "MAX_RULES", "MAX_FIGURES",
    "MAX_WARNINGS", "MAX_CAVEATS",
]
