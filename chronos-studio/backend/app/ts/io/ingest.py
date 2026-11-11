"""
ts/io/ingest.py — the S0 pipeline stage: raw bytes → SessionData.

This is the single orchestrator for ingestion. The API layer calls
`ingest_table()` and gets back a dataset plus the session-side series object;
it never parses anything itself.

Column mapping is *assisted but overridable*: we guess time/target/exogenous,
and every guess carries a confidence plus the evidence, so the UI can show
"we think it's column B because 100% of its values parse as dates".
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.core.errors import (
    ColumnMappingError,
    EmptyDatasetError,
    InsufficientDataError,
    NonNumericSeriesError,
    TimeIndexError,
)
from app.core.schemas import (
    Confidence,
    DataClass,
    DataQualityReport,
    FrequencySpec,
    ParseIssue,
    SessionData,
)
from app.ts.io.readers import RawTable, read_any, read_pasted_text
from app.ts.io.timeindex import (
    CALENDAR_SPECS,
    _SPEC_BY_CODE,
    build_from_year_month,
    detect_time_column,
    detect_year_month_columns,
    infer_frequency,
    parse_time_column,
    regularise,
)
from app.ts.io.validate import (
    DataClassVerdict,
    build_quality_report,
    coerce_numeric,
    detect_data_class,
    profile_columns,
)

MIN_OBSERVATIONS = 4


@dataclass
class ColumnMapping:
    """What the user (or the assistant) decided about the columns."""

    time_column: str | None = None
    target_column: str | None = None
    exogenous_columns: list[str] = field(default_factory=list)
    frequency_code: str | None = None
    row_order_is_time: bool = False
    start_date: str | None = None     # for row_order_is_time
    dayfirst: bool | None = None
    year_column: str | None = None
    month_column: str | None = None
    agg_duplicate: Literal["mean", "first", "sum", "last"] = "mean"
    #: Kept so an event-mode result can name the source file. Optional everywhere else.
    name: str | None = None
    source: Literal["upload", "example", "url", "paste"] | None = None
    #: The file lists events rather than values: one row per occurrence, no value column. Counts per
    #: period are derived from the timestamps. Set automatically for a single date column, and
    #: overridable so a user looking at a one-column file is never stuck.
    event_mode: bool = False
    #: An explicit data-class override from "Change mapping". Three places in the app already told
    #: the user "you can switch the data class in Column mapping", but no such field existed, so
    #: the count branch was unreachable for a whole-number series that Part 1 called continuous.
    #: Empty/None means "keep the detected verdict".
    data_class: str | None = None


@dataclass
class IngestResult:
    """Everything S0 produces. `series` never leaves the server."""

    dataset: SessionData
    series: pd.Series                      # regular DatetimeIndex, float, NaNs for gaps
    exogenous: pd.DataFrame | None
    raw: RawTable
    mapping: ColumnMapping
    issues: list[ParseIssue]
    profiles: list[Any]
    candidate_frequencies: list[FrequencySpec]


# --------------------------------------------------------------------------- #
# Column mapping assistant
# --------------------------------------------------------------------------- #
def suggest_mapping(
    df: pd.DataFrame, dayfirst: bool | None = None
) -> tuple[ColumnMapping, list[ParseIssue], list[str]]:
    """
    Propose time / target / exogenous columns and explain the reasoning.
    A beginner should be able to press "accept" and be right most of the time.
    """
    notes: list[str] = []
    issues: list[ParseIssue] = []

    # 1. Explicit year + month/quarter columns beat everything else.
    ym = detect_year_month_columns(df)
    if ym is not None:
        year_col, period_col = ym
        mapping = ColumnMapping(
            year_column=year_col, month_column=period_col,
            dayfirst=dayfirst,
            target_column=_first_numeric_column(df, exclude={year_col, period_col}),
        )
        notes.append(
            f"Found separate '{year_col}'"
            + (f" and '{period_col}'" if period_col else "")
            + " columns, which is the most reliable way to build a monthly/quarterly index."
        )
        mapping.exogenous_columns = _remaining_numeric(df, {
            year_col, period_col, mapping.target_column
        })
        return mapping, issues, notes

    # 2. A parseable time column.
    time_col, score = detect_time_column(df)
    if time_col and score >= 0.5:
        remaining = [c for c in df.columns if c != time_col]
        target = _first_numeric_column(df, exclude={time_col})
        mapping = ColumnMapping(
            time_column=str(time_col), target_column=target,
            dayfirst=dayfirst,
        )
        mapping.exogenous_columns = _remaining_numeric(df, {time_col, target})
        notes.append(
            f"Column '{time_col}' looks like the time index "
            f"(confidence {score:.0%}: the values parse as dates and follow an order)."
        )
        if target is None and len(remaining) == 0:
            # A single date column with nothing else in it is a list of *events*, not a series of
            # values. Without this the file fails mapping with "no numeric column", which is a dead end
            # for the one input shape the point-process branch exists for.
            mapping.event_mode = True
            notes.append(
                "There is no numeric column, so each row is treated as one event: the file is read as "
                "event timestamps and the app counts events per period itself."
            )
        return mapping, issues, notes

    # 3. No time column: assume the rows are evenly spaced observations.
    target = _first_numeric_column(df, exclude=set())
    if target is None:
        raise NonNumericSeriesError(
            plain="None of the columns contain numbers, so there is nothing to analyse.",
            fix="Upload a file with a numeric column, or past the table again with the header row.",
        )
    mapping = ColumnMapping(
        target_column=target, row_order_is_time=True, start_date=None, dayfirst=dayfirst,
    )
    mapping.exogenous_columns = _remaining_numeric(df, {target})
    issues.append(ParseIssue(
        code="data.no_time_column", severity="warning",
        plain=(
            "No date column was found, so the rows were treated as equally spaced "
            "observations starting from row 1."
        ),
        fix=(
            "If the rows are monthly (say), set the frequency and a start date in "
            "Column mapping — otherwise the axis labels will be row numbers."
        ),
    ))
    return mapping, issues, notes


def _first_numeric_column(df: pd.DataFrame, exclude: set[str]) -> str | None:
    for col in df.columns:
        if col in exclude:
            continue
        try:
            numeric, _ = coerce_numeric(df[col], str(col))
        except NonNumericSeriesError:
            continue
        if numeric.notna().sum() >= max(2, 0.6 * df[col].notna().sum()):
            return str(col)
    return None


def _remaining_numeric(df: pd.DataFrame, exclude: set[str | None]) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        try:
            numeric, _ = coerce_numeric(df[col], str(col))
        except NonNumericSeriesError:
            continue
        if numeric.notna().sum() > 0:
            out.append(str(col))
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def ingest_table(
    table: RawTable,
    mapping: ColumnMapping | None = None,
    name: str | None = None,
    source: Literal["upload", "example", "url", "paste"] = "upload",
) -> IngestResult:
    """
    Run S0. With no `mapping`, the assistant proposes one; with a mapping, the
    user's choices win (and are validated).
    """
    df = table.df.copy()
    if df.empty:
        raise EmptyDatasetError()

    issues: list[ParseIssue] = []
    if mapping is None:
        mapping, guess_issues, notes = suggest_mapping(df)
        issues.extend(guess_issues)
    else:
        notes = []

    # ---- time index ------------------------------------------------------- #
    row_index, frequency, candidates, index_notes = _build_index(df, mapping)
    notes.extend(index_notes)

    # ---- target values ---------------------------------------------------- #
    # A file of nothing but dates is not a file with a missing value column — it is event data, one
    # row per event. The spec's `event-timestamp` branch exists for exactly this input, and refusing it
    # as a mapping error would mean the point-process models in Part 10 could never be reached from a
    # real upload. It is handled here, before the numeric coercion, by counting events per period.
    if mapping.event_mode or (not mapping.target_column or mapping.target_column not in df.columns):
        only_time = [c for c in df.columns if c == mapping.time_column]
        if mapping.event_mode or (len(df.columns) == 1 and only_time):
            if not only_time:
                time_col, score = detect_time_column(df)
                if not time_col or score < 0.5:
                    raise ColumnMappingError(
                        plain="The value column you chose is not in the file.",
                        fix="Re-open Column mapping and pick the column holding the numbers.",
                    )
                mapping.time_column = str(time_col)
            return _ingest_events(
                df, mapping, frequency=None, row_index=row_index, issues=issues,
                notes=notes, table=table,
            )
        if not mapping.target_column or mapping.target_column not in df.columns:
            raise ColumnMappingError(
                plain="The value column you chose is not in the file.",
                fix="Re-open Column mapping and pick the column holding the numbers.",
            )
    numeric, coerce_issues = coerce_numeric(df[mapping.target_column], mapping.target_column)
    issues.extend(coerce_issues)
    numeric.index = row_index

    unreadable = int(numeric.index.isna().sum())
    if unreadable:
        notes.append(f"{unreadable} row(s) with unreadable dates were dropped.")
        numeric = numeric[~numeric.index.isna()]


    # ---- regularise ------------------------------------------------------- #
    reg = regularise(numeric, frequency, dedupe=mapping.agg_duplicate)
    issues.extend(ParseIssue(
        code="data.note", severity="info", plain=n,
    ) for n in reg.notes if n)

    series = reg.series if reg.series is not None else numeric
    if series is None or series.dropna().empty:
        raise EmptyDatasetError(
            plain="The chosen value column is empty after cleaning.",
            fix="Check that the column contains numbers, not just the header.",
        )
    if int(series.notna().sum()) < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            plain=f"Only {int(series.notna().sum())} usable observations were found.",
            fix="Upload a longer series — at least 10 points, ideally two full seasonal cycles.",
        )

    # ---- exogenous -------------------------------------------------------- #
    exog: pd.DataFrame | None = None
    exog_cols = [c for c in mapping.exogenous_columns if c in df.columns]
    if exog_cols:
        exog = pd.DataFrame(index=reg.index)
        for c in exog_cols:
            try:
                col, _ = coerce_numeric(df[c], c)
                col.index = row_index
                col = col[~col.index.isna()]
                exog[c] = regularise(col, frequency, dedupe=mapping.agg_duplicate).series
            except Exception:  # pragma: no cover - a bad extra column must not kill S0
                issues.append(ParseIssue(
                    code="data.exog_skipped", severity="warning",
                    plain=f"The column '{c}' could not be aligned to the time index and was ignored.",
                    fix="Remove it from the file, or fix its values.",
                ))
        if exog.shape[1] == 0:
            exog = None

    # ---- data class ------------------------------------------------------- #
    verdict = detect_data_class(series)
    requested_class = (mapping.data_class or "").strip().lower()
    if requested_class:
        try:
            chosen = DataClass(requested_class)
        except ValueError:
            issues.append(ParseIssue(
                code="data.class_override_invalid", severity="warning",
                plain=(f"'{mapping.data_class}' is not a data class this app knows, so the "
                       f"detected class ({verdict.data_class.value}) was kept."),
                fix="Choose continuous, count, intermittent, binary or proportion, or event timestamp.",
            ))
        else:
            verdict = DataClassVerdict(
                chosen,
                {**verdict.evidence, "detected_class": verdict.data_class.value,
                 "user_override": True},
                (f"Data class set to {chosen.value.replace('_', ' ')} in Column mapping "
                 f"(detected: {verdict.data_class.value.replace('_', ' ')})."),
            )

    # ---- quality report --------------------------------------------------- #
    report = build_quality_report(
        raw_table_rows=len(df),
        prepared=series,
        df_all=df,
        time_column=mapping.time_column or (mapping.year_column or "row"),
        target_column=mapping.target_column,
        exogenous_columns=exog_cols,
        frequency=reg.frequency,
        data_class=verdict.data_class,
        data_class_evidence=verdict.evidence,
        issues=[*issues, *table_notes_as_issues(table)],
        n_expected_observations=len(reg.index),
        dropped_rows=table.dropped_blank_rows,
        dropped_reason="blank rows" if table.dropped_blank_rows else "",
    )
    report.n_duplicate_timestamps = reg.n_duplicates
    if reg.n_duplicates:
        report.issues.append(ParseIssue(
            code="data.duplicates", severity="warning",
            plain=f"{reg.n_duplicates} duplicate timestamp(s) were combined using the {mapping.agg_duplicate}.",
            fix="Check your source file — duplicates usually mean two records for one period.",
        ))
    if reg.n_reindexed:
        report.issues.append(ParseIssue(
            code="data.gaps_inserted", severity="warning",
            plain=(
                f"{reg.n_reindexed} observation(s) were missing from the calendar and were "
                "inserted as blanks, so the seasonal pattern stays aligned."
            ),
            fix="Decide in Step 2 whether to impute them or leave them out.",
        ))

    dataset = SessionData(
        dataset_id=uuid.uuid4().hex[:12],
        name=name or table.source,
        source=source,
        time_column=report.time_column or "time",
        target_column=mapping.target_column,
        exogenous_columns=exog_cols,
        frequency=reg.frequency,
        n_observations=int(series.notna().sum()),
        time_start=str(series.index.min()),
        time_end=str(series.index.max()),
        data_class=verdict.data_class,
        quality=report,
        content_hash=SessionData.hash_values(
            [None if pd.isna(v) else float(v) for v in series.to_numpy()]
        ),
        preview_rows=_preview(series, exog),
    )

    if notes:
        dataset.quality.issues.append(ParseIssue(
            code="data.mapping_note", severity="info", plain=" ".join(notes)
        ))
    dataset.quality.plain_summary = (
        f"{dataset.quality.plain_summary} {verdict.plain}"
    )

    return IngestResult(
        dataset=dataset,
        series=series,
        exogenous=exog,
        raw=table,
        mapping=mapping,
        issues=dataset.quality.issues,
        profiles=profile_columns(df, report.time_column),
        candidate_frequencies=candidates or [reg.frequency],
    )


def _ingest_events(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    frequency,
    row_index,
    issues: list[ParseIssue],
    notes: list[str],
    table=None,
) -> IngestResult:
    """
    Event-timestamp ingestion: one row per event, counted into periods.

    Two series come out and they answer different questions:

    * `series` — **counts per period**, on a regular calendar index. This is what lets the S9/S10 count
      branch, the gate and the plots operate on event data without knowing it came from timestamps.
    * the timestamps themselves, kept in `session.objects["event_times"]` by the API layer, because the
      renewal and Hawkes models need the *intervals*, not the counts.

    Deriving one from the other is deliberately one-way: intervals → counts. A count series cannot
    recover the within-period timing, and pretending otherwise would let a point-process model be
    fitted to data it cannot see.
    """
    time_col = mapping.time_column
    if not time_col or time_col not in df.columns:
        raise ColumnMappingError(
            plain="No date column was found, so there is no event timing to work from.",
            fix="Re-open Column mapping and pick the column holding the dates.",
        )
    parsed = pd.to_datetime(df[time_col], errors="coerce", dayfirst=bool(mapping.dayfirst))
    n_bad = int(parsed.isna().sum())
    if n_bad:
        notes.append(f"{n_bad} row(s) with unreadable dates were dropped.")
    parsed = parsed.dropna()
    if parsed.empty:
        raise EmptyDatasetError(
            plain="None of the values in the date column could be read as dates.",
            fix="Check the date format, or set day-first in Column mapping.",
        )
    # Events per period. A dataset of timestamps usually has several events in one period, which is
    # exactly the information the count branch needs; the calendar is inferred from the span.
    from app.ts.io.timeindex import event_calendar  # local import: avoids a cycle at module load

    # The calendar for counting events is chosen from the *rate*, not from the modal spacing of unique
    # dates: five events across two months are a daily count series with mostly zeros, and inferring a
    # quarterly grid from their spacing would collapse the whole file into one observation.
    spec = frequency if frequency is not None else event_calendar(pd.DatetimeIndex(parsed.to_numpy()))
    # Each row is one event, so the value is 1.0 and the timestamp is *allowed to repeat*: three rows
    # dated the same day are three events. Only the bucket is aggregated, never the raw rows.
    stamp = pd.Series(1.0, index=pd.DatetimeIndex(sorted(parsed.to_numpy())))
    code = spec.code if hasattr(spec, "code") else "MS"
    # A daily code on daily-stamped events would make the count series a list of ones; the events are
    # counted into the calendar the dates actually imply, and the count is what the rest of the app
    # treats as "the series".
    counts = stamp.resample(code).sum()
    # An interval with no events is a zero, not a missing value: for counts, absence is a measurement.
    counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq=code),
                            fill_value=0.0)
    counts = counts.fillna(0.0)
    notes.append(
        f"The file lists {len(parsed)} events with no value column, so events were counted into "
        f"{counts.size} periods of the inferred calendar."
    )
    report = build_quality_report(
        raw_table_rows=int(len(df)),
        prepared=counts.astype(float),
        df_all=df,
        time_column=str(time_col),
        target_column="events",
        exogenous_columns=[],
        frequency=spec,
        data_class=DataClass.EVENT_TIMESTAMP,
        data_class_evidence={
            "event_rows": float(len(parsed)),
            "periods": float(counts.size),
            "no_value_column": True,
        },
        issues=list(issues),
    )
    report.plain_summary = (
        f"{len(parsed)} events over {counts.size} periods. Each row is one occurrence, so the series "
        "shown is the number of events per period."
    )
    dataset = SessionData(
        dataset_id=uuid.uuid4().hex[:12],
        name=mapping.name or "event timestamps",
        source=mapping.source or "upload",
        time_column=str(time_col),
        target_column="events",
        exogenous_columns=[],
        frequency=spec,
        n_observations=int(counts.size),
        time_start=str(counts.index.min()),
        time_end=str(counts.index.max()),
        data_class=DataClass.EVENT_TIMESTAMP,
        quality=report,
        content_hash=SessionData.hash_values([float(v) for v in counts.to_numpy()]),
        # The wire form is {time, value} for every dataset, event or not: the preview table reads
        # `value`, so a bespoke "events" key rendered as a column of dashes.
        preview_rows=_preview(counts.astype(float), None),
    )
    return IngestResult(
        dataset=dataset,
        series=counts.astype(float),
        exogenous=None,
        raw=table,
        mapping=mapping,
        issues=report.issues,
        profiles={},
        candidate_frequencies=[spec],
    )


def ingest_bytes(
    data: bytes,
    filename: str = "upload.csv",
    mapping: ColumnMapping | None = None,
    sheet_name: str | int | None = None,
    name: str | None = None,
    source: Literal["upload", "example", "url", "paste"] = "upload",
) -> IngestResult:
    table = read_any(data, filename=filename, sheet_name=sheet_name)
    return ingest_table(table, mapping=mapping, name=name or filename, source=source)


def ingest_text(text: str, mapping: ColumnMapping | None = None) -> IngestResult:
    table = read_pasted_text(text)
    return ingest_table(table, mapping=mapping, name="pasted table", source="paste")


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _build_index(
    df: pd.DataFrame, mapping: ColumnMapping
) -> tuple[pd.DatetimeIndex, FrequencySpec, list[FrequencySpec], list[str]]:
    """
    Build one datetime per row (len == len(df), NaT where unreadable), plus the
    calendar. Keeping the row alignment explicit avoids the classic bug where
    dropped rows silently shift every value onto the wrong date.
    """
    notes: list[str] = []

    if mapping.row_order_is_time:
        n = len(df)
        spec = (
            _SPEC_BY_CODE[mapping.frequency_code]
            if mapping.frequency_code and mapping.frequency_code in _SPEC_BY_CODE
            else _SPEC_BY_CODE["MS"]
        )
        start = (
            pd.Timestamp(mapping.start_date) if mapping.start_date
            else pd.Timestamp("2000-01-01")
        )
        index = pd.date_range(start, periods=n, freq=spec.code)
        if not mapping.start_date:
            notes.append(
                "No start date was given, so the axis begins at 2000-01-01. "
                "Set the real start date in Column mapping for correct labels."
            )
        return index, spec, [spec], notes

    if mapping.year_column:
        if mapping.year_column not in df.columns:
            raise ColumnMappingError(
                plain=f"The year column '{mapping.year_column}' is not in this file.",
                fix="Pick the year column from the list of columns in this file.",
            )
        has_companion = bool(mapping.month_column and mapping.month_column in df.columns)
        months = df[mapping.month_column] if has_companion else None
        is_quarter = bool(
            has_companion
            and str(mapping.month_column).lower().startswith(("q", "quart"))
        )
        idx = build_from_year_month(
            df[mapping.year_column],
            quarters=months if is_quarter else None,
            months=None if is_quarter else months,
        )
        spec = _SPEC_BY_CODE["QS"] if is_quarter else (
            _SPEC_BY_CODE["MS"] if has_companion else _SPEC_BY_CODE["YS"]
        )
        notes.append(
            f"Built the time index from '{mapping.year_column}'"
            + (f" + '{mapping.month_column}'." if has_companion else ".")
        )
        return pd.DatetimeIndex(idx), spec, [spec], notes

    if mapping.time_column and mapping.time_column not in df.columns:
        raise ColumnMappingError(
            plain=f"The time column '{mapping.time_column}' is not in this file.",
            fix="Column mapping lists the columns this file actually has — pick one of those.",
            context={"requested": mapping.time_column,
                     "available": [str(c) for c in df.columns]},
        )
    if not mapping.time_column:
        raise TimeIndexError(
            plain="No time column was chosen and the rows could not be treated as evenly spaced.",
            fix="Pick the time column in Column mapping, or say the rows are equally spaced.",
        )

    idx, parse_notes, bad = parse_time_column(df[mapping.time_column], mapping.time_column)
    notes.extend(parse_notes)
    if bad >= len(df):
        raise TimeIndexError()

    if mapping.frequency_code and mapping.frequency_code in _SPEC_BY_CODE:
        spec = _SPEC_BY_CODE[mapping.frequency_code]
        notes.append(f"You set the frequency to {spec.label.lower()}.")
        return pd.DatetimeIndex(idx), spec, [spec], notes

    valid = pd.DatetimeIndex(idx.dropna())
    spec, candidates = infer_frequency(valid)
    return pd.DatetimeIndex(idx), spec, candidates, notes


def table_notes_as_issues(table: RawTable) -> list[ParseIssue]:
    return [ParseIssue(code="data.loader_note", severity="info", plain=n) for n in table.notes]


def _preview(series: pd.Series, exog: pd.DataFrame | None, n: int = 8) -> list[dict[str, Any]]:
    head = series.head(n).reset_index()
    head.columns = ["time", "value"]
    rows = head.to_dict(orient="records")
    if exog is not None:
        for i, row in enumerate(rows):
            for col in exog.columns:
                val = exog[col].iloc[i]
                row[col] = None if pd.isna(val) else float(val)
    for row in rows:
        row["time"] = str(row["time"])
        if row.get("value") is not None and pd.isna(row["value"]):
            row["value"] = None
    return rows


def available_calendars() -> list[dict[str, Any]]:
    """For the UI's frequency picker — the same list the inference uses."""
    return [
        {"code": s.code, "label": s.label, "period": s.period}
        for s in CALENDAR_SPECS
    ]


__all__ = [
    "ColumnMapping",
    "IngestResult",
    "ingest_bytes",
    "ingest_text",
    "ingest_table",
    "suggest_mapping",
    "available_calendars",
    "MIN_OBSERVATIONS",
]
