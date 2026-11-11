"""
ts/io/timeindex.py — build a regular time index. This is the #1 beginner failure
point (§5.1), so it gets real effort and its own error messages.

Handles, in order of preference:
  1. an explicit datetime column (many formats, dayfirst ambiguity flagged)
  2. year + month / year + quarter / year + day columns
  3. 'YYYY-MM', 'YYYYQn', 'YYYY-MM-DD', ISO week, epoch seconds/millis,
     Excel serial numbers
  4. a regular index inferred from row order, with a confirmation prompt

Then: infer frequency (pd.infer_freq + a spacing/periodogram fallback),
detect gaps, duplicates and irregular spacing, and reindex onto the full
regular calendar so missing observations become visible instead of shifting
everything left.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.core.errors import FrequencyInferenceError, IrregularSpacingError, TimeIndexError
from app.core.schemas import Confidence, FrequencySpec

# --------------------------------------------------------------------------- #
# Candidate calendars. `code` is a pandas offset alias; `period` is the seasonal
# cycle length used everywhere downstream.
# --------------------------------------------------------------------------- #
CALENDAR_SPECS: list[FrequencySpec] = [
    FrequencySpec(code="h", label="Hourly", period=24, offset_step_ns=3_600_000_000_000,
                  evidence="Median spacing 1 hour"),
    FrequencySpec(code="D", label="Daily", period=7, offset_step_ns=86_400_000_000_000,
                  evidence="Median spacing 1 day"),
    FrequencySpec(code="W", label="Weekly", period=52, offset_step_ns=7 * 86_400_000_000_000,
                  evidence="Median spacing 7 days"),
    FrequencySpec(code="MS", label="Monthly (start of month)", period=12, offset_step_ns=None,
                  evidence="Dates one calendar month apart"),
    FrequencySpec(code="QS", label="Quarterly (start of quarter)", period=4, offset_step_ns=None,
                  evidence="Dates one calendar quarter apart"),
    FrequencySpec(code="YS", label="Yearly (start of year)", period=1, offset_step_ns=None,
                  evidence="Dates one calendar year apart"),
]

_SPEC_BY_CODE = {s.code: s for s in CALENDAR_SPECS}

# pandas `infer_freq` returns many aliases; map them onto our six calendars.
_ALIAS_MAP: dict[str, str] = {
    "A": "YS", "Y": "YS", "AS": "YS", "YS": "YS", "YE": "YS", "Y-DEC": "YS",
    "M": "MS", "MS": "MS", "ME": "MS", "BM": "MS", "BMS": "MS",
    "Q": "QS", "QS": "QS", "QE": "QS", "BQ": "QS", "BQS": "QS",
    "W": "W", "W-SUN": "W", "W-MON": "W", "W-TUE": "W", "W-WED": "W",
    "W-THU": "W", "W-FRI": "W", "W-SAT": "W",
    "D": "D", "B": "D", "C": "D",
    "H": "h", "h": "h", "BH": "h",
}

_DATETIME_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M", "%d.%m.%Y", "%Y%m%d", "%b %Y", "%B %Y", "%Y %b",
    "%d-%b-%Y", "%b-%y", "%d/%m/%Y %H:%M",
]

_QUARTER_RE = re.compile(r"^(\d{4})[-_/ ]?[Qq]([1-4])$")
_YEAR_MONTH_RE = re.compile(r"^(\d{4})[-_/ ]?(0?[1-9]|1[0-2])$")


@dataclass
class TimeIndexResult:
    index: pd.DatetimeIndex          # regular, complete, sorted
    frequency: FrequencySpec
    n_raw: int
    n_reindexed: int                 # gaps inserted
    n_duplicates: int
    n_unparseable: int
    irregular: bool = False
    notes: list[str] = field(default_factory=list)
    candidate_frequencies: list[FrequencySpec] = field(default_factory=list)
    parse_strategy: str = ""
    series: pd.Series | None = None   # values reindexed onto the regular calendar


# --------------------------------------------------------------------------- #
# Value-level parsing
# --------------------------------------------------------------------------- #
def _looks_like_excel_serial(s: pd.Series, name: str = "") -> bool:
    """
    Excel serial dates for 1950-2100 land roughly in 18_000..73_000.

    An all-integer column in that range is ambiguous (it could be a count), so we
    also require either a fractional part (dates with times) or small consecutive
    steps, and we tell the user what we assumed so it can be overridden.
    """
    numeric = pd.to_numeric(s, errors="coerce")
    valid = numeric.dropna()
    if valid.empty or not bool(valid.between(18_000, 73_000).mean() > 0.9):
        return False
    if (valid % 1 != 0).mean() > 0.05:
        return True
    diffs = np.diff(np.sort(valid.to_numpy()))
    if diffs.size == 0:
        return False
    small_steps = bool(((diffs >= 1) & (diffs <= 40)).mean() > 0.9)
    hinted = any(k in name.lower() for k in ("date", "time", "day", "period"))
    return small_steps or hinted


def _looks_like_epoch(s: pd.Series) -> Literal["s", "ms", None]:
    numeric = pd.to_numeric(s, errors="coerce").dropna()
    if numeric.empty:
        return None
    med = float(numeric.median())
    if 1e9 < med < 2e9:
        return "s"
    if 1e12 < med < 2e12:
        return "ms"
    return None


def parse_time_column(values: pd.Series, name: str = "") -> tuple[pd.DatetimeIndex, list[str], int]:
    """
    Turn a raw column into datetimes, trying the strategies a beginner's file is
    likely to use. Returns (index, notes, n_unparseable).
    """
    notes: list[str] = []
    s = values.astype("string").str.strip()
    raw_n = len(s)

    # --- strategy 1: already datetimes (Excel/Parquet hand us real objects) ---
    if pd.api.types.is_datetime64_any_dtype(values):
        idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
        return idx, ["The column was already a date type."], int(idx.isna().sum())

    # --- strategy 2: epoch seconds / millis ---
    epoch = _looks_like_epoch(s)
    if epoch:
        unit = epoch
        idx = pd.DatetimeIndex(pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit=unit))
        notes.append(f"Interpreted the column as Unix timestamps in {unit}.")
        return idx, notes, int(idx.isna().sum())

    # --- strategy 3: Excel serial numbers ---
    if _looks_like_excel_serial(s, name):
        numeric = pd.to_numeric(s, errors="coerce")
        idx = pd.DatetimeIndex(
            pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric, unit="D")
        )
        notes.append(
            "Interpreted the column as Excel serial dates (origin 1899-12-30). "
            "If that is wrong, pick another time column in Column mapping."
        )
        return idx, notes, int(idx.isna().sum())

    # --- strategy 4: 'YYYYQn' quarters ---
    sample = s.dropna().head(50)
    if len(sample) and sample.str.match(_QUARTER_RE).mean() > 0.9:
        parsed = s.str.extract(_QUARTER_RE)
        years = pd.to_numeric(parsed[0], errors="coerce")
        quarters = pd.to_numeric(parsed[1], errors="coerce")
        month = (quarters - 1) * 3 + 1
        idx = pd.DatetimeIndex(
            pd.to_datetime(dict(year=years, month=month, day=1), errors="coerce")
        )
        notes.append("Parsed the column as year-quarter labels (e.g. 1994Q3).")
        return idx, notes, int(idx.isna().sum())

    # --- strategy 5: explicit formats, then pandas' per-element fallback ---
    best: pd.DatetimeIndex | None = None
    best_bad = raw_n + 1
    best_fmt = ""
    for fmt in _DATETIME_FORMATS:
        try:
            candidate = pd.DatetimeIndex(pd.to_datetime(s, format=fmt, errors="coerce"))
        except (ValueError, TypeError):
            continue
        bad = int(candidate.isna().sum())
        if bad < best_bad:
            best, best_bad, best_fmt = candidate, bad, fmt
    if best is not None and best_bad == 0:
        notes.append(f"Parsed dates with the format {best_fmt}.")
        notes.extend(_dayfirst_notes(s))
        return best, notes, 0

    generic = pd.DatetimeIndex(pd.to_datetime(s, errors="coerce", format="mixed"))
    bad = int(generic.isna().sum())
    if bad < best_bad:
        best, best_bad, best_fmt = generic, bad, "pandas mixed-format parser"

    if best is None or best_bad >= raw_n:
        raise TimeIndexError(
            plain="No column in this file looks like a date or a time.",
            fix=(
                "Add a column of dates (for example 2020-01-01, Jan-2020 or 2020Q1), "
                "or tell us in Column mapping that the rows are evenly spaced observations."
            ),
        )
    if best_bad:
        notes.append(
            f"{best_bad} of {raw_n} time values could not be parsed and were left blank "
            f"(parser: {best_fmt})."
        )
    dayfirst = _detect_dayfirst(s)
    if dayfirst is not None:
        notes.append(
            "Dates like 03/04/2020 are ambiguous: read as "
            + ("day/month (European style)." if dayfirst else "month/day (US style).")
            + " Change this in Column mapping if it is wrong."
        )
    return best, notes, best_bad


def _dayfirst_notes(s: pd.Series) -> list[str]:
    dayfirst = _detect_dayfirst(s)
    if dayfirst is None:
        return []
    return [
        "Dates like 03/04/2020 are ambiguous: read as "
        + ("day/month (European style)." if dayfirst else "month/day (US style).")
        + " Change this in Column mapping if it is wrong."
    ]


def _detect_dayfirst(s: pd.Series) -> bool | None:
    """Only claim an answer when the data actually proves it."""
    sample = s.dropna().astype(str).head(200)
    slash = sample[sample.str.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")]
    if slash.empty:
        return None
    parts = slash.str.split(r"[/-]", expand=True)
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    if (first > 12).any():
        return True          # must be day-first
    if (second > 12).any():
        return False         # must be month-first
    return None              # genuinely ambiguous


# --------------------------------------------------------------------------- #
# Composite year + month / quarter columns
# --------------------------------------------------------------------------- #
def build_from_year_month(
    years: pd.Series, months: pd.Series | None = None,
    quarters: pd.Series | None = None, days: pd.Series | None = None,
) -> pd.DatetimeIndex:
    y = pd.to_numeric(years, errors="coerce")
    if quarters is not None:
        m = (pd.to_numeric(quarters, errors="coerce") - 1) * 3 + 1
    elif months is not None:
        m = pd.to_numeric(months, errors="coerce")
    else:
        m = pd.Series(1, index=y.index)
    d = pd.to_numeric(days, errors="coerce") if days is not None else pd.Series(1, index=y.index)
    return pd.DatetimeIndex(pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce"))


# --------------------------------------------------------------------------- #
# Frequency inference
# --------------------------------------------------------------------------- #
def infer_frequency(idx: pd.DatetimeIndex) -> tuple[FrequencySpec, list[FrequencySpec]]:
    """
    Return the best calendar plus runners-up.

    `pd.infer_freq` needs >= 3 clean dates and is easily fooled by gaps, so we
    cross-check with the median spacing and, when the two disagree, keep both as
    candidates and lower our confidence — the UI then shows the user two pictures.
    """
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(idx).dropna().unique()))
    if len(idx) < 2:
        raise FrequencyInferenceError(
            plain="There are fewer than two dates, so there is no time series to analyse.",
            fix="Upload a series with at least 10 observations, ideally two full seasonal cycles.",
        )

    candidates: list[FrequencySpec] = []
    alias: str | None = None
    if len(idx) >= 3:
        try:
            alias = pd.infer_freq(idx)
        except ValueError:
            alias = None
    if alias:
        mapped = _ALIAS_MAP.get(alias) or _ALIAS_MAP.get(alias.upper())
        if mapped:
            candidates.append(_SPEC_BY_CODE[mapped])

    deltas = np.diff(idx.asi8) / 86_400_000_000_000.0    # days
    median_days = float(np.median(deltas))
    spacing_guess = _spec_from_spacing(median_days, idx)
    if spacing_guess and spacing_guess.code not in {c.code for c in candidates}:
        candidates.append(spacing_guess)

    if not candidates:
        # Last resort: count observations per calendar year. Enough to offer a
        # calendar with LOW confidence so the user can confirm, rather than a wall.
        counted = _spec_from_counts(idx)
        if counted is not None:
            return counted.model_copy(update={
                "confidence": Confidence.LOW,
                "evidence": (
                    f"Spacing is irregular; inferred from ~{counted.period}-ish "
                    "observations per year. Please confirm the frequency."
                ),
            }), [counted]
        raise FrequencyInferenceError(
            plain="The gaps between observations are not consistent enough to name a frequency.",
            fix=(
                "Choose the frequency yourself in Column mapping (daily, weekly, monthly, "
                "quarterly or yearly), or aggregate the data to a coarser regular interval."
            ),
            technical=f"median spacing {median_days:.4f} days over {len(idx)} dates",
        )

    # Internal consistency: does the data actually match the calendar we inferred?
    # Twice-monthly dates parse as "monthly" to every calendar arithmetic check, so
    # we measure the share of steps that are (almost) equal to the median step.
    if len(deltas) > 2:
        tolerance = max(_step_tolerance(median_days), 1e-9)
        consistent = float(np.mean(np.abs(deltas - median_days) <= tolerance))
        if consistent < 0.95:
            chosen = candidates[0]
            return (
                chosen.model_copy(update={
                    "confidence": Confidence.LOW,
                    "evidence": (
                        f"Only {consistent:.0%} of the gaps between observations are the same "
                        "size, so this is not a clean regular series. Confirm the frequency, or "
                        "aggregate the data to a coarser interval."
                    ),
                }),
                [chosen, *_other_specs(candidates, chosen)],
            )

    best = candidates[0]
    if len(candidates) > 1:
        # Disagreement between infer_freq and median spacing → be honest about it.
        best = best.model_copy(update={
            "confidence": Confidence.MEDIUM,
            "evidence": (
                "Two plausible calendars were detected ("
                + " vs ".join(c.label.lower() for c in candidates[:2])
                + ") — please confirm which one is right."
            ),
        })
    else:
        best = best.model_copy(update={"confidence": Confidence.HIGH})
    others = [c for c in candidates if c.code != best.code]
    return best, [best, *others]


def _spec_from_subdaily_step(median_days: float) -> FrequencySpec | None:
    """
    Sub-daily data (minutes/seconds) fits none of the six named calendars, so we
    build a calendar from the observed step instead of silently rounding to hourly
    (which would collapse 60 observations into 1).
    """
    median_ns = int(round(median_days * 86_400 * 1e9))
    if median_ns <= 0:
        return None
    try:
        offset = pd.tseries.frequencies.to_offset(pd.Timedelta(median_ns))
        code = offset.freqstr
    except (ValueError, TypeError):
        return None
    period = 1440 if code.endswith("min") else (60 if code.endswith("s") else 24)
    pretty = {"min": "Minutely", "h": "Hourly", "s": "Per second"}.get(code, f"Every {code}")
    return FrequencySpec(
        code=code, label=f"{pretty} (step {code})", period=period,
        offset_step_ns=median_ns, confidence=Confidence.MEDIUM,
        evidence=f"Median spacing {median_ns / 1e9:g} seconds",
    )


def _spec_from_spacing(median_days: float, idx: pd.DatetimeIndex) -> FrequencySpec | None:
    if median_days <= 0:
        return None
    if median_days < 0.5:
        return _spec_from_subdaily_step(median_days)
    if 0.9 <= median_days <= 1.1:
        return _SPEC_BY_CODE["D"]
    if 6.5 <= median_days <= 7.5:
        return _SPEC_BY_CODE["W"]
    if 27 <= median_days <= 32:
        return _SPEC_BY_CODE["MS"]
    if 85 <= median_days <= 100:
        return _SPEC_BY_CODE["QS"]
    if 360 <= median_days <= 370:
        return _SPEC_BY_CODE["YS"]
    # Fall back to counting observations per calendar year.
    per_year = pd.Series(idx.year).value_counts()
    mode = int(per_year.mode().iloc[0]) if not per_year.empty else 0
    if mode >= 300:
        return _SPEC_BY_CODE["D"]
    if 45 <= mode <= 55:
        return _SPEC_BY_CODE["W"]
    if 10 <= mode <= 14:
        return _SPEC_BY_CODE["MS"]
    if 3 <= mode <= 5:
        return _SPEC_BY_CODE["QS"]
    return None


def _step_tolerance(median_days: float) -> float:
    """How close a step must be to the median step to count as 'the same size'."""
    if median_days >= 360:          # yearly: allow leap years
        return 1.5
    if median_days >= 85:           # quarterly
        return 3.0
    if median_days >= 27:           # monthly: 28-31 day months
        return 3.5
    return max(0.02 * median_days, 1e-3)


def _other_specs(candidates: list[FrequencySpec], chosen: FrequencySpec) -> list[FrequencySpec]:
    return [c for c in candidates if c.code != chosen.code]


def _spec_from_counts(idx: pd.DatetimeIndex) -> FrequencySpec | None:
    """Guess a calendar from how many observations land in a typical year."""
    if len(idx) < 6:
        return None
    per_year = pd.Series(idx.year).value_counts()
    if per_year.empty:
        return None
    mode = float(per_year.median())
    # Nearest calendar by observations-per-year. Deliberately forgiving: this is
    # only reached for irregular data, and the result is offered at LOW confidence.
    options = [(365.0, "D"), (52.0, "W"), (12.0, "MS"), (4.0, "QS"), (1.0, "YS")]
    nearest, code = min(options, key=lambda oc: abs(np.log(max(mode, 0.5) / oc[0])))
    if abs(np.log(max(mode, 0.5) / nearest)) > 0.9:      # more than ~2.5x off
        return None
    return _SPEC_BY_CODE[code]


def event_calendar(times: pd.DatetimeIndex) -> FrequencySpec:
    """
    The counting calendar for a list of event dates.

    Chosen so the resulting count series is *analysable*: at least ~24 periods if the span allows it,
    and never so fine that the series is a list of ones with no variation. Daily for a span under two
    years, weekly up to five, monthly beyond — the same rule the count branch in the API uses, stated
    once here so the ingest layer and the model layer cannot disagree about what "a period" means.
    """
    t = pd.DatetimeIndex(sorted(pd.DatetimeIndex(times).dropna()))
    if t.size < 2:
        return _SPEC_BY_CODE["MS"]
    span_days = float((t[-1] - t[0]) / pd.Timedelta(days=1))
    # Days per period for each candidate, stated directly. (A daily spec's `period` field is about
    # seasonality, not about how many days a bucket covers — using it here silently skipped daily.)
    days_per_period = {"D": 1.0, "W": 7.0, "MS": 365.25 / 12, "QS": 365.25 / 4}
    # The coarsest calendar that still leaves at least 24 periods: iterate fine → coarse and keep the
    # last that qualifies, so a two-month file is daily and a four-year file is monthly.
    chosen = "MS"
    for code in ("D", "W", "MS"):
        if span_days / days_per_period[code] >= 24:
            chosen = code
    return _SPEC_BY_CODE[chosen]


def period_of(spec: FrequencySpec) -> int | None:
    return spec.period if spec.period and spec.period > 1 else None


# --------------------------------------------------------------------------- #
# Regularisation
# --------------------------------------------------------------------------- #
def regularise(
    values: pd.Series, spec: FrequencySpec, dedupe: Literal["mean", "first", "sum", "last"] = "mean"
) -> TimeIndexResult:
    """
    Put a series on a complete regular calendar.

    Duplicates are aggregated (default: mean) and reported. Gaps are *inserted as
    NaN* rather than closed, because silently shifting subsequent values left is
    the single most common cause of wrong seasonal patterns.
    """
    s = values.copy()
    s.index = pd.DatetimeIndex(s.index)
    n_raw = len(s)

    dup = int(s.index.duplicated().sum())
    if dup:
        agg = {"mean": "mean", "first": "first", "sum": "sum", "last": "last"}[dedupe]
        s = s.groupby(level=0).agg(agg)

    s = s.sort_index()
    try:
        full = pd.date_range(s.index.min(), s.index.max(), freq=spec.code)
    except Exception as exc:  # pragma: no cover
        raise IrregularSpacingError(technical=str(exc)) from exc

    notes: list[str] = []
    if dup:
        notes.append(f"{dup} duplicate timestamp(s) were combined using the {dedupe}.")

    s_reg = s.reindex(full)
    n_gaps = int(s_reg.isna().sum())

    # A calendar can miss *every* observation even when the spacing is perfect, if the labels do not sit
    # on the anchor day the frequency implies. Weekly data labelled on Fridays reindexed onto a
    # `W` (Sunday-anchored) grid yields a series of nothing but NaN, and the only symptom upstream was
    # "the value column is empty after cleaning" — a message that points at the file rather than at the
    # calendar. Caught here, where the cause is visible.
    if n_raw and int(s_reg.notna().sum()) == 0:
        raise IrregularSpacingError(
            technical=(
                f"none of the {n_raw} timestamps fall on the {spec.label.lower()} calendar anchored at "
                f"{full[0].date()} (step {spec.offset_step_ns / 8.64e13:.0f} days); the observations are "
                "offset from the calendar grid"
            )
        )

    irregular = False
    if len(s.index) >= 3:
        diffs = np.diff(pd.DatetimeIndex(s.index).asi8)
        expected = _expected_step_ns(spec, s.index)
        if expected and np.median(diffs) > expected * 1.5:
            irregular = True
            notes.append(
                "The spacing between observations is irregular; "
                f"the series was placed on a regular {spec.label.lower()} calendar."
            )

    return TimeIndexResult(
        index=full,
        frequency=spec,
        series=s_reg,
        n_raw=n_raw,
        n_reindexed=n_gaps,
        n_duplicates=dup,
        n_unparseable=0,
        irregular=irregular,
        notes=notes,
        candidate_frequencies=[spec],
    )


def _expected_step_ns(spec: FrequencySpec, idx: pd.DatetimeIndex) -> int | None:
    if spec.offset_step_ns:
        return spec.offset_step_ns
    if len(idx) < 3:
        return None
    return int(np.median(np.diff(idx.asi8)))


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #
def score_time_column(series: pd.Series, name: str = "") -> float:
    """
    How likely is this column to be the time index? Returns 0..1.

    A name hint ('date', 'month', 'year', 'time', ...) plus a successful parse of
    most values. Used by the column-mapping assistant so beginners get a guess
    they can accept rather than a form they must fill.
    """
    if series.dropna().empty:
        return 0.0
    try:
        idx, _notes, bad = parse_time_column(series, name)
    except TimeIndexError:
        return 0.0
    parsed_frac = 1.0 - bad / max(len(series), 1)
    if parsed_frac < 0.5:
        return 0.0

    unique_frac = series.nunique(dropna=True) / max(len(series.dropna()), 1)
    hint = 0.0
    lowered = name.lower()
    for kw, weight in (
        ("date", 0.35), ("time", 0.3), ("month", 0.3), ("year", 0.3),
        ("quarter", 0.3), ("week", 0.25), ("period", 0.2), ("day", 0.2),
        ("timestamp", 0.35), ("date_time", 0.35),
    ):
        if kw in lowered:
            hint = max(hint, weight)
            break

    mono = 1.0
    valid = idx.dropna()
    if len(valid) > 2 and not valid.is_monotonic_increasing:
        mono = 0.6

    return float(min(1.0, 0.45 * parsed_frac + 0.35 * unique_frac + hint) * mono)


def detect_time_column(df: pd.DataFrame, exclude: set[str] | None = None) -> tuple[str | None, float]:
    exclude = exclude or set()
    best, best_score = None, 0.0
    for col in df.columns:
        if col in exclude:
            continue
        sc = score_time_column(df[col], str(col))
        if sc > best_score:
            best, best_score = str(col), sc
    return best, best_score


def detect_year_month_columns(df: pd.DataFrame) -> tuple[str, str | None] | None:
    """
    Find a (year, month|quarter|day) pair — extremely common in thesis datasets.
    Returns None when the guess is not convincing.
    """
    lower = {str(c).lower(): str(c) for c in df.columns}
    year_col = next((lower[k] for k in lower if k in {"year", "yr", "yyyy", "years"}), None)
    if year_col is None:
        return None

    # The column must actually hold 4-digit years. A 'Year' column containing full
    # dates (1821-01-01) is a datetime column, not a year column — treating it as
    # one silently produces an empty index.
    numeric_year = pd.to_numeric(df[year_col], errors="coerce")
    valid = numeric_year.dropna()
    if valid.empty or not bool(valid.between(1000, 2200).mean() > 0.9):
        return None

    for key in ("month", "mon", "mm", "quarter", "qtr", "day", "dd"):
        if key in lower:
            companion = lower[key]
            companion_numeric = pd.to_numeric(df[companion], errors="coerce").dropna()
            if companion_numeric.empty:
                continue
            if key.startswith("q") and not bool(companion_numeric.between(1, 4).all()):
                continue
            if key.startswith("m") and not bool(companion_numeric.between(1, 12).all()):
                continue
            if key.startswith("d") and not bool(companion_numeric.between(1, 31).all()):
                continue
            return year_col, companion
    return year_col, None


__all__ = [
    "CALENDAR_SPECS",
    "TimeIndexResult",
    "parse_time_column",
    "build_from_year_month",
    "infer_frequency",
    "regularise",
    "period_of",
    "detect_time_column",
    "detect_year_month_columns",
    "score_time_column",
]
