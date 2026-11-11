"""
ts/io/validate.py — turn raw text columns into a validated numeric series and
produce the DataQualityReport that Parts 2-7 consume.

Two things here are load-bearing:

`detect_data_class` decides the *entire downstream branch* (§5.8). A sequence of
monthly event counts must never be handed to ETS/ARIMA without the app saying
why, so this runs at ingest and its evidence travels with the dataset.

`build_quality_report` never recommends anything — it only measures. Advice
belongs in the advisor (Part 3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import NonNumericSeriesError
from app.core.schemas import (
    ColumnProfile,
    Confidence,
    DataClass,
    DataQualityReport,
    MissingPattern,
    ParseIssue,
)

# --------------------------------------------------------------------------- #
# Numeric coercion
# --------------------------------------------------------------------------- #
_CURRENCY_JUNK = re.compile(r"[\s\u00a0$€£¥₱,%]")
_PAREN_NEG = re.compile(r"^\((.*)\)$")


def coerce_numeric(series: pd.Series, name: str = "value") -> tuple[pd.Series, list[ParseIssue]]:
    """
    Turn a text column into floats, tolerating what real spreadsheets contain:
    currency symbols, thousands separators, '%', European decimals, '(1,234)'
    for negatives, and the usual missing-value tokens.

    Returns (numeric, issues). Never silently drops information: every coercion
    that changed a value is reported.
    """
    issues: list[ParseIssue] = []
    s = series.astype("string").str.strip()

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), issues

    # European decimal style: '1.234,56' → '1234.56'
    euro = s.str.match(r"^-?\d{1,3}(\.\d{3})+,\d+$", na=False)
    if euro.mean() > 0.3:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        issues.append(ParseIssue(
            code="data.european_decimals", severity="info",
            plain="Some numbers used a comma as the decimal separator; they were converted.",
            fix="No action needed, but export as CSV with '.' decimals to avoid ambiguity.",
            rows_affected=int(euro.sum()),
        ))
    else:
        # Thousands separators: '1,234' or '1 234' → 1234. Only strip commas that
        # sit inside a number, so '1,5' style values are not corrupted.
        thousands = s.str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", na=False)
        if thousands.any():
            s = s.where(~thousands, s.str.replace(",", "", regex=False))
            issues.append(ParseIssue(
                code="data.thousands_separator", severity="info",
                plain="Numbers containing thousands separators were cleaned.",
                rows_affected=int(thousands.sum()),
            ))

    paren = s.str.match(_PAREN_NEG, na=False)
    if paren.any():
        inner = s[paren].str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        s = s.where(~paren, inner)
        issues.append(ParseIssue(
            code="data.paren_negatives", severity="info",
            plain="Accounting-style values in parentheses were read as negative numbers.",
            rows_affected=int(paren.sum()),
        ))

    cleaned = s.str.replace(_CURRENCY_JUNK, "", regex=True)
    cleaned = cleaned.replace({
        "": pd.NA, "-": pd.NA, "—": pd.NA, ".": pd.NA,
        "nan": pd.NA, "NaN": pd.NA, "NA": pd.NA, "N/A": pd.NA, "n/a": pd.NA,
        "null": pd.NA, "NULL": pd.NA, "None": pd.NA, "#N/A": pd.NA, "nil": pd.NA,
    })

    numeric = pd.to_numeric(cleaned, errors="coerce")
    unparseable = cleaned.notna() & numeric.isna()
    n_bad = int(unparseable.sum())
    if n_bad:
        examples = cleaned[unparseable].dropna().unique()[:3].tolist()
        if numeric.notna().sum() == 0:
            raise NonNumericSeriesError(
                plain=f"The column '{name}' contains no numbers we can read.",
                fix=(
                    "Pick a numeric column, or clean the text first "
                    f"(examples of what we found: {', '.join(map(str, examples))})."
                ),
            )
        issues.append(ParseIssue(
            code="data.unparseable_values", severity="warning",
            plain=(
                f"{n_bad} value(s) in '{name}' were not numbers and were set to missing "
                f"(examples: {', '.join(map(str, examples))})."
            ),
            fix="Check those rows in your spreadsheet if the series looks wrong.",
            rows_affected=n_bad,
        ))
    return numeric.astype(float), issues


# --------------------------------------------------------------------------- #
# Data class detection (§5.8) — this gates the whole model registry
# --------------------------------------------------------------------------- #
@dataclass
class DataClassVerdict:
    data_class: DataClass
    evidence: dict[str, float | str | bool]
    plain: str


def detect_data_class(
    values: pd.Series,
    has_explicit_target: bool = True,
    intermittent_zero_threshold: float = 0.30,
) -> DataClassVerdict:
    """
    continuous | count | intermittent | binary_or_proportion | event_timestamp

    Order matters: zeros first (intermittent beats count), then integrality, then
    range. Each threshold is a documented default, not a hidden rule.
    """
    if not has_explicit_target:
        return DataClassVerdict(
            DataClass.EVENT_TIMESTAMP,
            {"has_target": False},
            "There is no value column, so each row is an event timestamp — a point process.",
        )

    v = values.dropna().to_numpy(dtype=float)
    n = v.size
    if n == 0:
        return DataClassVerdict(DataClass.CONTINUOUS, {"n": 0}, "No values to classify.")

    n_zeros = int((v == 0).sum())
    pct_zeros = n_zeros / n
    all_int = bool(np.all(np.equal(np.mod(v, 1), 0)))
    n_neg = int((v < 0).sum())
    n_unique = int(np.unique(v).size)
    mean = float(np.mean(v))
    var = float(np.var(v, ddof=1)) if n > 1 else 0.0
    disp = var / mean if mean > 0 else float("nan")

    evidence: dict[str, float | str | bool] = {
        "n": float(n),
        "n_zeros": float(n_zeros),
        "pct_zeros": round(pct_zeros, 4),
        "all_integer": all_int,
        "n_negative": float(n_neg),
        "n_unique": float(n_unique),
        "mean": round(mean, 6),
        "variance": round(var, 6),
        "dispersion_ratio": None if np.isnan(disp) else round(disp, 4),
        "min": round(float(np.min(v)), 6),
        "max": round(float(np.max(v)), 6),
    }

    # Binary first: a 0/1 series has many zeros but is not "intermittent demand".
    if all_int and set(np.unique(v).tolist()) <= {0.0, 1.0}:
        return DataClassVerdict(
            DataClass.BINARY_OR_PROPORTION, evidence,
            "Only 0s and 1s appear, so this is a binary outcome; a proportion or binomial model applies.",
        )

    if pct_zeros > intermittent_zero_threshold:
        return DataClassVerdict(
            DataClass.INTERMITTENT, evidence,
            (
                f"{pct_zeros:.0%} of the observations are zero, so this is intermittent demand. "
                "Classical ARIMA/ETS are the wrong tool here; Croston/TSB-type methods and "
                "MASE-based evaluation are."
            ),
        )

    if all_int and n_neg == 0:
        # Non-negative whole numbers *are* count data formally, but large-magnitude
        # integers (passenger numbers, monthly deaths) are treated as continuous in
        # practice and by every mainstream forecasting package. So: small counts or
        # many zeros go to the count branch; large-magnitude integers stay
        # continuous, with the count branch one click away.
        median_value = float(np.median(v))
        unique_frac = n_unique / max(n, 1)
        # High dispersion and near-unique integer values say "a measured quantity
        # that happens to land on integers" (passengers, deaths), not "events".
        large_magnitude = bool(
            median_value >= 20 and n_unique >= 20 and pct_zeros < 0.10
        )
        evidence["median"] = round(median_value, 4)
        if large_magnitude:
            evidence["count_branch_available"] = True
            return DataClassVerdict(
                DataClass.CONTINUOUS, evidence,
                (
                    "Values are whole numbers but large in magnitude "
                    f"(median {median_value:,.0f}), so they are treated as continuous — "
                    "this is what forecasting packages do with passenger numbers or monthly "
                    "death counts. If these are genuine event counts, switch the data class to "
                    "'count' in Column mapping to unlock Poisson/negative-binomial models."
                ),
            )
        ambiguous = bool(unique_frac > 0.5 and n_unique > 30)
        shape = "overdispersed" if (not np.isnan(disp) and disp > 1.5) else (
            "equidispersed" if (not np.isnan(disp) and disp >= 0.7) else "underdispersed")
        evidence["unique_frac"] = round(unique_frac, 4)
        evidence["count_ambiguity"] = ambiguous
        text = (
            "Values are non-negative whole numbers, so this is treated as count data "
            + (f"({shape}: variance/mean = {disp:.2f}). " if not np.isnan(disp) else ". ")
            + "Poisson or negative-binomial models apply, and MAPE is not a valid metric."
        )
        if ambiguous:
            text += (
                " The values are very spread out, so they could be rounded measurements rather "
                "than event counts — you can change the data class in Column mapping."
            )
        return DataClassVerdict(DataClass.COUNT, evidence, text)

    return DataClassVerdict(
        DataClass.CONTINUOUS, evidence,
        "Values look continuous (fractional or many distinct values), so continuous-time models apply.",
    )


# --------------------------------------------------------------------------- #
# Missing-data audit
# --------------------------------------------------------------------------- #
def missing_pattern(series: pd.Series, label: str = "value") -> MissingPattern:
    mask = series.isna()
    n_missing = int(mask.sum())
    if n_missing == 0:
        return MissingPattern(index=label, n_missing=0, pct_missing=0.0, longest_run=0,
                              plain="No gaps in this column.")

    longest, longest_start, run, run_start = 0, None, 0, None
    for i, is_na in enumerate(mask.to_numpy()):
        if is_na:
            if run == 0:
                run_start = i
            run += 1
            if run > longest:
                longest, longest_start = run, run_start
        else:
            run = 0

    gaps_over_two = _count_runs(mask.to_numpy(), min_len=2)
    pos = series.index[longest_start] if longest_start is not None else None
    plain = (
        f"{n_missing} missing value(s) ({mask.mean():.1%}); the longest unbroken gap is "
        f"{longest} observation(s)"
        + (f" starting {pos}." if pos is not None else ".")
    )
    if longest >= 2:
        plain += " A gap this long usually means the series stopped, so imputing it needs care."
    return MissingPattern(
        index=label, n_missing=n_missing, pct_missing=round(float(mask.mean()), 4),
        longest_run=longest,
        longest_run_start=str(pos) if pos is not None else None,
        gaps_over_two=gaps_over_two, plain=plain,
    )


def _count_runs(mask: np.ndarray, min_len: int) -> int:
    count, run = 0, 0
    for flag in mask:
        if flag:
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 0
    if run >= min_len:
        count += 1
    return count


# --------------------------------------------------------------------------- #
# Column profiling / mapping assistant
# --------------------------------------------------------------------------- #
def profile_columns(df: pd.DataFrame, time_col: str | None = None) -> list[ColumnProfile]:
    out: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        n_non_null = int(s.notna().sum())
        numeric, _ = (pd.Series(dtype=float), None)
        try:
            numeric, _issues = coerce_numeric(s, str(col))
        except NonNumericSeriesError:
            numeric = pd.Series(dtype=float, index=s.index)
        is_numeric = numeric.notna().sum() >= max(0.5 * max(n_non_null, 1), 1)
        out.append(ColumnProfile(
            name=str(col),
            dtype=str(s.dtype),
            n_non_null=n_non_null,
            n_missing=int(s.isna().sum()),
            pct_missing=round(float(s.isna().mean()), 4),
            n_unique=int(s.nunique(dropna=True)),
            is_numeric=bool(is_numeric),
            is_datetime_like=bool(col == time_col),
            sample_values=[str(x) for x in s.dropna().head(3).tolist()],
            role_guess="time" if col == time_col else ("target" if is_numeric else "ignore"),
            role_confidence=Confidence.HIGH if col == time_col else (
                Confidence.MEDIUM if is_numeric else Confidence.LOW
            ),
        ))
    return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def robust_outlier_count(values: pd.Series, z_cutoff: float = 3.5) -> int:
    """
    Count outliers with the median-absolute-deviation rule (Iglewicz & Hoaglin).
    Robust to the very outliers we are looking for, unlike a plain z-score.
    """
    v = values.dropna().to_numpy(dtype=float)
    if v.size < 5:
        return 0
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        return 0
    modified_z = 0.6745 * (v - med) / mad
    return int(np.sum(np.abs(modified_z) > z_cutoff))


def longest_constant_run(values: pd.Series) -> int:
    v = values.to_numpy()
    longest = run = 1 if len(v) else 0
    for i in range(1, len(v)):
        run = run + 1 if (v[i] == v[i - 1]) else 1
        longest = max(longest, run)
    return int(longest) if len(v) else 0


def build_quality_report(
    *,
    raw_table_rows: int,
    prepared: pd.Series,
    df_all: pd.DataFrame,
    time_column: str,
    target_column: str,
    exogenous_columns: list[str],
    frequency,
    data_class: DataClass,
    data_class_evidence: dict[str, Any],
    issues: list[ParseIssue],
    n_expected_observations: int | None = None,
    dropped_rows: int = 0,
    dropped_reason: str = "",
) -> DataQualityReport:
    values = prepared.dropna()
    n = int(prepared.size)
    const_run = longest_constant_run(prepared)
    # A run of zeros in intermittent/sparse data is the phenomenon, not a defect.
    if (
        const_run >= 6
        and data_class not in (DataClass.INTERMITTENT, DataClass.EVENT_TIMESTAMP)
        and const_run < 0.5 * max(n, 1)
    ):
        issues.append(ParseIssue(
            code="data.constant_run", severity="warning",
            plain=f"{const_run} identical values in a row — the series may be stale or truncated.",
            fix="Check whether the data were actually recorded during that period.",
            rows_affected=const_run,
        ))
    if n and n < 2 * (frequency.period or 1):
        issues.append(ParseIssue(
            code="data.short_series", severity="warning",
            plain=(
                f"Only {n} observations, which is less than two full seasonal cycles "
                f"({2 * (frequency.period or 1)}). Seasonal models will be unreliable."
            ),
            fix="Extend the series, or plan on a non-seasonal model.",
        ))

    return DataQualityReport(
        n_rows_raw=raw_table_rows,
        n_rows_used=n,
        n_columns=int(df_all.shape[1]),
        dropped_rows=dropped_rows,
        dropped_reason=dropped_reason,
        time_column=time_column,
        target_column=target_column,
        exogenous_columns=exogenous_columns,
        time_start=str(prepared.index.min()) if n else None,
        time_end=str(prepared.index.max()) if n else None,
        frequency=frequency,
        n_observations=n,
        n_expected_observations=n_expected_observations,
        n_gaps=int(n_expected_observations - n) if n_expected_observations else 0,
        is_regular=bool(n_expected_observations in (None, n)),
        data_class=data_class,
        data_class_evidence=data_class_evidence,
        n_missing_total=int(prepared.isna().sum()),
        missing_by_column=[missing_pattern(prepared, target_column)],
        n_zeros=int((values == 0).sum()),
        pct_zeros=round(float((values == 0).mean()) if n else 0.0, 4),
        n_negatives=int((values < 0).sum()),
        n_constant_runs=const_run,
        n_duplicate_timestamps=0,          # set by the caller from TimeIndexResult
        n_outlier_candidates=robust_outlier_count(prepared),
        issues=issues,
        plain_summary=_plain_summary(prepared, frequency, data_class),
    )


def _plain_summary(values: pd.Series, frequency, data_class: DataClass) -> str:
    v = values.dropna()
    if v.empty:
        return "The series is empty after cleaning."
    n = len(v)
    first, last = float(v.iloc[0]), float(v.iloc[-1])
    direction = "risen" if last > first else ("fallen" if last < first else "stayed level")
    change = (last - first) / abs(first) * 100 if first else float("nan")
    missing = int(values.isna().sum())
    bits = [
        f"{n} {frequency.label.lower()} observations from {values.index.min():%Y-%m-%d} "
        f"to {values.index.max():%Y-%m-%d}.",
        f"It has {direction} from {first:,.3g} to {last:,.3g} "
        + (f"({change:+.1f}%)." if np.isfinite(change) else "."),
    ]
    if missing:
        bits.append(f"{missing} observation(s) are missing.")
    if data_class is not DataClass.CONTINUOUS:
        bits.append(f"Detected data class: {data_class.value.replace('_', ' ')}.")
    return " ".join(bits)


__all__ = [
    "coerce_numeric",
    "detect_data_class",
    "DataClassVerdict",
    "missing_pattern",
    "profile_columns",
    "robust_outlier_count",
    "longest_constant_run",
    "build_quality_report",
]
