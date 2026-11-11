"""
ts/io/readers.py — read a table out of almost anything (§5.1).

Rules:
  * every reader returns a pandas DataFrame with string column names and the
    raw values untouched — never coerce types here. Type coercion is the time
    index parser's and validator's job, so we can report *why* something failed.
  * an optional dependency that is absent must produce an actionable error, not
    an ImportError traceback.
  * nothing in this module logs user data at INFO.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import pandas as pd

from app.core.config import get_settings
from app.core.errors import (
    EmptyDatasetError,
    FileTooLargeError,
    UnsupportedFormatError,
)


@dataclass
class RawTable:
    """A loaded table plus everything we know about how it was loaded."""

    df: pd.DataFrame
    source: str
    format: str
    delimiter: str | None = None
    sheet_name: str | None = None
    encoding: str | None = None
    notes: list[str] = field(default_factory=list)
    n_rows_raw: int = 0
    dropped_blank_rows: int = 0

    def __post_init__(self) -> None:
        if self.n_rows_raw == 0:
            self.n_rows_raw = len(self.df)


# --------------------------------------------------------------------------- #
# Delimited text
# --------------------------------------------------------------------------- #
_CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
_CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]


def _decode_text(data: bytes) -> tuple[str, str]:
    """Try encodings in order; report which one worked so the UI can say so."""
    last_err: Exception | None = None
    for enc in _CANDIDATE_ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError as e:  # pragma: no cover - depends on input
            last_err = e
    raise UnsupportedFormatError(
        plain="The text file's character encoding could not be read.",
        fix="Save the file as 'CSV UTF-8' and upload again.",
        technical=str(last_err),
    )


def sniff_delimiter(sample: str) -> str:
    """
    Pick the delimiter by counting candidates outside quotes on the first lines.
    `csv.Sniffer` is tried first but is unreliable on single-column files, so we
    fall back to the most frequent candidate.
    """
    sample_lines = "\n".join(sample.splitlines()[:20])
    if not sample_lines.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample_lines, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        counts = {d: sample_lines.count(d) for d in _CANDIDATE_DELIMITERS}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] > 0 else ","


def read_csv_bytes(data: bytes, source: str = "<upload>") -> RawTable:
    text, encoding = _decode_text(data)
    delimiter = sniff_delimiter(text)
    df = pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        dtype=str,              # keep raw text; coercion happens later
        keep_default_na=True,
        na_values=["", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "-", ".", "#N/A"],
        skipinitialspace=True,
        engine="python",
    )
    df.columns = [str(c).strip() for c in df.columns]
    n_before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    dropped = n_before - len(df)
    table = RawTable(
        df=df, source=source, format="csv" if delimiter == "," else "delimited",
        delimiter=delimiter, encoding=encoding, n_rows_raw=n_before,
        dropped_blank_rows=dropped,
    )
    if dropped:
        table.notes.append(f"{dropped} blank row(s) removed at load time.")
    if df.empty:
        raise EmptyDatasetError()
    return table


def read_pasted_text(text: str) -> RawTable:
    """Clipboard paste: same path as CSV, but allows tab-separated Excel copies."""
    if not text.strip():
        raise EmptyDatasetError(
            plain="You pasted nothing.",
            fix="Copy the table again from your spreadsheet, including the header row.",
        )
    return read_csv_bytes(text.encode("utf-8"), source="<pasted>")


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def read_excel_bytes(
    data: bytes, source: str = "<upload>", sheet_name: str | int | None = None
) -> RawTable:
    try:
        xls = pd.ExcelFile(io.BytesIO(data))
    except Exception as exc:  # pragma: no cover - depends on the file
        raise UnsupportedFormatError(
            plain="That Excel file could not be opened.",
            fix="Re-save it as .xlsx (not .xls) or export the sheet to CSV.",
            technical=str(exc),
        ) from exc

    sheets = xls.sheet_names
    chosen = sheet_name if sheet_name is not None else _pick_best_sheet(xls)
    df = xls.parse(chosen, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    n_before = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    table = RawTable(
        df=df, source=source, format="excel", sheet_name=str(chosen),
        n_rows_raw=n_before, dropped_blank_rows=n_before - len(df),
    )
    if len(sheets) > 1:
        table.notes.append(
            f"The workbook has {len(sheets)} sheets ({', '.join(sheets)}); "
            f"'{chosen}' was read. You can change this in Column mapping."
        )
    if df.empty:
        raise EmptyDatasetError(
            plain=f"The sheet '{chosen}' has no data rows.",
            fix="Pick a different sheet in Column mapping.",
        )
    return table


def _pick_best_sheet(xls: pd.ExcelFile) -> str:
    """Choose the sheet with the most populated cells — usually the data one."""
    best, best_score = xls.sheet_names[0], -1
    for name in xls.sheet_names:
        try:
            probe = xls.parse(name, nrows=200, dtype=str)
            score = probe.notna().sum().sum()
        except Exception:  # pragma: no cover
            score = -1
        if score > best_score:
            best, best_score = name, score
    return best


# --------------------------------------------------------------------------- #
# JSON / JSON-lines / arrays
# --------------------------------------------------------------------------- #
def read_json_bytes(data: bytes, source: str = "<upload>") -> RawTable:
    text, encoding = _decode_text(data)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # Try newline-delimited JSON before giving up.
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            payload = rows
        except json.JSONDecodeError:
            raise UnsupportedFormatError(
                plain="That JSON file is not valid JSON.",
                fix="Check for a missing comma or bracket, or export as CSV.",
                technical=str(exc),
            ) from exc

    notes: list[str] = []
    # pandas `DataFrame.to_json()` defaults, and several export tools, produce
    # orient='columns' ( {col: {row: value}} ) or orient='index' ( {row: {col: value}} ).
    if isinstance(payload, dict) and payload and all(
        isinstance(v, dict) for v in payload.values()
    ):
        outer = list(payload.keys())
        # orient='index'  → outer keys are row labels ("0","1","2", …)
        # orient='columns'→ outer keys are column names
        sequential_labels = all(str(k).isdigit() for k in outer) and [
            int(k) for k in outer
        ] == list(range(len(outer)))
        if sequential_labels:
            frame = pd.DataFrame.from_dict(payload, orient="index")
            note = "Read a row-oriented (orient='index') JSON object as a table."
        else:
            frame = pd.DataFrame(payload)
            note = "Read a column-oriented (orient='columns') JSON object as a table."
        frame.columns = [str(c).strip() for c in frame.columns]
        return RawTable(df=frame, source=source, format="json", encoding=encoding,
                        notes=[note], n_rows_raw=len(frame))

    if isinstance(payload, dict):
        # Try the usual containers first, then any list-valued key.
        for key in ("data", "records", "rows", "values", "results", "items"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                notes.append(f"Read the '{key}' array from the JSON object.")
                break
        else:
            list_keys = [k for k, v in payload.items() if isinstance(v, list)]
            if len(list_keys) == 1:
                payload = payload[list_keys[0]]
                notes.append(f"Read the '{list_keys[0]}' array from the JSON object.")
            else:
                payload = [payload]

    if not isinstance(payload, list) or not payload:
        raise EmptyDatasetError(
            plain="The JSON file contains no rows.",
            fix="Wrap your records in a JSON array, or export the data as CSV.",
        )

    if all(isinstance(x, dict) for x in payload):
        df = pd.DataFrame(payload)
    elif all(isinstance(x, (int, float)) for x in payload):
        df = pd.DataFrame({"value": payload})
        notes.append("Found a flat list of numbers; named the column 'value'.")
    elif all(isinstance(x, list) and len(x) == 2 for x in payload):
        df = pd.DataFrame(payload, columns=["time", "value"])
        notes.append("Found a list of [time, value] pairs.")
    elif all(isinstance(x, list) for x in payload):
        width = len(payload[0])
        df = pd.DataFrame(payload, columns=[f"V{i + 1}" for i in range(width)])
        notes.append(
            f"Found a JSON array of {width} numbers per row; the columns were named "
            "V1, V2, … — rename them in Column mapping."
        )
    else:
        df = pd.DataFrame(payload)
        notes.append("Converted a nested JSON array into a table; check the columns.")

    df.columns = [str(c).strip() for c in df.columns]
    return RawTable(
        df=df, source=source, format="json", encoding=encoding,
        notes=notes, n_rows_raw=len(df),
    )


# --------------------------------------------------------------------------- #
# Parquet / Stata / SPSS / RDS
# --------------------------------------------------------------------------- #
def read_parquet_bytes(data: bytes, source: str = "<upload>") -> RawTable:
    try:
        df = pd.read_parquet(io.BytesIO(data))
    except ImportError as exc:
        raise UnsupportedFormatError(
            plain="Reading Parquet needs the 'pyarrow' package, which is not installed.",
            fix="Install it (pip install pyarrow), or export the data to CSV.",
            technical=str(exc),
        ) from exc
    df.columns = [str(c).strip() for c in df.columns]
    return RawTable(df=df, source=source, format="parquet", n_rows_raw=len(df))


def read_stata_bytes(data: bytes, source: str = "<upload>") -> RawTable:
    try:
        df = pd.read_stata(io.BytesIO(data))
    except Exception as exc:
        raise UnsupportedFormatError(
            plain="That Stata file could not be read.",
            fix="In Stata, run: export delimited using \"mydata.csv\", then upload the CSV.",
            technical=str(exc),
        ) from exc
    df.columns = [str(c).strip() for c in df.columns]
    return RawTable(df=df, source=source, format="stata", n_rows_raw=len(df))


def read_spss_bytes(data: bytes, source: str = "<upload>") -> RawTable:
    try:
        import pyreadstat  # noqa: PLC0415
    except ImportError as exc:
        raise UnsupportedFormatError(
            plain="Reading SPSS files needs the 'pyreadstat' package, which is not installed.",
            fix="Install it (pip install pyreadstat), or in SPSS use File → Save As → CSV.",
            technical=str(exc),
        ) from exc
    try:
        df, _meta = pyreadstat.read_sav(io.BytesIO(data))
    except Exception as exc:  # pragma: no cover
        raise UnsupportedFormatError(
            plain="That SPSS file could not be read.",
            fix="In SPSS use File → Save As → CSV, then upload the CSV.",
            technical=str(exc),
        ) from exc
    df.columns = [str(c).strip() for c in df.columns]
    return RawTable(df=df, source=source, format="spss", n_rows_raw=len(df))


def read_rds_bytes(data: bytes, source: str = "<upload>") -> RawTable:
    try:
        import pyreadr  # noqa: PLC0415
    except ImportError as exc:
        raise UnsupportedFormatError(
            plain="Reading .rds files needs the 'pyreadr' package, which is not installed.",
            fix="Install it (pip install pyreadr), or in R run: write.csv(mydata, 'mydata.csv').",
            technical=str(exc),
        ) from exc
    path = Path(get_settings().upload_dir_resolved()) / f"{abs(hash(data))}.rds"
    try:
        path.write_bytes(data)
        result = pyreadr.read_r(str(path))
        frame = next(iter(result.values()))
        frame.columns = [str(c).strip() for c in frame.columns]
        return RawTable(df=frame, source=source, format="rds", n_rows_raw=len(frame))
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_READERS = {
    ".csv": read_csv_bytes,
    ".tsv": read_csv_bytes,
    ".txt": read_csv_bytes,
    ".xlsx": read_excel_bytes,
    ".xls": read_excel_bytes,
    ".ods": read_excel_bytes,
    ".json": read_json_bytes,
    ".parquet": read_parquet_bytes,
    ".dta": read_stata_bytes,
    ".sav": read_spss_bytes,
    ".rds": read_rds_bytes,
}


def read_any(data: bytes, filename: str = "upload.csv", sheet_name: str | int | None = None) -> RawTable:
    """
    Single entry point. Chooses a reader by extension (and, for Excel, sheet).
    Enforces the size limit before doing any work.
    """
    settings = get_settings()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise FileTooLargeError(
            plain=f"That file is {len(data) / 1e6:.1f} MB; the limit is {settings.max_upload_mb} MB.",
            fix="Upload a shorter date range, or split the series into several files.",
        )
    if not data:
        raise EmptyDatasetError(
            plain="The uploaded file is empty.",
            fix="Check the file on your computer and upload it again.",
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in _READERS:
        raise UnsupportedFormatError(
            plain=f"'{suffix or 'unknown'}' files are not supported.",
            fix=(
                "Supported formats: CSV, TSV, TXT, Excel (.xlsx/.xls), JSON, Parquet, "
                "Stata (.dta), SPSS (.sav), R (.rds) — or paste the table directly."
            ),
        )
    if suffix in {".xlsx", ".xls", ".ods"}:
        return read_excel_bytes(data, source=filename, sheet_name=sheet_name)
    return _READERS[suffix](data, source=filename)


def supported_extensions() -> Iterable[str]:
    return sorted(s.upper().lstrip(".") for s in _READERS)


__all__ = [
    "RawTable",
    "read_any",
    "read_csv_bytes",
    "read_excel_bytes",
    "read_json_bytes",
    "read_parquet_bytes",
    "read_stata_bytes",
    "read_spss_bytes",
    "read_rds_bytes",
    "read_pasted_text",
    "sniff_delimiter",
    "supported_extensions",
]
