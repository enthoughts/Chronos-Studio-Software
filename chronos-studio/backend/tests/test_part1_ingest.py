"""
tests/test_part1_ingest.py — Part 1 verification.

The ingestion layer is where beginners fail, so these tests are behavioural:
each one encodes a real file shape that people actually upload, plus the failure
message a non-expert needs to see when it goes wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.errors import (
    ChronosError,
    EmptyDatasetError,
    NonNumericSeriesError,
    TimeIndexError,
    UnsupportedFormatError,
)
from app.core.schemas import DataClass, FrequencySpec
from app.ts.io.ingest import (
    ColumnMapping,
    ingest_bytes,
    ingest_table,
    ingest_text,
    suggest_mapping,
)
from app.ts.io.readers import read_any, read_json_bytes, sniff_delimiter
from app.ts.io.timeindex import (
    build_from_year_month,
    detect_time_column,
    infer_frequency,
    parse_time_column,
    regularise,
)
from app.ts.io.validate import (
    coerce_numeric,
    detect_data_class,
    longest_constant_run,
    missing_pattern,
    robust_outlier_count,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "app" / "assets" / "examples"
CSV = 'text/csv'


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _monthly(n: int = 60, start: str = "2015-01-01", col: str = "Date", val: str = "Value") -> bytes:
    idx = pd.date_range(start, periods=n, freq="MS")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: idx.strftime("%Y-%m-%d"), val: 100 + np.arange(n) + rng.normal(0, 2, n)})
    return _csv(df)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def test_csv_with_semicolons_and_decimal_commas():
    text = "Date;Umsatz\n2020-01-01;1.234,5\n2020-02-01;1.300,0\n2020-03-01;1.410,25\n2020-04-01;1.500,0\n"
    r = ingest_bytes(text.encode("utf-8"), "german.csv")
    assert r.dataset.frequency.code == "MS"
    assert r.dataset.n_observations == 4
    vals = r.series.dropna().tolist()
    assert vals[0] == pytest.approx(1234.5)
    assert any("decimal" in i.plain.lower() for i in r.issues)


def test_tab_separated_and_tsv_extension():
    text = "Date\tValue\n2020-01-01\t10\n2020-02-01\t12\n2020-03-01\t14\n2020-04-01\t16\n"
    r = ingest_bytes(text.encode(), "x.tsv")
    assert r.dataset.n_observations == 4


def test_json_array_of_objects():
    payload = json.dumps(
        [{"date": f"2020-{m:02d}-01", "v": m * 10} for m in range(1, 13)]
    ).encode()
    r = ingest_bytes(payload, "x.json")
    assert r.dataset.n_observations == 12
    assert r.dataset.frequency.code == "MS"


def test_json_wrapped_in_data_key_and_flat_number_list():
    wrapped = json.dumps({"data": [{"t": "2020-01-01", "v": 1},
                                   {"t": "2020-02-01", "v": 2},
                                   {"t": "2020-03-01", "v": 3},
                                   {"t": "2020-04-01", "v": 4}]}).encode()
    assert ingest_bytes(wrapped, "w.json").dataset.n_observations == 4

    t = read_json_bytes(json.dumps([1, 2, 3, 4, 5]).encode())
    assert list(t.df.columns) == ["value"]


def test_excel_roundtrip(tmp_path):
    import openpyxl  # noqa: F401  (raises if absent, which the app reports)

    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"Month": idx, "Sales": np.arange(24) * 3 + 50})
    path = tmp_path / "book.xlsx"
    df.to_excel(path, index=False, sheet_name="Data")
    r = ingest_bytes(path.read_bytes(), "book.xlsx")
    assert r.dataset.n_observations == 24
    assert r.dataset.frequency.code == "MS"


def test_multi_sheet_excel_picks_the_populated_sheet(tmp_path):
    idx = pd.date_range("2020-01-01", periods=30, freq="MS")
    big = pd.DataFrame({"Month": idx, "Value": np.arange(30)})
    small = pd.DataFrame({"note": ["read me"]})
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path) as xw:
        small.to_excel(xw, sheet_name="Cover", index=False)
        big.to_excel(xw, sheet_name="Series", index=False)
    r = ingest_bytes(path.read_bytes(), "multi.xlsx")
    assert r.dataset.n_observations == 30


def test_unsupported_extension_has_an_actionable_message():
    with pytest.raises(UnsupportedFormatError) as ei:
        read_any(b"whatever", "slides.pptx")
    assert ".pptx" in ei.value.plain
    assert "CSV" in ei.value.fix


def test_empty_file_and_empty_body():
    with pytest.raises(ChronosError):
        read_any(b"", "x.csv")
    with pytest.raises(EmptyDatasetError):
        ingest_text("   \n  \n")


def test_sniffer_handles_single_column():
    assert sniff_delimiter("value\n1\n2\n3\n") == ","


# --------------------------------------------------------------------------- #
# Time index parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected_first",
    [
        (["2020-01-01", "2020-02-01"], "2020-01-01"),
        (["2020/01/01", "2020/02/01"], "2020-01-01"),
        (["01.01.2020", "01.02.2020"], "2020-01-01"),
        (["2020-01-01 00:00:00", "2020-02-01 00:00:00"], "2020-01-01"),
    ],
)
def test_common_date_formats(raw, expected_first):
    idx, _notes, bad = parse_time_column(pd.Series(raw))
    assert bad == 0
    assert str(idx[0].date()) == expected_first


def test_quarter_labels():
    idx, notes, bad = parse_time_column(pd.Series(["1994Q1", "1994Q2", "1994Q3", "1994Q4"]))
    assert bad == 0
    assert [f"{d.year}-{d.month:02d}" for d in idx] == ["1994-01", "1994-04", "1994-07", "1994-10"]
    assert any("quarter" in n.lower() for n in notes)


def test_epoch_seconds_and_millis():
    secs = pd.Series([1609459200, 1609545600, 1609632000, 1609718400])
    idx, notes, _ = parse_time_column(secs)
    assert str(idx[0].date()) == "2021-01-01"
    assert any("Unix" in n for n in notes)

    millis = secs * 1000
    idx2, _, _ = parse_time_column(millis)
    assert str(idx2[0].date()) == "2021-01-01"


def test_excel_serial_dates():
    idx, notes, bad = parse_time_column(pd.Series([43831, 43832, 43833, 43834]), "date")
    assert bad == 0
    assert str(idx[0].date()) == "2020-01-01"
    assert any("Excel serial" in n for n in notes)


def test_year_and_month_columns_build_a_clean_index():
    idx = build_from_year_month(pd.Series([2020, 2020, 2020, 2021]), pd.Series([1, 2, 3, 1]))
    assert [f"{d.year}-{d.month:02d}" for d in idx] == ["2020-01", "2020-02", "2020-03", "2021-01"]


def test_dayfirst_ambiguity_is_reported_only_when_provable():
    _idx, notes, _ = parse_time_column(pd.Series(["03/04/2020", "04/04/2020", "13/04/2020"]))
    assert any("ambiguous" in n.lower() or "day/month" in n.lower() for n in notes)

    _idx2, notes2, _ = parse_time_column(pd.Series(["2020-01-01", "2020-02-01"]))
    assert not any("ambiguous" in n.lower() for n in notes2)


def test_only_parsed_part_of_a_messy_date_column_is_used():
    idx, notes, bad = parse_time_column(pd.Series(["2020-01-01", "not a date", "2020-03-01"]))
    assert bad == 1
    assert idx.isna().sum() == 1
    assert any("could not be parsed" in n for n in notes)


def test_no_time_column_raises_with_a_usable_instruction():
    with pytest.raises(TimeIndexError) as ei:
        parse_time_column(pd.Series(["apple", "banana", "cherry"]), "notes")
    assert "date" in ei.value.fix.lower()


def test_a_column_named_date_holding_words_is_not_quietly_accepted():
    """A weak name hint must not turn a text column into a time index."""
    with pytest.raises(TimeIndexError):
        parse_time_column(pd.Series(["apple", "banana", "cherry", "date"]), "date")


# --------------------------------------------------------------------------- #
# Frequency inference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "freq,expected",
    [("MS", "MS"), ("QS", "QS"), ("YS", "YS"), ("D", "D"), ("W", "W")],
)
def test_frequency_inference(freq, expected):
    idx = pd.date_range("2020-01-01", periods=40, freq=freq)
    spec, _candidates = infer_frequency(idx)
    assert spec.code == expected


def test_frequency_inference_with_gaps_still_lands_on_monthly():
    """Two missing months must not shift the calendar to something exotic."""
    idx = pd.date_range("2020-01-01", periods=24, freq="MS").delete([5, 6])
    spec, candidates = infer_frequency(idx)
    assert spec.code == "MS"
    assert spec in candidates


def test_semimonthly_data_is_reported_at_low_confidence_not_rounded_away():
    """Twice-monthly dates fit no standard calendar: we offer one, and say we are unsure."""
    firsts = pd.date_range("2020-01-01", periods=24, freq="MS")
    idx = pd.DatetimeIndex(sorted([*firsts, *(d + pd.Timedelta(days=14) for d in firsts)]))
    spec, candidates = infer_frequency(idx)
    assert spec.confidence == "low"
    assert "confirm" in spec.evidence.lower()
    assert candidates


def test_irregular_dates_get_a_low_confidence_calendar_not_a_crash():
    rng = np.random.default_rng(1)
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(rng.choice(1500, 60, replace=False), origin="2016-01-01", unit="D")))
    spec, _cands = infer_frequency(idx)
    assert spec.code in {"D", "W", "MS", "QS", "YS"}
    assert spec.confidence == "low"


def test_regularise_inserts_gaps_instead_of_shifting_values():
    vals = pd.Series(
        [1.0, 2.0, 10.0, 11.0],
        index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-05-01", "2020-06-01"]),
    )
    spec = FrequencySpec(code="MS", label="Monthly", period=12, confidence="high")
    r = regularise(vals, spec)
    assert len(r.series) == 6
    assert r.n_reindexed == 2
    assert pd.isna(r.series.loc["2020-03-01"])
    assert r.series.loc["2020-05-01"] == 10.0     # value stayed on its own month


def test_regularise_aggregates_duplicates_and_says_so():
    vals = pd.Series(
        [1.0, 3.0, 5.0],
        index=pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"]),
    )
    spec = FrequencySpec(code="MS", label="Monthly", period=12, confidence="high")
    r = regularise(vals, spec, dedupe="mean")
    assert r.n_duplicates == 1
    assert r.series.loc["2020-01-01"] == pytest.approx(2.0)
    assert any("duplicate" in n.lower() for n in r.notes)


# --------------------------------------------------------------------------- #
# Numeric coercion
# --------------------------------------------------------------------------- #
def test_currency_percent_and_parens_are_cleaned():
    s = pd.Series(["$1,234.50", "€ 99", "45%", "(120)", "₱1,000"])
    numeric, issues = coerce_numeric(s, "revenue")
    assert numeric.tolist() == pytest.approx([1234.5, 99.0, 45.0, -120.0, 1000.0])
    assert any(i.code == "data.paren_negatives" for i in issues)


def test_text_column_raises_with_examples_of_what_was_found():
    with pytest.raises(NonNumericSeriesError) as ei:
        coerce_numeric(pd.Series(["alpha", "beta"]), "notes")
    assert "notes" in ei.value.plain
    assert "alpha" in ei.value.fix


def test_partially_bad_column_reports_instead_of_dropping_silently():
    numeric, issues = coerce_numeric(pd.Series(["1", "2", "N/A", "4", "oops"]), "v")
    assert numeric.notna().sum() == 3
    assert any(i.code == "data.unparseable_values" and i.rows_affected == 1 for i in issues)


# --------------------------------------------------------------------------- #
# Data class detection — this gates the model registry (§5.8)
# --------------------------------------------------------------------------- #
def test_data_class_detection_across_the_branches():
    rng = np.random.default_rng(7)
    assert detect_data_class(pd.Series(rng.normal(0, 1, 100))).data_class is DataClass.CONTINUOUS
    # low-rate counts (no heavy tail) → count branch
    assert detect_data_class(pd.Series(rng.poisson(4, 80).astype(float))).data_class is DataClass.COUNT
    # high-magnitude integers (passenger numbers, monthly deaths) → continuous,
    # with the count branch offered as an explicit override
    high = detect_data_class(pd.Series(rng.poisson(150, 80).astype(float)))
    assert high.data_class is DataClass.CONTINUOUS
    assert high.evidence["count_branch_available"] is True
    assert detect_data_class(
        pd.Series(np.where(rng.random(100) < 0.3, 4, 0).astype(float))
    ).data_class is DataClass.INTERMITTENT
    assert detect_data_class(pd.Series(rng.integers(0, 2, 60).astype(float))).data_class is (
        DataClass.BINARY_OR_PROPORTION
    )


def test_zero_heavy_binary_is_binary_not_intermittent():
    """A 0/1 series is often >30% zeros; it must not be misfiled as intermittent demand."""
    s = pd.Series(np.r_[np.zeros(70), np.ones(30)])
    assert detect_data_class(s).data_class is DataClass.BINARY_OR_PROPORTION


def test_intermittent_message_names_the_consequence():
    s = pd.Series(np.where(np.arange(100) % 4 == 0, 5.0, 0.0))
    verdict = detect_data_class(s)
    assert verdict.data_class is DataClass.INTERMITTENT
    assert "Croston" in verdict.plain and "MASE" in verdict.plain
    assert verdict.evidence["pct_zeros"] > 0.3


# --------------------------------------------------------------------------- #
# Quality measures
# --------------------------------------------------------------------------- #
def test_missing_pattern_finds_the_longest_run_and_its_date():
    idx = pd.date_range("2020-01-01", periods=10, freq="MS")
    s = pd.Series([1, 2, np.nan, np.nan, np.nan, 6, 7, 8, 9, 10], index=idx, dtype=float)
    mp = missing_pattern(s)
    assert mp.n_missing == 3
    assert mp.longest_run == 3
    assert "2020-03" in (mp.longest_run_start or "")
    assert mp.gaps_over_two == 1


def test_robust_outlier_count_ignores_a_heavy_tail_it_did_not_create():
    rng = np.random.default_rng(3)
    clean = pd.Series(rng.normal(0, 1, 500))
    assert robust_outlier_count(clean) <= 5
    spiked = pd.Series(np.r_[rng.normal(0, 1, 500), [40.0, -45.0]])
    assert robust_outlier_count(spiked) >= 2


def test_longest_constant_run():
    assert longest_constant_run(pd.Series([1, 1, 1, 2, 3, 3, 4])) == 3


# --------------------------------------------------------------------------- #
# End-to-end ingest
# --------------------------------------------------------------------------- #
def test_full_ingest_of_a_clean_monthly_file():
    r = ingest_bytes(_monthly(72), "sales.csv")
    ds = r.dataset
    assert ds.n_observations == 72
    assert ds.frequency.code == "MS"
    assert ds.frequency.period == 12
    assert ds.quality.is_regular
    assert ds.content_hash and len(ds.content_hash) == 16
    assert len(ds.preview_rows) > 0
    assert ds.quality.plain_summary


def test_row_order_fallback_warns_loudly_about_the_axis():
    df = pd.DataFrame({"Value": np.arange(30.0)})
    r = ingest_table(read_json_bytes(df.to_json().encode()), name="roworder")
    assert r.dataset.n_observations == 30
    assert any("row" in i.plain.lower() for i in r.issues)


def test_year_month_columns_beat_date_parsing():
    df = pd.DataFrame({"Year": [2019, 2019, 2020, 2020], "Month": [1, 7, 1, 7], "Sales": [1.0, 2, 3, 4]})
    mapping, _issues, notes = suggest_mapping(df)
    assert mapping.year_column == "Year" and mapping.month_column == "Month"
    assert mapping.target_column == "Sales"
    r = ingest_table(read_json_bytes(df.to_json().encode()))
    assert r.dataset.frequency.code == "MS"
    assert r.dataset.n_observations == 4
    assert notes


def test_a_date_valued_year_column_is_not_treated_as_a_year_column():
    """Regression: a column named 'Year' holding 1821-01-01 must not become a year index."""
    idx = pd.date_range("1821-01-01", periods=20, freq="YS")
    df = pd.DataFrame({"Year": idx.strftime("%Y-%m-%d"), "Trappings": np.arange(20.0) + 100})
    r = ingest_table(read_json_bytes(df.to_json().encode()))
    assert r.dataset.n_observations == 20
    assert r.dataset.frequency.code == "YS"


def test_too_short_series_is_refused_with_advice():
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-02-01"], "Value": [1.0, 2.0]})
    with pytest.raises(ChronosError) as ei:
        ingest_table(read_json_bytes(df.to_json().encode()))
    assert ei.value.code == "prep.insufficient_data"
    assert "seasonal" in ei.value.fix.lower()


def test_short_series_gets_a_warning_even_when_it_loads():
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=8, freq="MS").strftime("%Y-%m-%d"),
        "Value": np.arange(8.0),
    })
    r = ingest_table(read_json_bytes(df.to_json().encode()))
    assert any(i.code == "data.short_series" for i in r.issues)


def test_exogenous_columns_are_carried_alongside_the_target():
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    df = pd.DataFrame({
        "Date": idx.strftime("%Y-%m-%d"),
        "Sales": np.arange(36.0) * 2 + 10,
        "Price": np.linspace(9.9, 11.0, 36),
        "Promo": np.tile([0, 1], 18),
    })
    r = ingest_bytes(_csv(df), "exog.csv")
    assert r.dataset.target_column == "Sales"
    assert set(r.dataset.exogenous_columns) == {"Price", "Promo"}
    assert r.exogenous is not None and r.exogenous.shape[1] == 2


def test_duplicate_timestamps_are_reported_in_the_quality_report():
    df = pd.DataFrame({
        "Date": ["2020-01-01", "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        "Value": [1.0, 3.0, 4.0, 5.0, 6.0],
    })
    r = ingest_bytes(_csv(df), "dupes.csv")
    assert r.dataset.quality.n_duplicate_timestamps == 1
    assert any(i.code == "data.duplicates" for i in r.issues)


def test_invalid_mapping_raises_a_mapping_error_not_a_stack_trace():
    csv = _monthly(24)
    with pytest.raises(ChronosError) as ei:
        ingest_bytes(csv, "x.csv", mapping=ColumnMapping(time_column="Nope", target_column="Value"))
    assert ei.value.code == "data.column_mapping"
    assert "Nope" in ei.value.plain


# --------------------------------------------------------------------------- #
# Bundled examples — each must load and be detected as documented
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("example_id", [
    "airpassengers", "usaccdeaths", "lynx", "wwwusage", "johnsonjohnson", "sunspots",
    "ausbeer", "electricity_daily", "retail_sales", "outbreak_counts", "spare_parts",
    "structural_break", "random_walk", "white_noise",
])
def test_every_bundled_example_loads(example_id):
    manifest = json.loads((EXAMPLES / "manifest.json").read_text())
    meta = {m["id"]: m for m in manifest}[example_id]
    r = ingest_bytes((EXAMPLES / meta["file"]).read_bytes(), meta["file"], name=meta["title"])
    assert r.dataset.n_observations == meta["n_rows"]
    assert r.dataset.data_class.value == meta["data_class"], (
        f"{example_id}: detected {r.dataset.data_class.value}, manifest says {meta['data_class']}"
    )
    assert r.dataset.frequency.code == meta["frequency"]
    assert meta["teaches"], "every example must say what it teaches"


def test_intermittent_example_really_is_intermittent():
    r = ingest_bytes((EXAMPLES / "spare_parts.csv").read_bytes(), "spare_parts.csv")
    assert r.dataset.data_class is DataClass.INTERMITTENT
    assert r.dataset.quality.pct_zeros > 0.5


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #
def test_data_formats_endpoint_lists_readers_and_calendars(client):
    body = client.get("/api/v1/data/formats").json()
    assert "CSV" in body["extensions"] and "XLSX" in body["extensions"]
    codes = [c["code"] for c in body["calendars"]]
    assert {"MS", "QS", "YS", "D", "W"} <= set(codes)


def test_examples_endpoint_returns_teaching_notes(client):
    examples = client.get("/api/v1/data/examples").json()["examples"]
    assert len(examples) >= 12
    assert all(e["teaches"] for e in examples)
    assert all(e["available"] for e in examples)


def test_upload_sets_the_dataset_and_completes_s0(client, session_id):
    r = client.post(
        "/api/v1/data/upload",
        files={"file": ("sales.csv", _monthly(48), CSV)},
        headers={"X-Session-Id": session_id},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["dataset"]["n_observations"] == 48
    assert body["dataset"]["frequency"]["code"] == "MS"
    assert body["next"] == "S1"

    state = client.get(f"/api/v1/sessions/{session_id}").json()
    assert state["pipeline"]["stages"]["S0"]["status"] == "complete"
    assert state["dataset"]["content_hash"]


def test_load_example_endpoint(client, session_id):
    r = client.post(
        "/api/v1/data/example",
        json={"example_id": "airpassengers"},
        headers={"X-Session-Id": session_id},
    )
    assert r.status_code == 201
    assert r.json()["dataset"]["n_observations"] == 144
    assert r.json()["dataset"]["source"] == "example"


def test_unknown_example_is_a_friendly_404(client, session_id):
    r = client.post(
        "/api/v1/data/example",
        json={"example_id": "nope"},
        headers={"X-Session-Id": session_id},
    )
    assert r.status_code == 400
    assert "nope" in r.json()["error"]["plain"]


def test_paste_endpoint(client, session_id):
    text = "Date,Value\n" + "\n".join(
        f"2020-{m:02d}-01,{m * 5}" for m in range(1, 13)
    )
    r = client.post(
        "/api/v1/data/paste",
        json={"text": text},
        headers={"X-Session-Id": session_id},
    )
    assert r.status_code == 201
    assert r.json()["dataset"]["n_observations"] == 12


def test_inspect_then_remap_flow(client, session_id):
    csv = _monthly(36)
    ins = client.post("/api/v1/data/inspect", files={"file": ("a.csv", csv, CSV)})
    assert ins.status_code == 200
    guess = ins.json()["suggested_mapping"]
    assert guess["target_column"] == "Value"

    client.post("/api/v1/data/upload", files={"file": ("a.csv", csv, CSV)},
                headers={"X-Session-Id": session_id})
    remap = client.post(
        "/api/v1/data/remap",
        json={"mapping": {"time_column": "Date", "target_column": "Value"}},
        headers={"X-Session-Id": session_id},
    )
    assert remap.status_code == 200
    assert remap.json()["dataset"]["n_observations"] == 36


def test_remap_without_upload_is_a_clear_error(client, session_id):
    r = client.post(
        "/api/v1/data/remap",
        json={"mapping": {"time_column": "Date"}},
        headers={"X-Session-Id": session_id},
    )
    assert r.status_code == 422
    assert "upload" in r.json()["error"]["fix"].lower()


def test_current_dataset_endpoint(client, session_id):
    assert client.get("/api/v1/data/current", headers={"X-Session-Id": session_id}).json()["dataset"] is None
    client.post("/api/v1/data/example", json={"example_id": "lynx"},
                headers={"X-Session-Id": session_id})
    body = client.get("/api/v1/data/current", headers={"X-Session-Id": session_id}).json()
    assert body["dataset"]["n_observations"] == 114
    assert body["series_head"][0]["value"] is not None


def test_missing_session_header_is_rejected(client):
    r = client.get("/api/v1/data/current")
    assert r.status_code == 422


def test_upload_requires_a_session(client):
    r = client.post("/api/v1/data/upload", files={"file": ("a.csv", _monthly(20), CSV)})
    assert r.status_code == 422
