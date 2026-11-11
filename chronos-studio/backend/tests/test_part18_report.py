"""
tests/test_part18_report.py — §5.13 / §8 Part 18 Reports & Finalization.

The point of these tests is not "a file was returned". They pin that the verdict
is one of the three exact strings, that every weighted row names a result and a
panel, that manuscript placeholders are filled from real numbers, that the HTML
is genuinely self-contained, and that every advertised deliverable opens as the
file type it claims.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import struct
import warnings
import zipfile
from pathlib import Path
from typing import Any

import pytest
from docx import Document
from openpyxl import load_workbook

from app.charts import exporters as E
from app.main import app
from app.ts.report.prose import PLACEHOLDER_EXAMPLE, TEMPLATES, fill_template
from app.ts.report.scorecard import NEARLY_TEXT, NOT_READY_TEXT, READY_TEXT
from app.ts.report.utils import fallback_for

BASE = "/api/v1"
ROOT = Path(__file__).resolve().parents[1]
SELECTED = ["fig.series", "table.descriptives"]
SECTION_TITLES = ("Methods", "Results", "Model Selection", "Diagnostics", "Forecast Accuracy")
ALL_FILES = (
    "chronos-report.html", "chronos-report.docx", "chronos-report.pdf",
    "chronos-tables.tex", "chronos-tables.xlsx", "reproduce_analysis.py",
    "reproduce_analysis.R", "session.chronos", "figures.zip", "status.json", "README.txt",
)


def _require_static_exports() -> None:
    """Skip the publication-export contract when its deliberately optional deps are absent."""
    import importlib.util

    if importlib.util.find_spec("kaleido") is None:
        pytest.skip("Kaleido is not installed; 300-dpi PNG/PDF exports degrade by design.")
    if importlib.util.find_spec("playwright") is None:
        pytest.skip("Playwright is not installed; print-CSS PDF export degrades by design.")
    from app.charts.exporters import _chrome_path

    try:
        _chrome_path()
    except Exception as exc:
        pytest.skip(f"Local Chromium is unavailable: {exc}")


def _force_offline_svg(monkeypatch) -> None:
    """Simulate the standalone wheelhouse, which has no Kaleido or Chromium."""
    def missing():
        raise ModuleNotFoundError("kaleido")

    monkeypatch.setattr(E, "_ensure_server", missing)


def _bad_const(value: str) -> None:
    raise AssertionError(f"non-finite JSON constant: {value}")


def _json(response: Any) -> dict[str, Any]:
    return json.loads(response.text, parse_constant=_bad_const)


def _post(client: Any, path: str, payload: dict[str, Any] | None, headers: dict[str, str]) -> Any:
    r = client.post(f"{BASE}{path}", json=payload or {}, headers=headers)
    assert r.status_code == 200, (path, r.text[:500])
    return r


def _get(client: Any, path: str, headers: dict[str, str]) -> Any:
    r = client.get(f"{BASE}{path}", headers=headers)
    assert r.status_code == 200, (path, r.text[:500])
    return r


@pytest.fixture(scope="module")
def report_run(client) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sid = client.post(f"{BASE}/sessions").json()["session_id"]
        h = {"X-Session-Id": sid}
        assert client.post(f"{BASE}/data/example", json={"example_id": "traffic_spend"}, headers=h).status_code == 201
        for path, payload in [
            ("profile", {}), ("advice", {}), ("decompose", {"method": "stl"}),
            ("tests", {}),
        ]:
            _post(client, f"/analysis/{path}", payload, h)
        # S6 → S7 → S8 → S9 is order-sensitive: GET /analysis/suggest needs
        # the gate, while POST /analysis/gate needs the next-step advisor.
        for path in ("correlate", "next", "facts"):
            _get(client, f"/analysis/{path}", h)
        _post(client, "/analysis/gate", {}, h)
        _post(client, "/analysis/suggest", {}, h)
        _get(client, "/analysis/suggest", h)
        _post(client, "/analysis/fit", {"model_ids": ["ets", "arima", "drift", "mean"]}, h)
        _post(client, "/analysis/select", {
            "horizon": 12, "folds": 4, "models": ["ets", "arima", "drift", "mean"],
        }, h)
        _post(client, "/analysis/diagnose", {}, h)
        _post(client, "/analysis/forecast", {"horizon": 6}, h)
        _post(client, "/analysis/evaluate", {
            "horizon": 6, "folds": 3, "n_paths": 2000, "reps": 3,
        }, h)
        basket = client.post(
            f"{BASE}/analysis/catalog/report",
            json={"action": "set", "item_ids": SELECTED}, headers=h,
        )
        assert basket.status_code == 200, basket.text[:400]
        assert basket.json()["count"] == len(SELECTED)
        finalized = client.post(f"{BASE}/export/finalize", headers=h)
        assert finalized.status_code == 200, finalized.text[:500]
        return {"sid": sid, "headers": h, "finalize": finalized.json()}


def test_export_router_and_source_boundaries(client):
    paths = app.openapi()["paths"]
    assert "/api/v1/export/status" in paths
    assert "/api/v1/export/finalize" in paths
    assert "/api/v1/export/{kind}" in paths
    assert client.get(f"{BASE}/health").json()["stages"] == 16
    assert [s["code"] for s in client.get(f"{BASE}/pipeline").json()["stages"]] == [f"S{i}" for i in range(16)]
    for name in (
        "__init__.py", "schemas.py", "utils.py", "scorecard.py", "prose.py",
        "bundle.py", "html.py", "docx.py", "pdf.py", "latex.py", "xlsx.py",
        "script.py", "archive.py", "figures.py", "all.py",
    ):
        assert (ROOT / "app" / "ts" / "report" / name).is_file()
    assert (ROOT / "app" / "api" / "export.py").is_file()
    assert '"app.api.export"' in (ROOT / "app" / "main.py").read_text()
    assert "app.api.export" not in (ROOT / "app" / "api" / "catalog.py").read_text()
    assert "playwright" in (ROOT / "pyproject.toml").read_text()


def test_empty_session_status_and_finalize_gate(client, session_id):
    h = {"X-Session-Id": session_id}
    r = _get(client, "/export/status", h)
    d = _json(r)
    assert d["stage_status"] == "not_started"
    assert d["scorecard"]["verdict"] == NOT_READY_TEXT
    assert d["scorecard"]["score"] == 0
    assert len(d["scorecard"]["items"]) == 10
    assert sum(x["weight"] for x in d["scorecard"]["items"]) == 100
    assert len(d["scorecard"]["next_steps"]) == 10
    assert all(x["result_ids"] and x["panel_id"] and x["panel_label"] for x in d["scorecard"]["items"])
    assert fallback_for("S5") in " ".join(x["evidence"] for x in d["scorecard"]["items"])
    bad = client.post(f"{BASE}/export/finalize", headers=h)
    assert bad.status_code == 409
    assert bad.json()["error"]["code"] == "session.stage_order"


def test_exact_verdict_strings_and_template_fallback():
    assert READY_TEXT == "Analysis complete — ready for manuscript writing"
    assert NEARLY_TEXT == "Nearly ready — 2 items outstanding"
    assert NOT_READY_TEXT == "Not yet — see blockers"
    assert PLACEHOLDER_EXAMPLE == "[PLACEHOLDER: statistic]"
    assert len(TEMPLATES) == 5
    assert all("[PLACEHOLDER:" in template for _, _, template, _ in TEMPLATES)
    filled = fill_template("[PLACEHOLDER: n] [PLACEHOLDER: missing]", {"n": "12"})
    assert filled.startswith("12 the app has not computed that yet — open the")
    assert "[PLACEHOLDER" not in filled


def test_nearly_ready_verdict_names_both_panels(client, report_run):
    """Exactly one or two outstanding checks must use the pinned §5.13 verdict."""
    from app.core.session import get_store
    from app.ts.report.scorecard import build_scorecard

    session = get_store().get(report_run["sid"])
    sc = build_scorecard(session)
    assert sc.status == "ready"
    # Remove the evidence for two weighted rows without recomputing any statistic.
    stage = session.state.stages["S1"]
    original_status = stage.status
    original_caveats = list(session.state.caveats)
    stage.status = "not_started"
    session.state.caveats.clear()
    try:
        nearly = build_scorecard(session)
        assert nearly.status == "nearly"
        assert nearly.verdict == NEARLY_TEXT
        assert len(nearly.outstanding) == 2
        assert nearly.blockers == []
        assert [x.number for x in nearly.next_steps] == [1, 2]
        assert {x.panel_id for x in nearly.next_steps} == {"S1", "S8"}
        assert all(x.panel_label and x.result_ids for x in nearly.next_steps)
        assert 0 < nearly.score < 100
    finally:
        stage.status = original_status
        session.state.caveats[:] = original_caveats
    assert build_scorecard(session).verdict == READY_TEXT


def test_catalog_cache_is_invalidated_after_a_data_change(monkeypatch, report_run):
    """A stale catalogue-id cache must not survive a new series version."""
    from app.charts import completeness as C
    from app.core.session import get_store
    from app.ts.report.scorecard import (
        CATALOG_CACHE_KEY,
        CATALOG_FINGERPRINT_KEY,
        _all_ids,
    )

    session = get_store().get(report_run["sid"])
    objects = session.objects
    missing = object()
    saved = {key: objects.get(key, missing) for key in (CATALOG_CACHE_KEY, CATALOG_FINGERPRINT_KEY)}
    original_version = session.state.series_version
    calls: list[str] = []

    def changed_catalog(_session: Any, *, mode: str = "light") -> dict[str, Any]:
        calls.append(mode)
        return {"scorecard": {}, "plots": [], "tables": [{"id": "catalog.after.change"}]}

    try:
        with monkeypatch.context() as patch:
            patch.setattr(C, "build_catalog", changed_catalog)
            objects.pop(CATALOG_CACHE_KEY, None)
            objects.pop(CATALOG_FINGERPRINT_KEY, None)
            assert "catalog.after.change" in _all_ids(session)
            assert len(calls) == 1
            # An unchanged session is allowed to reuse the fingerprinted cache.
            assert "catalog.after.change" in _all_ids(session)
            assert len(calls) == 1
            # A preprocessing change increments the series version and must rebuild.
            session.state.series_version += 1
            assert "catalog.after.change" in _all_ids(session)
            assert len(calls) == 2
    finally:
        session.state.series_version = original_version
        for key, value in saved.items():
            if value is missing:
                objects.pop(key, None)
            else:
                objects[key] = value


def test_a_genuine_catalogue_failure_is_not_reported_as_ready(monkeypatch, client, report_run):
    """A broken Part 16 catalogue must fail visibly, not become an empty ready report."""
    from fastapi.testclient import TestClient

    from app.charts import completeness as C
    from app.core.session import get_store
    from app.ts.report.bundle import build_bundle
    from app.ts.report.scorecard import CATALOG_CACHE_KEY, CATALOG_FINGERPRINT_KEY

    session = get_store().get(report_run["sid"])
    objects = session.objects
    missing = object()
    saved = {key: objects.get(key, missing) for key in (CATALOG_CACHE_KEY, CATALOG_FINGERPRINT_KEY)}

    def broken_catalog(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("catalogue is broken")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(C, "build_catalog", broken_catalog)
            objects.pop(CATALOG_CACHE_KEY, None)
            objects.pop(CATALOG_FINGERPRINT_KEY, None)
            with pytest.raises(RuntimeError, match="catalogue is broken"):
                build_bundle(session, mode="print")
            # The FastAPI boundary turns a real builder defect into a visible
            # 500; it does not return an empty catalogue with a ready verdict.
            with TestClient(app, raise_server_exceptions=False) as visible:
                r = visible.get(f"{BASE}/export/status", headers=report_run["headers"])
            assert r.status_code == 500
            assert "catalogue is broken" in r.text
    finally:
        for key, value in saved.items():
            if value is missing:
                objects.pop(key, None)
            else:
                objects[key] = value


def test_status_read_survives_a_concurrent_finalization_insertion():
    """A status read must not crash when finalization adds a new stage bucket."""
    from types import SimpleNamespace

    from app.charts import completeness as C
    from app.core.session import Session

    session = Session()

    class RaceResult:
        id = "race.trigger"
        kind = "narrative"
        plain = ""
        caption = ""

        def __init__(self, owner: Any) -> None:
            self.owner = owner
            self.fired = False

        @property
        def title(self) -> str:
            if not self.fired:
                self.fired = True
                # Emulate POST /export/finalize adding the S15 result bucket
                # while GET /export/status is reading the catalogue.
                self.owner.results["S15"] = {}
            return "Concurrent insertion"

    session.results["S1"]["race.trigger"] = RaceResult(session)
    session.results["S2"]["other"] = SimpleNamespace(
        id="other", kind="narrative", title="Other", plain="", caption="",
    )
    catalog = C.build_catalog(session, mode="print")
    assert catalog["plots"] == [] and catalog["tables"] == []
    assert "S15" in session.results
    assert {r.id for r in session.all_results()} == {"race.trigger", "other"}
    assert session.get_result("other") is not None


def test_manuscript_facts_use_catalog_rows_and_panel_fallbacks():
    """Template slots come from catalogue rows or named fallbacks, never guesses."""
    from app.core.session import Session
    from app.ts.report.prose import build_facts, manuscript_sections
    from app.ts.report.scorecard import build_scorecard
    from app.ts.report.utils import fallback_for

    session = Session()
    scorecard = build_scorecard(session)
    catalog = {"plots": [], "tables": [{"id": "table.descriptives", "rows": [
        {"quantity": "Observations (n)", "value": 123},
        {"quantity": "Mean", "value": 4.5},
        {"quantity": "Standard deviation", "value": 1.2},
    ]}]}
    facts = build_facts(session, catalog, scorecard)
    assert facts["n_observations"] == "123"
    assert facts["series_mean"] == "4.500"
    assert facts["series_sd"] == "1.200"
    sections = manuscript_sections(session, catalog, scorecard)
    assert len(sections) == 5
    assert all("[PLACEHOLDER" not in p for section in sections for p in section["paragraphs"])
    empty = manuscript_sections(session, {"plots": [], "tables": []}, scorecard)
    text = " ".join(p for section in empty for p in section["paragraphs"])
    assert fallback_for("S0") in text
    assert "[PLACEHOLDER" not in text


def test_finalization_does_not_pollute_travelling_caveats():
    """Report cards must not create the evidence for their own caveats check."""
    from app.core.schemas import NarrativeResult, Stage
    from app.core.session import Session
    from app.ts.report.scorecard import build_scorecard, finalize_session

    session = Session()
    dummy = NarrativeResult(
        id="dummy.quality", stage=Stage.S1_PROFILE, title="Stored result",
        plain="A stored result.", why="Regression input.", method="Constructed in the test.",
        paragraphs=["A stored paragraph."], reproducible_snippet="print('ok')",
    )
    session.add_result(dummy)
    session.state.caveats.append("Existing assumption")
    before = list(session.state.caveats)
    first = build_scorecard(session)
    assert first.status == "not_ready"

    finalized = finalize_session(session)
    assert finalized.status == first.status
    assert session.state.caveats == before
    assert build_scorecard(session).score == first.score
    cards = list(session.results[Stage.S15_FINALIZE].values())
    assert {card.id for card in cards} == {
        "report.readiness", "report.scorecard", "report.next_steps",
    }
    assert all(card.caveats == [] for card in cards)


def test_readiness_uses_stored_evidence_not_completed_stage_status_alone():
    """All stages marked complete without results still cannot be ready."""
    from app.core.schemas import Stage
    from app.core.session import Session
    from app.ts.report.scorecard import build_scorecard

    session = Session()
    for stage in Stage:
        session.mark_stage(stage, "complete")
    assert all(s.status == "complete" for s in session.state.stages.values())
    scorecard = build_scorecard(session)
    assert scorecard.status == "not_ready"
    assert scorecard.score == 0
    assert scorecard.verdict == NOT_READY_TEXT
    assert len(scorecard.next_steps) == 10


def test_rerunning_an_upstream_stage_marks_the_report_stale():
    """The pinned state machine must visibly invalidate downstream stages."""
    from app.core.schemas import PipelineState, Stage

    state = PipelineState(session_id="regression")
    state.mark(Stage.S14_EVALUATE, "complete")
    state.mark(Stage.S15_FINALIZE, "complete")
    state.mark(Stage.S13_FORECAST, "complete")
    assert state.stages["S13"].status == "complete"
    assert state.stages["S14"].status == "stale"
    assert state.stages["S15"].status == "stale"
    assert "Needs re-running" in state.stages["S15"].message


def test_full_run_readiness_and_finalize(client, report_run):
    h = report_run["headers"]
    d = _json(_get(client, "/export/status", h))
    s = d["scorecard"]
    assert s["verdict"] == READY_TEXT
    assert s["status"] == "ready" and s["ready"] is True
    assert s["score"] == s["max_score"] == 100
    assert s["outstanding"] == [] and s["blockers"] == [] and s["next_steps"] == []
    assert d["stage_status"] == "complete"
    assert d["selection_count"] == len(SELECTED)
    assert all(x["passed"] and x["available_result_ids"] and x["panel_label"] for x in s["items"])
    assert report_run["finalize"]["scorecard_verdict"] == READY_TEXT
    assert report_run["finalize"]["result_ids"] == [
        "report.readiness", "report.scorecard", "report.next_steps",
    ]
    results = _json(_get(client, f"/sessions/{report_run['sid']}/results", h))["S15"]
    assert {x["id"] for x in results} == set(report_run["finalize"]["result_ids"])
    assert all(ast.parse(x["reproducible_snippet"]) for x in results)


def test_selected_scope_status_and_export_index(client, report_run):
    h = report_run["headers"]
    d = _json(_get(client, "/export/status?mode=dark&scope=selected", h))
    assert d["mode"] == "dark" and d["scope"] == "selected"
    assert d["selection_count"] == len(SELECTED)
    assert d["catalog_counts"]["items"] == len(SELECTED)
    assert [x["kind"] for x in d["exports"]] == [
        "html", "docx", "pdf", "latex", "xlsx", "python", "r", "chronos", "figures", "all",
    ]
    assert "key" in d["privacy_note"] or "Gemini" in d["privacy_note"]
    index = _json(_get(client, "/export", h))
    assert index["stage"] == "S15"
    assert index["session_dep"] == "X-Session-Id"
    assert len(index["exports"]) == 10


def test_html_is_self_contained_and_filled(client, report_run, monkeypatch):
    # The standalone build has no Chromium. Force the pure-SVG path so this
    # contract is exercised in exactly that environment.
    _force_offline_svg(monkeypatch)
    h = report_run["headers"]
    r = _get(client, "/export/html?scope=selected&mode=print", h)
    assert r.headers["content-disposition"].startswith("inline;")
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-report-verdict"] == "ready"
    assert r.headers["x-report-ready"] == "true"
    text = r.text
    assert text.startswith("<!doctype html>")
    assert "data:image/svg+xml;base64," in text
    assert "Static figure unavailable" not in text
    assert "<script" not in text and "<link" not in text
    assert "http://" not in text and "https://" not in text
    assert "[PLACEHOLDER" not in text
    for title in SECTION_TITLES:
        assert title in text
    assert READY_TEXT in text
    assert "2.894" in text and "180" in text
    forced = _get(client, "/export/html?scope=selected&mode=print&download=1", h)
    assert forced.headers["content-disposition"].startswith("attachment;")
    assert b"GEMINI_API_KEY" not in forced.content


def test_docx_has_numbered_figures_tables_and_filled_prose(client, report_run):
    h = report_run["headers"]
    r = _get(client, "/export/docx?scope=selected", h)
    assert r.content[:2] == b"PK"
    doc = Document(io.BytesIO(r.content))
    body = "\n".join(
        [p.text for p in doc.paragraphs]
        + [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
    )
    assert "[PLACEHOLDER" not in body
    for title in SECTION_TITLES:
        assert title in body
    assert "Figure 1." in body and "Table 1." in body
    assert READY_TEXT in body
    assert "Manuscript-readiness scorecard" in body


def test_pdf_is_printed_from_html(client, report_run):
    _require_static_exports()
    r = _get(client, "/export/pdf?scope=selected", report_run["headers"])
    assert r.content.startswith(b"%PDF")
    assert len(r.content) > 3000
    assert r.headers["x-report-kind"] == "pdf"


def test_latex_snippets_are_thesis_ready(client, report_run):
    r = _get(client, "/export/latex?scope=selected", report_run["headers"])
    text = r.text
    assert "booktabs" in text and "longtable" in text
    assert "\\begin{longtable}" in text and "\\caption" in text and "\\label" in text
    assert "[PLACEHOLDER" not in text
    assert "report.scorecard" in text


def test_xlsx_has_one_sheet_per_table_and_readme(client, report_run):
    r = _get(client, "/export/xlsx?scope=selected", report_run["headers"])
    wb = load_workbook(io.BytesIO(r.content), read_only=True, data_only=True)
    assert wb.sheetnames[0] == "README"
    assert "Scorecard" in wb.sheetnames
    assert len(wb.sheetnames) == 3  # README + Scorecard + one selected catalogue table
    assert wb["README"]["B2"].value == READY_TEXT
    assert wb["Scorecard"].max_row >= 11
    assert wb[wb.sheetnames[2]].max_row > 1
    assert not any(
        "[PLACEHOLDER" in str(cell.value)
        for ws in wb.worksheets for row in ws.iter_rows() for cell in row
    )


def test_python_reproducible_script_parses(client, report_run):
    r = _get(client, "/export/python?scope=selected", report_run["headers"])
    text = r.text
    ast.parse(text)
    assert "CHRONOS_BASE_URL" in text and "/export/finalize" in text
    assert "stage_calls" in text and "data_hash" in text
    assert "GEMINI_API_KEY" not in text


def test_r_reproducible_script_is_optional_equivalent(client, report_run):
    r = _get(client, "/export/r?scope=selected", report_run["headers"])
    text = r.text
    assert "jsonlite" in text and "httr" in text
    assert "/data/example" in text and "/export/finalize" in text
    assert "GEMINI_API_KEY" not in text


def test_chronos_archive_contains_hash_settings_and_results(client, report_run):
    h = report_run["headers"]
    r = _get(client, "/export/chronos?scope=selected", h)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        expected = {
            "manifest.json", "data_hash.txt", "settings.json", "pipeline.json",
            "results.json", "catalog.json", "scorecard.json", "manuscript.json",
            "caveats.json", "data.csv", "README.txt",
        }
        assert expected <= names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["data_hash"] == zf.read("data_hash.txt").decode().strip()
        assert manifest["verdict"] == READY_TEXT
        settings = json.loads(zf.read("settings.json"))
        assert {"seed", "content_hash", "stage_calls", "library_versions", "preprocessing", "mapping"} <= set(settings)
        results = json.loads(zf.read("results.json"))
        assert any(x["id"] == "report.readiness" for x in results)
        catalog = json.loads(zf.read("catalog.json"))
        assert len(catalog["plots"]) >= 35 and len(catalog["tables"]) >= 30
        manuscript = json.loads(zf.read("manuscript.json"))
        assert len(manuscript) == 5
        assert all("[PLACEHOLDER" not in " ".join(x["paragraphs"]) for x in manuscript)
        assert b"GEMINI_API_KEY" not in b"".join(zf.read(n) for n in names if n.endswith(".json"))
    state = _json(_get(client, f"/sessions/{report_run['sid']}", h))
    assert state["pipeline"]["stages"]["S15"]["status"] == "complete"


def test_figures_are_png_svg_pdf_at_publication_resolution(client, report_run):
    _require_static_exports()
    r = _get(client, "/export/figures?scope=selected", report_run["headers"])
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        assert "errors.txt" not in names
        assert "captions.json" in names and "dpi.txt" in names and "manifest.json" in names
        stem = "fig.series"
        png = zf.read(f"figures/{stem}.png")
        svg = zf.read(f"figures/{stem}.svg")
        pdf = zf.read(f"figures/{stem}.pdf")
        width, _height = struct.unpack(">II", png[16:24])
        assert width == 3750  # 300 dpi at 1200 CSS px
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert b"<svg" in svg[:2000]
        assert pdf.startswith(b"%PDF")
        assert zf.read("dpi.txt").strip() == b"300"


def test_everything_bundle_contains_all_deliverables(client, report_run):
    _require_static_exports()
    r = _get(client, "/export/all?scope=selected", report_run["headers"])
    assert r.content[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        assert set(ALL_FILES) <= names
        assert "errors.txt" not in names
        status = json.loads(zf.read("status.json"))
        assert status["verdict"] == READY_TEXT
        assert status["stage_status"] == "complete"
        assert zf.read("chronos-report.html").startswith(b"<!doctype html>")
        assert zf.read("chronos-report.pdf").startswith(b"%PDF")
        assert b"GEMINI_API_KEY" not in zf.read("session.chronos")

def test_all_catalogue_html_embeds_every_offline_vector_figure(client, report_run, monkeypatch):
    """The guide's all-items HTML scope must not degrade to error callouts."""
    _force_offline_svg(monkeypatch)
    r = _get(client, "/export/html?scope=all&mode=print", report_run["headers"])
    text = r.text
    assert text.startswith("<!doctype html>")
    assert text.count("data:image/svg+xml;base64,") >= 35
    assert "Static figure unavailable" not in text
    assert "<script" not in text and "<link" not in text
    assert "http://" not in text and "https://" not in text


def test_figures_zip_keeps_svg_and_reports_png_pdf_limits(client, report_run, monkeypatch):
    """Optional publication formats fail explicitly; vector SVG still ships."""
    _force_offline_svg(monkeypatch)
    r = _get(client, "/export/figures?scope=selected", report_run["headers"])
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        assert "figures/fig.series.svg" in names
        assert b"<svg" in zf.read("figures/fig.series.svg")[:2000]
        assert "figures/fig.series.png" not in names
        assert "figures/fig.series.pdf" not in names
        errors = zf.read("errors.txt").decode("utf-8")
        assert "Kaleido" in errors
        assert "fig.series png" in errors and "fig.series pdf" in errors

