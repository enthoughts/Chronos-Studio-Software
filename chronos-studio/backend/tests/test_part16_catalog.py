"""
tests/test_part16_catalog.py — §5.10 / §8 Part 16 completeness pass.

The failure modes being pinned here are the ones a pretty screenshot hides:

* a scorecard that claims 35 plots but cannot name them,
* a figure with no caption, alt text or data-behind table,
* an "export" that is only the interactive PNG modebar,
* a new S16 stage smuggled into the 16-stage contract,
* a GARCH/PMF/VAR family shown as a generic empty card,
* a dark/print mode that quietly keeps the light theme, and
* a Part 18 whole-report router duplicating Part 16's per-item exports.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import re
import struct
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.charts import exporters as E
from app.charts.svg_fallback import SVG_NS
from app.charts.theme import (
    DASHES,
    FONT_STACK,
    LEGEND_ITEMS_PER_ROW,
    LEGEND_ROW_HEIGHT,
    PALETTE,
    PALETTE_VARIANTS,
    TITLE_TOP_MARGIN,
    TITLE_Y,
    apply,
    colours,
    restyled,
    to_json,
    with_alpha,
)
from app.main import app

BASE = "/api/v1"
ROOT = Path(__file__).resolve().parents[1]
ALL_FORMATS = ("png", "svg", "pdf", "xlsx", "csv")
STAGES = [f"S{i}" for i in range(16)]


def _bad_const(value: str) -> None:
    raise AssertionError(f"non-finite JSON constant: {value}")


def _item(catalog: dict[str, Any], item_id: str) -> dict[str, Any]:
    for item in catalog["plots"] + catalog["tables"]:
        if item["id"] == item_id:
            return item
    raise AssertionError(f"missing catalogue item {item_id}")


def _prepared(client: TestClient, example: str) -> tuple[str, dict[str, str]]:
    sid = client.post(f"{BASE}/sessions").json()["session_id"]
    headers = {"X-Session-Id": sid}
    assert client.post(f"{BASE}/data/example", json={"example_id": example}, headers=headers).status_code == 201
    assert client.post(f"{BASE}/analysis/profile", json={}, headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/advice", json={}, headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/decompose", json={"method": "stl"}, headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/tests", json={}, headers=headers).status_code == 200
    assert client.get(f"{BASE}/analysis/correlate", headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/gate", json={}, headers=headers).status_code == 200
    assert client.post(f"{BASE}/analysis/suggest", json={}, headers=headers).status_code == 200
    return sid, headers


def _models_ready(
    client: TestClient,
    headers: dict[str, str],
    fit_models: list[str],
    select_models: list[str] | None = None,
    *,
    horizon: int = 6,
    folds: int = 3,
) -> None:
    fit = client.post(f"{BASE}/analysis/fit", json={"model_ids": fit_models}, headers=headers)
    assert fit.status_code == 200, fit.text[:400]
    sel = client.post(
        f"{BASE}/analysis/select",
        json={"horizon": horizon, "folds": folds, "models": select_models or fit_models},
        headers=headers,
    )
    assert sel.status_code == 200, sel.text[:500]
    diag = client.post(f"{BASE}/analysis/diagnose", json={}, headers=headers)
    assert diag.status_code == 200, diag.text[:400]


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


def _forecast(client: TestClient, headers: dict[str, str], horizon: int = 6) -> None:
    r = client.post(f"{BASE}/analysis/forecast", json={"horizon": horizon}, headers=headers)
    assert r.status_code == 200, r.text[:600]


@pytest.fixture(scope="module")
def full_run(client) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sid, headers = _prepared(client, "traffic_spend")
        _models_ready(
            client, headers,
            ["ets", "arima", "drift", "mean", "var"],
            ["ets", "arima", "drift", "mean"],
        )
        _forecast(client, headers)
        evaluated = client.post(
            f"{BASE}/analysis/evaluate",
            json={"horizon": 6, "folds": 3, "n_paths": 2000, "reps": 3},
            headers=headers,
        )
        assert evaluated.status_code == 200, evaluated.text[:700]
        r = client.get(f"{BASE}/analysis/catalog?mode=light", headers=headers)
    assert r.status_code == 200, r.text[:500]
    return {"sid": sid, "headers": headers, "catalog": r.json(), "text": r.text}


@pytest.fixture(scope="module")
def branch_catalogs(client) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cases = [
        ("variance", "volatile_returns", ["garch", "naive"]),
        ("count", "outbreak_counts", ["poisson", "neg_binomial"]),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, example, models in cases:
            sid, headers = _prepared(client, example)
            _models_ready(client, headers, models)
            _forecast(client, headers)
            r = client.get(f"{BASE}/analysis/catalog?mode=light", headers=headers)
            assert r.status_code == 200, r.text[:400]
            out[label] = {"sid": sid, "headers": headers, "catalog": r.json()}
    return out


# --------------------------------------------------------------------------- #
# Scorecard and contract
# --------------------------------------------------------------------------- #
def test_catalog_meets_the_35_plot_30_table_scorecard(full_run):
    catalog = full_run["catalog"]
    score = catalog["scorecard"]
    json.loads(full_run["text"], parse_constant=_bad_const)
    assert score["plots"] >= 35
    assert score["tables"] >= 30
    assert score["min_plots"] == 35 and score["min_tables"] == 30
    assert all(score[k] for k in (
        "plots_ok", "tables_ok", "captions_ok", "alt_text_ok", "data_behind_ok",
    ))
    assert len(catalog["groups"]) == 9
    assert set(score["groups"]) == set(catalog["groups"])
    ids = [x["id"] for x in catalog["plots"] + catalog["tables"]]
    assert len(ids) == len(set(ids))
    assert all(x["stage"] in STAGES for x in catalog["plots"] + catalog["tables"])


def test_every_item_is_captioned_accessible_and_has_downloadable_data(full_run):
    catalog = full_run["catalog"]
    items = catalog["plots"] + catalog["tables"]
    assert len(items) >= 65
    for item in items:
        assert item["caption"].strip()
        assert len(item["alt_text"].strip()) > 20
        assert item["rows"], item["id"]
        assert item["columns"], item["id"]
        assert set(item["formats"]) == set(ALL_FORMATS)
        assert item["dpi"] == 300
        assert item["report_insertable"] is True
        assert item["source"].strip()
        figure = item["figure_json"]
        assert isinstance(figure.get("data"), list) and figure["data"], item["id"]
        assert figure.get("layout", {}).get("font", {}).get("family") == FONT_STACK
    for plot in catalog["plots"]:
        assert plot["kind"] == "plot"
        assert len(plot["alt_text"]) > 20


def test_catalog_is_strict_finite_json_and_groups_are_not_fabricated(client, session_id):
    headers = {"X-Session-Id": session_id}
    r = client.get(f"{BASE}/analysis/catalog", headers=headers)
    assert r.status_code == 200
    catalog = json.loads(r.text, parse_constant=_bad_const)
    assert catalog["scorecard"]["plots"] == 0
    assert catalog["scorecard"]["tables"] == 0
    assert not catalog["scorecard"]["plots_ok"]
    assert not _has_item(catalog, "models.garch_variance_path")
    assert all(v == {"plots": 0, "tables": 0} for v in catalog["scorecard"]["groups"].values())


def _has_item(catalog: dict[str, Any], item_id: str) -> bool:
    return any(x["id"] == item_id for x in catalog["plots"] + catalog["tables"])


def test_catalog_does_not_add_a_stage_and_part18_consumes_it(full_run):
    paths = app.openapi()["paths"]
    assert "/api/v1/analysis/catalog" in paths
    assert "/api/v1/analysis/catalog/report" in paths
    assert "/api/v1/analysis/catalog/{item_id}/download" in paths
    # Part 18 mounts the whole-report router; Part 16 still owns per-item exports.
    assert "/api/v1/export/status" in paths
    assert "/api/v1/export/finalize" in paths
    assert "/api/v1/export/{kind}" in paths
    assert client_health_stages(full_run) == 16
    assert [s["code"] for s in client_get(full_run, "/pipeline")["stages"]] == STAGES
    results = client_get(full_run, f"/sessions/{full_run['sid']}/results")
    assert sorted(results.keys()) == sorted(STAGES)
    state = client_get(full_run, f"/sessions/{full_run['sid']}")
    assert state["pipeline"]["stages"]["S15"]["status"] == "not_started"


def client_get(full_run: dict[str, Any], path: str) -> dict[str, Any]:
    # Local import keeps the module-scope fixture data serializable.
    from fastapi.testclient import TestClient

    c = TestClient(app)
    r = c.get(f"{BASE}{path}")
    assert r.status_code == 200, r.text[:300]
    return r.json()


def client_health_stages(full_run: dict[str, Any]) -> int:
    return int(client_get(full_run, "/health")["stages"])


# --------------------------------------------------------------------------- #
# Styling, modes and single-source rule
# --------------------------------------------------------------------------- #
def test_dark_light_print_modes_use_the_single_theme(full_run):
    from fastapi.testclient import TestClient

    c = TestClient(app)
    headers = full_run["headers"]
    figures = {}
    for mode in ("light", "dark", "print"):
        r = c.get(f"{BASE}/analysis/catalog?mode={mode}", headers=headers)
        assert r.status_code == 200
        payload = json.loads(r.text, parse_constant=_bad_const)
        assert payload["scorecard"]["styling"]["single_source"] == "app.charts.theme"
        assert payload["scorecard"]["styling"]["colorblind_safe"] is True
        assert set(payload["scorecard"]["styling"]["dark_light_print"]) == {"light", "dark", "print"}
        figures[mode] = _item(payload, "fig.series")["figure_json"]
    assert figures["light"]["layout"]["paper_bgcolor"] == "#ffffff"
    assert figures["dark"]["layout"]["paper_bgcolor"] == "#18212b"
    assert figures["print"]["layout"]["paper_bgcolor"] == "#ffffff"
    assert figures["print"]["layout"]["font"]["color"] == "#000000"
    assert figures["dark"]["layout"]["font"]["color"] == "#e8edf3"
    for fig in figures.values():
        assert fig["layout"]["title"]["text"]
        assert fig["layout"]["font"]["family"] == FONT_STACK


def test_part16_modules_use_shared_theme_not_ad_hoc_plotly_figures():
    completeness = (ROOT / "app/charts/completeness.py").read_text()
    exporters = (ROOT / "app/charts/exporters.py").read_text()
    catalog = (ROOT / "app/api/catalog.py").read_text()
    main = (ROOT / "app/main.py").read_text()
    assert "go.Figure" not in completeness
    assert "from app.charts import primitives as P" in completeness
    assert "from app.charts.theme import" in completeness
    assert "restyled" in exporters
    assert "app.api.export" not in catalog
    assert '"app.api.catalog"' in main
    assert '"app.api.export"' in main  # Part 18 consumes the catalogue here
    assert "Part 18 — whole-report builders" in main


# --------------------------------------------------------------------------- #
# Report basket
# --------------------------------------------------------------------------- #
def test_report_basket_add_remove_set_clear_and_refuse_unknown(full_run):
    from fastapi.testclient import TestClient

    c = TestClient(app)
    h = full_run["headers"]
    fig_id, tab_id = "fig.series", "table.observations"
    r = c.post(f"{BASE}/analysis/catalog/report", json={"action": "add", "item_id": fig_id}, headers=h)
    assert r.status_code == 200 and r.json()["count"] == 1
    r = c.post(f"{BASE}/analysis/catalog/report", json={"action": "add", "item_id": tab_id}, headers=h)
    assert r.json()["count"] == 2
    r = c.post(f"{BASE}/analysis/catalog/report", json={"action": "remove", "item_id": fig_id}, headers=h)
    assert r.json()["item_ids"] == [tab_id]
    r = c.post(f"{BASE}/analysis/catalog/report", json={"action": "set", "item_ids": [fig_id, tab_id]}, headers=h)
    assert r.json()["count"] == 2
    r = c.get(f"{BASE}/analysis/catalog/report", headers=h)
    assert r.json()["count"] == 2
    r = c.post(f"{BASE}/analysis/catalog/report", json={"action": "add", "item_id": "not.computed"}, headers=h)
    assert r.status_code == 404 and r.json()["error"]["code"] == "catalog.item_not_found"
    r = c.post(f"{BASE}/analysis/catalog/report", json={"action": "clear"}, headers=h)
    assert r.json()["count"] == 0
    assert c.get(f"{BASE}/analysis/catalog", headers=h).json()["report"]["count"] == 0


# --------------------------------------------------------------------------- #
# Real exports
# --------------------------------------------------------------------------- #
def test_all_five_exports_are_real_files_at_300_dpi(full_run):
    _require_static_exports()
    from fastapi.testclient import TestClient

    c = TestClient(app)
    h = full_run["headers"]
    catalog = full_run["catalog"]
    table = _item(catalog, "table.observations")
    for item_id, fmt in [
        ("fig.series", "png"), ("fig.series", "svg"), ("fig.series", "pdf"),
        ("fig.series", "xlsx"), ("fig.series", "csv"),
        ("table.observations", "png"), ("table.observations", "xlsx"),
        ("table.observations", "csv"),
    ]:
        r = c.get(
            f"{BASE}/analysis/catalog/{item_id}/download?format={fmt}&mode=print",
            headers=h,
        )
        assert r.status_code == 200, (item_id, fmt, r.text[:300])
        assert r.headers["content-disposition"].startswith("attachment;")
        assert r.headers["x-catalog-dpi"] == "300"
        assert r.headers["x-catalog-item"] == item_id
        data = r.content
        assert data
        if fmt == "png":
            assert data[:8] == b"\x89PNG\r\n\x1a\n"
            width, height = struct.unpack(">II", data[16:24])
            assert (width, height) == (3750, 2500)
        elif fmt == "svg":
            assert b"<svg" in data[:3000]
        elif fmt == "pdf":
            assert data.startswith(b"%PDF")
        elif fmt == "xlsx":
            assert data[:2] == b"PK"
        elif fmt == "csv":
            assert b"," in data
    xlsx = c.get(
        f"{BASE}/analysis/catalog/table.observations/download?format=xlsx",
        headers=h,
    ).content
    wb = load_workbook(io.BytesIO(xlsx))
    assert "Data" in wb.sheetnames and "About this item" in wb.sheetnames
    assert wb["Data"].max_row == len(table["rows"]) + 1
    assert wb["About this item"]["A1"].value == "Catalogue item"
    csv = c.get(
        f"{BASE}/analysis/catalog/table.observations/download?format=csv",
        headers=h,
    ).content
    assert b"Time,Value" in csv
    bad_fmt = c.get(f"{BASE}/analysis/catalog/fig.series/download?format=gif", headers=h)
    assert bad_fmt.status_code == 400
    assert bad_fmt.json()["error"]["code"] == "catalog.unsupported_format"
    bad_id = c.get(f"{BASE}/analysis/catalog/not.here/download?format=png", headers=h)
    assert bad_id.status_code == 404
    assert bad_id.json()["error"]["code"] == "catalog.item_not_found"


# --------------------------------------------------------------------------- #
# Per-family §8 gaps
# --------------------------------------------------------------------------- #
def test_garch_variance_path_is_a_real_per_family_figure(branch_catalogs):
    catalog = branch_catalogs["variance"]["catalog"]
    item = _item(catalog, "models.garch_variance_path")
    rows = [r for r in item["rows"] if r.get("variance") is not None]
    assert len(rows) >= 20
    assert all(r["variance"] > 0 and r["sd"] > 0 for r in rows)
    assert any(r.get("persistence") is not None for r in rows)
    assert "GARCH" in item["caption"]
    assert len(item["figure_json"]["data"]) >= 2


def test_count_pmf_ribbons_carry_probabilities_and_disclosure(branch_catalogs):
    catalog = branch_catalogs["count"]["catalog"]
    item = _item(catalog, "forecast.count_pmf_ribbons")
    rows = [r for r in item["rows"] if r.get("probability") is not None]
    assert rows
    sums: dict[int, float] = {}
    for row in rows:
        sums[int(row["step"])] = sums.get(int(row["step"]), 0.0) + float(row["probability"])
    assert all(math.isclose(v, 1.0, abs_tol=0.02) for v in sums.values()), sums
    assert all("empirical" in row for row in rows)
    assert "PMF" in item["caption"]
    assert "continuous predictive distribution" not in item["caption"]
    assert len(item["figure_json"]["data"]) >= 1


def test_var_irf_and_fevd_include_bands_and_cholesky_ordering(full_run):
    catalog = full_run["catalog"]
    irf = _item(catalog, "models.var_irf")
    fevd = _item(catalog, "models.var_fevd")
    assert any(r.get("n_boot", 0) > 0 for r in irf["rows"])
    assert all(r.get("ordering") for r in irf["rows"])
    assert any(r.get("lower_90") is not None and r.get("upper_90") is not None for r in irf["rows"])
    assert "bootstrap" in irf["caption"]
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in fevd["rows"]:
        if row.get("share") is not None:
            grouped.setdefault((str(row["model"]), str(row["target"])), []).append(float(row["share"]))
    assert grouped
    assert all(math.isclose(sum(vals), 1.0, abs_tol=1e-6) for vals in grouped.values())
    assert all(r.get("ordering") for r in fevd["rows"] if r.get("share") is not None)
    assert "ordering" in fevd["caption"]

# --------------------------------------------------------------------------- #
# One styling source, enforced statically and at restyle time
# --------------------------------------------------------------------------- #
def _hardcoded_colours(text: str) -> list[str]:
    """Hex/rgb literals and Excel-style six-digit hex colours outside theme.py."""
    out: list[str] = []
    patterns = (
        r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b",
        r"rgba?\(\s*\d{1,3}\s*,",
        r"(?<![0-9A-Za-z_])[0-9a-fA-F]{6}(?![0-9A-Za-z_])",
    )
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            token = m.group()
            if (pattern.startswith("#") or pattern.startswith("rgba")
                    or any(ch in "abcdefABCDEF" for ch in token)):
                out.append(token)
    return out


_DASH_LITERAL = re.compile(
    r'"dash"\s*:\s*"(?:solid|dash|dot|dashdot|longdash|longdashdot)"'
    r'|get\(\s*"dash"\s*,\s*"(?:solid|dash|dot|dashdot|longdash|longdashdot)"'
)


def test_every_chart_module_takes_styling_from_theme_only():
    files = sorted((ROOT / "app").rglob("*.py"))
    assert files
    for path in files:
        if path.name == "theme.py":
            continue
        text = path.read_text()
        bad = _hardcoded_colours(text)
        assert not bad, (str(path), bad[:8])
        assert not _DASH_LITERAL.search(text), str(path)
        if "go.Figure" in text:
            assert "from app.charts.theme import" in text, str(path)
            assert "apply(" in text and "to_json" in text, str(path)


def test_restyled_modes_remap_trace_colours_and_print_dashes():
    figure = {
        "data": [{
            "type": "scatter", "mode": "lines", "x": [1, 2], "y": [1, 2],
            "line": {"color": PALETTE[0], "width": 2},
            "fillcolor": with_alpha(PALETTE[0], 0.14),
        }],
        "layout": {"height": 360, "title": {"text": "Preview"}},
    }
    light = restyled(figure, "light")
    dark = restyled(figure, "dark")
    printed = restyled(figure, "print")
    assert light["data"][0]["line"]["color"] == PALETTE[0]
    assert dark["data"][0]["line"]["color"] == PALETTE_VARIANTS["dark"][0]
    assert dark["layout"]["paper_bgcolor"] == colours("dark")["paper"]
    assert dark["data"][0]["fillcolor"].startswith("rgba(")
    for fig in (light, dark, printed):
        assert fig["layout"]["font"]["family"] == FONT_STACK
        assert fig["layout"]["title"]["text"]
    p_colour = printed["data"][0]["line"]["color"]
    assert p_colour.startswith("#") and len(p_colour) == 7
    assert p_colour[1:3] == p_colour[3:5] == p_colour[5:7]
    assert printed["data"][0]["line"]["dash"] in DASHES
    m = re.match(r"rgba\((\d+),(\d+),(\d+),", printed["data"][0]["fillcolor"])
    assert m and m.group(1) == m.group(2) == m.group(3)



def test_restyled_preserves_title_geometry_and_hidden_legends():
    """A dark/print restyle must recolour ink, not move or resurrect the legend."""
    title = (
        "<b>Seasonal profile</b><br>"
        "<span style='font-size:11.5px;color:#4a5a6a'>Peak in Mar, trough in Oct.</span>"
    )
    figure = {
        "data": [{
            "type": "bar", "x": ["Jan", "Feb"], "y": [1, 2],
            "marker": {"color": PALETTE[0]},
        }],
        "layout": {
            "height": 260,
            "margin": {"l": 58, "r": 18, "t": TITLE_TOP_MARGIN, "b": 46},
            "showlegend": False,
            "title": {"text": title, "x": 0, "xanchor": "left", "y": 0.97, "yanchor": "top"},
        },
    }
    for mode in ("light", "dark", "print"):
        layout = restyled(figure, mode)["layout"]
        assert layout["title"]["text"].startswith("<b>Seasonal profile</b>")
        assert layout["title"]["y"] == TITLE_Y
        assert layout["margin"]["t"] == TITLE_TOP_MARGIN
        assert layout["height"] == 260
        assert layout["showlegend"] is False
        assert layout["paper_bgcolor"] == colours(mode)["paper"]
        assert layout["legend"]["font"]["color"] == colours(mode)["muted"]


def test_apply_reserves_top_margin_for_wrapped_horizontal_legends():
    """Plotly.js anchors top legends above the plot; long CUSUM legends must fit."""
    import plotly.graph_objects as go

    count = LEGEND_ITEMS_PER_ROW * 4 + 1
    fig = go.Figure()
    for i in range(count):
        fig.add_trace(go.Scatter(
            x=[1, 2], y=[i, i + 1], mode="lines", name=f"model {i} CUSUM",
        ))
    apply(fig, title="Residual CUSUM paths", subtitle="Dashed lines are 95% bounds.",
          height=260)
    layout = to_json(fig)["layout"]
    rows = math.ceil(count / LEGEND_ITEMS_PER_ROW)
    assert layout["margin"]["t"] == TITLE_TOP_MARGIN + rows * LEGEND_ROW_HEIGHT
    assert layout["height"] == 260 + rows * LEGEND_ROW_HEIGHT
    assert layout["title"]["y"] == TITLE_Y
    assert layout["showlegend"] is True

    hidden = go.Figure(go.Bar(x=["Jan", "Feb"], y=[1, 2]))
    apply(hidden, title="Seasonal profile", height=260, legend=False)
    plain = to_json(hidden)["layout"]
    assert plain["margin"]["t"] == TITLE_TOP_MARGIN
    assert plain["height"] == 260
    assert plain["showlegend"] is False

def test_catalog_tolerates_theta_without_a_residual_battery(client):
    """Theta can have no one-step residuals; the catalogue must not dereference None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _sid, h = _prepared(client, "traffic_spend")
        _models_ready(client, h, ["ets", "arima", "theta", "drift", "mean"])
        _forecast(client, h)
        evaluated = client.post(
            f"{BASE}/analysis/evaluate",
            json={"horizon": 6, "folds": 3, "n_paths": 2000, "reps": 3},
            headers=h,
        )
        assert evaluated.status_code == 200, evaluated.text[:500]
        catalog = client.get(f"{BASE}/analysis/catalog?mode=light", headers=h)
        status = client.get(f"{BASE}/export/status", headers=h)
    assert catalog.status_code == 200, catalog.text[:500]
    assert _item(catalog.json(), "diagnostics.residual_acf")
    assert status.status_code == 200, status.text[:500]

def test_svg_fallback_keeps_the_report_vector_without_kaleido(monkeypatch):
    """A missing browser renderer must not remove figures from the HTML report."""
    def missing():
        raise ModuleNotFoundError("kaleido")

    monkeypatch.setattr(E, "_ensure_server", missing)
    figure = {
        "data": [
            {"type": "scatter", "x": [1, 2, 3], "y": [2, 4, 3], "mode": "lines+markers",
             "line": {"color": "#0072B2", "width": 2}, "name": "Observed"},
            {"type": "bar", "x": ["A", "B", "C"], "y": [2, 5, 3],
             "marker": {"color": "#D55E00"}},
        ],
        "layout": {"title": {"text": "Offline fallback"}, "showlegend": True},
    }
    data = E.static_image_bytes(figure, "svg", mode="print", title="Offline fallback")
    root = ET.fromstring(data)
    assert root.tag == f"{{{SVG_NS}}}svg"
    assert b"<script" not in data and b"<link" not in data
    assert b"href=" not in data
    assert b"Static vector figure unavailable" not in data
    assert b"<circle" in data and b"<rect" in data


def test_every_catalogue_plot_has_an_offline_svg_fallback(full_run, monkeypatch):
    """All 49 catalogue figures must render as valid, self-contained vector SVG."""
    def missing():
        raise ModuleNotFoundError("kaleido")

    monkeypatch.setattr(E, "_ensure_server", missing)
    plots = full_run["catalog"]["plots"]
    assert len(plots) >= 35
    for item in plots:
        data, media_type = E.download_bytes(item, "svg", mode="light")
        assert media_type == "image/svg+xml"
        root = ET.fromstring(data)
        assert root.tag == f"{{{SVG_NS}}}svg", item["id"]
        assert b"Static vector figure unavailable" not in data, item["id"]
        assert b"<script" not in data and b"<link" not in data, item["id"]
        assert b"href=" not in data, item["id"]

