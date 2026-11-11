"""
Regression tests for the debugging sweep across Parts 0–13.

Each test here is named after a defect that was actually observed in the running app, not after a
feature. The point of keeping them together is that these bugs crossed module boundaries: a calendar
guess in Part 1 produced an empty chart panel in Part 2, a nonsensical sentence in Part 4 and a
dead-end gate in Part 7. A test placed in only one part's file would not have caught the chain.

Run: pytest tests/test_debug_regressions.py -p no:warnings
"""

from __future__ import annotations

import ast
import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ts.decompose.components import scorecard, seasonal_finding, trend_finding
from app.ts.decompose.figures import component_grid
from app.ts.decompose.methods import cf_filter, decompose, hp_filter, ucm
from app.ts.eda.figures import lag_plot_chart as lag_plots
from app.ts.gate.readiness import GateFacts, assess
from app.ts.preprocess.advise import advise_preprocessing, build_context

B = "/api/v1"


def _client() -> tuple[TestClient, dict[str, str]]:
    c = TestClient(app)
    sid = c.post(f"{B}/sessions").json()["session_id"]
    return c, {"X-Session-Id": sid}


def _monthly(n: int = 120) -> pd.Series:
    return pd.Series(
        np.arange(n, dtype=float) + 10.0,
        index=pd.date_range("2015-01-01", periods=n, freq="MS"),
        name="y",
    )


def _constant(n: int = 40) -> pd.Series:
    return pd.Series(
        np.full(n, 5.0),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
        name="y",
    )


def _paste_constant(c: TestClient, h: dict[str, str], value: float = 5.0, n: int = 60) -> None:
    text = "date,value\n" + "".join(
        f"2020-{1 + i // 28:02d}-{1 + i % 28:02d},{value}\n" for i in range(n)
    )
    r = c.post(f"{B}/data/paste", json={"text": text, "name": "constant.csv"}, headers=h)
    assert r.status_code in (200, 201), r.text


# --------------------------------------------------------------------------- #
# 1. Part 2 — a nominal period larger than the series made an empty "r = nan" panel
# --------------------------------------------------------------------------- #
def test_lag_plots_omit_lags_longer_than_the_series():
    """wwwusage: 100 minute observations, nominal daily period 1440. `s.iloc[:-1440]` is empty."""
    s = pd.Series(
        100.0 + np.arange(100, dtype=float),
        index=pd.date_range("2020-01-01", periods=100, freq="min"),
    )
    chart = lag_plots(s, period=1440)
    ann = chart.figure_json["layout"].get("annotations", [])
    texts = [a.get("text", "") for a in ann]
    assert not any("nan" in t.lower() for t in texts), texts
    # No lag-1440 subplot title either: an empty panel is not a lag plot.
    assert not any("1440" in t for t in texts), texts
    assert any("lag 1" in t for t in texts)


def test_lag_plots_refuse_a_constant_window_in_words_not_in_nan():
    chart = lag_plots(_constant(40), period=None)
    texts = [a.get("text", "") for a in chart.figure_json["layout"].get("annotations", [])]
    assert not any("nan" in t.lower() for t in texts), texts
    assert any("not available" in t or "n/a" in t for t in texts), texts


def test_profile_endpoint_has_no_nan_annotation_on_the_minute_series():
    c, h = _client()
    assert c.post(f"{B}/data/example", json={"example_id": "wwwusage"}, headers=h).status_code in (200, 201)
    r = c.post(f"{B}/analysis/profile", json={}, headers=h)
    assert r.status_code == 200
    body = json.dumps(r.json())
    assert "r = nan" not in body
    assert '"nan"' not in body.lower().replace("nanoid", "")


# --------------------------------------------------------------------------- #
# 2. Part 3 — `None` was interpolated into a sentence as the word "None"
# --------------------------------------------------------------------------- #
def test_seasonal_advice_says_untested_not_none():
    s = _constant(60)
    advice = advise_preprocessing(
        s, eda_facts={}, name="constant", frequency_label="Daily",
        period=None, data_class="continuous",
    )
    blob = json.dumps([r.model_dump(mode="json") for r in advice.results], default=str)
    # No sentence may contain Python's repr of an unanswered question.
    assert "helps: None" not in blob
    assert "helps: True" not in blob and "helps: False" not in blob


def test_advice_endpoint_never_prints_none_for_the_seasonal_hint():
    c, h = _client()
    _paste_constant(c, h)
    c.post(f"{B}/analysis/profile", json={}, headers=h)
    r = c.post(f"{B}/analysis/advice", json={}, headers=h)
    assert r.status_code == 200
    assert "helps: None" not in json.dumps(r.json())


# --------------------------------------------------------------------------- #
# 3. Part 4 — `np.isfinite(None)` crashed HP/CF on any series without drift
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [hp_filter, cf_filter])
def test_trend_finding_handles_a_decomposition_without_strength_measures(factory):
    s = pd.Series(
        np.random.default_rng(3).normal(0, 1, 120),
        index=pd.date_range("2015-01-01", periods=120, freq="MS"),
    )
    dec = factory(s)
    assert dec.trend_strength is None
    finding = trend_finding(dec)
    assert finding.plain.startswith("The trend")


def test_hp_and_cf_decompose_endpoints_do_not_500_on_white_noise():
    c, h = _client()
    assert c.post(f"{B}/data/example", json={"example_id": "white_noise"}, headers=h).status_code in (200, 201)
    c.post(f"{B}/analysis/profile", json={}, headers=h)
    for method in ("hp", "cf"):
        r = c.post(f"{B}/analysis/decompose", json={"method": method}, headers=h)
        assert r.status_code == 200, (method, r.status_code, r.text[:200])


def test_hp_decompose_does_not_500_on_a_constant_series():
    c, h = _client()
    _paste_constant(c, h)
    c.post(f"{B}/analysis/profile", json={}, headers=h)
    r = c.post(f"{B}/analysis/decompose", json={"method": "hp"}, headers=h)
    assert r.status_code == 200, r.text[:200]


# --------------------------------------------------------------------------- #
# 4. Part 4 — HP/CF claimed seasonal facts about a component they do not estimate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [hp_filter, cf_filter])
def test_cycle_methods_refuse_to_invent_a_seasonal_shape(factory):
    s = pd.Series(
        10 + np.sin(np.arange(120) / 3.0) + np.arange(120) * 0.05,
        index=pd.date_range("2015-01-01", periods=120, freq="MS"),
    )
    dec = factory(s)
    f = seasonal_finding(dec, s, 12)
    assert f.statistic is None
    assert f.table == []
    assert "no seasonal component" in f.plain.lower()
    assert "seasonality repeats" not in f.plain.lower()


def test_scorecard_names_the_cycle_what_it_is():
    s = _monthly(120)
    dec = hp_filter(s)
    sc = scorecard(dec, s, 12)
    labels = [r["component"] for r in sc["rows"]]
    assert "Cycle" in labels and "Seasonality" not in labels
    assert "cycle" in sc["plain"].lower()


def test_component_grid_labels_the_second_panel_cycle_for_hp():
    s = _monthly(120)
    dec = hp_filter(s)
    chart = component_grid(dec, "test")
    names = [t.get("name", "") for t in chart.figure_json["data"]]
    assert "Cycle" in names and "Seasonal" not in names
    assert "cycle" in chart.alt_text.lower()


def test_ucm_without_a_seasonal_term_does_not_describe_one():
    """A 100-point minute series has a nominal period of 1440; UCM declines the seasonal term."""
    s = pd.Series(
        10 + np.arange(100, dtype=float) * 0.1,
        index=pd.date_range("2020-01-01", periods=100, freq="min"),
    )
    dec = ucm(s, period=1440)
    assert dec.period is None
    f = seasonal_finding(dec, s, 1440)
    assert f.statistic is None
    assert "1440" not in f.plain or "too short" in f.plain
    assert "peak position" not in f.plain.lower()


# --------------------------------------------------------------------------- #
# 5. Part 4 — the "reproducible snippet" hard-coded STL for every method
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("method", "must_contain"),
    [
        ("stl", "STL("),
        ("classical", "seasonal_decompose"),
        ("ucm", "UnobservedComponents"),
        ("hp", "hpfilter"),
        ("cf", "cffilter"),
    ],
)
def test_reproducible_snippet_matches_the_method_that_ran(method, must_contain):
    c, h = _client()
    assert c.post(f"{B}/data/example", json={"example_id": "airpassengers"}, headers=h).status_code in (200, 201)
    c.post(f"{B}/analysis/profile", json={}, headers=h)
    r = c.post(f"{B}/analysis/decompose", json={"method": method}, headers=h)
    assert r.status_code == 200, (method, r.status_code, r.text[:200])
    snippets = [
        res["reproducible_snippet"]
        for res in r.json()["results"]
        if res.get("reproducible_snippet") and str(res.get("id", "")).startswith("stat.decomp.")
    ]
    assert snippets, f"no snippet emitted for {method}"
    for snip in snippets:
        assert must_contain in snip, (method, snip[:200])
        if method != "stl":
            assert "STL(" not in snip, (method, snip[:200])
        ast.parse(snip)  # the snippet must still be runnable Python


# --------------------------------------------------------------------------- #
# 6. Part 6 — a constant series was a 500 instead of a plain-English refusal
# --------------------------------------------------------------------------- #
def test_correlate_refuses_a_constant_series_in_plain_language():
    c, h = _client()
    _paste_constant(c, h)
    c.post(f"{B}/analysis/profile", json={}, headers=h)
    r = c.get(f"{B}/analysis/correlate", headers=h)
    assert r.status_code == 400, r.text[:300]
    err = r.json()["error"]
    assert err["code"] == "correlate.constant_series"
    assert "constant" in err["plain"].lower()
    assert err["fix"] and "Step 0" in err["fix"]


# --------------------------------------------------------------------------- #
# 7. Part 7/8 — too few seasonal cycles blocked *all* modelling, not seasonal models
# --------------------------------------------------------------------------- #
def test_few_seasonal_cycles_is_a_caveat_not_a_full_stop():
    g = assess(GateFacts(n=100, period=1440, profile_ran=True, breaks_tested=True,
                         stationarity_verdict="stationary", n_agreeing=3,
                         correlogram_pattern="white_noise", ljung_box_p=0.4,
                         variance_ratio=0.1, variance_growth_pct=5,
                         seasonal_strength=0.0, signal_to_noise=0.5))
    length = next(c for c in g.checklist if c.key == "length")
    assert length.state == "fail"
    assert length.severity == "caveat"
    assert length not in g.blockers
    assert g.verdict != "not_ready", g.headline
    assert any("seasonal" in t.lower() for t in g.caveats_travelling)


def test_the_short_minute_series_can_actually_be_fitted():
    """The whole point: wwwusage is a bundled teaching dataset, not a dead end."""
    c, h = _client()
    assert c.post(f"{B}/data/example", json={"example_id": "wwwusage"}, headers=h).status_code in (200, 201)
    c.post(f"{B}/analysis/profile", json={}, headers=h)
    c.post(f"{B}/analysis/advice", json={}, headers=h)
    c.post(f"{B}/analysis/decompose", json={"method": "hp"}, headers=h)
    c.post(f"{B}/analysis/tests", json={}, headers=h)
    c.get(f"{B}/analysis/correlate", headers=h)
    gate = c.post(f"{B}/analysis/gate", json={}, headers=h)
    assert gate.status_code == 200
    assert c.get(f"{B}/analysis/gate", headers=h).json()["verdict"] != "not_ready"
    s = c.get(f"{B}/analysis/suggest", headers=h)
    assert s.status_code == 200 and s.json()["suggestions"], "candidates must be offered"
    assert c.post(f"{B}/analysis/suggest", json={}, headers=h).status_code == 200
    fit = c.post(f"{B}/analysis/fit", json={"model_ids": ["mean", "naive", "drift"]}, headers=h)
    assert fit.status_code == 200, fit.text[:300]
    fits = c.get(f"{B}/analysis/fits", headers=h).json()["fits"]
    assert len(fits) >= 3


def test_a_blocker_refusal_names_the_number_not_just_the_checklist_label():
    """'not ready to model: Enough observations to fit a seasonal model' read as a contradiction."""
    from app.api.models import _gate_check

    class _S:
        objects = {
            "gate_verdict": {
                "payload": {
                    "verdict": "not_ready",
                    "blockers": [{"key": "length", "label": "Enough observations to fit a model",
                                  "value": "12 observations"}],
                }
            }
        }

    with pytest.raises(Exception) as exc:
        _gate_check(_S())
    assert "12 observations" in str(exc.value)


# --------------------------------------------------------------------------- #
# 8. Sweep-level: no non-finite numbers leak into any JSON response
# --------------------------------------------------------------------------- #
def test_no_nan_literals_in_a_full_pipeline_on_the_minute_series():
    c, h = _client()
    assert c.post(f"{B}/data/example", json={"example_id": "wwwusage"}, headers=h).status_code in (200, 201)
    endpoints = [
        ("post", "profile", {}), ("post", "advice", {}), ("post", "decompose", {"method": "hp"}),
        ("post", "tests", {}), ("post", "gate", {}),
    ]
    for verb, path, payload in endpoints:
        r = getattr(c, verb)(f"{B}/analysis/{path}", json=payload, headers=h)
        assert r.status_code == 200, (path, r.status_code, r.text[:200])
        assert "NaN" not in r.text and "Infinity" not in r.text, path
    for path in ("correlate", "facts", "suggest"):
        r = c.get(f"{B}/analysis/{path}", headers=h)
        assert r.status_code == 200, (path, r.status_code)
        assert "NaN" not in r.text and "Infinity" not in r.text, path
    r = c.get(f"{B}/pipeline", headers=h)
    assert r.status_code == 200 and "NaN" not in r.text


# --------------------------------------------------------------------------- #
# 9. Parts 1 ↔ 8 — the model layer overruled the ingest layer's data class
# --------------------------------------------------------------------------- #
def test_whole_number_but_continuous_examples_are_not_sent_to_count_models():
    """
    `detect_data_class` (Part 1) deliberately treats large-magnitude whole numbers as continuous.
    `classify_data_class` (Part 8) used to overrule that verdict for *any* non-negative integer
    series, which put Poisson / NB / INGARCH on wwwusage (mean 137, sd 40, minute-by-minute usage)
    and on eight other bundled teaching datasets. The screenshot that exposed this showed
    "Count data with variance 11.67× the mean" over a series of 83–228.
    """
    from app.ts.models.facts import classify_data_class
    from app.core.schemas import DataClass

    c = TestClient(app)
    for ex in ("wwwusage", "airpassengers", "lynx", "sunspots", "usaccdeaths"):
        sid = c.post(f"{B}/sessions").json()["session_id"]
        h = {"X-Session-Id": sid}
        load = c.post(f"{B}/data/example", json={"example_id": ex}, headers=h)
        assert load.json()["dataset"]["data_class"] == "continuous", ex
        for p, pay in (("profile", {}), ("advice", {}), ("decompose", {"method": "hp"}),
                       ("tests", {})):
            c.post(f"{B}/analysis/{p}", json=pay, headers=h)
        c.get(f"{B}/analysis/correlate", headers=h)
        c.post(f"{B}/analysis/gate", json={}, headers=h)
        s = c.get(f"{B}/analysis/suggest", headers=h).json()
        assert s["data_class"] == "continuous", (ex, s["data_class"])
        ids = [x["model_id"] for x in s["suggestions"]]
        assert not ({"poisson", "neg_binomial", "ingarch", "inar", "croston"} & set(ids)), (
            ex, ids
        )
        assert "arima" in ids or "ets" in ids, (ex, ids)
    # The unit-level rule: an explicit ingest verdict wins over integrality.
    y = pd.Series(np.arange(80, 220, dtype=float))
    dc, why = classify_data_class(y, DataClass.CONTINUOUS)
    assert dc is DataClass.CONTINUOUS and "declared continuous" in why


def test_binary_series_with_many_zeros_is_not_called_intermittent():
    from app.core.schemas import DataClass
    from app.ts.models.facts import classify_data_class

    y = pd.Series([0.0] * 70 + [1.0] * 30)
    dc, _ = classify_data_class(y, DataClass.BINARY_OR_PROPORTION)
    assert dc is DataClass.BINARY_OR_PROPORTION


def test_the_count_branch_is_one_click_away_as_the_app_promises():
    """
    Three strings in the app say "you can override the data class in Column mapping", but
    `ColumnMapping` had no such field, so the promise was unreachable. Both directions must work.
    """
    c, h = _client()
    load = c.post(f"{B}/data/example", json={"example_id": "wwwusage"}, headers=h).json()
    mapping = dict(load["mapping"])

    mapping["data_class"] = "count"
    r = c.post(f"{B}/data/remap", json={"mapping": mapping}, headers=h)
    assert r.status_code == 200, r.text[:200]
    assert r.json()["dataset"]["data_class"] == "count"
    assert r.json()["dataset"]["quality"]["data_class_evidence"]["user_override"] is True

    mapping["data_class"] = "continuous"
    r = c.post(f"{B}/data/remap", json={"mapping": mapping}, headers=h)
    assert r.json()["dataset"]["data_class"] == "continuous"

    mapping["data_class"] = "banana"
    r = c.post(f"{B}/data/remap", json={"mapping": mapping}, headers=h)
    assert r.status_code == 200
    assert r.json()["dataset"]["data_class"] == "continuous"
    assert any(i["code"] == "data.class_override_invalid"
               for i in r.json()["dataset"]["quality"]["issues"])


# --------------------------------------------------------------------------- #
# 10. Parts 2–13 — impossible p-values in prose
# --------------------------------------------------------------------------- #
def test_p_value_helper_never_claims_zero_or_puts_equals_before_an_inequality():
    from app.core.formatting import p_number, p_phrase

    assert p_phrase(0.0) == "p < 0.0001"
    assert p_phrase(3.4e-120) == "p < 0.0001"
    assert p_phrase(0.0003703) == "p = 0.00037"
    assert p_phrase(0.0075) == "p = 0.0075"
    assert p_phrase(0.0076, 3) == "p = 0.008"
    assert "not available" in p_phrase(None)
    assert "not available" in p_phrase(float("nan"))
    assert p_number(0.0) == "< 0.0001"
    assert p_number(0.0003703) == "0.00037"
    assert p_number(None) == "—"


def test_live_results_contain_only_grammatical_p_values():
    """
    `f"{p:.3f}"` printed `p = 0.000`; `f"{p:.1e}"` printed `0.0e+00`; and the first fix
    prepended `p = ` to `< 0.0001`, producing `p = < 0.0001` in fourteen diagnostic strings
    and in the selection report's "why this model won" paragraph. Every prose path has to
    use the same complete comparison.
    """
    import re

    c, h = _client()
    c.post(f"{B}/data/example", json={"example_id": "traffic_spend"}, headers=h)
    texts: list[str] = []
    for path, payload in (("profile", {}), ("advice", {}), ("decompose", {"method": "stl"}),
                          ("decompose", {"method": "hp"}), ("tests", {}), ("gate", {}),
                          ("suggest", {})):
        texts.append(c.post(f"{B}/analysis/{path}", json=payload, headers=h).text)
    for path in ("correlate", "next", "suggest", "facts"):
        texts.append(c.get(f"{B}/analysis/{path}", headers=h).text)
    texts.append(c.post(
        f"{B}/analysis/fit",
        json={"model_ids": ["ets", "arima", "drift", "mean"]},
        headers=h,
    ).text)
    texts.append(c.post(
        f"{B}/analysis/select",
        json={"horizon": 12, "folds": 4, "models": ["ets", "arima", "drift", "mean"]},
        headers=h,
    ).text)
    texts.append(c.get(f"{B}/analysis/selection", headers=h).text)
    texts.append(c.post(f"{B}/analysis/diagnose", json={}, headers=h).text)
    texts.append(c.get(f"{B}/analysis/diagnostics", headers=h).text)

    blob = " ".join(texts)
    bad = re.compile(r"p\s*=\s*(?:0\.0{3,}|<|nan)|0\.0e\+00|p\s*=\s*<", re.IGNORECASE)
    assert not bad.search(blob), [blob[max(0, m.start() - 80):m.end() + 50]
                                 for m in bad.finditer(blob)][:3]
    assert "p < 0.0001" in blob, "the small p-values must still be quoted honestly"


# --------------------------------------------------------------------------- #
# 11. Part 4 — an explicit method request must not silently become another method
# --------------------------------------------------------------------------- #
def test_unavailable_decomposition_refuses_with_a_usable_alternative():
    """
    On a 100-point minute series the calendar suggests period 1440. The UI greys out STL
    and pre-selects UCM, so the non-programmer never reaches a dead end. The API keeps the
    stricter contract: an *explicit* `method="stl"` returns a measured refusal rather than
    quietly running HP and labelling it STL, which would break reproducibility. The reply
    names methods that actually work here.
    """
    c, h = _client()
    c.post(f"{B}/data/example", json={"example_id": "wwwusage"}, headers=h)
    c.post(f"{B}/analysis/profile", json={}, headers=h)
    methods = c.get(f"{B}/analysis/decompose/methods", headers=h).json()["methods"]
    stl = next(m for m in methods if m["method"] == "stl")
    assert stl["available"] is False
    assert "two full cycles" in stl["why_not"].lower()

    r = c.post(f"{B}/analysis/decompose", json={"method": "stl"}, headers=h)
    assert r.status_code == 400, r.text[:300]
    err = r.json()["error"]
    assert "two full cycles" in err["plain"].lower()
    assert "Hodrick–Prescott" in err["fix"] and "Unobserved components" in err["fix"]

    ok = c.post(f"{B}/analysis/decompose", json={"method": "ucm"}, headers=h)
    assert ok.status_code == 200, ok.text[:300]


def test_p_value_table_columns_do_not_use_a_fixed_point_format():
    """
    The fit-comparison table still printed `0.000` in its Ljung–Box p column after the prose
    fix. A table cell is not prose, but `0.000` still claims an impossible exact zero. Backend
    tables mark p-value columns with `format="p"`; the frontend's universal table renderer
    turns those into `< 0.0001` or `0.008`.
    """
    import re

    c, h = _client()
    sid = h["X-Session-Id"]
    c.post(f"{B}/data/example", json={"example_id": "wwwusage"}, headers=h)
    for path, payload in (("profile", {}), ("advice", {}), ("decompose", {"method": "ucm"}),
                          ("tests", {})):
        c.post(f"{B}/analysis/{path}", json=payload, headers=h)
    c.get(f"{B}/analysis/correlate", headers=h)
    c.post(f"{B}/analysis/gate", json={}, headers=h)
    c.post(f"{B}/analysis/suggest", json={}, headers=h)
    c.post(f"{B}/analysis/fit", json={"model_ids": ["ets", "arima", "drift", "mean"]}, headers=h)
    c.post(f"{B}/analysis/select", json={"horizon": 6, "folds": 4,
                                         "models": ["ets", "arima", "drift", "mean"]}, headers=h)
    c.post(f"{B}/analysis/diagnose", json={}, headers=h)

    results = c.get(f"{B}/sessions/{sid}/results", headers=h).json()
    label = re.compile(r"(?:\bp\b|p-value)", re.IGNORECASE)
    seen = 0
    for stage, items in results.items():
        for result in items:
            if result.get("kind") != "table":
                continue
            for col in result.get("columns") or []:
                if label.search(col.get("label", "")):
                    seen += 1
                    assert col.get("format") == "p", (stage, col)
                    assert col.get("dtype") == "number", (stage, col)
    assert seen >= 5, f"expected the p-value columns from S5, S10 and S12; saw {seen}"
