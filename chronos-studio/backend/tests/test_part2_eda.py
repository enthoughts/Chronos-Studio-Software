"""
tests/test_part2_eda.py — Part 2 verification.

The EDA layer is where a beginner forms their first (and often only) impression of
the series, so the tests focus on the failure modes that produce *confidently wrong*
descriptions:

  * an undetrended seasonal test that cannot see seasonality through a trend,
  * a seasonal profile reported in detrended units (unreadable for a user),
  * "the swing is growing" asserted on data with no seasonality at all,
  * an outlier panel that flags 10% of the series,
  * a narrator that states a number which was never computed,
  * a chart factory that lets a handler build its own Plotly figure.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.charts import primitives as pr
from app.charts.theme import PALETTE, base_layout, colours
from app.core.schemas import DataClass, FigureResult, NarrativeResult, Stage, TableResult
from app.ts.eda.descriptives import describe, normality_battery, rolling_cv_stability, runs_test
from app.ts.eda.narrate import NarrativeFacts, assert_grounded, narrate, numbers_in
from app.ts.eda.outliers import MAX_FLAG_RATE, detect_outliers
from app.ts.eda.profile import run_eda
from app.ts.eda.seasonal import (
    amplitude_trend,
    labels_for,
    month_year_matrix,
    seasonal_profile,
    stl_strength,
    trend_hint,
)

EXAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "app" / "assets" / "examples"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def air() -> pd.Series:
    s = pd.read_csv(EXAMPLES / "airpassengers.csv", index_col=0)["Passengers"]
    s.index = pd.to_datetime(s.index)
    return s


@pytest.fixture(scope="module")
def noise() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(
        rng.normal(50, 8, 180), index=pd.date_range("2015-01-01", periods=180, freq="MS")
    )


@pytest.fixture(scope="module")
def additive() -> pd.Series:
    """Trend + a stable additive season + noise: the well-behaved textbook case."""
    idx = pd.date_range("2010-01-01", periods=132, freq="MS")
    t = np.arange(132)
    vals = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * (t % 12) / 12) + np.random.default_rng(2).normal(0, 3, 132)
    return pd.Series(vals, index=idx)


@pytest.fixture(scope="module")
def eda_air(air) -> object:
    return run_eda(air, name="AirPassengers", frequency_label="Monthly",
                   freq_code="MS", period=12, data_class=DataClass.CONTINUOUS)


# --------------------------------------------------------------------------- #
# Chart factory (§5.10 consistency rule)
# --------------------------------------------------------------------------- #
def test_chart_palette_is_colourblind_safe_and_fixed():
    assert PALETTE[0] == "#0072B2"          # Okabe–Ito blue
    assert len(set(PALETTE)) == len(PALETTE)


def test_base_layout_sets_the_house_style():
    layout = base_layout(title="T", x_title="time", y_title="value")
    assert layout["hovermode"] == "x unified"
    assert layout["xaxis"]["title"]["text"] == "time"
    assert layout["paper_bgcolor"]


def test_print_mode_collapses_to_greys():
    c = colours("print")
    assert c["series"] == "#000000"
    assert c["seasonal"] != c["series"]        # still distinguishable by grey level


def test_every_chart_returns_json_alt_text_and_data(air):
    charts = [
        pr.time_series(air, title="x"),
        pr.distribution_panel(air),
        pr.stem_plot(range(12), [1.0] + [0.1] * 11, title="ACF", band=0.16),
        pr.bars(["a", "b"], [1.0, 2.0], title="bars", reference=1.5),
        pr.rolling_panel(air, 12, title="rolling"),
    ]
    for ch in charts:
        assert isinstance(ch.figure_json, dict)
        assert "data" in ch.figure_json and len(ch.figure_json["data"]) > 0
        assert ch.alt_text and len(ch.alt_text) > 20
        # alt text must carry a number, not just a label
        assert any(ch.isdigit() for ch in ch.alt_text)


def test_alt_text_describes_the_series_with_real_numbers(air):
    ch = pr.time_series(air, title="Passengers")
    assert "144 observations" in ch.alt_text
    assert "1949" in ch.alt_text and "1960" in ch.alt_text
    assert str(int(air.min())) in ch.alt_text or "104" in ch.alt_text


def test_missing_values_are_drawn_as_gaps_not_joined(air):
    gappy = air.copy()
    gappy.iloc[10:14] = np.nan
    ch = pr.time_series(gappy, title="with gaps")
    trace = ch.figure_json["data"][0]
    assert trace.get("connectgaps") in (False, None)
    assert "missing" in ch.alt_text.lower()


def test_empty_figure_does_not_crash_the_factory():
    ch = pr.time_series(pd.Series(dtype=float), title="empty")
    assert ch.figure_json["data"]
    assert "no observations" in ch.alt_text.lower()


def test_heatmap_data_behind_round_trips():
    matrix = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["2020", "2021"], columns=["Q1", "Q2"])
    ch = pr.heatmap(matrix, title="grid")
    assert len(ch.data_behind) == 2
    assert ch.data_behind[0]["Q1"] == 1.0


# --------------------------------------------------------------------------- #
# Descriptives
# --------------------------------------------------------------------------- #
def test_descriptives_are_numerically_correct(air):
    d = describe(air, "Monthly", 12)
    assert d.n == 144
    assert d.mean == pytest.approx(float(air.mean()), rel=1e-9)
    assert d.median == pytest.approx(float(air.median()), rel=1e-9)
    assert d.sd == pytest.approx(float(air.std(ddof=1)), rel=1e-9)
    assert d.minimum == float(air.min()) and d.maximum == float(air.max())
    assert d.total_change == pytest.approx(float(air.iloc[-1] - air.iloc[0]))
    assert d.cagr == pytest.approx(12.0, abs=0.2)


def test_descriptives_bands_are_plain_english(air):
    d = describe(air, "Monthly", 12)
    assert "variable" in d.coefficient_of_variation_band
    assert d.skew_band in {"roughly symmetric", "moderately skewed right", "strongly skewed right"}


def test_normality_battery_says_why_it_does_not_matter(air):
    nb = normality_battery(air.to_numpy())
    assert "p_value" in nb["jarque_bera"]
    assert "raw values" in nb["plain"]
    assert "residuals" in nb["plain"]


def test_cagr_is_undefined_without_positive_endpoints():
    idx = pd.date_range("2015-01-01", periods=36, freq="MS")
    s = pd.Series(np.linspace(10, -5, 36), index=idx)
    d = describe(s, "Monthly", 12)
    assert d.cagr is None


def test_runs_test_detects_persistence_and_alternation():
    idx = pd.date_range("2015-01-01", periods=200, freq="MS")
    # persistence: a random walk of a smoothed sequence wanders in long stretches
    rng = np.random.default_rng(3)
    smooth = np.convolve(rng.normal(0, 1, 205), np.ones(6) / 6, "valid")
    assert runs_test(pd.Series(np.cumsum(smooth), index=idx))["z"] < -5

    # A perfectly monotone series has zero runs variance: report that instead of a
    # meaningless z, rather than dividing by zero.
    monotone = runs_test(pd.Series(np.arange(200.0), index=idx))
    assert monotone["available"] is False
    assert monotone["all_same_sign"] is True

    alternating = pd.Series([0.0, 1.0] * 100, index=idx)
    assert runs_test(alternating)["z"] > 0


def test_rolling_stability_flags_a_growing_spread(air):
    out = rolling_cv_stability(air, 12)
    assert out["available"]
    assert out["ratio"] > 1.5
    assert "log" in out["plain"] or "Box" in out["plain"]


# --------------------------------------------------------------------------- #
# Seasonality — the trap this module exists to avoid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "code,period,expected",
    [
        ("MS", 12, ["Jan", "Feb", "Mar"]),
        ("QS", 4, ["Q1", "Q2", "Q3"]),
        ("D", 7, ["Mon", "Tue", "Wed"]),
        ("h", 24, ["00h", "01h", "02h"]),
    ],
)
def test_season_labels_are_human(code, period, expected):
    labels = labels_for(code, period)
    assert labels[:3] == expected
    assert len(labels) == period


def test_seasonal_profile_reports_original_units(air):
    """Regression: the table must be in the units the user reads, not detrended."""
    p = seasonal_profile(air, 12, "MS")
    assert p.available
    assert all(m > 100 for m in p.means), "means must be passenger counts, not deviations"
    assert p.peak_label == "Jul"
    assert p.amplitude > 0


def test_seasonal_test_sees_through_a_trend(air):
    """
    The key regression: testing the RAW values of a rising seasonal series gives
    p ≈ 0.43 (no seasonality). Detrending first gives p ≈ 1e-17.
    """
    p = seasonal_profile(air, 12, "MS")
    assert p.kw_p_value is not None and p.kw_p_value < 1e-6
    assert p.strength is not None and p.strength > 0.9


def test_no_seasonality_is_reported_as_absent(noise):
    p = seasonal_profile(noise, 12, "MS")
    assert p.available
    assert p.kw_p_value is not None and p.kw_p_value > 0.05
    assert "no convincing difference" in p.plain


def test_growing_seasonality_is_only_claimed_when_significant(noise):
    """A non-significant seasonal test must not be followed by 'the swing is growing'."""
    p = seasonal_profile(noise, 12, "MS")
    assert p.amplitude_growing is None


def test_additive_seasonality_is_not_called_growing(additive):
    p = seasonal_profile(additive, 12, "MS")
    assert p.amplitude_growing is not True
    assert p.strength > 0.5


def test_multiplicative_seasonality_is_flagged_as_growing(air):
    p = seasonal_profile(air, 12, "MS")
    assert p.amplitude_growing is True
    assert p.amplitude_late > p.amplitude_early


def test_stl_strength_bounds(air, noise):
    assert stl_strength(air, 12) > 0.8
    assert stl_strength(noise, 12) < 0.5
    assert stl_strength(air.iloc[:10], 12) is None      # too short


def test_amplitude_trend_needs_enough_years(additive):
    growing, early, late = amplitude_trend(additive.iloc[:18], 12, labels_for("MS", 12))
    assert growing is None and early is None and late is None


def test_seasonal_profile_declines_gracefully():
    short = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    p = seasonal_profile(short, 12, "MS")
    assert p.available is False
    assert "no seasonal cycle" in p.plain.lower()


def test_month_year_matrix_shape(air):
    m = month_year_matrix(air, 12, "MS")
    assert m.shape == (12, 12)
    assert list(m.columns)[:2] == ["Jan", "Feb"]
    assert m.index.name == "year"


def test_trend_hint_is_descriptive_not_inferential(air):
    t = trend_hint(air)
    assert t["available"]
    assert t["slope_per_period"] > 0
    assert 0.7 < t["r_squared"] <= 1.0
    assert "p_value" not in t, "S1 must not pretend to have tested anything"


# --------------------------------------------------------------------------- #
# Outliers — labelled, not just flagged
# --------------------------------------------------------------------------- #
def test_injected_spike_is_found_and_called_a_spike(noise):
    spiked = noise.copy()
    spiked.iloc[50] = spiked.iloc[50] + 12 * float(noise.std())
    out, summary = detect_outliers(spiked, period=12)
    flagged = {o.time[:10]: o for o in out}
    hit = flagged.get(str(spiked.index[50])[:10])
    assert hit is not None, f"injected 12-sigma spike was missed; flagged {sorted(flagged)}"
    assert hit.type == "spike"
    assert hit.robust_z > 10
    # the injected point must be the most extreme thing reported, by a distance
    assert max(o.robust_z for o in out) == pytest.approx(hit.robust_z)


def test_level_shift_is_labelled_not_deleted():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    t = np.arange(120)
    vals = np.where(t < 60, 100.0, 165.0) + 12 * np.sin(2 * np.pi * (t % 12) / 12) + rng.normal(0, 3, 120)
    out, summary = detect_outliers(pd.Series(vals, index=idx), period=12)
    kinds = [o.type for o in out]
    assert "level_shift" in kinds
    shift = next(o for o in out if o.type == "level_shift")
    assert "step dummy" in shift.plain
    assert "Do not delete" in shift.plain or "not delete" in shift.plain.lower()


def test_flag_rate_is_never_exceeded():
    """A tight remainder must not be presented as 10% of the series being anomalous."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2010-01-01", periods=400, freq="MS")
    t = np.arange(400)
    vals = 100 + 20 * np.sin(2 * np.pi * (t % 12) / 12) + rng.normal(0, 0.5, 400)
    out, summary = detect_outliers(pd.Series(vals, index=idx), period=12)
    cap = max(2, int(np.ceil(MAX_FLAG_RATE * 400))) + 3     # + up to 3 level shifts
    assert len(out) <= cap
    assert summary["n_flagged"] == len(out)
    assert summary["n_flagged"] <= 400 * MAX_FLAG_RATE + 3


def test_capping_is_applied_and_disclosed_when_everything_looks_extreme():
    """
    When the remainder really is swamped (here: a level shift plus a tiny, very tight
    noise floor), the panel keeps the most extreme points and says that it did.
    """
    rng = np.random.default_rng(11)
    idx = pd.date_range("2010-01-01", periods=240, freq="MS")
    t = np.arange(240)
    vals = 100 + 5 * np.tanh((t - 120) / 3) + rng.standard_t(1.5, 240) * 0.5
    out, summary = detect_outliers(pd.Series(vals, index=idx), period=12, z_cutoff=2.5)
    assert summary["capped"] is True
    assert summary["n_flagged"] <= max(2, int(np.ceil(MAX_FLAG_RATE * 240)))
    assert "most extreme" in summary["plain"]


def test_outlier_panel_is_quiet_for_clean_data(noise):
    """Random noise must not produce a wall of "anomalies" — the panel has to stay honest."""
    _out, summary = detect_outliers(noise, period=12)
    assert summary["n_flagged"] <= 1
    assert summary["plain"]


def test_unusual_for_season_flag_is_separate_from_type(noise):
    spiked = noise.copy()
    spiked.iloc[62] = spiked.iloc[62] + 15 * float(noise.std())
    out, _ = detect_outliers(spiked, period=12)
    assert out
    assert all(isinstance(o.unusual_for_season, bool) for o in out)
    assert out[0].type in {"spike", "level_shift", "temporary_change", "isolated_run"}


def test_outliers_on_tiny_series_do_not_crash():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    out, summary = detect_outliers(s, period=None)
    assert out == [] and summary["available"] is False


# --------------------------------------------------------------------------- #
# Narrator + grounding (§5.14 foundation)
# --------------------------------------------------------------------------- #
def test_narrative_cites_only_computed_numbers(eda_air):
    narrative = next(r for r in eda_air.results if isinstance(r, NarrativeResult))
    assert 3 <= len(narrative.paragraphs) <= 8
    allowed = {k: float(v) for k, v in narrative.numbers_cited.items()
               if isinstance(v, (int, float))}
    text = " ".join(narrative.paragraphs)
    ungrounded = assert_grounded(text, allowed)
    assert ungrounded == [], f"narrative cites numbers that were never computed: {ungrounded}"


def test_narrative_is_not_ai_drafted(eda_air):
    narrative = next(r for r in eda_air.results if isinstance(r, NarrativeResult))
    assert narrative.ai_drafted is False


def test_narrative_mentions_seasonality_and_trend(eda_air):
    text = " ".join(next(r for r in eda_air.results if isinstance(r, NarrativeResult)).paragraphs)
    assert "seasonal" in text.lower()
    assert "trend is" in text.lower() or "upward" in text.lower()


def test_narrative_warns_about_intermittent_data():
    idx = pd.date_range("2015-01-01", periods=96, freq="MS")
    rng = np.random.default_rng(0)
    vals = np.where(rng.random(96) < 0.7, 0.0, rng.integers(1, 8, 96).astype(float))
    out = run_eda(pd.Series(vals, index=idx), name="Spares", frequency_label="Monthly",
                  freq_code="MS", period=12, data_class=DataClass.INTERMITTENT)
    text = " ".join(next(r for r in out.results if isinstance(r, NarrativeResult)).paragraphs)
    assert "intermittent" in text.lower()
    assert "Croston" in text or "MASE" in text


def test_narrator_is_deterministic(air):
    facts = NarrativeFacts(
        series_name="x", n=120, n_missing=0, time_start="2010-01-01", time_end="2019-12-01",
        frequency_label="Monthly", period=12, data_class=DataClass.CONTINUOUS,
        first=100.0, last=150.0, pct_change=50.0, mean=125.0, median=124.0, sd=10.0,
        cv=0.08, minimum=90.0, maximum=160.0, skewness=0.1,
        n_outliers=0, outlier_types={}, n_zeros=0, pct_zeros=0.0,
        n_negatives=0, n_gaps=0, constant_run=1,
    )
    assert narrate(facts) == narrate(facts)


def test_grounding_checker_catches_an_invented_p_value():
    allowed = {"adf_p": 0.612, "n": 144.0}
    assert assert_grounded("The ADF p-value is 0.612.", allowed) == []
    invented = assert_grounded("The ADF p-value is 0.037 and n was 144.", allowed)
    assert "0.037" in " ".join(invented)


def test_numbers_in_normalises_thousands_separators():
    assert "1234.5" in numbers_in("revenue was 1,234.5 units")


# --------------------------------------------------------------------------- #
# The S1 stage output
# --------------------------------------------------------------------------- #
def test_eda_emits_the_expected_result_set(eda_air):
    ids = {r.id for r in eda_air.results}
    expected = {
        "narr.series_description", "table.descriptives", "fig.series",
        "fig.rolling_stats", "fig.distribution", "fig.outliers",
        "fig.lag_plots", "table.seasonal_profile", "fig.seasonal_profile",
        "fig.month_year_heatmap", "fig.seasonal_subseries", "fig.seasonal_box",
        "stat.kruskalwal1is_seasonal", "stat.runs_test",
    }
    assert expected <= ids


def test_every_result_is_on_stage_s1_and_explains_itself(eda_air):
    for r in eda_air.results:
        assert r.stage == Stage.S1_PROFILE or r.stage == "S1"
        assert r.plain and r.why and r.method
        assert len(r.plain) < 220, "plain text must stay one sentence"


def test_every_figure_has_alt_text_and_exportable_json(eda_air):
    figs = [r for r in eda_air.results if isinstance(r, FigureResult)]
    assert len(figs) >= 8
    for f in figs:
        assert f.alt_text and len(f.alt_text) > 20
        assert f.figure_json["data"], f"{f.id} has no traces"
        json.dumps(f.figure_json)          # must be JSON-serialisable for the wire


def test_every_snippet_is_valid_python(eda_air):
    import ast

    for r in eda_air.results:
        if r.reproducible_snippet:
            ast.parse(r.reproducible_snippet)


def test_profiling_a_non_seasonal_series_says_so(noise):
    """The cards may still appear, but nothing may claim a seasonal pattern exists."""
    out = run_eda(noise, name="noise", frequency_label="Monthly", freq_code="MS",
                  period=12, data_class=DataClass.CONTINUOUS)
    ids = {r.id for r in out.results}
    assert "fig.series" in ids
    kw = next((r for r in out.results if r.id == "stat.kruskalwal1is_seasonal"), None)
    assert kw is not None and kw.p_value > 0.05
    assert kw.band == "none"
    assert "no evidence" in kw.interpretation
    # and no result may advise acting on a seasonal pattern
    for r in out.results:
        assert "log transform" not in r.plain.lower()


def test_eda_facts_are_flat_numbers_for_the_advisor(eda_air):
    for key, value in eda_air.facts.items():
        assert isinstance(value, (int, float, bool, str)), f"{key} is not flat"
    assert eda_air.facts["seasonal_strength"] > 0.9
    assert eda_air.facts["variance_ratio"] > 1.5
    assert eda_air.facts["n"] == 144


def test_short_series_is_refused_gracefully():
    idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    out = run_eda(pd.Series([1.0, 2.0, 3.0], index=idx), name="tiny")
    assert out.results == []
    assert out.warnings


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #
def test_profile_endpoint_runs_and_is_idempotent(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    r1 = client.post("/api/v1/analysis/profile", json={}, headers=h)
    assert r1.status_code == 200
    n1 = len(r1.json()["results"])
    assert n1 >= 12
    r2 = client.post("/api/v1/analysis/profile", json={}, headers=h)
    assert len(r2.json()["results"]) == n1, "re-running must replace, not duplicate"
    assert client.get("/api/v1/analysis/profile/results", headers=h).json()["status"] == "complete"


def test_profile_requires_data(client, session_id):
    r = client.post("/api/v1/analysis/profile", json={}, headers={"X-Session-Id": session_id})
    assert r.status_code == 400
    assert "Load a file" in r.json()["error"]["fix"]


def test_facts_endpoint_exposes_what_was_computed(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "ausbeer"}, headers=h)
    client.post("/api/v1/analysis/profile", json={}, headers=h)
    body = client.get("/api/v1/analysis/facts", headers=h).json()
    assert body["eda"]["n"] == 120
    assert body["dataset"]["frequency"] == "QS"
    assert body["seed"] == 20260101


def test_series_endpoint_decimates_for_display_only(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "electricity_daily"}, headers=h)
    body = client.get("/api/v1/analysis/series?limit=100", headers=h).json()
    assert body["n"] == 730
    assert body["n_displayed"] <= 120
    assert body["decimated"] is True
    assert "All statistics use the full series" in body["note"]


def test_eda_endpoint_accepts_a_custom_window(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "retail_sales"}, headers=h)
    r = client.post("/api/v1/analysis/profile", json={"window": 6}, headers=h)
    assert r.status_code == 200
    body = client.get("/api/v1/analysis/profile/results", headers=h).json()
    result = run_eda(
        pd.read_csv(EXAMPLES / "retail_sales.csv", index_col=0).iloc[:, 0].pipe(
            lambda s: s.set_axis(pd.to_datetime(s.index))
        ),
        name="retail", frequency_label="Monthly", freq_code="MS", period=12,
        data_class=DataClass.CONTINUOUS, window=6,
    )
    rolling = next(x for x in result.results if x.id == "fig.rolling_stats")
    assert "6" in rolling.plain or "6" in rolling.title
