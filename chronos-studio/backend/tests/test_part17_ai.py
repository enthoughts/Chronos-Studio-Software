"""Part 17 — grounded, offline-first AI assistant contracts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from app.ai.client import (
    GeminiStreamError,
    build_generation_config,
    clean_key,
    config_response,
    parse_image,
    resolve_mode,
    sdk_available,
)
from app.ai.digest import build_digest
from app.ai.grounding import sanitize_numbers, validate_grounding
from app.ai.narrative import generate_offline
from app.ai.prompts import DRAFT_LABEL, FALLBACK_TEMPLATE, SYSTEM_PROMPT
from app.ai.schemas import AIChatRequest, AnalysisDigest
from app.core.config import get_settings
from app.core.schemas import (
    ChecklistItem,
    FigureResult,
    NarrativeResult,
    Stage,
    StatResult,
    TableColumn,
    TableResult,
    Verdict,
    VerdictResult,
)
from app.core.session import get_store

STAT_ID = "stat.adf_trend"
TABLE_ID = "table.acf_lags"
FIG_ID = "fig.seasonal_profile"
NARR_ID = "narr.gate_summary"
VERDICT_ID = "verdict.gate"
KEY = "SECRET_TEST_KEY_123"
PNG_1X1 = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lZkAAAAASUVORK5CYII="
)


def _headers(sid: str, key: bool = False) -> dict[str, str]:
    h = {"X-Session-Id": sid}
    if key:
        h["X-Gemini-Key"] = KEY
    return h


def _results() -> list:
    return [
        StatResult(
            id=STAT_ID,
            stage=Stage.S5_TEST_STATION,
            title="Augmented Dickey–Fuller test",
            plain="The series still has a drift (non-stationarity).",
            why="A drifting average can make a forecast look better than it is.",
            technical="unit-root test (ADF)",
            method="statsmodels adfuller with a constant and trend",
            value=-2.13,
            p_value=0.237,
            statistic_name="ADF",
            interpretation="No evidence against a unit root.",
            caveats=["The test has low power on a short series."],
        ),
        TableResult(
            id=TABLE_ID,
            stage=Stage.S6_CORRELATE,
            title="Autocorrelation by lag",
            plain="Nearby observations are strongly related (autocorrelation).",
            why="Persistent correlation guides the differencing decision.",
            technical="sample autocorrelation function",
            method="statsmodels acf with adjusted sample size",
            columns=[
                TableColumn(key="lag", label="Lag", dtype="integer"),
                TableColumn(key="acf", label="ACF", dtype="number"),
            ],
            rows=[{"lag": 1, "acf": 0.81}, {"lag": 2, "acf": 0.72}],
            caption="ACF (autocorrelation) by lag.",
        ),
        FigureResult(
            id=FIG_ID,
            stage=Stage.S1_PROFILE,
            title="Seasonal profile",
            plain="The seasonal shape repeats across the year (seasonality).",
            why="A repeating shape suggests a seasonal model may help.",
            technical="seasonal mean profile",
            method="grouped monthly means from the stored Plotly figure",
            figure_json={
                "data": [
                    {
                        "name": "Original series",
                        "type": "scatter",
                        "x": ["Jan", "Feb", "Mar", "Apr"],
                        "y": [12.5, 13.25, 14.0, 13.5],
                    }
                ],
                "layout": {
                    "xaxis": {"title": {"text": "Month"}},
                    "yaxis": {"title": {"text": "Value"}},
                    "height": 260,
                },
            },
            caption="Mean value by month.",
            alt_text="A line chart of the mean value by month.",
        ),
        NarrativeResult(
            id=NARR_ID,
            stage=Stage.S8_GATE,
            title="Readiness summary",
            plain="The gate is ready with caveats.",
            why="Caveats must travel with the model choice.",
            technical="readiness narrative",
            method="deterministic gate narrative",
            paragraphs=["The readiness checks passed with caveats."],
            numbers_cited={"n_models": 4.0},
        ),
        VerdictResult(
            id=VERDICT_ID,
            stage=Stage.S8_GATE,
            title="Ready to model?",
            plain="Ready with caveats (readiness verdict).",
            why="Some diagnostics need attention before submission.",
            technical="weighted checklist verdict",
            method="gate checklist and blocker reconciliation",
            verdict=Verdict.READY_WITH_CAVEATS,
            headline="Ready with caveats.",
            checklist=[ChecklistItem(key="n", label="Sample size", passed=True, value="ok")],
        ),
    ]


def _seeded_session(client):
    sid = client.post("/api/v1/sessions").json()["session_id"]
    s = get_store().get(sid)
    s.objects["series"] = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    s.objects["eda_facts"] = {"mean": 3.5, "std": 1.7, "n_gaps": 0}
    s.objects["next_advice"] = {
        "ranked": [
            {
                "id": "next.diff.guardrail",
                "stage": "S3",
                "recommend": "Difference the series before fitting ARMA.",
                "because": "The correlogram persists and the unit-root p-value is 0.03.",
                "how_to": "Open Prepare data, preview first differencing, then confirm Apply.",
                "provenance": [STAT_ID, TABLE_ID],
                "confidence": "high",
                "kind": "recommendation",
                "rank": 9.0,
            },
            {
                "id": "next.transform.guardrail",
                "stage": "S3",
                "recommend": "Check a log transform only if variance grows with level.",
                "because": "The variance path has not been computed.",
                "how_to": "Open Prepare data and preview transform.log.",
                "provenance": [FIG_ID],
                "confidence": "medium",
                "kind": "recommendation",
                "rank": 7.0,
            },
        ],
        "facts": {"ljung_box_p": 0.03, "r1": 0.81},
    }
    for result in _results():
        s.add_result(result)
    for code in ("S0", "S1", "S5", "S6", "S7", "S8"):
        s.state.stages[code].status = "complete"
    s.state.current_stage = Stage.S8_GATE
    s.state.caveats = ["Seasonal period is inferred from the calendar."]
    return sid, s


def _events(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        event = "message"
        data: dict | None = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if data is not None:
            out.append((event, data))
    return out


def _concat_chunks(events: list[tuple[str, dict]]) -> str:
    buf = ""
    for event, data in events:
        if event == "reset":
            buf = ""
        elif event == "chunk":
            buf += data["text"]
    return buf


def _done(events: list[tuple[str, dict]]) -> dict:
    return next(data for event, data in events if event == "done")


def test_digest_subclass_keeps_part0_contract():
    from app.core.schemas import AnalysisDigest as CoreDigest

    assert issubclass(AnalysisDigest, CoreDigest)
    assert FALLBACK_TEMPLATE in SYSTEM_PROMPT
    assert "result id" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT
    assert "technical term in parentheses" in SYSTEM_PROMPT


def test_config_is_offline_ready_and_never_returns_key(client):
    sid = client.post("/api/v1/sessions").json()["session_id"]
    r = client.get("/api/v1/ai/config", headers=_headers(sid, key=True))
    assert r.status_code == 200
    body = r.json()
    assert body["offline_ready"] is True
    assert body["temperature"] <= 0.3
    assert body["max_requests_per_session"] == get_settings().ai_max_requests_per_session
    assert body["allow_data_values_by_default"] is False
    assert "never logged" in body["privacy_note"]
    assert KEY not in json.dumps(body)
    assert clean_key(KEY) == KEY


def test_preview_hides_raw_values_and_key_and_can_acknowledge(client):
    sid, _ = _seeded_session(client)
    r = client.post("/api/v1/ai/preview", headers=_headers(sid, key=True), json={
        "question": "Explain the stationarity result",
        "action": "explain",
        "result_id": STAT_ID,
        "acknowledge": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["api_key_in_payload"] is False
    assert body["preview_acknowledged"] is True
    assert body["digest"]["data_values"] is None
    assert body["data_values_included"] is False
    assert body["estimated_prompt_tokens"] > 0
    assert FALLBACK_TEMPLATE in body["system_prompt"]
    assert KEY not in json.dumps(body)
    raw = json.dumps(body)
    assert "12.5" not in raw and "13.25" not in raw  # figure trace values

    with_values = client.post("/api/v1/ai/preview", headers=_headers(sid, key=True), json={
        "question": "Show the values",
        "action": "ask",
        "include_values": True,
    })
    assert with_values.status_code == 200
    wv = with_values.json()
    assert wv["data_values_included"] is True
    assert len(wv["digest"]["data_values"]) <= get_settings().ai_max_values
    assert KEY not in json.dumps(wv)


def test_digest_and_number_grounding_validator(client):
    sid, s = _seeded_session(client)
    digest = build_digest(s, panel_id="S8", focus_result_id=STAT_ID)
    assert digest.data_values is None
    assert digest.data_values_included is False
    assert STAT_ID in digest.result_ids
    assert digest.top_results[0].id == STAT_ID
    assert digest.stage_verdicts["S5"] == "complete"
    assert digest.facts_used
    assert 0.237 in digest.numbers.values() or any(
        abs(v - 0.237) < 1e-9 for v in digest.numbers.values()
    )

    bad = validate_grounding("A made-up value is 987654.321", digest)
    assert bad.grounded is False
    assert "987654.321" in bad.ungrounded_numbers
    assert bad.missing_citations is True
    assert "987654.321" not in sanitize_numbers("A made-up value is 987654.321", digest)

    good = validate_grounding(
        f"The unit-root test reports {digest.numbers.get(f'{STAT_ID}.p_value', 0.237)} [{STAT_ID}]",
        digest,
    )
    assert good.grounded is True
    assert good.citations == [STAT_ID]

    fallback = validate_grounding(
        "the app has not computed that yet — open the Diagnostics panel", digest
    )
    assert fallback.grounded is True
    assert fallback.fallback_used is True


def test_offline_stream_is_grounded_cites_result_and_does_not_store_key(client):
    sid, s = _seeded_session(client)
    r = client.post("/api/v1/ai/chat/stream", headers=_headers(sid, key=True), json={
        "question": "Explain this result",
        "action": "explain",
        "result_id": STAT_ID,
        "panel_id": "S5",
        "mode": "offline",
    })
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert any(event == "meta" for event, _ in events)
    done = _done(events)
    assert _concat_chunks(events) == done["text"]
    assert done["mode"] == "offline"
    assert done["grounding"]["grounded"] is True
    assert STAT_ID in done["citations"]
    assert done["usage"]["requests"] == 1
    assert done["usage"]["remaining"] == get_settings().ai_max_requests_per_session - 1
    assert done["usage"]["prompt_tokens"] > 0
    assert KEY not in r.text
    assert KEY not in str(s.objects.get("ai_state", {}))

    history = client.get("/api/v1/ai/history", headers=_headers(sid)).json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]
    assert history["messages"][1]["citations"] == [STAT_ID]
    assert history["usage"]["requests"] == 1


def test_what_next_uses_ranked_rules_and_clickable_actions(client):
    sid, _ = _seeded_session(client)
    r = client.post("/api/v1/ai/chat/stream", headers=_headers(sid), json={
        "question": "What should I do next?",
        "action": "what_next",
        "panel_id": "S7",
        "mode": "offline",
    })
    assert r.status_code == 200
    done = _done(_events(r.text))
    assert "ranked advisor rules" in done["text"]
    assert "next.diff.guardrail" in done["text"]
    assert "Difference the series" in done["text"]
    assert "the app has not computed" not in done["text"]
    assert done["grounding"]["grounded"] is True
    assert STAT_ID in done["citations"] or TABLE_ID in done["citations"]
    assert done["actions"]
    assert all(a["kind"] in {"open_panel", "explain_result"} for a in done["actions"])
    assert all(a["stage"].startswith("S") for a in done["actions"])
    assert any("Prepare data" in a["label"] for a in done["actions"])


def test_what_next_without_rules_uses_exact_fallback(client):
    sid, s = _seeded_session(client)
    s.objects.pop("next_advice", None)
    r = client.post("/api/v1/ai/chat/stream", headers=_headers(sid), json={
        "question": "What next?",
        "action": "what_next",
        "panel_id": "S7",
        "mode": "offline",
    })
    done = _done(_events(r.text))
    assert done["text"] == "the app has not computed that yet — open the What next panel"
    assert done["grounding"]["fallback_used"] is True
    assert done["grounding"]["grounded"] is True
    assert done["citations"] == []


def test_write_up_is_labelled_and_uses_computed_numbers(client):
    sid, _ = _seeded_session(client)
    r = client.post("/api/v1/ai/chat/stream", headers=_headers(sid), json={
        "question": "Write it up",
        "action": "write_up",
        "panel_id": "S15",
        "mode": "offline",
    })
    done = _done(_events(r.text))
    assert DRAFT_LABEL in done["text"]
    assert done["ai_drafted"] is True
    assert done["draft_label"] == DRAFT_LABEL
    for heading in ("Methods", "Results", "Diagnostics", "Discussion"):
        assert heading in done["text"]
    assert STAT_ID in done["citations"]
    assert done["grounding"]["grounded"] is True


def test_interpret_plot_from_figure_payload_or_image(client):
    sid, _ = _seeded_session(client)
    preview = client.post("/api/v1/ai/preview", headers=_headers(sid), json={
        "question": "Interpret this plot",
        "action": "interpret_plot",
        "result_id": FIG_ID,
        "image_data_url": PNG_1X1,
    })
    assert preview.status_code == 200
    assert preview.json()["image_attached"] is True
    assert preview.json()["image_bytes"] > 0
    assert PNG_1X1 not in json.dumps(preview.json())

    r = client.post("/api/v1/ai/chat/stream", headers=_headers(sid), json={
        "question": "Interpret this plot",
        "action": "interpret_plot",
        "result_id": FIG_ID,
        "panel_id": "S1",
        "mode": "offline",
        "image_data_url": PNG_1X1,
    })
    done = _done(_events(r.text))
    assert "figure-data payload" in done["text"]
    assert "Original series" in done["text"]
    assert "image was attached" in done["text"]
    assert FIG_ID in done["citations"]
    assert done["grounding"]["grounded"] is True
    assert "12.5" not in done["text"]  # no raw trace values by default


def test_explain_result_endpoint_is_wired_to_offline_generator(client):
    sid, _ = _seeded_session(client)
    r = client.post(
        f"/api/v1/sessions/{sid}/results/{STAT_ID}/explain",
        headers=_headers(sid),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["grounded"] is True
    assert STAT_ID in body["citations"]
    assert body["offline_answer"]
    assert "never logged" in body["privacy_note"]


def test_explain_unknown_result_is_still_not_an_error(client):
    sid = client.post("/api/v1/sessions").json()["session_id"]
    r = client.post(f"/api/v1/sessions/{sid}/results/nope/explain", headers=_headers(sid))
    assert r.status_code == 200
    assert r.json()["found"] is False


def test_hard_cap_privacy_delete_and_clear_history(client):
    sid, s = _seeded_session(client)
    cap = get_settings().ai_max_requests_per_session
    s.ai_requests = cap
    limited = client.post("/api/v1/ai/chat/stream", headers=_headers(sid), json={
        "question": "One more?", "action": "ask", "mode": "offline",
    })
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "ai.request_cap"

    fresh, _ = _seeded_session(client)
    stream = client.post("/api/v1/ai/chat/stream", headers=_headers(fresh), json={
        "question": "Hello", "action": "ask", "mode": "offline",
    })
    assert stream.status_code == 200
    before = client.get("/api/v1/ai/history", headers=_headers(fresh)).json()
    assert len(before["messages"]) == 2
    cleared = client.delete("/api/v1/ai/history", headers=_headers(fresh))
    assert cleared.status_code == 200
    after = client.get("/api/v1/ai/history", headers=_headers(fresh)).json()
    assert after["messages"] == []
    deleted = client.delete("/api/v1/ai/session", headers=_headers(fresh))
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get("/api/v1/ai/history", headers=_headers(fresh)).status_code == 404


def test_preview_required_before_first_live_request(client, monkeypatch):
    sid, _ = _seeded_session(client)
    monkeypatch.setattr("app.api.ai.resolve_mode", lambda *_a, **_k: "gemini")
    r = client.post("/api/v1/ai/chat/stream", headers=_headers(sid, key=True), json={
        "question": "Use live AI", "action": "ask", "mode": "gemini",
    })
    assert r.status_code == 428
    assert r.json()["error"]["code"] == "ai.preview_required"


def test_gemini_rate_limit_degrades_to_offline_without_partial_history(client, monkeypatch):
    sid, _ = _seeded_session(client)
    ack = client.post("/api/v1/ai/preview", headers=_headers(sid, key=True), json={
        "question": "Hello", "action": "ask", "acknowledge": True,
    })
    assert ack.json()["preview_acknowledged"] is True
    monkeypatch.setattr("app.api.ai.resolve_mode", lambda *_a, **_k: "gemini")
    settings = get_settings()
    monkeypatch.setattr(settings, "ai_min_interval_seconds", 0.0)

    async def failing(**_kwargs):
        raise GeminiStreamError(
            rate_limited=True,
            code="ai.rate_limited",
            plain="The free AI quota has been used up for now.",
            fix="The assistant switched to offline mode.",
        )
        yield "", None  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("app.api.ai.stream_gemini", failing)
    r = client.post("/api/v1/ai/chat/stream", headers=_headers(sid, key=True), json={
        "question": "Explain the ADF result",
        "action": "explain",
        "result_id": STAT_ID,
        "mode": "gemini",
    })
    events = _events(r.text)
    assert any(event == "error" for event, _ in events)
    assert any(event == "reset" for event, _ in events)
    done = _done(events)
    assert done["mode"] == "degraded"
    assert done["usage"]["rate_limited"] is True
    assert done["grounding"]["grounded"] is True
    assert STAT_ID in done["citations"]
    assert _concat_chunks(events) == done["text"]
    history = client.get("/api/v1/ai/history", headers=_headers(sid)).json()
    assert history["messages"][-1]["role"] == "assistant"
    assert history["messages"][-1]["mode"] == "degraded"


def test_clean_cancel_saves_no_partial_assistant_message(client):
    from app.api.ai import ai_chat_stream

    sid, s = _seeded_session(client)

    class DisconnectedRequest:
        def __init__(self) -> None:
            self.calls = 0

        async def is_disconnected(self) -> bool:
            self.calls += 1
            return True

    async def run():
        response = await ai_chat_stream(
            AIChatRequest(question="Explain this", action="explain", result_id=STAT_ID, mode="offline"),
            DisconnectedRequest(),
            s,
            None,
            None,
        )
        collected: list[str] = []
        async for event in response.body_iterator:
            collected.append(event)
        return collected, __import__("app.ai.state", fromlist=["history_model"]).history_model(s)

    collected, history = asyncio.run(run())
    assert not any("event: done" in item for item in collected)
    assert [m.role for m in history.messages] == []
    assert s.ai_requests == 0
    assert s.objects["ai_state"]["waiting"] == 0


def test_generation_config_uses_low_temperature_and_seed_when_supported():
    class ConfigWithSeed:
        model_fields = {"seed": None, "temperature": None}

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ConfigWithoutSeed(ConfigWithSeed):
        model_fields = {}

    types = SimpleNamespace(GenerateContentConfig=ConfigWithSeed)
    cfg = build_generation_config(types, "system prompt", 20260101)
    assert cfg.kwargs["system_instruction"] == "system prompt"
    assert cfg.kwargs["temperature"] <= 0.3
    assert cfg.kwargs["seed"] == 20260101

    types2 = SimpleNamespace(GenerateContentConfig=ConfigWithoutSeed)
    cfg2 = build_generation_config(types2, "system prompt", 20260101)
    assert "seed" not in cfg2.kwargs


def test_offline_generator_handles_eli15_audience_and_simplify(client):
    sid, s = _seeded_session(client)
    digest = build_digest(s, panel_id="S5", focus_result_id=STAT_ID, eli15=True)
    for action, audience in (("eli15", "beginner"), ("simplify", "peer"), ("audience", "reviewer")):
        request = SimpleNamespace(
            action=action, audience=audience, eli15=action == "eli15",
            result_id=STAT_ID, panel_id="S5", question="Explain it simply",
        )
        answer = generate_offline(digest, request)
        report = validate_grounding(answer.text, digest)
        assert report.grounded is True
        assert STAT_ID in answer.citations
        assert "(" in answer.text and ")" in answer.text


def test_parse_image_rejects_unsupported_and_too_large():
    image = parse_image(PNG_1X1)
    assert image is not None
    assert image.mime == "image/png"
    assert image.size > 0
    with pytest.raises(Exception):
        parse_image("data:text/plain;base64,QQ==")


def test_sdk_capability_is_not_cached(monkeypatch):
    """Installing google-genai at runtime must change the capability result."""
    import app.ai.client as client

    monkeypatch.setattr(
        client.importlib.util, "find_spec",
        lambda name: object() if name == "google.genai" else None,
    )
    assert sdk_available() is True
    monkeypatch.setattr(client.importlib.util, "find_spec", lambda name: None)
    assert sdk_available() is False


def test_live_mode_requires_both_sdk_and_key(monkeypatch):
    import app.ai.client as client

    monkeypatch.setattr(client, "sdk_available", lambda: True)
    assert resolve_mode("auto", "DUMMY_KEY") == "gemini"
    assert resolve_mode("offline", "DUMMY_KEY") == "offline"
    monkeypatch.setattr(client, "sdk_available", lambda: False)
    assert resolve_mode("auto", "DUMMY_KEY") == "offline"
    assert resolve_mode("gemini", "DUMMY_KEY") == "degraded"


def test_config_response_reports_live_only_with_sdk_and_key(monkeypatch):
    import app.ai.client as client

    monkeypatch.setattr(client, "sdk_available", lambda: True)
    ready = config_response("DUMMY_KEY")
    assert ready["sdk_available"] is True
    assert ready["live_ready"] is True
    assert "DUMMY_KEY" not in json.dumps(ready)
    monkeypatch.setattr(client, "sdk_available", lambda: False)
    blocked = config_response("DUMMY_KEY")
    assert blocked["sdk_available"] is False
    assert blocked["live_ready"] is False
