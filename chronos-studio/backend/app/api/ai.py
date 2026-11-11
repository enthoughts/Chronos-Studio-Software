"""
api/ai.py — Part 17's grounded assistant endpoints.

Routes (all under /api/v1):
  GET    /ai/config         capability + quota + privacy settings
  POST   /ai/preview        exact non-secret payload before first live use
  GET    /ai/history        per-session chat history + usage
  DELETE /ai/history        clear assistant history only
  DELETE /ai/session        privacy control: hard-delete the whole session
  POST   /ai/chat/stream    SSE: meta / chunk / reset / error / done

The live Gemini path is entirely optional. Offline mode uses the deterministic
narrative generator and works with no SDK, no key and no network call.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import aclosing
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ai.client import (
    GeminiStreamError,
    build_payload,
    build_preview,
    build_system,
    config_response,
    model_name,
    parse_image,
    resolve_key,
    resolve_mode,
    stream_gemini,
)
from app.ai.digest import build_digest
from app.ai.grounding import validate_grounding
from app.ai.narrative import generate_offline
from app.ai.prompts import DRAFT_LABEL, estimate_tokens, panel_label
from app.ai.schemas import (
    AIAction,
    AIChatRequest,
    AIConfigResponse,
    AIHistoryMessage,
    AIHistoryResponse,
    AIPayloadPreview,
    AIPreviewRequest,
    AIStreamChunk,
    AIStreamDone,
    AIStreamError,
    AIStreamMeta,
    AIUsage,
    AIMode,
)
from app.ai.state import (
    acknowledge_preview,
    ai_state,
    append_message,
    check_request_cap,
    clear_history,
    history_model,
    mark_degraded,
    mark_recovered,
    preview_acknowledged,
    privacy_delete,
    queued_request,
    record_request,
    add_usage,
    usage_model,
)
from app.api.deps import session_dep
from app.core.config import get_settings
from app.core.errors import AIPreviewRequiredError
from app.core.session import Session

router = APIRouter(prefix="/ai", tags=["ai"])


def _sse(event: str, data: BaseModel | dict[str, Any]) -> str:
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n"
    )


def _chunks(text: str, max_chars: int = 96) -> list[str]:
    parts = re.findall(r"\S+\s*|\n", text)
    out: list[str] = []
    buf = ""
    for part in parts:
        if buf and len(buf) + len(part) > max_chars:
            out.append(buf)
            buf = part
        else:
            buf += part
    if buf:
        out.append(buf)
    return out or [text]


def _ack_header(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "ack", "acknowledged"}


def _open_action(request: AIChatRequest, result_id: str | None) -> AIAction:
    return AIAction(
        label=f"Open {panel_label(request.panel_id)}",
        stage=request.panel_id,
        kind="explain_result" if result_id else "open_panel",
        result_id=result_id,
        description="User-confirmed navigation only; the assistant cannot mutate the analysis.",
    )


@router.get("/config", response_model=AIConfigResponse)
async def ai_config(
    x_gemini_key: str | None = Header(default=None, alias="X-Gemini-Key"),
) -> AIConfigResponse:
    """Capability flags. No key value is ever returned."""
    return AIConfigResponse(**config_response(resolve_key(x_gemini_key)))


@router.post("/preview", response_model=AIPayloadPreview)
async def ai_preview(
    body: AIPreviewRequest,
    session: Session = Depends(session_dep),
    x_gemini_key: str | None = Header(default=None, alias="X-Gemini-Key"),
) -> AIPayloadPreview:
    """Show exactly what a live request would send, before the first use."""
    key = resolve_key(x_gemini_key)
    image = parse_image(body.image_data_url)
    digest = build_digest(
        session,
        panel_id=body.panel_id,
        focus_result_id=body.result_id,
        include_values=bool(body.include_values or get_settings().ai_allow_data_values_by_default),
        eli15=body.eli15,
        prefer_kind="figure" if body.action == "interpret_plot" else None,
        max_values=get_settings().ai_max_values,
    )
    if body.acknowledge:
        acknowledge_preview(session)
    return build_preview(
        digest,
        body,
        key=key,
        image=image,
        preview_acknowledged=preview_acknowledged(session),
    )


@router.get("/history", response_model=AIHistoryResponse)
async def ai_history(session: Session = Depends(session_dep)) -> AIHistoryResponse:
    return history_model(session)


@router.delete("/history")
async def ai_clear_history(session: Session = Depends(session_dep)) -> dict[str, Any]:
    clear_history(session)
    return {"cleared": True, "usage": usage_model(session).model_dump(mode="json")}


@router.delete("/session")
async def ai_delete_session(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """Delete-my-session: hard-delete data, results, fitted objects and chat."""
    sid = session.id
    return {"deleted": privacy_delete(session), "session_id": sid}


@router.post("/chat/stream")
async def ai_chat_stream(
    body: AIChatRequest,
    request: Request,
    session: Session = Depends(session_dep),
    x_gemini_key: str | None = Header(default=None, alias="X-Gemini-Key"),
    x_preview_ack: str | None = Header(default=None, alias="X-Chronos-AI-Preview-Ack"),
) -> StreamingResponse:
    """SSE chat with clean cancel, grounding validation and offline fallback."""
    key = resolve_key(x_gemini_key)
    mode = resolve_mode(body.mode, key)
    image = parse_image(body.image_data_url)
    if _ack_header(x_preview_ack) and not preview_acknowledged(session):
        acknowledge_preview(session)
    if mode == "gemini" and not preview_acknowledged(session):
        raise AIPreviewRequiredError()
    check_request_cap(session)

    settings = get_settings()
    digest = build_digest(
        session,
        panel_id=body.panel_id,
        focus_result_id=body.result_id,
        include_values=bool(body.include_values or settings.ai_allow_data_values_by_default),
        eli15=body.eli15,
        prefer_kind="figure" if body.action == "interpret_plot" else None,
        max_values=settings.ai_max_values,
    )
    offline = generate_offline(digest, body, image_attached=image is not None)
    system_prompt = build_system(body)
    payload = build_payload(digest, body)
    estimated_prompt = estimate_tokens(system_prompt, payload)
    planned_model = model_name(body.model, mode) if mode == "gemini" else None
    state = ai_state(session)
    queued = int(state.get("waiting", 0))

    async def events() -> AsyncIterator[str]:
        nonlocal mode, planned_model
        used_mode: AIMode = mode
        live = mode == "gemini"
        rate_limited = False
        if mode == "degraded":
            mark_degraded(session)
        if live and queued > 2:
            # A free-tier queue must not make the user wait indefinitely.
            live = False
            used_mode = "degraded"
            planned_model = None
            mark_degraded(session)

        async with queued_request(session, live=live) as ai:
            cap_error = ai.get("cap_error")
            if cap_error:
                yield _sse("error", AIStreamError(**cap_error))
                return
            if await request.is_disconnected():
                return
            record_request(session)
            append_message(session, AIHistoryMessage(
                role="user",
                content=body.question,
                action=body.action,
                panel_id=body.panel_id,
                mode="offline",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            yield _sse("meta", AIStreamMeta(
                mode=used_mode,
                model=planned_model,
                action=body.action,
                panel_id=body.panel_id,
                citations=offline.citations,
                usage=usage_model(session, int(ai.get("waiting", 0))),
            ))

            full_text = ""
            usage_seen = (0, 0)
            cancelled = False
            if live:
                try:
                    async with aclosing(stream_gemini(
                        key=key or "",
                        model=planned_model or settings.gemini_model,
                        system_prompt=system_prompt,
                        payload=payload,
                        image=image,
                        seed=session.seed,
                    )) as live_stream:
                        async for text, usage in live_stream:
                            if await request.is_disconnected():
                                cancelled = True
                                break
                            if usage is not None:
                                usage_seen = (
                                    max(usage_seen[0], usage[0]),
                                    max(usage_seen[1], usage[1]),
                                )
                            if text:
                                full_text += text
                                yield _sse("chunk", AIStreamChunk(text=text))
                            await asyncio.sleep(0)
                    if not full_text.strip():
                        raise GeminiStreamError(
                            rate_limited=False,
                            code="ai.empty_reply",
                            plain="The live AI returned an empty answer.",
                            fix="The assistant switched to offline mode.",
                        )
                except GeminiStreamError as exc:
                    rate_limited = exc.rate_limited
                    used_mode = "degraded"
                    planned_model = None
                    mark_degraded(session, exc.rate_limited)
                    yield _sse("error", AIStreamError(
                        code=exc.code, plain=exc.plain, fix=exc.fix
                    ))
                    yield _sse("reset", {"text": ""})
                    full_text = ""
                    usage_seen = (0, 0)
                except Exception:
                    # No raw exception text, key or SDK detail is ever streamed.
                    used_mode = "degraded"
                    planned_model = None
                    mark_degraded(session)
                    yield _sse("error", AIStreamError(
                        code="ai.unavailable",
                        plain="The live AI is not reachable right now.",
                        fix="The assistant switched to offline mode.",
                    ))
                    yield _sse("reset", {"text": ""})
                    full_text = ""
                    usage_seen = (0, 0)

            if cancelled:
                # Clean cancel: no partial assistant message is saved.
                return

            if used_mode != "gemini" or not full_text:
                used_mode = "degraded" if mode != "offline" else "offline"
                full_text = ""
                for chunk in _chunks(offline.text):
                    if await request.is_disconnected():
                        return
                    full_text += chunk
                    yield _sse("chunk", AIStreamChunk(text=chunk))
                    await asyncio.sleep(0.005)

            if await request.is_disconnected():
                return

            prompt_tokens = usage_seen[0] or estimated_prompt
            response_tokens = usage_seen[1] or estimate_tokens(full_text)
            add_usage(session, prompt_tokens, response_tokens)
            grounding = validate_grounding(full_text, digest)
            citations = grounding.citations or offline.citations
            actions = list(offline.actions)
            if not actions:
                actions.append(_open_action(
                    body,
                    body.result_id or (digest.result_ids[0] if digest.result_ids else None),
                ))
            if used_mode == "gemini":
                mark_recovered(session)
            usage: AIUsage = usage_model(session, int(ai.get("waiting", 0)))
            usage.degraded = used_mode == "degraded" or usage.degraded
            usage.rate_limited = rate_limited or usage.rate_limited
            done = AIStreamDone(
                text=full_text,
                grounding=grounding,
                citations=citations,
                actions=actions,
                usage=usage,
                mode=used_mode,
                ai_drafted=body.action == "write_up",
                draft_label=DRAFT_LABEL if body.action == "write_up" else "",
            )
            append_message(session, AIHistoryMessage(
                role="assistant",
                content=full_text,
                action=body.action,
                panel_id=body.panel_id,
                citations=citations,
                grounded=grounding.grounded,
                ungrounded_numbers=grounding.ungrounded_numbers,
                mode=used_mode,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            yield _sse("done", done)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
