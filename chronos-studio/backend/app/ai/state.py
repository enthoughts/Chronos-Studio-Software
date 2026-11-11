"""
ai/state.py — per-session chat history, usage counters and free-tier queue.

Everything lives in the in-memory Session and dies with its TTL or a hard delete.
No API key, image bytes or raw data values are stored here.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.errors import AIRequestCapError
from app.core.session import Session, get_store

from .client import estimated_cost
from .schemas import (
    AIHistoryMessage,
    AIHistoryResponse,
    AIUsage,
)

STATE_KEY = "ai_state"
MAX_HISTORY = 40
_GLOBAL_LOCK = threading.RLock()


def _new_state() -> dict[str, Any]:
    return {
        "messages": [],
        "prompt_tokens": 0,
        "response_tokens": 0,
        "preview_acknowledged": False,
        "last_live": 0.0,
        "waiting": 0,
        "degraded": False,
        "rate_limited": False,
        "lock": asyncio.Lock(),
        "loop": None,
    }


def ai_state(session: Session) -> dict[str, Any]:
    """Get/create state, rebinding the asyncio lock if the event loop changed."""
    loop = asyncio.get_running_loop()
    with _GLOBAL_LOCK:
        state = session.objects.get(STATE_KEY)
        if not isinstance(state, dict):
            state = _new_state()
            session.objects[STATE_KEY] = state
        if state.get("loop") is not loop:
            state["loop"] = loop
            state["lock"] = asyncio.Lock()
        return state


def usage_model(session: Session, queued: int = 0) -> AIUsage:
    state = ai_state(session)
    settings = get_settings()
    prompt = int(state.get("prompt_tokens", 0))
    response = int(state.get("response_tokens", 0))
    return AIUsage(
        requests=int(session.ai_requests),
        remaining=max(0, settings.ai_max_requests_per_session - int(session.ai_requests)),
        max_requests=settings.ai_max_requests_per_session,
        prompt_tokens=prompt,
        response_tokens=response,
        estimated_cost_usd=estimated_cost(prompt, response),
        queued=max(0, int(queued)),
        degraded=bool(state.get("degraded", False)),
        rate_limited=bool(state.get("rate_limited", False)),
    )


def history_model(session: Session) -> AIHistoryResponse:
    state = ai_state(session)
    messages = [AIHistoryMessage(**m) for m in state.get("messages", [])]
    return AIHistoryResponse(
        messages=messages,
        usage=usage_model(session, int(state.get("waiting", 0))),
        preview_acknowledged=bool(state.get("preview_acknowledged", False)),
        session_id=session.id,
    )


def append_message(session: Session, message: AIHistoryMessage) -> None:
    state = ai_state(session)
    state["messages"].append(message.model_dump(mode="json"))
    if len(state["messages"]) > MAX_HISTORY:
        state["messages"] = state["messages"][-MAX_HISTORY:]


def clear_history(session: Session) -> None:
    state = ai_state(session)
    state["messages"] = []
    state["degraded"] = False
    state["rate_limited"] = False


def acknowledge_preview(session: Session) -> None:
    ai_state(session)["preview_acknowledged"] = True


def preview_acknowledged(session: Session) -> bool:
    return bool(ai_state(session).get("preview_acknowledged", False))


def check_request_cap(session: Session) -> None:
    cap = get_settings().ai_max_requests_per_session
    if int(session.ai_requests) >= cap:
        raise AIRequestCapError()


def record_request(session: Session) -> None:
    session.ai_requests = int(session.ai_requests) + 1


def add_usage(session: Session, prompt_tokens: int, response_tokens: int) -> None:
    state = ai_state(session)
    state["prompt_tokens"] = int(state.get("prompt_tokens", 0)) + max(0, prompt_tokens)
    state["response_tokens"] = int(state.get("response_tokens", 0)) + max(0, response_tokens)


def mark_degraded(session: Session, rate_limited: bool = False) -> None:
    state = ai_state(session)
    state["degraded"] = True
    if rate_limited:
        state["rate_limited"] = True


def mark_recovered(session: Session) -> None:
    """A successful live answer clears the last degradation indicator."""
    state = ai_state(session)
    state["degraded"] = False
    state["rate_limited"] = False


@asynccontextmanager
async def queued_request(session: Session, *, live: bool) -> AsyncIterator[dict[str, Any]]:
    """Serialise chat requests and space live calls to protect the free tier."""
    state = ai_state(session)
    state["waiting"] = int(state.get("waiting", 0)) + 1
    state["cap_error"] = None
    lock: asyncio.Lock = state["lock"]
    try:
        await lock.acquire()
        try:
            try:
                check_request_cap(session)
            except AIRequestCapError as exc:
                # The cap can be reached while this request waited in the queue.
                # Surface it as an SSE event instead of breaking the stream.
                state["cap_error"] = {
                    "code": exc.code,
                    "plain": exc.plain,
                    "fix": exc.fix,
                }
                yield state
                return
            if live:
                elapsed = time.monotonic() - float(state.get("last_live", 0.0))
                wait = get_settings().ai_min_interval_seconds - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
                state["last_live"] = time.monotonic()
            yield state
        finally:
            if lock.locked():
                lock.release()
    finally:
        state["waiting"] = max(0, int(state.get("waiting", 0)) - 1)


def privacy_delete(session: Session) -> bool:
    """Hard-delete the whole analysis, including this assistant state."""
    return get_store().delete(session.id)


__all__ = [
    "STATE_KEY", "MAX_HISTORY", "ai_state", "usage_model", "history_model",
    "append_message", "clear_history", "acknowledge_preview",
    "preview_acknowledged", "check_request_cap", "record_request",
    "add_usage", "mark_degraded", "mark_recovered", "queued_request",
    "privacy_delete",
]
