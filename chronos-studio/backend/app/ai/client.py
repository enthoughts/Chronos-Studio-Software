"""
ai/client.py — optional Gemini streaming client, with offline-safe degradation.

The SDK is imported lazily. Without `google-genai` or a key, every code path
returns an offline/degraded answer; the app remains fully usable. The API key is
used only to construct a short-lived client for one request and is never logged,
stored in a session object, echoed in an error, or included in a payload preview.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib.util
import inspect
import os
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.core.config import get_settings
from app.core.errors import AIKeyInvalidError, AIPayloadTooLargeError, ChronosError

from .prompts import (
    build_system_prompt,
    estimate_tokens,
    preview_contents,
    user_payload,
)
from .schemas import (
    AIActionName,
    AIAudience,
    AIMode,
    AIModelChoice,
    AIPayloadPreview,
    AIRequestMode,
    AnalysisDigest,
)

IMAGE_RE = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$")
MAX_IMAGE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ImagePayload:
    data: bytes
    mime: str

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class GeminiStreamError(Exception):
    rate_limited: bool
    code: str
    plain: str
    fix: str

    def __str__(self) -> str:  # pragma: no cover - defensive
        return self.plain


def sdk_available() -> bool:
    """True only when the optional google-genai SDK is importable.

    Deliberately uncached: installing the optional dependency while the server
    is running must be reflected on the next capability request. A stale False
    previously made Live AI look permanently unavailable after installation.
    """
    try:
        return importlib.util.find_spec("google.genai") is not None
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False


def clean_key(raw: str | None) -> str | None:
    """Validate a per-request key without ever returning or logging its value."""
    if raw is None:
        return None
    key = raw.strip()
    if not key:
        return None
    if len(key) > 256 or any(ch.isspace() or ord(ch) < 32 for ch in key):
        raise AIKeyInvalidError()
    return key


def resolve_key(header_key: str | None) -> str | None:
    """Browser-session key wins; an env-provided key is the deployment fallback."""
    supplied = clean_key(header_key)
    if supplied is not None:
        return supplied
    settings = get_settings()
    try:
        return clean_key(os.getenv(settings.gemini_api_key_env))
    except AIKeyInvalidError:
        # A malformed deployment key must not break the offline path.
        return None


def model_name(choice: AIModelChoice | None, mode: AIMode) -> str | None:
    if mode not in {"gemini", "degraded"}:
        return None
    settings = get_settings()
    return settings.gemini_model_pro if choice == "pro" else settings.gemini_model


def resolve_mode(requested: AIRequestMode, key: str | None) -> AIMode:
    if requested == "offline":
        return "offline"
    if sdk_available() and key:
        return "gemini"
    return "degraded" if requested == "gemini" else "offline"


def parse_image(data_url: str | None) -> ImagePayload | None:
    """Accept a small inline PNG/JPEG/WebP image; never persist the bytes."""
    if not data_url:
        return None
    match = IMAGE_RE.match(data_url.strip())
    if match is None:
        raise ChronosError(
            plain="That plot image is not a supported PNG, JPEG or WebP data URL.",
            fix="Take a smaller screenshot, or use the stored figure-data payload instead.",
            code="ai.image_unsupported",
        )
    mime = "image/jpeg" if match.group(1) == "image/jpeg" else match.group(1)
    try:
        data = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ChronosError(
            plain="That plot image could not be read.",
            fix="Attach a fresh PNG, JPEG or WebP screenshot, or use the stored figure data.",
            code="ai.image_unreadable",
        ) from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise AIPayloadTooLargeError()
    return ImagePayload(data=data, mime=mime)


def build_system(request: Any) -> str:
    action: AIActionName = getattr(request, "action", "ask")
    audience: AIAudience = getattr(request, "audience", "beginner")
    return build_system_prompt(action, audience, bool(getattr(request, "eli15", False)))


def build_payload(digest: AnalysisDigest, request: Any) -> str:
    return user_payload(
        digest,
        str(getattr(request, "question", "") or ""),
        str(getattr(request, "panel_id", "S15") or "S15"),
    )


def build_preview(
    digest: AnalysisDigest,
    request: Any,
    *,
    key: str | None,
    image: ImagePayload | None,
    preview_acknowledged: bool = False,
) -> AIPayloadPreview:
    """Exactly the non-secret live payload, plus an honest image summary."""
    mode = resolve_mode(getattr(request, "mode", "auto"), key)
    system_prompt = build_system(request)
    payload = build_payload(digest, request)
    return AIPayloadPreview(
        mode=mode,
        model=model_name(getattr(request, "model", None), mode),
        system_prompt=system_prompt,
        contents=preview_contents(payload, image_attached=image is not None),
        digest=digest,
        question=str(getattr(request, "question", "") or ""),
        panel_id=str(getattr(request, "panel_id", "S15") or "S15"),
        estimated_prompt_tokens=estimate_tokens(system_prompt, payload),
        api_key_in_payload=False,
        data_values_included=bool(digest.data_values),
        image_attached=image is not None,
        image_bytes=image.size if image is not None else 0,
        image_mime=image.mime if image is not None else None,
        preview_acknowledged=preview_acknowledged,
    )


def build_generation_config(types: Any, system_prompt: str, seed: int) -> Any:
    """Temperature ≤ 0.3 and a seed where the installed SDK supports one."""
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": min(float(settings.gemini_temperature), 0.3),
        "max_output_tokens": 1200,
    }
    fields = getattr(types.GenerateContentConfig, "model_fields", {})
    if "seed" in fields:
        kwargs["seed"] = seed
    try:
        return types.GenerateContentConfig(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return types.GenerateContentConfig(**kwargs)


def _gemini_parts(types: Any, payload: str, image: ImagePayload | None) -> list[Any]:
    parts: list[Any] = [types.Part(text=payload)]
    if image is not None:
        try:
            parts.append(types.Part.from_bytes(data=image.data, mime_type=image.mime))
        except (AttributeError, TypeError):
            parts.append(
                types.Part(inline_data=types.Blob(data=image.data, mime_type=image.mime))
            )
    return parts


def _usage_from(chunk: Any) -> tuple[int, int] | None:
    usage = getattr(chunk, "usage_metadata", None)
    if usage is None:
        return None
    prompt = (
        getattr(usage, "prompt_token_count", None)
        or getattr(usage, "input_token_count", None)
        or 0
    )
    response = (
        getattr(usage, "candidates_token_count", None)
        or getattr(usage, "output_token_count", None)
        or 0
    )
    try:
        return int(prompt or 0), int(response or 0)
    except (TypeError, ValueError):
        return None


def _rate_limited(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(term in text for term in (
        "429", "resource_exhausted", "resource exhausted", "quota",
        "rate_limit", "rate limit", "too many requests",
    ))


async def stream_gemini(
    *,
    key: str,
    model: str,
    system_prompt: str,
    payload: str,
    image: ImagePayload | None,
    seed: int,
) -> AsyncIterator[tuple[str, tuple[int, int] | None]]:
    """Yield (text, optional usage) from Gemini; errors are sanitized."""
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - SDK missing in offline deployments
        raise GeminiStreamError(
            rate_limited=False,
            code="ai.sdk_missing",
            plain="The Gemini SDK is not installed on this server.",
            fix="Use offline mode, or install the optional `ai` dependency.",
        ) from exc
    stream: Any = None
    try:
        client = genai.Client(api_key=key)
        config = build_generation_config(types, system_prompt, seed)
        parts = _gemini_parts(types, payload, image)
        contents = [types.Content(role="user", parts=parts)]
        stream = client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        if inspect.isawaitable(stream):
            stream = await stream
        if not hasattr(stream, "__aiter__"):
            raise TypeError("Gemini stream is not async-iterable")
        iterator = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=60)
            except StopAsyncIteration:
                break
            text = getattr(chunk, "text", None) or ""
            usage = _usage_from(chunk)
            if text or usage:
                yield text, usage
    except asyncio.TimeoutError as exc:
        raise GeminiStreamError(
            rate_limited=False,
            code="ai.timeout",
            plain="The live AI took too long to answer.",
            fix="The assistant switched to offline mode. Try a shorter question.",
        ) from exc
    except GeminiStreamError:
        raise
    except Exception as exc:  # no key or raw SDK text is ever surfaced
        limited = _rate_limited(exc)
        raise GeminiStreamError(
            rate_limited=limited,
            code="ai.rate_limited" if limited else "ai.unavailable",
            plain=(
                "The free AI quota has been used up for now."
                if limited else "The live AI is not reachable right now."
            ),
            fix=(
                "The assistant switched to offline mode; its answers still use only computed results."
                if limited else "The assistant switched to offline mode. Check the key in Settings later."
            ),
        ) from exc
    finally:
        # A user cancel must also close the SDK iterator, not just our generator.
        close = getattr(stream, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass


def estimated_cost(prompt_tokens: int, response_tokens: int) -> float:
    settings = get_settings()
    return round(
        (prompt_tokens * settings.ai_input_usd_per_million
         + response_tokens * settings.ai_output_usd_per_million) / 1_000_000,
        6,
    )


def config_response(key: str | None) -> dict[str, Any]:
    settings = get_settings()
    return {
        "offline_ready": True,
        "sdk_available": sdk_available(),
        "env_key_present": settings.has_gemini_key(),
        "live_ready": sdk_available() and bool(key),
        "default_model": settings.gemini_model,
        "pro_model": settings.gemini_model_pro,
        "temperature": min(settings.gemini_temperature, 0.3),
        "max_requests_per_session": settings.ai_max_requests_per_session,
        "max_requests_per_minute": settings.ai_max_requests_per_minute,
        "min_interval_seconds": settings.ai_min_interval_seconds,
        "max_values": settings.ai_max_values,
        "allow_data_values_by_default": settings.ai_allow_data_values_by_default,
        "input_usd_per_million": settings.ai_input_usd_per_million,
        "output_usd_per_million": settings.ai_output_usd_per_million,
    }


__all__ = [
    "IMAGE_RE", "MAX_IMAGE_BYTES", "ImagePayload", "GeminiStreamError",
    "sdk_available", "clean_key", "resolve_key", "model_name", "resolve_mode",
    "parse_image", "build_system", "build_payload", "build_preview",
    "build_generation_config", "stream_gemini", "estimated_cost",
    "config_response",
]
