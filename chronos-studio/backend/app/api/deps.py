"""api/deps.py — shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Header

from app.core.session import Session, get_store


def session_dep(x_session_id: str = Header(..., alias="X-Session-Id")) -> Session:
    """Every analysis endpoint carries the session in a header, not the URL."""
    return get_store().get(x_session_id)


__all__ = ["session_dep"]
