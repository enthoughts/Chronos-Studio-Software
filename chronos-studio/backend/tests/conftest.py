"""Shared fixtures. Kept deliberately small — each part adds its own."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def session_id(client) -> str:
    r = client.post("/api/v1/sessions")
    assert r.status_code == 201
    return r.json()["session_id"]
