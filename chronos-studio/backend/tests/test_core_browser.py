"""The standalone build must not ship a developer-machine browser path."""

from __future__ import annotations

from pathlib import Path

from app.core.browser import chromium_candidates, find_chromium


def test_find_chromium_honours_an_explicit_executable(tmp_path, monkeypatch):
    executable = tmp_path / "chrome"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setenv("CHRONOS_CHROME", str(executable))
    monkeypatch.delenv("CHROME_PATH", raising=False)
    assert find_chromium() == str(executable)


def test_browser_lookup_has_no_hardcoded_sandbox_cache_path(monkeypatch):
    monkeypatch.delenv("CHRONOS_CHROME", raising=False)
    monkeypatch.delenv("CHROME_PATH", raising=False)
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    source = Path(__file__).resolve().parents[1] / "app" / "core" / "browser.py"
    assert "chromium-1248" not in source.read_text(encoding="utf-8")
    assert "/home/user/.cache" not in source.read_text(encoding="utf-8")
    assert all("chromium-1248" not in candidate for candidate in chromium_candidates())
