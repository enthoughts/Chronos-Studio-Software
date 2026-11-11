"""Locate a local Chromium/Chrome without embedding a machine-specific path.

The offline wheelhouse omits both Kaleido and a browser binary.  Publication
PNG/PDF exports are therefore optional: when a user installs a system Chromium
or points ``CHRONOS_CHROME`` at one, these helpers find it.  Playwright's cache
is searched relative to the current user and ``PLAYWRIGHT_BROWSERS_PATH``; no
developer-sandbox path is shipped in the standalone package.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SYSTEM_CANDIDATES = (
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
)


def chromium_candidates() -> list[str]:
    """Return executable paths to check, in priority order."""
    candidates: list[str] = [
        os.environ.get("CHRONOS_CHROME") or "",
        os.environ.get("CHROME_PATH") or "",
    ]
    roots: list[Path] = []
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if override and override != "0":
        roots.append(Path(override))
    try:
        roots.append(Path.home() / ".cache" / "ms-playwright")
    except (OSError, RuntimeError):
        pass
    patterns = (
        "chromium-*/chrome-linux*/chrome",
        "chromium-*/chrome-linux*/headless_shell",
        "chromium_headless_shell-*/chrome-linux*/headless_shell",
        "chromium_headless_shell-*/chrome-linux*/chrome-headless-shell",
    )
    for root in roots:
        try:
            for pattern in patterns:
                candidates.extend(str(path) for path in sorted(root.glob(pattern)))
        except (OSError, ValueError):
            continue
    candidates.extend(SYSTEM_CANDIDATES)
    return [candidate for candidate in candidates if candidate]


def find_chromium() -> str | None:
    """Return an existing executable, or ``None`` when no local browser is installed."""
    for candidate in chromium_candidates():
        try:
            if Path(candidate).is_file():
                return candidate
        except OSError:
            continue
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    return None


__all__ = ["SYSTEM_CANDIDATES", "chromium_candidates", "find_chromium"]
