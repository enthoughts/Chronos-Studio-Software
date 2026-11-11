"""Print-CSS PDF via the locally installed Chromium (no weasyprint needed)."""

from __future__ import annotations

from typing import Any

from app.core.browser import find_chromium
from app.core.errors import ChronosError
from app.ts.report.bundle import ReportBundle
from app.ts.report.html import render_html


def chrome_path() -> str:
    found = find_chromium()
    if found:
        return found
    raise ChronosError(
        code="export.chromium_missing",
        plain="PDF export needs a local Chromium, but none was found.",
        fix="Install Chromium or set CHRONOS_CHROME to its executable path, then try again.",
        technical="The PDF is printed from the self-contained HTML using print-CSS; no network renderer is used.",
    )


def render_pdf(bundle: ReportBundle) -> bytes:
    html = render_html(bundle).decode("utf-8")
    executable = chrome_path()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - dependency is installed in this environment
        raise ChronosError(
            code="export.playwright_missing",
            plain="PDF export needs the Playwright Python package.",
            fix="Install playwright, then try the PDF export again.",
            technical=str(exc),
        ) from exc
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=executable,
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            try:
                page = browser.new_page(viewport={"width": 1200, "height": 1600})
                page.emulate_media(media="print")
                page.set_content(html, wait_until="load", timeout=60_000)
                page.wait_for_timeout(250)
                data = page.pdf(
                    format="A4", print_background=True, prefer_css_page_size=True,
                    margin={"top": "14mm", "bottom": "14mm", "left": "14mm", "right": "14mm"},
                )
            finally:
                browser.close()
    except ChronosError:
        raise
    except Exception as exc:
        raise ChronosError(
            code="export.pdf_failed",
            plain="The HTML report could not be printed to PDF.",
            fix="Download the one-file HTML and print it from your browser, or try again.",
            technical=f"{type(exc).__name__}: {exc}",
        ) from exc
    if not data.startswith(b"%PDF"):
        raise ChronosError(
            code="export.pdf_empty",
            plain="The PDF renderer returned an empty file.",
            fix="Download the one-file HTML and print it from your browser.",
        )
    return bytes(data)


__all__ = ["chrome_path", "render_pdf"]
