"""
Part 19 — static verification of the deployment configuration and docs.

    PYTHONPATH=. pytest -q tests/test_part19_deploy.py

The build sandbox has **no Docker daemon**, so a live `docker compose up`
cannot be executed here (stated in METHODS.md §13). These tests validate
everything that can be checked without one: YAML validity, build contexts,
Dockerfile COPY sources, dependency consistency, nginx/SPA proxying, the
single-container free-tier path, env documentation, and the accessibility
primitives in the frontend source.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
FRONTEND = ROOT / "frontend"

COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
PYPROJECT = tomllib.loads((BACKEND / "pyproject.toml").read_text())


# --------------------------------------------------------------------------- #
# Compose
# --------------------------------------------------------------------------- #
def test_compose_has_both_services_with_existing_build_contexts():
    services = COMPOSE["services"]
    assert {"backend", "frontend"} <= set(services)
    for name, svc in services.items():
        ctx = ROOT / svc["build"]["context"]
        assert ctx.is_dir(), f"{name}: missing build context {ctx}"
        assert (ctx / "Dockerfile").is_file(), f"{name}: missing Dockerfile"


def test_compose_frontend_waits_for_a_healthy_backend():
    frontend = COMPOSE["services"]["frontend"]
    assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"
    assert COMPOSE["services"]["backend"]["healthcheck"]["test"]
    assert frontend["ports"] == ["8080:80"]
    assert COMPOSE["services"]["backend"]["expose"] == ["8000"]
    assert frontend["restart"] == "unless-stopped"


def test_compose_has_no_hardcoded_secret():
    blob = (ROOT / "docker-compose.yml").read_text()
    assert "GEMINI_API_KEY=${GEMINI_API_KEY:-}" in blob
    assert not re.search(r"sk-[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_-]{20,}", blob)


def test_compose_sets_privacy_ttl_and_bounded_uploads():
    backend = COMPOSE["services"]["backend"]
    env = "\n".join(backend["environment"])
    assert "CHRONOS_SESSION_TTL_HOURS" in env
    assert "CHRONOS_UPLOAD_DIR=/tmp/chronos_uploads" in env
    assert "CHRONOS_DEBUG=false" in env
    assert backend.get("tmpfs"), "uploads should live on a bounded tmpfs"
    assert any("chronos_uploads" in str(x) for x in backend["tmpfs"])


# --------------------------------------------------------------------------- #
# Dockerfiles
# --------------------------------------------------------------------------- #
def _copy_sources(dockerfile: Path) -> list[str]:
    out: list[str] = []
    for line in dockerfile.read_text().splitlines():
        if re.search(r"\sCOPY\s", line) and "--from=" in line:
            continue                         # source lives in another build stage
        m = re.match(r"\s*COPY\s+(.+?)\s+\S+\s*$", line)
        if m:
            out.extend(m.group(1).split())
    return [s for s in out if not s.startswith("$")]


@pytest.mark.parametrize("rel", ["backend/Dockerfile", "frontend/Dockerfile"])
def test_every_dockerfile_copy_source_exists(rel):
    df = ROOT / rel
    context = df.parent
    for src in _copy_sources(df):
        # Globs such as package-lock.json* are resolved, not taken literally.
        matches = list(context.glob(src)) if any(c in src for c in "*?[") else [context / src]
        assert matches, f"{rel}: COPY source {src!r} does not exist in {context}"


def test_backend_dockerfile_is_non_root_healthy_and_uses_deploy_overlay():
    text = (BACKEND / "Dockerfile").read_text()
    assert re.search(r"^USER chronos", text, re.M)
    assert "useradd" in text and "EXPOSE 8000" in text
    assert "HEALTHCHECK" in text and "/api/v1/health" in text
    assert "requirements-deploy.txt" in text
    assert "pip install --no-cache-dir -r requirements-deploy.txt" in text
    assert "ARG INSTALL_BROWSER=0" in text
    assert "scripts/gen_examples.py" in text


def test_frontend_dockerfile_serves_static_and_uses_a_real_nginx_config():
    text = (FRONTEND / "Dockerfile").read_text()
    assert "npm run build" in text
    assert "/usr/share/nginx/html" in text
    assert "nginx.conf" in text and "printf" not in text
    assert "EXPOSE 80" in text


@pytest.mark.parametrize("rel", [
    "backend/.dockerignore", "frontend/.dockerignore",
])
def test_dockerignore_excludes_generated_files(rel):
    text = (ROOT / rel).read_text()
    assert "node_modules/" in text or "__pycache__/" in text
    assert ".env" in text


# --------------------------------------------------------------------------- #
# Dependency consistency (the Part 0 → Part 19 mismatch)
# --------------------------------------------------------------------------- #
def _pkg(name: str) -> str:
    return re.split(r"[<>=!~\[ ]", name.strip())[0].lower().replace("_", "-")


def test_requirements_covers_every_core_dependency_in_pyproject():
    core = [_pkg(d) for d in PYPROJECT["project"]["dependencies"]]
    reqs = {_pkg(l) for l in (BACKEND / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#") and not l.startswith("-r")}
    missing = [d for d in core if d not in reqs]
    assert not missing, f"requirements.txt is missing {missing}"


def test_deploy_overlay_installs_reported_optional_extras():
    text = (BACKEND / "requirements-deploy.txt").read_text()
    assert "-r requirements.txt" in text
    for pkg in ("python-docx", "playwright", "scikit-learn", "google-genai"):
        assert pkg in text, f"deploy image must install {pkg}"
    extras = PYPROJECT["project"]["optional-dependencies"]
    for group in ("report", "ml", "ai"):
        assert group in extras


def test_pyproject_declares_coverage_gate_and_golden_marker():
    cov = PYPROJECT["tool"]["coverage"]
    assert cov["report"]["fail_under"] >= 80
    assert any("app/ts" in s for s in cov["run"]["source"])
    markers = "\n".join(PYPROJECT["tool"]["pytest"]["ini_options"]["markers"])
    assert "golden" in markers and "slow" in markers


# --------------------------------------------------------------------------- #
# nginx / SPA / single-container free tier
# --------------------------------------------------------------------------- #
def test_nginx_proxies_api_to_backend_and_falls_back_to_index():
    text = (FRONTEND / "nginx.conf").read_text()
    assert "proxy_pass         http://backend:8000" in text
    assert "try_files $uri $uri/ /index.html" in text
    assert "client_max_body_size" in text
    assert "gzip" in text and "proxy_read_timeout" in text


def test_single_container_free_tier_files_are_complete():
    df = (ROOT / "Dockerfile.single").read_text()
    assert "node:20-alpine AS web" in df and "python:3.11-slim" in df
    assert "nginx" in df and "gettext-base" in df
    assert "requirements-deploy.txt" in df
    assert "/app/single-container/entrypoint.sh" in df
    entry = (ROOT / "scripts/single-container/entrypoint.sh").read_text()
    assert "envsubst '${PORT}'" in entry
    assert "uvicorn app.main:app --host 127.0.0.1 --port 8000" in entry
    assert "exec nginx -g 'daemon off;'" in entry
    tpl = (ROOT / "scripts/single-container/nginx.conf.template").read_text()
    assert "listen       ${PORT}" in tpl
    assert "proxy_pass         http://127.0.0.1:8000" in tpl
    vercel = json.loads((FRONTEND / "vercel.json").read_text())
    assert any("/api/" in r["source"] for r in vercel["rewrites"])
    assert any(r["destination"] == "/index.html" for r in vercel["rewrites"])


def test_env_example_documents_every_deploy_variable():
    env = (ROOT / ".env.example").read_text()
    for key in (
        "GEMINI_API_KEY", "CHRONOS_GEMINI_MODEL", "CHRONOS_DEBUG",
        "CHRONOS_ALLOW_ALL_ORIGINS", "CHRONOS_SESSION_TTL_HOURS",
        "CHRONOS_UPLOAD_DIR", "CHRONOS_CHROME", "X13AS_PATH", "INSTALL_BROWSER",
    ):
        assert key in env, f".env.example is missing {key}"


def test_browser_code_never_hardcodes_a_host():
    offenders: list[str] = []
    for path in (FRONTEND / "src").rglob("*"):
        if path.suffix not in {".ts", ".tsx"}:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("*", "//", "/*")):
                continue                      # comments may mention "localhost"
            if re.search(r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)", line):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {stripped[:80]}")
    assert not offenders, f"browser-facing absolute host: {offenders}"


# --------------------------------------------------------------------------- #
# Docs, glossary and accessibility primitives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [
    "README.md", "METHODS.md", "GLOSSARY.md", "CONTRIBUTING.md",
    "CHANGELOG.md", "DEPLOY.md",
])
def test_required_docs_exist_and_are_substantive(name):
    p = ROOT / name
    assert p.is_file() and p.stat().st_size > 1500, name


def test_deploy_guide_covers_the_required_free_tier_platforms():
    text = (ROOT / "DEPLOY.md").read_text().lower()
    for term in ("docker compose up", "render", "railway", "hf spaces",
                 "vercel", "fly.io", "health", "gemini_api_key"):
        assert term in text, f"DEPLOY.md should mention {term}"


def test_methods_document_is_committee_grade():
    text = (ROOT / "METHODS.md").read_text()
    for term in ("adfuller", "KPSS", "SARIMAX", "ExponentialSmoothing",
                 "MASE", "CRPS", "golden", "Fenwick", "Theil", "WCAG",
                 "Performance pass", "Known limitations"):
        assert term in text, f"METHODS.md is missing {term}"
    assert text.count("\n|") > 20, "METHODS.md should tabulate the safeguards and references"


def test_glossary_has_120_terms_in_both_outputs():
    md = (ROOT / "GLOSSARY.md").read_text()
    ts = (FRONTEND / "src" / "copy" / "glossary.ts").read_text()
    md_terms = re.findall(r"^\*\*.+\*\* \(`[a-z0-9_]+`\) — ", md, re.M)
    ts_terms = re.findall(r'^  "([a-z0-9_]+)": Object\.freeze', ts, re.M)
    assert len(md_terms) >= 120 and len(ts_terms) >= 120
    assert len(md_terms) == len(ts_terms), "docs and UI glossaries disagree"
    assert len(set(ts_terms)) == len(ts_terms), "duplicate glossary slug"
    assert f"export const GLOSSARY_TERM_COUNT = {len(ts_terms)};" in ts


def test_glossary_generator_is_the_single_source():
    gen = (ROOT / "scripts/generate_glossary.py").read_text()
    assert "len(TERMS) >= 120" in gen
    assert "GLOSSARY.md" in gen and "glossary.ts" in gen
    assert (BACKEND / "scripts/generate_golden.py").is_file()


def test_accessibility_primitives_are_present_in_source():
    theme = (FRONTEND / "src" / "styles" / "theme.css").read_text()
    assert ":focus-visible" in theme
    assert "prefers-reduced-motion: reduce" in theme

    ribbon = (FRONTEND / "src" / "components" / "Ribbon.tsx").read_text()
    assert 'role="toolbar"' in ribbon
    assert "ribbonNextIndex" in ribbon and "tabIndex" in ribbon
    assert "isEssentialPanel" in ribbon          # Simplify view

    figure = (FRONTEND / "src" / "charts" / "PlotlyFigure.tsx").read_text()
    assert 'role="img"' in figure and "aria-describedby" in figure
    assert "figureValueSummary" in figure

    card = (FRONTEND / "src" / "components" / "ResultCard.tsx").read_text()
    assert "<caption" in card and "tableValueSummary" in card
    assert 'className="result-tech"' in card      # plain-language mode

    app = (FRONTEND / "src" / "App.tsx").read_text()
    assert "skip-link" in app and 'data-simplify' in app and 'data-plain' in app


def test_part19_beginner_help_history_and_error_library_are_wired():
    app = (FRONTEND / "src" / "App.tsx").read_text()
    for term in ("<HelpDrawer", "<ErrorLibraryPanel", "<GlossarySelection",
                 'data-simplify', 'data-plain', "onOpenHelp"):
        assert term in app, f"App.tsx is missing {term}"

    stepper = (FRONTEND / "src" / "components" / "Stepper.tsx").read_text()
    assert "stepNavigation" in stepper
    assert 'data-testid="step-back"' in stepper and 'data-testid="step-next"' in stepper

    history = (FRONTEND / "src" / "components" / "HistoryControls.tsx").read_text()
    assert "undoHistory" in history and "redoHistory" in history
    assert 'data-testid="history-undo"' in history and 'data-testid="history-redo"' in history

    help_copy = (FRONTEND / "src" / "copy" / "beginner.ts").read_text()
    assert "PANEL_HELP" in help_copy and "DEFAULT_PANEL_HELP" in help_copy
    errors = (FRONTEND / "src" / "copy" / "errors.ts").read_text()
    assert "ERROR_LIBRARY" in errors and "errorToastPayload" in errors
    glossary_text = (FRONTEND / "src" / "components" / "GlossaryText.tsx").read_text()
    assert "glossarySegments" in glossary_text and "GlossaryTip" in glossary_text


def test_backend_demo_and_redo_are_mounted():
    from app.main import app
    paths = set(app.openapi()["paths"])
    for p in ("/api/v1/demo/scenarios", "/api/v1/demo/start",
              "/api/v1/demo/status", "/api/v1/demo/cancel",
              "/api/v1/preprocess/redo"):
        assert p in paths, f"missing route {p}"


def test_changelog_records_part_19():
    text = (ROOT / "CHANGELOG.md").read_text()
    assert "Part 19" in text and "Performance" in text and "Golden" in text
