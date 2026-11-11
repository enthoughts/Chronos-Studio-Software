"""
core/config.py — one place for every setting.

Free-tier-first (§4.1): no paid service is required to run the app. Optional
features are enabled by the presence of a package or an env var, and every one
of them must degrade gracefully when absent.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.browser import find_chromium

BACKEND_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = BACKEND_ROOT / "app" / "assets"


def _load_dotenv() -> int:
    """
    Read `backend/.env` into the environment, if it exists.

    Free-tier-first (§4.1) means the app must run with no `.env` at all, so this is best-effort and
    silent: no file, unreadable file, or malformed line is not an error. It deliberately does **not**
    override an already-set variable — a shell export wins over the file, which is what you want when
    a value is being overridden for one run. Nothing here is ever logged or returned by an endpoint:
    `has_gemini_key()` answers yes/no, and the key itself never leaves this process.
    """
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return 0
    loaded = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:  # pragma: no cover - permissions, transient IO
        return 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value
            loaded += 1
    return loaded


DOTENV_LOADED = _load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _has_module(name: str) -> bool:
    """True if an optional dependency is importable. Never raises."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _local_chromium() -> bool:
    """Whether a local browser is available for print-CSS PDF/static figures."""
    return find_chromium() is not None


class Settings(BaseModel):
    app_name: str = "Chronos Studio"
    version: str = "0.1.0"
    debug: bool = _env_bool("CHRONOS_DEBUG", True)

    # CORS: the preview host is a different origin, so this must be permissive
    # in dev and explicit in production.
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv(
                "CHRONOS_ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://0.0.0.0:5173",
            ).split(",")
            if o.strip()
        ]
    )
    allow_all_origins: bool = _env_bool("CHRONOS_ALLOW_ALL_ORIGINS", True)

    # Uploads
    max_upload_mb: int = 64
    upload_dir: Path = Path(os.getenv("CHRONOS_UPLOAD_DIR", "/tmp/chronos_uploads"))
    allowed_extensions: set[str] = Field(
        default_factory=lambda: {
            ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json",
            ".parquet", ".dta", ".sav", ".ods", ".rds",
        }
    )

    # Sessions
    session_ttl_hours: int = 6
    max_sessions: int = 200
    default_seed: int = 20260101

    # AI assistant (Google AI Studio / Gemini). The key is never persisted.
    gemini_api_key_env: str = "GEMINI_API_KEY"
    gemini_model: str = os.getenv("CHRONOS_GEMINI_MODEL", "gemini-2.5-flash")
    gemini_model_pro: str = "gemini-2.5-pro"
    gemini_temperature: float = 0.2
    ai_max_requests_per_session: int = 60
    ai_max_requests_per_minute: int = 15
    ai_min_interval_seconds: float = 4.0
    ai_max_values: int = 40
    ai_allow_data_values_by_default: bool = False
    # Indicative list prices for the token/cost counter. Free-tier bills may be
    # zero; these are configuration values, not a billing guarantee.
    ai_input_usd_per_million: float = 0.30
    ai_output_usd_per_million: float = 2.50

    # Feature flags — resolved at runtime by `optional_features()`
    flags: dict[str, bool] = Field(default_factory=dict)

    def has_gemini_key(self) -> bool:
        return bool(os.getenv(self.gemini_api_key_env, "").strip())

    def upload_dir_resolved(self) -> Path:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        return self.upload_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def optional_features() -> dict[str, bool]:
    """
    Which optional extras are actually installed. The UI greys out the rest and
    tells the user how to enable them — it never crashes on a missing package.
    """
    return {
        "pmdarima_auto_arima": _has_module("pmdarima"),
        "arch_garch": _has_module("arch"),
        "prophet": _has_module("prophet"),
        "statsforecast": _has_module("statsforecast"),
        "sktime": _has_module("sktime"),
        "tbats": _has_module("tbats"),
        "lightgbm": _has_module("lightgbm"),
        "xgboost": _has_module("xgboost"),
        "scikit_learn": _has_module("sklearn"),
        "python_docx": _has_module("docx"),
        "weasyprint": _has_module("weasyprint"),
        "playwright": _has_module("playwright"),
        "report_chromium": _local_chromium(),
        "pyreadstat": _has_module("pyreadstat"),
        "pyreadr": _has_module("pyreadr"),
        "kaleido": _has_module("kaleido"),
        "x13as_binary": bool(os.getenv("X13AS_PATH", "").strip()),
        "gemini": _has_module("google.genai") and get_settings().has_gemini_key(),
    }


__all__ = ["Settings", "get_settings", "optional_features", "ASSETS_DIR", "BACKEND_ROOT"]
