#!/usr/bin/env bash
# First-run launcher for the Chronos Studio standalone .deb.
# Creates a private Python 3.12 environment from the bundled offline wheels,
# then starts the single-process desktop app. No network access is required.
set -euo pipefail

APP_DIR="${CHRONOS_APP_DIR:-/opt/chronos-studio}"
DATA_DIR="${CHRONOS_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/chronos-studio}"
PYTHON_BIN="${CHRONOS_PYTHON_BIN:-python3.12}"
WHEELS_ARCHIVE="${CHRONOS_WHEELS_ARCHIVE:-$APP_DIR/standalone/wheels.tar.xz}"
VENV="$DATA_DIR/venv"
WHEELS_DIR="$DATA_DIR/wheels"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
        PYTHON_BIN=python3
    else
        echo "ERROR: Python 3.12 with venv support is required." >&2
        echo "Install it with: sudo apt install python3.12 python3.12-venv" >&2
        exit 1
    fi
fi

if [ ! -f "$WHEELS_ARCHIVE" ]; then
    echo "ERROR: offline wheel archive not found at $WHEELS_ARCHIVE" >&2
    exit 1
fi

MARKER="$VENV/.chronos-runtime"
STAMP="$(sha256sum "$WHEELS_ARCHIVE" "$APP_DIR/standalone/requirements-runtime.txt" | sha256sum | cut -d' ' -f1)"
if [ ! -x "$VENV/bin/python" ] || [ ! -f "$MARKER" ] || [ "$(cat "$MARKER" 2>/dev/null)" != "$STAMP" ]; then
    echo "→ Preparing Chronos Studio in $DATA_DIR (first run or runtime update)"
    mkdir -p "$DATA_DIR"
    rm -rf "$VENV"
    "$PYTHON_BIN" -m venv "$VENV"
    rm -rf "$WHEELS_DIR"
    mkdir -p "$WHEELS_DIR"
    tar -xJf "$WHEELS_ARCHIVE" -C "$WHEELS_DIR"
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$VENV/bin/pip" install \
        --no-index --find-links "$WHEELS_DIR" --no-cache-dir \
        -r "$APP_DIR/standalone/requirements-runtime.txt"
    printf '%s\n' "$STAMP" > "$MARKER"
fi

export CHRONOS_APP_DIR="$APP_DIR"
export CHRONOS_DATA_DIR="$DATA_DIR"
export CHRONOS_STANDALONE=1
exec "$VENV/bin/python" "$APP_DIR/standalone/chronos_studio.py" \
    --app-dir "$APP_DIR" --data-dir "$DATA_DIR" "$@"
