#!/usr/bin/env bash
# One-time local install for Linux Mint / Ubuntu-based systems.
# Usage:
#   bash scripts/install-local-linux-mint.sh
#   PYTHON_BIN=python3.12 bash scripts/install-local-linux-mint.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "== Chronos Studio local install =="
echo "Project: $ROOT"
echo "Python:  $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Install Python 3.11+ and python3-venv." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
print(f"Detected Python {major}.{minor}")
if (major, minor) < (3, 11):
    raise SystemExit("ERROR: Python 3.11 or newer is required.")
if (major, minor) >= (3, 14):
    print("WARNING: Python 3.14 may not have wheels for scipy/statsmodels yet.")
    print("         If a wheel build fails, rerun with PYTHON_BIN=python3.12 or python3.13.")
PY

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: Node.js and npm are required. Install Node 20 LTS first." >&2
  exit 1
fi
NODE_MAJOR="$(node -p 'Number(process.versions.node.split(".")[0])')"
echo "Detected Node $(node -v), npm $(npm -v)"
if [ "$NODE_MAJOR" -lt 18 ]; then
  echo "ERROR: Node 18+ is required; Node 20 LTS is recommended." >&2
  exit 1
fi

if [ ! -x backend/.venv/bin/python ]; then
  echo "→ creating backend/.venv"
  "$PYTHON_BIN" -m venv backend/.venv
else
  echo "→ reusing existing backend/.venv"
fi

source backend/.venv/bin/activate
export PIP_DISABLE_PIP_VERSION_CHECK=1
python -m pip install --upgrade pip
echo "→ installing backend dependencies"
python -m pip install -r backend/requirements-deploy.txt

if [ ! -f backend/.env ]; then
  echo "→ creating backend/.env from .env.example (offline mode; no key)"
  cp .env.example backend/.env
  chmod 600 backend/.env
else
  echo "→ keeping existing backend/.env"
fi

echo "→ generating example datasets (if needed)"
(cd backend && python scripts/gen_examples.py)

echo "→ compiling backend"
(cd backend && python -m compileall -q app)

if [ ! -d frontend/node_modules ]; then
  echo "→ installing frontend dependencies"
  (cd frontend && npm ci --no-audit --no-fund)
else
  echo "→ reusing existing frontend/node_modules"
fi

echo
echo "Install complete."
echo "Run:  bash scripts/run-local.sh"
echo "Open: http://localhost:5173"
echo "Stop: Ctrl+C in that terminal"
