#!/usr/bin/env bash
# Start Chronos Studio locally after install-local-linux-mint.sh has run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -f backend/.venv/bin/activate ]; then
  echo "ERROR: backend/.venv is missing. Run:" >&2
  echo "  bash scripts/install-local-linux-mint.sh" >&2
  exit 1
fi

source backend/.venv/bin/activate
exec bash scripts/dev.sh
