#!/usr/bin/env bash
# Start both halves of Chronos Studio for local development.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d frontend/node_modules ]; then
  echo "→ installing frontend dependencies (first run only)"
  (cd frontend && npm install)
fi

echo "→ backend  http://localhost:8000/docs"
(cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) &
BACKEND=$!
trap 'kill $BACKEND 2>/dev/null || true' EXIT

echo "→ frontend http://localhost:5173"
(cd frontend && npm run dev)
