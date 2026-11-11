#!/bin/sh
# Single-container entrypoint: serve the SPA and proxy /api to a local uvicorn.
# Used by Dockerfile.single (Fly.io / HF Spaces free tier). See DEPLOY.md.
set -eu

: "${PORT:=8080}"
export PORT

# Render the nginx config with the platform-supplied port (Fly 8080, HF 7860).
envsubst '${PORT}' < /app/single-container/nginx.conf.template \
  > /etc/nginx/conf.d/default.conf

mkdir -p /tmp/chronos_uploads

# Backend binds to loopback only: nothing outside this container can reach it
# directly, and the browser talks exclusively to nginx on $PORT.
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# nginx in the foreground so the container receives SIGTERM correctly.
exec nginx -g 'daemon off;'
