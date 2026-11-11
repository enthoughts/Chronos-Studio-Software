# Chronos Studio backend — slim, non-root, free-tier friendly (Part 19).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHRONOS_DEBUG=false \
    CHRONOS_ALLOW_ALL_ORIGINS=true \
    CHRONOS_SESSION_TTL_HOURS=6 \
    CHRONOS_UPLOAD_DIR=/tmp/chronos_uploads

# 1 = also install Chromium so PDF reports and 300-dpi figure exports work.
# 0 (default) keeps the image small for free tiers; those two export kinds
# then degrade into errors.txt while every other export still works.
ARG INSTALL_BROWSER=0

WORKDIR /app

# libgomp1 is needed by some scientific wheels; curl by the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
 && if [ "$INSTALL_BROWSER" = "1" ]; then \
      apt-get install -y --no-install-recommends chromium fonts-liberation ; \
    fi \
 && rm -rf /var/lib/apt/lists/*

# Install the deploy overlay (core + report/ML/AI extras), not requirements.txt
# alone — see requirements-deploy.txt for the rationale.
COPY requirements.txt requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app ./app
COPY scripts ./scripts

# Example datasets are generated at build time so the image needs no network.
RUN python scripts/gen_examples.py || true

RUN useradd -m -u 10001 chronos \
 && mkdir -p /tmp/chronos_uploads \
 && chown -R chronos:chronos /app /tmp/chronos_uploads
USER chronos

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

LABEL org.opencontainers.image.title="Chronos Studio backend" \
      org.opencontainers.image.description="Time-series analysis API" \
      org.opencontainers.image.source="https://github.com/your-org/chronos-studio"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
