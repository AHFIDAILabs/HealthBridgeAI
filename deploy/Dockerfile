# ── HealthBridgeAI Webhook Service ──────────────────────────────────────────
# Lightweight — no BGE-M3. Only validates WhatChimp payloads and publishes
# to Pub/Sub.  Target: ~300 MB image, <2s cold-start.
#
# Build context: repo root (pass --file deploy/Dockerfile from root)

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
# curl: Cloud Run health checks during rollout
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# Copy only the package manifest first (layer cache hit if deps unchanged)
COPY pyproject.toml .

# Install the package itself plus its core dependencies.
# Exclude heavy ML libs that are only needed by the processor.
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "gunicorn>=22.0.0" \
    "httpx>=0.27.0" \
    "google-cloud-pubsub>=2.21.0" \
    "google-cloud-secret-manager>=2.20.0" \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.3.0" \
    "pyyaml>=6.0.1" \
    "structlog>=24.2.0" \
    "tenacity>=8.3.0"

# ── Application source ────────────────────────────────────────────────────────
COPY src/ src/
RUN pip install --no-deps -e .

# ── Non-root user (Cloud Run security best practice) ─────────────────────────
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8080

# Gunicorn + Uvicorn worker: single process per container (Cloud Run scales
# horizontally), 2 threads for burst handling within the process.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "30", \
     "--log-level", "info", \
     "healthbridgeai.main:app"]
