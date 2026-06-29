"""Health and readiness probe endpoints for Cloud Run."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])

_VERSION = "1.0.0"


@router.get("/health", status_code=200)
async def health() -> JSONResponse:
    """
    Liveness probe — always 200 if the process is alive.
    Cloud Run uses this to decide whether to restart the container.
    """
    return JSONResponse({"status": "ok", "version": _VERSION})


@router.get("/ready", status_code=200)
async def ready(request: Request) -> JSONResponse:
    """
    Readiness probe — verifies all critical dependencies before accepting traffic.

    Checks performed:
      pinecone  — describe_index() round-trip (network call)
      pubsub    — topic_path is configured (config check, no RPC overhead)
      whatchamp — API key is present (config check, no outbound HTTP)

    Returns 200 when all checks pass, 503 if any fail.
    Cloud Run uses this to gate traffic routing.
    """
    checks: dict[str, str] = {}
    healthy = True

    # Pinecone index reachability — real network round-trip
    try:
        stats = await request.app.state.pinecone.describe_index()
        vector_count = stats.get("total_vector_count", "?")
        checks["pinecone"] = f"ok ({vector_count} vectors)"
    except Exception as exc:
        checks["pinecone"] = f"error: {str(exc)[:120]}"
        healthy = False
        log.warning("health.pinecone_unreachable", error=str(exc))

    # Pub/Sub topic — verify the topic path is configured (no gRPC overhead per probe)
    try:
        topic_path: str = getattr(request.app.state, "topic_path", "")
        if not topic_path:
            raise ValueError("topic_path not set on app.state")
        checks["pubsub"] = f"ok ({topic_path})"
    except Exception as exc:
        checks["pubsub"] = f"error: {str(exc)[:120]}"
        healthy = False
        log.warning("health.pubsub_not_configured", error=str(exc))

    # WhatChimp — verify the adapter has an API key (no outbound HTTP per probe)
    try:
        messaging = getattr(request.app.state, "messaging", None)
        if messaging is None or not getattr(messaging, "_api_key", ""):
            raise ValueError("WhatChimp adapter not initialised or API key missing")
        checks["whatchamp"] = "ok"
    except Exception as exc:
        checks["whatchamp"] = f"error: {str(exc)[:120]}"
        healthy = False
        log.warning("health.whatchamp_not_configured", error=str(exc))

    status_code = 200 if healthy else 503
    return JSONResponse(
        {
            "status": "ready" if healthy else "not_ready",
            "checks": checks,
            "version": _VERSION,
        },
        status_code=status_code,
    )
