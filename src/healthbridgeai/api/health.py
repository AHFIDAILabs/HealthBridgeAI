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
    Readiness probe — verifies Pinecone connectivity before accepting traffic.
    Returns 200 when healthy, 503 when a dependency is unavailable.
    Cloud Run uses this to gate traffic routing.
    """
    checks: dict[str, str] = {}
    healthy = True

    # Pinecone index reachability
    try:
        stats = await request.app.state.pinecone.describe_index()
        vector_count = stats.get("total_vector_count", "?")
        checks["pinecone"] = f"ok ({vector_count} vectors)"
    except Exception as exc:
        checks["pinecone"] = f"error: {str(exc)[:120]}"
        healthy = False
        log.warning("health.pinecone_unreachable", error=str(exc))

    status_code = 200 if healthy else 503
    return JSONResponse(
        {
            "status": "ready" if healthy else "not_ready",
            "checks": checks,
            "version": _VERSION,
        },
        status_code=status_code,
    )
