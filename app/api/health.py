"""
Enterprise RAG OS — Health Check Endpoints
=============================================

Purpose:
    Health, readiness, and liveness endpoints for monitoring and orchestration.

    - GET /health        — Basic liveness (is the process alive?)
    - GET /health/ready  — Readiness (can it serve traffic? are dependencies up?)

Why separate liveness and readiness?
    Kubernetes (and most orchestrators) distinguish between:
    - Liveness: "Is the process stuck?" → If unhealthy, restart the container.
    - Readiness: "Can it handle requests?" → If not ready, remove from load balancer.

    A service can be alive but not ready (e.g., during startup while loading models).
"""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.config.settings import get_settings
from app.core.lifecycle import get_uptime_seconds
from app.schemas.health import ComponentHealth, ComponentStatus, HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Check",
    description="Basic health check. Returns 200 if the application is running.",
)
async def health_check() -> HealthResponse:
    """
    Liveness probe endpoint.

    This endpoint is intentionally lightweight — no dependency checks.
    If this returns 200, the process is alive. If it doesn't respond,
    the orchestrator should restart it.
    """
    settings = get_settings()
    return HealthResponse(
        status=ComponentStatus.HEALTHY,
        version=__version__,
        environment=settings.environment.value,
        uptime_seconds=round(get_uptime_seconds(), 2),
        components=[],
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness Check",
    description="Detailed health check including dependency status.",
)
async def readiness_check() -> HealthResponse:
    """
    Readiness probe endpoint.

    Checks the health of all critical dependencies (vector store,
    LLM provider, etc.). The service should only receive traffic
    when all critical dependencies are healthy.

    In Phase 1, only basic components are checked. Later phases
    will add vector store, embedding model, and LLM health checks.
    """
    settings = get_settings()
    components: list[ComponentHealth] = []

    # Configuration check
    components.append(
        ComponentHealth(
            name="configuration",
            status=ComponentStatus.HEALTHY,
            message=f"Environment: {settings.environment.value}",
        )
    )

    # Vector store (placeholder — Phase 2)
    components.append(
        ComponentHealth(
            name="vector_store",
            status=ComponentStatus.NOT_CONFIGURED,
            message="Vector store not initialized (Phase 2)",
        )
    )

    # Embedding model (placeholder — Phase 2)
    components.append(
        ComponentHealth(
            name="embedding_model",
            status=ComponentStatus.NOT_CONFIGURED,
            message="Embedding model not loaded (Phase 2)",
        )
    )

    # LLM provider (placeholder — Phase 4)
    components.append(
        ComponentHealth(
            name="llm_provider",
            status=ComponentStatus.NOT_CONFIGURED,
            message="LLM provider not configured (Phase 4)",
        )
    )

    # Determine overall status
    statuses = [c.status for c in components]
    if ComponentStatus.UNHEALTHY in statuses:
        overall = ComponentStatus.UNHEALTHY
    elif ComponentStatus.DEGRADED in statuses:
        overall = ComponentStatus.DEGRADED
    else:
        overall = ComponentStatus.HEALTHY

    return HealthResponse(
        status=overall,
        version=__version__,
        environment=settings.environment.value,
        uptime_seconds=round(get_uptime_seconds(), 2),
        components=components,
    )
