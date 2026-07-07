"""
Enterprise RAG OS — Health Check Schemas
==========================================

Purpose:
    Schemas for health, readiness, and liveness endpoints.

Why health checks?
    - Kubernetes uses /health/live for liveness probes (is the process alive?)
    - Kubernetes uses /health/ready for readiness probes (can it serve traffic?)
    - Docker uses HEALTHCHECK for container health monitoring
    - Load balancers use health endpoints to route traffic
    - Monitoring systems use health endpoints for alerting
"""

from __future__ import annotations

import enum

from pydantic import Field

from app.schemas.base import BaseSchema


class ComponentStatus(enum.StrEnum):
    """Health status of an individual component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    NOT_CONFIGURED = "not_configured"


class ComponentHealth(BaseSchema):
    """Health status of a single system component."""

    name: str = Field(description="Component name")
    status: ComponentStatus = Field(description="Current status")
    message: str | None = Field(default=None, description="Status details")
    latency_ms: float | None = Field(default=None, description="Response time")


class HealthResponse(BaseSchema):
    """
    Health check response.

    The overall status is determined by the worst component status:
    - All healthy → healthy
    - Any degraded → degraded
    - Any unhealthy → unhealthy
    """

    status: ComponentStatus = Field(description="Overall system status")
    version: str = Field(description="Application version")
    environment: str = Field(description="Current environment")
    uptime_seconds: float = Field(description="Seconds since startup")
    components: list[ComponentHealth] = Field(
        default_factory=list,
        description="Individual component health statuses",
    )
