"""
Enterprise RAG OS — Application Lifecycle
============================================

Purpose:
    Startup and shutdown hooks for the FastAPI application.
    These run once when the app starts/stops and handle:
    - Logging initialization
    - Resource allocation/cleanup
    - Health check for critical dependencies
    - Graceful shutdown (close connections, flush logs)

Why lifecycle hooks?
    Resources like database connections, ML model loading, and
    background tasks must be initialized ONCE at startup and
    cleaned up at shutdown. Doing this per-request is wasteful
    and error-prone.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from app.logging.logger import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import FastAPI

logger = get_logger(__name__)

# Global startup time for uptime calculation
_startup_time: float = 0.0


def get_uptime_seconds() -> float:
    """Return seconds since application startup."""
    if _startup_time == 0.0:
        return 0.0
    return time.time() - _startup_time


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Everything before `yield` runs at startup.
    Everything after `yield` runs at shutdown.

    This is the modern FastAPI pattern replacing the deprecated
    @app.on_event("startup") and @app.on_event("shutdown") decorators.
    """
    global _startup_time

    # ── STARTUP ──────────────────────────────────────────────────────────
    setup_logging()
    _startup_time = time.time()

    logger.info(
        "Application starting",
        app_name="Enterprise RAG OS",
    )

    # Future: Initialize vector store connections, load ML models,
    # start background workers, etc.

    logger.info("Application startup complete")

    yield  # Application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info("Application shutting down")

    # Future: Close database connections, flush caches,
    # stop background workers, etc.

    logger.info("Application shutdown complete")
