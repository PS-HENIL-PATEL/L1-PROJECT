"""
Enterprise RAG OS — Middleware
================================

Purpose:
    FastAPI middleware for cross-cutting concerns:
    - Request ID injection (correlation ID for tracing)
    - Request timing (latency measurement)
    - Global exception handling (consistent error responses)

Architecture:
    Middleware wraps every request in a processing pipeline:

    Request ──▶ RequestID ──▶ Timing ──▶ Route Handler ──▶ Response
                   │            │                            │
                   └─ Sets      └─ Measures                  └─ Includes
                   correlation     elapsed                     X-Request-ID
                   ID in           time                        X-Process-Time
                   context                                     headers

Why middleware instead of dependencies?
    Dependencies are per-route. Middleware applies globally to ALL requests
    including static files, health checks, and error responses. Correlation
    IDs and timing must be universal.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.exceptions import EnterpriseRAGError
from app.logging.logger import correlation_id_var, get_logger
from app.utils.ids import generate_short_id

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique request ID into every request.

    The ID is:
    1. Read from the incoming X-Request-ID header (if present, for distributed tracing)
    2. Or generated as a new short ID
    3. Stored in a ContextVar (accessible throughout the request lifecycle)
    4. Added to the response headers

    This enables end-to-end request tracing across logs, monitoring, and debugging.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Use existing request ID (from upstream proxy) or generate new one
        request_id = request.headers.get("X-Request-ID", generate_short_id("req"))

        # Store in context variable — this propagates to all loggers automatically
        correlation_id_var.set(request_id)

        # Store on request state for route handlers to access
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures and logs request processing time.

    Adds X-Process-Time header (in milliseconds) to every response.
    Also logs slow requests (>1000ms) as warnings for monitoring.
    """

    SLOW_REQUEST_THRESHOLD_MS = 1000.0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"

        # Log slow requests as warnings
        if elapsed_ms > self.SLOW_REQUEST_THRESHOLD_MS:
            logger.warning(
                "Slow request detected",
                path=request.url.path,
                method=request.method,
                elapsed_ms=round(elapsed_ms, 2),
            )
        else:
            logger.info(
                "Request completed",
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                elapsed_ms=round(elapsed_ms, 2),
            )

        return response


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers on the FastAPI app.

    Maps our custom exception hierarchy to HTTP responses.
    Also catches unhandled exceptions to prevent 500 errors
    from leaking internal details.
    """

    @app.exception_handler(EnterpriseRAGError)
    async def enterprise_rag_error_handler(
        request: Request, exc: EnterpriseRAGError
    ) -> JSONResponse:
        """Handle all custom application errors."""
        logger.error(
            "Application error",
            error_code=exc.error_code,
            detail=exc.detail,
            status_code=exc.status_code,
            context=exc.context,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all for unhandled exceptions.

        In production, this prevents internal details from leaking to clients.
        In development, the full traceback is included in the response.
        """
        logger.exception(
            "Unhandled exception",
            error_type=type(exc).__name__,
            detail=str(exc),
            path=request.url.path,
        )
        # Don't expose internal details in production
        from app.config.settings import get_settings

        settings = get_settings()
        detail = str(exc) if settings.debug else "An internal error occurred."

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": detail,
                    "status": 500,
                    "context": {},
                }
            },
        )
