"""
Enterprise RAG OS — Application Factory
==========================================

Purpose:
    Creates and configures the FastAPI application instance. This is the
    entry point for the entire system.

Architecture:
    Uses the Application Factory pattern: a function that creates and
    returns a fully configured FastAPI app. This enables:
    - Different configurations for testing, development, production
    - Clean dependency injection
    - Testable app creation (tests can create isolated app instances)

    ┌─────────────────────────────────────────────────────────────────┐
    │                     create_app()                                │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
    │  │ Settings  │  │ Lifespan │  │Middleware │  │   Routers    │  │
    │  │ Loading   │→ │ Setup    │→ │ Stack     │→ │  Registration│  │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

Entry Points:
    - Development: uvicorn app.main:app --reload
    - Production:  uvicorn app.main:app --workers 4
    - Programmatic: from app.main import create_app; app = create_app()

Dependencies:
    - fastapi
    - uvicorn
    - All app.core modules (middleware, lifecycle, dependencies)
    - All app.api routers
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __app_name__, __version__
from app.api.router import root_router
from app.config.settings import get_settings
from app.core.lifecycle import lifespan
from app.core.middleware import (
    RequestIDMiddleware,
    TimingMiddleware,
    register_exception_handlers,
)


def create_app() -> FastAPI:
    """
    Application factory.

    Creates a fully configured FastAPI instance with:
    - Metadata (title, version, description)
    - Lifespan management (startup/shutdown)
    - CORS middleware
    - Request ID middleware
    - Timing middleware
    - Global exception handlers
    - All API routes

    Returns:
        Configured FastAPI application.
    """
    settings = get_settings()

    # ── Create FastAPI instance ───────────────────────────────────────────
    application = FastAPI(
        title=__app_name__,
        version=__version__,
        description=(
            "A Production-Ready, Agentic, Multi-Modal, Explainable RAG System "
            "with Advanced Retrieval, Evaluation, Observability, and Enterprise Features."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # Let custom exception handlers manage debug output to ensure JSON format
        # debug=settings.debug,
    )

    # ── Middleware Stack ──────────────────────────────────────────────────
    # Order matters! Middleware executes in REVERSE registration order.
    # Last registered = first to execute.
    #
    # Execution order for a request:
    #   1. RequestIDMiddleware (sets correlation ID)
    #   2. TimingMiddleware (starts timer)
    #   3. CORSMiddleware (handles CORS headers)
    #   4. Route handler
    #   5. CORSMiddleware (adds CORS headers)
    #   6. TimingMiddleware (stops timer, logs)
    #   7. RequestIDMiddleware (adds X-Request-ID header)

    # CORS (must be added before custom middleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    # Timing (measures total request processing time)
    application.add_middleware(TimingMiddleware)

    # Request ID (correlation ID for distributed tracing)
    application.add_middleware(RequestIDMiddleware)

    # ── Exception Handlers ────────────────────────────────────────────────
    register_exception_handlers(application)

    # ── Routes ────────────────────────────────────────────────────────────
    application.include_router(root_router)

    # ── Serve UI ──────────────────────────────────────────────────────────
    import os
    from fastapi.staticfiles import StaticFiles

    ui_dir = os.path.join(settings.project_root, "app", "ui")
    if os.path.exists(ui_dir):
        application.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    return application


# ── Application Instance ──────────────────────────────────────────────────────
# This is what uvicorn imports: `uvicorn app.main:app`
app = create_app()


def run_server() -> None:
    """
    Run the development server.

    This is the entry point for the `rag-server` console script
    defined in pyproject.toml.
    """
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers if not settings.server.reload else 1,
        log_level=settings.logging.level.value.lower(),
    )


if __name__ == "__main__":
    run_server()
