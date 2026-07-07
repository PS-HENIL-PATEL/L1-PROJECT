"""
Enterprise RAG OS — Structured Logging
=======================================

Purpose:
    Configures structured logging for the entire application using structlog.
    Produces machine-parseable JSON logs in production and human-readable
    colored logs in development.

Why structlog?
    Python's built-in logging module outputs unstructured text by default.
    At scale, you need structured logs (key-value pairs) that can be:
    - Ingested by log aggregation systems (ELK, Datadog, Splunk, CloudWatch)
    - Queried with structured queries ("show all logs where user_id=X and latency>500ms")
    - Correlated across services using request IDs

    structlog wraps Python's logging module, adding:
    - Automatic key-value binding (logger.info("query", user_id="123", latency_ms=45))
    - Processor pipelines (add timestamp, add log level, format as JSON)
    - Context variables (correlation ID from middleware propagates automatically)
    - Zero-config integration with Python's logging ecosystem

Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
    │ Application  │────▶│  structlog   │────▶│  Processors   │
    │  Code        │     │  Logger      │     │  Pipeline     │
    └─────────────┘     └──────────────┘     └───────┬───────┘
                                                      │
                                              ┌───────▼───────┐
                                              │   Renderer    │
                                              │  (JSON/Text)  │
                                              └───────┬───────┘
                                                      │
                                         ┌────────────┼────────────┐
                                         │            │            │
                                    ┌────▼───┐  ┌────▼───┐  ┌────▼────┐
                                    │ Console │  │  File  │  │ External│
                                    │ Handler │  │Handler │  │ (future)│
                                    └────────┘  └────────┘  └─────────┘

Dependencies:
    - structlog
    - logging (stdlib)

Usage:
    from app.logging.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Document processed", document_id="abc", chunks=42, latency_ms=123.4)
    logger.warning("Slow retrieval", query="...", latency_ms=2500)
    logger.error("LLM failed", provider="openai", error="timeout")
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog

from app.config.settings import LogFormat, get_settings

# ── Context Variables ─────────────────────────────────────────────────────────
# These are thread-safe and async-safe. Middleware sets them per-request.
# All log entries within that request automatically include these values.

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)


def add_correlation_id(
    _logger: logging.Logger, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Structlog processor that injects the correlation ID into every log entry.

    The correlation ID is set by the RequestIDMiddleware (see core/middleware.py)
    and stored in a ContextVar. This processor reads it and adds it to the log
    event dict, ensuring every log line from a given request shares the same ID.

    This enables request-level tracing: filter all logs by correlation_id to see
    the entire lifecycle of a single request across all components.
    """
    cid = correlation_id_var.get()
    if cid:
        event_dict["correlation_id"] = cid
    uid = user_id_var.get()
    if uid:
        event_dict["user_id"] = uid
    return event_dict


def setup_logging() -> None:
    """
    Configure the logging system for the entire application.

    Called once during application startup (see core/lifecycle.py).
    Sets up both structlog (for application code) and stdlib logging
    (for third-party libraries like uvicorn, httpx).

    The configuration adapts based on the LOG_FORMAT setting:
    - JSON: Structured JSON output for production log aggregation
    - TEXT: Colored, human-readable output for local development
    """
    settings = get_settings()
    log_settings = settings.logging

    # Map our LogLevel enum to Python's logging level
    log_level = getattr(logging, log_settings.level.value, logging.INFO)

    # ── Create log directory if needed ────────────────────────────────────
    log_file = Path(settings.project_root) / log_settings.file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Configure stdlib logging ──────────────────────────────────────────
    # This catches logs from uvicorn, httpx, and other third-party libs.
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicates on reload
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # ── Shared processors ─────────────────────────────────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # ── Renderer based on format ──────────────────────────────────────────
    if log_settings.format == LogFormat.JSON:
        structlog.processors.JSONRenderer()
        stdlib_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        structlog.dev.ConsoleRenderer(colors=True)
        stdlib_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )

    # Apply formatter to stdlib handlers
    for handler in root_logger.handlers:
        handler.setFormatter(stdlib_formatter)

    # ── Configure structlog ───────────────────────────────────────────────
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    This is the primary way to obtain a logger throughout the application.
    Each module should call this with __name__ to get a logger scoped to
    that module.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A structlog BoundLogger with all configured processors.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing document", doc_id="abc", size_bytes=1024)
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
