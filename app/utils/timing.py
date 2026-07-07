"""
Enterprise RAG OS — Performance Timing Utilities
==================================================

Purpose:
    Decorators and context managers for measuring execution time of
    functions and code blocks. Essential for identifying bottlenecks
    in the RAG pipeline (embedding, retrieval, generation).

Architecture:
    The @timed decorator logs execution time via structlog. In the
    observability dashboard (Phase 5), these timings feed into latency
    charts and performance monitoring.

    Pipeline Stage Timing:
    ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
    │ Chunking│→│Embedding │→│Retrieval │→│Reranking │→│ Generation │
    │  120ms  │ │  450ms   │ │   80ms   │ │  200ms   │ │   1200ms   │
    └─────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘
    Total: 2050ms  ← Each stage is independently timed and logged.

Usage:
    from app.utils.timing import timed, Timer

    @timed
    def embed_documents(docs):
        ...

    @timed(name="custom_retrieval")
    async def retrieve(query):
        ...

    with Timer("embedding_batch"):
        results = model.encode(texts)
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, ParamSpec, TypeVar

from app.logging.logger import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class Timer:
    """
    Context manager for timing code blocks.

    Records wall-clock time and logs the result. Can also be used
    to access the elapsed time programmatically.

    Example:
        with Timer("embedding") as t:
            embeddings = model.encode(texts)
        print(f"Took {t.elapsed_ms:.1f}ms")
    """

    def __init__(self, name: str = "operation") -> None:
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        logger.info(
            "Timer completed",
            operation=self.name,
            elapsed_ms=round(self.elapsed_ms, 2),
        )


def timed(
    func: Any = None,
    *,
    name: str | None = None,
) -> Any:
    """
    Decorator that logs function execution time.

    Works with both sync and async functions. Logs the function name,
    arguments (if small), and elapsed time in milliseconds.

    Args:
        func: The function to wrap (used when decorating without arguments).
        name: Custom operation name for logging. Defaults to function name.

    Returns:
        Wrapped function that logs its execution time.

    Example:
        @timed
        def process_document(doc_id: str) -> Document:
            ...

        @timed(name="llm_generation")
        async def generate_answer(prompt: str) -> str:
            ...
    """

    def decorator(fn: Any) -> Any:
        operation_name = name or fn.__qualname__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.info(
                        "Function completed",
                        operation=operation_name,
                        elapsed_ms=round(elapsed, 2),
                        status="success",
                    )
                    return result
                except Exception:
                    elapsed = (time.perf_counter() - start) * 1000
                    logger.error(
                        "Function failed",
                        operation=operation_name,
                        elapsed_ms=round(elapsed, 2),
                        status="error",
                    )
                    raise

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.info(
                    "Function completed",
                    operation=operation_name,
                    elapsed_ms=round(elapsed, 2),
                    status="success",
                )
                return result
            except Exception:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    "Function failed",
                    operation=operation_name,
                    elapsed_ms=round(elapsed, 2),
                    status="error",
                )
                raise

        return sync_wrapper

    # Support both @timed and @timed(name="...")
    if func is not None:
        return decorator(func)
    return decorator
