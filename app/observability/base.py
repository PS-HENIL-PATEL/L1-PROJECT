"""
Enterprise RAG OS — Base Observability Interface
===================================================

Purpose:
    Abstract base class for tracing and observability. Enables distributed
    tracing across pipeline stages and integration with observability
    platforms (LangSmith, MLflow, OpenTelemetry).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTracer(ABC):
    """Abstract base class for pipeline tracers."""

    @abstractmethod
    def start_span(self, name: str, metadata: dict[str, Any] | None = None) -> Any:
        """Start a new tracing span."""

    @abstractmethod
    def end_span(self, span: Any, metadata: dict[str, Any] | None = None) -> None:
        """End a tracing span."""

    @abstractmethod
    def log_event(self, name: str, data: dict[str, Any]) -> None:
        """Log a discrete event."""

    @abstractmethod
    def log_prompt(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a prompt-response pair for analysis."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this tracer implementation."""
