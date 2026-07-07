"""
Enterprise RAG OS — Base Analytics Collector Interface
========================================================

Purpose:
    Abstract base class for analytics collection. Tracks system metrics
    like query counts, latencies, token usage, and cache hit ratios
    for the analytics dashboard.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAnalyticsCollector(ABC):
    """Abstract base class for analytics data collection."""

    @abstractmethod
    async def track_query(self, data: dict[str, Any]) -> None:
        """Track a query event with metadata."""

    @abstractmethod
    async def track_document_upload(self, data: dict[str, Any]) -> None:
        """Track a document upload event."""

    @abstractmethod
    async def track_latency(self, operation: str, latency_ms: float) -> None:
        """Track operation latency."""

    @abstractmethod
    async def get_summary(self, time_range_hours: int = 24) -> dict[str, Any]:
        """Get analytics summary for a time range."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this analytics implementation."""
