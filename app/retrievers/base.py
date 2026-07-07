"""
Enterprise RAG OS — Base Retriever Interface
===============================================

Purpose:
    Abstract base class defining the contract for all retriever implementations.
    Any retrieval backend (dense, BM25, hybrid, multi-query) must implement
    this interface, making them interchangeable in the pipeline.

Design Pattern: Strategy Pattern
    The retriever is selected at runtime via configuration. The pipeline
    code interacts only with this interface, never with concrete implementations.
    This enables A/B testing, gradual migration, and zero-downtime swaps.

    Pipeline ──▶ BaseRetriever (interface)
                     │
         ┌───────────┼────────────┐
         ▼           ▼            ▼
    DenseRetriever  BM25      HybridRetriever
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """
    A single chunk returned by a retriever.

    Contains the text content, its source metadata, and the
    similarity/relevance score from the retrieval algorithm.
    """

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    page_number: int | None = None
    section: str | None = None


@dataclass
class RetrievalResult:
    """Complete result from a retrieval operation."""

    chunks: list[RetrievedChunk]
    query: str
    strategy: str
    latency_ms: float = 0.0
    total_candidates: int = 0


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.

    Every retriever (dense, sparse, hybrid) must implement this interface.
    The pipeline invokes `retrieve()` without knowing which implementation
    is behind it.

    Lifecycle:
        1. __init__: Configure the retriever (model, index, params)
        2. retrieve(): Execute a search query
        3. health_check(): Verify the retriever is operational
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query text.
            top_k: Maximum number of chunks to return.
            filters: Optional metadata filters.
            **kwargs: Additional retriever-specific parameters.

        Returns:
            RetrievalResult containing ranked chunks.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the retriever is operational.

        Returns:
            True if healthy, False otherwise.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this retriever implementation."""
