"""
Enterprise RAG OS — Base Reranker Interface
=============================================

Purpose:
    Abstract base class for reranking models. Rerankers take the top-K
    results from initial retrieval and re-score them using a more
    expensive but more accurate model (typically a cross-encoder).

Why Reranking?
    Bi-encoder retrieval (embedding similarity) is fast but imprecise.
    Cross-encoder reranking is slow but accurate. The two-stage approach
    gets the best of both: fast retrieval (top-20) then precise reranking
    (return top-5).

    Stage 1: Bi-encoder retrieval (fast, approximate)
    ┌─────────────────────────────────────────────┐
    │ Query embedding ←→ Document embeddings       │
    │ Returns top-20 candidates in ~50ms           │
    └──────────────────────┬──────────────────────┘
                           ▼
    Stage 2: Cross-encoder reranking (slow, precise)
    ┌─────────────────────────────────────────────┐
    │ Cross-encoder scores each (query, doc) pair  │
    │ Returns top-5 from the 20 candidates         │
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankedChunk:
    """A chunk with both original retrieval score and reranking score."""

    chunk_id: str
    document_id: str
    content: str
    original_score: float
    rerank_score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
    page_number: int | None = None
    section: str | None = None


class BaseReranker(ABC):
    """Abstract base class for all rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[Any],
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RankedChunk]:
        """
        Rerank a list of retrieved chunks.

        Args:
            query: The original search query.
            chunks: Retrieved chunks to rerank.
            top_k: Number of top results to return after reranking.

        Returns:
            List of RankedChunk sorted by rerank_score (descending).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this reranker implementation."""
