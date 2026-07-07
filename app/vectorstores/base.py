"""
Enterprise RAG OS — Base Vector Store Interface
=================================================

Purpose:
    Abstract base class for vector database backends. The vector store
    is the persistence layer for embeddings and enables similarity search.

Design: Repository Pattern
    The vector store abstraction follows the Repository pattern:
    - The pipeline works with an abstract interface
    - Concrete implementations (ChromaDB, Qdrant, FAISS) handle persistence
    - Swapping databases requires zero pipeline changes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorSearchResult:
    """A single result from a vector similarity search."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


class BaseVectorStore(ABC):
    """Abstract base class for all vector store backends."""

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add documents with embeddings to the vector store.

        Args:
            ids: Unique identifiers for each document.
            embeddings: Embedding vectors.
            documents: Raw text content.
            metadatas: Optional metadata dicts for each document.
        """

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """
        Search for similar documents by embedding vector.

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum results to return.
            filters: Optional metadata filters.

        Returns:
            List of VectorSearchResult sorted by similarity (descending).
        """

    @abstractmethod
    async def delete(
        self,
        ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Delete documents by ID."""

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of documents in the store."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is operational."""

    @abstractmethod
    async def list_sources(self) -> list[dict[str, Any]]:
        """Return a list of unique document sources and metadata."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this vector store implementation."""
