"""
Enterprise RAG OS — Base Embedding Provider Interface
=======================================================

Purpose:
    Abstract base class for embedding model providers. Embeddings convert
    text into dense numerical vectors that capture semantic meaning,
    enabling similarity search in vector databases.

Mathematical Intuition:
    An embedding function f: Text -> R^d maps text strings to points in
    d-dimensional space such that semantically similar texts are close
    together (measured by cosine similarity or dot product).

    f("What is machine learning?") ≈ f("Explain ML")
    f("What is machine learning?") ≠ f("Recipe for chocolate cake")

    Key properties:
    - Dimensionality (d): Higher d captures more nuance but costs more
      memory and compute. Common: 384 (MiniLM), 768 (BERT), 1536 (OpenAI).
    - Cosine similarity: cos(θ) = (a·b)/(||a||·||b||), range [-1, 1].
    - Normalization: Most models output unit vectors, so cos(a,b) = a·b.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EmbeddingResult:
    """Result from an embedding operation."""

    embeddings: list[list[float]]
    model: str
    dimension: int
    token_count: int = 0
    latency_ms: float = 0.0


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            EmbeddingResult with embedding vectors.
        """

    @abstractmethod
    async def embed_query(self, query: str, **kwargs: Any) -> list[float]:
        """
        Generate embedding for a single query.

        Some models use different encoding for queries vs. documents
        (e.g., Instructor, E5). This method handles query-specific encoding.

        Args:
            query: Query string to embed.

        Returns:
            Embedding vector as list of floats.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the embedding provider is operational."""
