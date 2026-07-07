"""
Enterprise RAG OS — Base Document Loader Interface
=====================================================

Purpose:
    Abstract base class for document loaders. A loader reads a file from
    a source (filesystem, URL, S3) and produces a raw Document object
    containing the text content and metadata.

    Loader → Parser → Chunker → Embedder → VectorStore
    ^^^^^^
    (this layer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class LoadedDocument:
    """Raw document loaded from a source."""

    content: str | bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    format: str = ""
    size_bytes: int = 0
    page_count: int | None = None


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    async def load(
        self,
        source: str | Path,
        **kwargs: Any,
    ) -> list[LoadedDocument]:
        """
        Load documents from a source.

        Args:
            source: File path, URL, or other source identifier.

        Returns:
            List of LoadedDocument objects.
        """

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported file extensions (e.g., ['.pdf', '.docx'])."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this loader implementation."""
