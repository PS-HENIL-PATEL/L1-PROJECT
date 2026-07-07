"""
Enterprise RAG OS — Base Parser Interface
============================================

Purpose:
    Abstract base class for document parsers. A parser extracts structured
    text and metadata from raw document content (e.g., extracting text
    from PDF binary data, parsing HTML into clean text).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedContent:
    """Structured content extracted from a document."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[str] | None = None
    sections: list[dict[str, str]] | None = None


class BaseParser(ABC):
    """Abstract base class for all document parsers."""

    @abstractmethod
    async def parse(
        self,
        content: str | bytes,
        format: str,
        **kwargs: Any,
    ) -> ParsedContent:
        """
        Parse raw content into structured text.

        Args:
            content: Raw document content (text or bytes).
            format: Document format (e.g., 'pdf', 'html').

        Returns:
            ParsedContent with extracted text and metadata.
        """

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported formats."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this parser implementation."""
