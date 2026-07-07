"""
Enterprise RAG OS — Base Chunking Strategy Interface
======================================================

Purpose:
    Abstract base class for all text chunking strategies. Chunking is the
    process of splitting documents into smaller pieces for embedding and
    retrieval. Different strategies produce different chunk qualities.

Why Multiple Strategies?
    There is no universally best chunking strategy. The optimal choice depends
    on document type, query patterns, and embedding model context window:

    ┌─────────────────────┬───────────────────────┬──────────────────────────┐
    │ Strategy            │ Best For              │ Trade-off                │
    ├─────────────────────┼───────────────────────┼──────────────────────────┤
    │ Recursive Character │ General-purpose        │ May split mid-sentence   │
    │ Semantic            │ Preserving meaning     │ Slower, needs embeddings │
    │ Markdown-aware      │ Structured docs        │ Only works with markdown │
    │ Token-based         │ LLM context budgeting  │ Requires tokenizer       │
    │ Sliding Window      │ Overlapping context    │ More chunks, more storage│
    │ Parent-Child        │ Hierarchical retrieval │ Complex implementation   │
    └─────────────────────┴───────────────────────┴──────────────────────────┘
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """
    A single chunk of text produced by a chunking strategy.

    Contains the text content plus metadata about its position
    in the original document (character offsets, page, section).
    """

    content: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    page_number: int | None = None
    section: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The full text to chunk.
            metadata: Document metadata to propagate to chunks.

        Returns:
            List of Chunk objects.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this chunking strategy."""

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Target chunk size (in characters or tokens)."""

    @property
    @abstractmethod
    def chunk_overlap(self) -> int:
        """Overlap between consecutive chunks."""
