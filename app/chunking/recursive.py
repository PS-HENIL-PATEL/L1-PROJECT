"""
Enterprise RAG OS — Recursive Character Chunker
=================================================

Purpose:
    Splits text recursively using a hierarchy of separators (e.g., double
    newline, single newline, space, empty string). This preserves semantic
    boundaries like paragraphs and sentences as much as possible.
"""

from __future__ import annotations

import re
from typing import Any

from app.chunking.base import BaseChunker, Chunk
from app.logging.logger import get_logger

logger = get_logger(__name__)


class RecursiveCharacterChunker(BaseChunker):
    """
    Splits text recursively using a list of separators.

    Tries to split on the first separator. If chunks are still too large,
    moves to the next separator, and so on, until chunks are within `chunk_size`.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size in characters.
            chunk_overlap: Overlap in characters.
            separators: List of separators to use (in order of priority).
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[Chunk]:
        """Split text into chunks."""
        meta = metadata or {}

        # Strip trailing/leading whitespace from the whole text
        text = text.strip()
        if not text:
            return []

        # Split text
        string_chunks = self._split_text(text, self.separators)

        # Now we have a list of chunks, but some might be very small.
        # We need to merge them up to chunk_size and handle overlap.
        merged_chunks = self._merge_splits(string_chunks, self.separators[-1])

        # Convert to Chunk objects with metadata and offsets
        final_chunks: list[Chunk] = []
        current_char = 0

        for i, text_chunk in enumerate(merged_chunks):
            # Calculate approx start_char in original text
            # (Finding exact offset requires more complex tracking if overlap is used)
            start_char = text.find(text_chunk[: min(20, len(text_chunk))], current_char)
            if start_char == -1:
                start_char = current_char

            end_char = start_char + len(text_chunk)

            # Update search position for next chunk, accounting for overlap
            # Next chunk might overlap with this one, so we don't jump all the way to end_char
            current_char = max(0, end_char - self._chunk_overlap)

            final_chunks.append(
                Chunk(
                    content=text_chunk,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=meta.copy(),
                )
            )

        return final_chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separators."""
        # Find the appropriate separator
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if re.search(re.escape(s), text):
                separator = s
                new_separators = separators[i + 1 :]
                break

        # If separator is empty string, split into characters
        if separator == "":
            return list(text)

        # Split by separator
        splits = text.split(separator)

        # Re-attach the separator to all but the last split so we don't lose it
        # (Alternatively, we can just join them back later, which we do in _merge_splits)
        good_splits = []
        for s in splits:
            if len(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    good_splits.extend(self._split_text(s, new_separators))
                else:
                    good_splits.append(s)

        return good_splits

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge smaller splits into chunks of target size with overlap."""
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for s in splits:
            _len = len(s)
            # If current split itself is larger than chunk_size (should be rare if separators work),
            # we just have to emit it as a large chunk.

            # If adding this split exceeds chunk_size, finish current chunk
            if total + _len + (len(separator) if len(current_doc) > 0 else 0) > self._chunk_size:  # noqa: SIM102
                if total > 0:
                    chunk = separator.join(current_doc)
                    if chunk:
                        docs.append(chunk)

                    # Handle overlap: keep last N elements that fit within overlap
                    while total > self._chunk_overlap or (
                        total + _len + (len(separator) if len(current_doc) > 0 else 0) > self._chunk_size and total > 0  # noqa: E501
                    ):
                        total -= len(current_doc[0]) + (len(separator) if len(current_doc) > 1 else 0)  # noqa: E501
                        current_doc.pop(0)

            current_doc.append(s)
            total += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            chunk = separator.join(current_doc)
            if chunk:
                docs.append(chunk)

        return docs

    @property
    def name(self) -> str:
        return "recursive_character_chunker"

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap
