"""
Tests — Phase 2 Ingestion Components
=======================================

Tests for parsers, chunkers, and local loader.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.chunking.recursive import RecursiveCharacterChunker
from app.loaders.local import LocalDirectoryLoader
from app.parsers.text import TextParser


class TestTextParser:
    """Test TextParser functionality."""

    @pytest.mark.asyncio
    async def test_parse_string(self) -> None:
        parser = TextParser()
        result = await parser.parse("Hello world!", format="txt")
        assert result.text == "Hello world!"
        assert result.metadata["parsed_format"] == "txt"

    @pytest.mark.asyncio
    async def test_parse_bytes(self) -> None:
        parser = TextParser()
        result = await parser.parse(b"Hello world!", format="md")
        assert result.text == "Hello world!"
        assert result.metadata["parsed_format"] == "md"

    @pytest.mark.asyncio
    async def test_unsupported_format(self) -> None:
        from app.core.exceptions import UnsupportedFormatError
        parser = TextParser()
        with pytest.raises(UnsupportedFormatError, match="does not support format"):
            await parser.parse("Hello", format="pdf")


class TestRecursiveCharacterChunker:
    """Test RecursiveCharacterChunker functionality."""

    def test_chunking_basic(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=2)
        text = "Hello world!"
        chunks = chunker.chunk(text)
        # "Hello worl", "world!" -> Overlap doesn't strictly mean exact length
        assert len(chunks) > 1

    def test_chunking_with_newlines(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=20, chunk_overlap=5)
        text = "Sentence one.\n\nSentence two.\n\nSentence three."
        chunks = chunker.chunk(text)
        assert len(chunks) == 3
        assert "Sentence one." in chunks[0].content
        assert "Sentence two." in chunks[1].content


class TestLocalDirectoryLoader:
    """Test LocalDirectoryLoader functionality."""

    @pytest.mark.asyncio
    async def test_load_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            (dir_path / "test1.txt").write_text("content 1", encoding="utf-8")
            (dir_path / "test2.md").write_text("content 2", encoding="utf-8")
            (dir_path / "sub").mkdir()
            (dir_path / "sub" / "test3.txt").write_text("content 3", encoding="utf-8")

            loader = LocalDirectoryLoader()
            docs = await loader.load(dir_path)
            assert len(docs) == 3

            # Check contents
            contents = {doc.content for doc in docs}
            assert "content 1" in contents
            assert "content 2" in contents
            assert "content 3" in contents

            # Check formats
            formats = {doc.format for doc in docs}
            assert formats == {"txt", "md"}

    @pytest.mark.asyncio
    async def test_non_recursive_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            (dir_path / "test1.txt").write_text("content 1", encoding="utf-8")
            (dir_path / "sub").mkdir()
            (dir_path / "sub" / "test3.txt").write_text("content 3", encoding="utf-8")

            loader = LocalDirectoryLoader()
            docs = await loader.load(dir_path, recursive=False)
            assert len(docs) == 1
            assert docs[0].content == "content 1"
