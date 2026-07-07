"""
Tests — Utility Functions
============================

Tests for hashing, timing, serialization, and ID generation utilities.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from app.utils.hashing import content_hash, file_hash
from app.utils.ids import generate_id, generate_short_id
from app.utils.serialization import (
    deserialize_json,
    serialize_json,
)
from app.utils.timing import Timer


class TestContentHash:
    """Test content hashing utility."""

    def test_string_hash(self) -> None:
        h = content_hash("Hello, world!")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_bytes_hash(self) -> None:
        h = content_hash(b"Hello, world!")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_deterministic(self) -> None:
        h1 = content_hash("test")
        h2 = content_hash("test")
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_custom_algorithm(self) -> None:
        h = content_hash("test", algorithm="md5")
        assert len(h) == 32  # MD5 hex digest


class TestFileHash:
    """Test file hashing utility."""

    def test_file_hash(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            f.flush()
            h = file_hash(Path(f.name))
            assert isinstance(h, str)
            assert len(h) == 64

    def test_file_hash_deterministic(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            f.flush()
            h1 = file_hash(Path(f.name))
            h2 = file_hash(Path(f.name))
            assert h1 == h2


class TestGenerateId:
    """Test UUID generation."""

    def test_generates_valid_uuid(self) -> None:
        id = generate_id()
        # Should not raise
        UUID(id)

    def test_generates_unique_ids(self) -> None:
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestGenerateShortId:
    """Test short ID generation."""

    def test_default_length(self) -> None:
        id = generate_short_id()
        assert len(id) == 12

    def test_with_prefix(self) -> None:
        id = generate_short_id("doc")
        assert id.startswith("doc_")
        # "doc_" (4) + 12 random chars
        assert len(id) == 16

    def test_custom_length(self) -> None:
        id = generate_short_id(length=8)
        assert len(id) == 8

    def test_uniqueness(self) -> None:
        ids = {generate_short_id("req") for _ in range(100)}
        assert len(ids) == 100


class TestSerialization:
    """Test JSON serialization with extended type support."""

    def test_serialize_datetime(self) -> None:
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        result = serialize_json({"ts": dt})
        assert "2024-01-15" in result

    def test_serialize_path(self) -> None:
        p = Path("/data/file.pdf")
        result = serialize_json({"path": p})
        parsed = json.loads(result)
        assert "file.pdf" in parsed["path"]

    def test_serialize_uuid(self) -> None:
        from uuid import uuid4

        uid = uuid4()
        result = serialize_json({"id": uid})
        parsed = json.loads(result)
        assert parsed["id"] == str(uid)

    def test_serialize_set(self) -> None:
        result = serialize_json({"tags": {"b", "a", "c"}})
        parsed = json.loads(result)
        assert parsed["tags"] == ["a", "b", "c"]  # Sorted

    def test_serialize_pretty(self) -> None:
        result = serialize_json({"key": "value"}, pretty=True)
        assert "\n" in result
        assert "  " in result

    def test_deserialize(self) -> None:
        data = deserialize_json('{"key": "value", "num": 42}')
        assert data["key"] == "value"
        assert data["num"] == 42


class TestTimer:
    """Test the Timer context manager."""

    def test_timer_measures_time(self) -> None:
        import time

        with Timer("test_op") as t:
            time.sleep(0.01)  # 10ms
        assert t.elapsed_ms > 5  # Should be at least 5ms
        assert t.elapsed_ms < 500  # Should be less than 500ms

    def test_timer_has_name(self) -> None:
        with Timer("my_operation") as t:
            pass
        assert t.name == "my_operation"
