"""
Enterprise RAG OS — Content Hashing Utilities
===============================================

Purpose:
    Content-addressable hashing for document deduplication and cache keys.
    When the same document is uploaded twice, we detect it instantly by
    comparing hashes instead of re-processing the entire file.

Why SHA-256?
    - Collision-resistant: 2^128 operations to find a collision (birthday attack).
    - Fast: Hardware-accelerated on modern CPUs (SHA-NI instruction set).
    - Standard: Universally supported, no compatibility concerns.
    - Not MD5: MD5 has known collision attacks — unacceptable for deduplication.
    - Not SHA-3: Marginally more secure but slower and unnecessary for our use case.

Usage:
    from app.utils.hashing import content_hash, file_hash

    hash1 = content_hash("Hello, world!")
    hash2 = file_hash(Path("report.pdf"))
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def content_hash(content: str | bytes, algorithm: str = "sha256") -> str:
    """
    Compute a hex digest hash of the given content.

    Args:
        content: String or bytes to hash.
        algorithm: Hash algorithm name (default: sha256).

    Returns:
        Hex-encoded hash string.

    Example:
        >>> content_hash("Hello")
        '185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969'
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.new(algorithm, content).hexdigest()


def file_hash(file_path: Path, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """
    Compute a hex digest hash of a file's contents.

    Reads the file in chunks to handle large files without loading
    everything into memory.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm name (default: sha256).
        chunk_size: Read buffer size in bytes.

    Returns:
        Hex-encoded hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file is not readable.
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()
