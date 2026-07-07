"""
Enterprise RAG OS — ID Generation Utilities
=============================================

Purpose:
    Generate unique identifiers for documents, chunks, sessions, and requests.
    Provides both full UUIDs (for database primary keys) and short IDs
    (for human-readable references in logs and UI).

Why UUID v4?
    - Universally unique without coordination (no central ID server needed).
    - 122 bits of randomness → collision probability is negligible.
    - Standard format recognized by every database and serialization library.

Why short IDs?
    - UUIDs are 36 characters — painful to read in logs and copy-paste.
    - Short IDs (12 chars) are sufficient for human identification within
      a single system while remaining collision-resistant for practical use.
    - Format: <prefix>_<random> (e.g., "doc_a1b2c3d4e5f6", "chk_x9y8z7w6v5u4")

Usage:
    from app.utils.ids import generate_id, generate_short_id

    doc_id = generate_id()                    # "a1b2c3d4-..."
    chunk_id = generate_short_id("chk")       # "chk_a1b2c3d4e5f6"
    request_id = generate_short_id("req")     # "req_x9y8z7w6v5u4"
"""

from __future__ import annotations

import secrets
import uuid


def generate_id() -> str:
    """
    Generate a UUID v4 string.

    Returns:
        UUID string in standard format (e.g., "550e8400-e29b-41d4-a716-446655440000").
    """
    return str(uuid.uuid4())


def generate_short_id(prefix: str = "", length: int = 12) -> str:
    """
    Generate a short, human-readable ID.

    Uses `secrets.token_hex` for cryptographically random bytes,
    then truncates to the desired length.

    Args:
        prefix: Optional prefix (e.g., "doc", "chk", "req").
                If provided, the result is "{prefix}_{random}".
        length: Number of hex characters in the random portion.

    Returns:
        Short ID string.

    Example:
        >>> generate_short_id("doc")
        'doc_a1b2c3d4e5f6'
        >>> generate_short_id()
        'a1b2c3d4e5f6'
    """
    random_part = secrets.token_hex(length // 2 + 1)[:length]
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part
