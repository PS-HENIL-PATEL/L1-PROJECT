"""
Enterprise RAG OS — Serialization Utilities
=============================================

Purpose:
    JSON and datetime serialization helpers that handle edge cases
    not covered by Python's built-in json module (datetime, Path,
    Pydantic models, UUID, bytes, sets, enums).

Why custom serialization?
    Python's json.dumps() fails on datetime, Path, UUID, and other common types.
    Rather than sprinkling custom encoders everywhere, we provide a single
    `serialize_json()` function that handles all common types consistently.

Usage:
    from app.utils.serialization import serialize_json, deserialize_json

    data = {"created_at": datetime.now(), "path": Path("/data/file.pdf")}
    json_str = serialize_json(data)
"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import UUID


class EnterpriseJSONEncoder(json.JSONEncoder):
    """
    Extended JSON encoder that handles common Python types.

    Supported types beyond stdlib:
        - datetime/date → ISO 8601 string
        - UUID → string
        - Path → string
        - bytes → UTF-8 string (with fallback to repr)
        - set/frozenset → list
        - Enum → value
        - Pydantic BaseModel → dict (via model_dump)
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, (Path, PurePosixPath, PureWindowsPath)):
            return str(o)
        if isinstance(o, bytes):
            try:
                return o.decode("utf-8")
            except UnicodeDecodeError:
                return repr(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o) if all(isinstance(x, str) for x in o) else list(o)
        if isinstance(o, Enum):
            return o.value
        # Pydantic model support
        if hasattr(o, "model_dump"):
            return o.model_dump()
        return super().default(o)


def serialize_json(data: Any, *, pretty: bool = False) -> str:
    """
    Serialize data to a JSON string with extended type support.

    Args:
        data: The data to serialize.
        pretty: If True, format with indentation for readability.

    Returns:
        JSON string.
    """
    return json.dumps(
        data,
        cls=EnterpriseJSONEncoder,
        indent=2 if pretty else None,
        ensure_ascii=False,
    )


def deserialize_json(json_str: str) -> Any:
    """
    Deserialize a JSON string to Python objects.

    Args:
        json_str: JSON string to parse.

    Returns:
        Parsed Python object (dict, list, str, int, float, bool, None).

    Raises:
        json.JSONDecodeError: If the string is not valid JSON.
    """
    return json.loads(json_str)
