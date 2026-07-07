"""
Enterprise RAG OS — Base Memory Interface
============================================

Purpose:
    Abstract base class for conversation memory. Memory enables multi-turn
    conversations by persisting context across requests within a session.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A single entry in conversation memory."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """Abstract base class for all memory implementations."""

    @abstractmethod
    async def add(self, session_id: str, entry: MemoryEntry) -> None:
        """Add an entry to session memory."""

    @abstractmethod
    async def get(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve recent memory entries for a session."""

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear all memory for a session."""

    @abstractmethod
    async def summarize(self, session_id: str) -> str:
        """Generate a summary of the conversation history."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this memory implementation."""
