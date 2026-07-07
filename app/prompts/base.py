"""
Enterprise RAG OS — Base Prompt Template
==========================================

Purpose:
    Abstract base class for prompt templates. Prompts are the interface
    between the retrieval pipeline and the LLM — they structure the
    context, instructions, and query into a format the model understands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptInput:
    """Input data for prompt rendering."""

    query: str
    context: list[str] = field(default_factory=list)
    chat_history: list[dict[str, str]] = field(default_factory=list)
    system_instructions: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def render(self, input: PromptInput) -> str:
        """
        Render the prompt template with the given input.

        Args:
            input: PromptInput containing query, context, and history.

        Returns:
            Fully rendered prompt string ready for LLM.
        """

    @abstractmethod
    def render_messages(self, input: PromptInput) -> list[dict[str, str]]:
        """
        Render as a list of message dicts for chat models.

        Returns:
            List of {"role": "...", "content": "..."} dicts.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this prompt template."""

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum tokens allocated for context in this template."""
