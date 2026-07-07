"""
Enterprise RAG OS — Base LLM Interface
========================================

Purpose:
    Abstract base class for all LLM providers. Defines a unified contract
    for text generation (single-shot and streaming) so the RAG pipeline
    never depends on a specific provider's SDK.

Architecture:
    BaseLLM (abstract)
        │
        ├── OpenAICompatibleLLM  (OpenAI, OpenRouter, Ollama, vLLM)
        ├── AnthropicLLM         (future)
        └── GoogleLLM            (future)

Design Decisions:
    - generate() returns a structured LLMResponse, not a raw string.
      This ensures token counts and latency are always captured.
    - generate_stream() yields string tokens for real-time UI streaming.
    - Every implementation must expose model_name for observability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class LLMResponse:
    """
    Structured response from an LLM generation call.

    Attributes:
        text: The generated text content.
        model: The model identifier that produced this response.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens consumed.
        latency_ms: Wall-clock time for the generation call.
        finish_reason: Why the model stopped (e.g., "stop", "length").
        metadata: Provider-specific metadata.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    Every LLM integration must implement generate() and generate_stream().
    The RAG pipeline invokes these methods without knowing which provider
    is behind them.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a single complete response.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters.

        Returns:
            LLMResponse with text, token counts, and metadata.
        """

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response, yielding tokens as they arrive.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters.

        Yields:
            Individual string tokens as they are generated.
        """
        yield ""  # pragma: no cover — abstract placeholder

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is reachable."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The identifier of the model being used."""
