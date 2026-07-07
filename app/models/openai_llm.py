"""
Enterprise RAG OS — OpenAI-Compatible LLM Provider
=====================================================

Purpose:
    Concrete LLM implementation using the OpenAI Python SDK.
    Supports any OpenAI-compatible API by swapping base_url:
        - OpenAI:      https://api.openai.com/v1
        - OpenRouter:  https://openrouter.ai/api/v1
        - Ollama:      http://localhost:11434/v1
        - vLLM:        http://localhost:8000/v1
        - LM Studio:   http://localhost:1234/v1

Design Decisions:
    - Uses openai.AsyncOpenAI for non-blocking I/O within FastAPI.
    - Retry logic with exponential backoff handles transient 429/5xx errors.
    - Token counts are extracted from the API response for cost tracking.
    - Streaming uses server-sent events (SSE) under the hood.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    RateLimitError,
)

from app.core.exceptions import (
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)
from app.logging.logger import get_logger
from app.models.llm import BaseLLM, LLMResponse

logger = get_logger(__name__)

# Default retry configuration
_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 1.0  # seconds


class OpenAICompatibleLLM(BaseLLM):
    """
    LLM provider for any OpenAI-compatible API.

    Examples:
        # OpenAI
        llm = OpenAICompatibleLLM(api_key="sk-...", model="gpt-4o-mini")

        # Ollama (local)
        llm = OpenAICompatibleLLM(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2",
        )

        # OpenRouter
        llm = OpenAICompatibleLLM(
            api_key="sk-or-...",
            base_url="https://openrouter.ai/api/v1",
            model="meta-llama/llama-3.1-8b-instruct",
        )
    """

    def __init__(
        self,
        api_key: str = "no-key",
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,  # We handle retries ourselves
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)

        logger.info(
            "OpenAI-compatible LLM initialized",
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a complete response from the LLM.

        Implements retry with exponential backoff for transient errors.
        """
        temp = temperature if temperature is not None else self._temperature
        tokens = max_tokens if max_tokens is not None else self._max_tokens

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            start_time = time.perf_counter()
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temp,
                    max_tokens=tokens,
                    **kwargs,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000
                choice = response.choices[0]
                usage = response.usage

                return LLMResponse(
                    text=choice.message.content or "",
                    model=response.model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    latency_ms=round(latency_ms, 2),
                    finish_reason=choice.finish_reason or "stop",
                )

            except RateLimitError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = _RETRY_DELAY_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM rate limited, retrying",
                        attempt=attempt,
                        delay=delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
                else:
                    raise LLMRateLimitError(
                        detail=f"Rate limited after {self._max_retries} retries: {e}"
                    ) from e

            except APIConnectionError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = _RETRY_DELAY_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM connection error, retrying",
                        attempt=attempt,
                        delay=delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
                else:
                    raise LLMConnectionError(
                        detail=f"Cannot connect to LLM after {self._max_retries} retries: {e}"
                    ) from e

            except Exception as e:
                logger.error(
                    "LLM generation failed",
                    error=str(e),
                    attempt=attempt,
                )
                raise LLMResponseError(
                    detail=f"LLM generation failed: {e}"
                ) from e

        # Should not reach here, but safety net
        raise LLMError(
            detail=f"LLM generation failed after {self._max_retries} retries"
        ) from last_error

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the LLM as they are generated.

        Yields individual string tokens for real-time UI rendering.
        """
        temp = temperature if temperature is not None else self._temperature
        tokens = max_tokens if max_tokens is not None else self._max_tokens

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temp,
                max_tokens=tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except RateLimitError as e:
            raise LLMRateLimitError(
                detail=f"Rate limited during streaming: {e}"
            ) from e
        except APIConnectionError as e:
            raise LLMConnectionError(
                detail=f"Connection lost during streaming: {e}"
            ) from e
        except Exception as e:
            raise LLMResponseError(
                detail=f"Streaming generation failed: {e}"
            ) from e

    async def health_check(self) -> bool:
        """
        Verify the LLM provider is reachable by sending a minimal request.
        """
        try:
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.warning("LLM health check failed", error=str(e))
            return False

    @property
    def model_name(self) -> str:
        return self._model
