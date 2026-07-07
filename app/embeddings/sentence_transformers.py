"""
Enterprise RAG OS — Sentence Transformers Embedding
=====================================================

Purpose:
    Generates embeddings locally using the sentence-transformers library.
    Great for privacy, local development, and avoiding API costs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sentence_transformers import SentenceTransformer

from app.core.exceptions import EmbeddingError
from app.embeddings.base import BaseEmbeddingProvider, EmbeddingResult
from app.logging.logger import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedding(BaseEmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Runs CPU-bound embedding generation in a separate thread to
    prevent blocking the FastAPI event loop.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model.

        Args:
            model_name: The HuggingFace model identifier.
        """
        self._model_name = model_name

        try:
            # Load model synchronously during initialization
            logger.info("Loading sentence-transformers model", model=model_name)
            self._model = SentenceTransformer(model_name)

            # Determine dimension from model
            # For SentenceTransformers, this is usually model.get_sentence_embedding_dimension()
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Model loaded successfully", dimension=self._dimension)
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            raise EmbeddingError(
                detail=f"Failed to load local model '{model_name}': {e}",
                context={"model": model_name}
            ) from e

    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,  # noqa: ARG002
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self._model_name,
                dimension=self._dimension,
            )

        start_time = time.perf_counter()

        try:
            # Embeddings are CPU bound, run in thread pool
            # Convert numpy arrays to nested lists of floats
            embeddings_np = await asyncio.to_thread(
                self._model.encode,
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            embeddings = embeddings_np.tolist()

        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise EmbeddingError(
                detail=f"Failed to generate embeddings: {e}"
            ) from e

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Rough token count estimation for local models since we don't have direct access
        # to the tokenizer's output length easily without extra work.
        # Approx 4 chars per token.
        estimated_tokens = sum(len(t) for t in texts) // 4

        return EmbeddingResult(
            embeddings=embeddings,
            model=self._model_name,
            dimension=self._dimension,
            token_count=estimated_tokens,
            latency_ms=round(latency_ms, 2),
        )

    async def embed_query(self, query: str, **kwargs: Any) -> list[float]:
        """Generate embedding for a single query."""
        # Some models use instructions. If using an instruction model (e.g. INSTRUCTOR),
        # this is where we'd prepend the query instruction. For MiniLM, it's the same.
        result = await self.embed([query], **kwargs)
        return result.embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def health_check(self) -> bool:
        """Check if model is loaded and functional."""
        try:
            await self.embed_query("test")
            return True
        except Exception as e:
            logger.error("Embedding health check failed", error=str(e))
            return False
