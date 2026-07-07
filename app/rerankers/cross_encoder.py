"""
Enterprise RAG OS — Cross-Encoder Reranker
=============================================

Purpose:
    Reranks documents using a Cross-Encoder model.
    A Cross-Encoder takes both the query and the document as a single
    input and outputs a highly accurate relevance score.
"""

from __future__ import annotations

import asyncio
from typing import Any

from sentence_transformers import CrossEncoder

from app.core.exceptions import EnterpriseRAGError
from app.logging.logger import get_logger
from app.rerankers.base import BaseReranker, RankedChunk

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using sentence-transformers Cross-Encoders.
    Runs in a thread pool to avoid blocking the event loop.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Initialize the Cross-Encoder.

        Args:
            model_name: The HuggingFace model identifier.
        """
        self._model_name = model_name
        try:
            logger.info("Loading Cross-Encoder model", model=model_name)
            self._model = CrossEncoder(model_name)
            logger.info("Cross-Encoder loaded successfully")
        except Exception as e:
            logger.error("Failed to load Cross-Encoder", error=str(e))
            raise EnterpriseRAGError(
                detail=f"Failed to load Cross-Encoder '{model_name}': {e}",
                error_code="MODEL_LOAD_ERROR"
            ) from e

    async def rerank(
        self,
        query: str,
        chunks: list[Any],  # Expected to be list[RetrievedChunk]
        top_k: int = 5,
        **_kwargs: Any,
    ) -> list[RankedChunk]:
        """
        Rerank a list of chunks based on cross-encoder similarity to query.

        Args:
            query: The search query.
            chunks: List of RetrievedChunk objects.
            top_k: Top K results to return.
            **kwargs: Extra arguments.

        Returns:
            List of RankedChunk objects.
        """
        if not chunks:
            return []

        if not hasattr(chunks[0], "content"):
            logger.warning("Reranker received invalid chunks, missing 'content' attr")
            return []

        # Prepare pairs of (query, chunk_content) for the model
        pairs = [(query, chunk.content) for chunk in chunks]

        try:
            # Predict scores using a thread pool
            scores_np = await asyncio.to_thread(
                self._model.predict,
                pairs,
                show_progress_bar=False
            )
            scores = scores_np.tolist()
        except Exception as e:
            logger.error("Cross-Encoder prediction failed", error=str(e))
            raise EnterpriseRAGError(
                detail=f"Cross-Encoder prediction failed: {e}",
                error_code="RERANK_ERROR"
            ) from e

        # Create ranked chunks
        ranked = []
        for _i, (chunk, score) in enumerate(zip(chunks, scores, strict=False)):
            ranked.append(
                RankedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    original_score=chunk.score,
                    rerank_score=score,
                    rank=0,  # Will be set after sorting
                    metadata=chunk.metadata,
                    page_number=chunk.page_number,
                    section=chunk.section,
                )
            )

        # Sort descending by rerank score
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)

        # Limit to top_k and update rank
        top_ranked = ranked[:top_k]
        for i, r_chunk in enumerate(top_ranked, 1):
            r_chunk.rank = i

        logger.debug(
            "Cross-Encoder reranking complete",
            in_chunks=len(chunks),
            out_chunks=len(top_ranked),
            top_score=top_ranked[0].rerank_score if top_ranked else None
        )

        return top_ranked

    @property
    def name(self) -> str:
        return f"cross_encoder_{self._model_name.replace('/', '_')}"
