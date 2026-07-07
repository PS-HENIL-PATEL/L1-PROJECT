"""
Enterprise RAG OS — Advanced Retrieval Pipeline
=================================================

Purpose:
    Coordinates the retrieval and reranking process.
    Provides a unified interface for fetching the most relevant chunks
    for a given query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.exceptions import PipelineError
from app.logging.logger import get_logger
from app.pipelines.base import BasePipeline, PipelineContext
from app.utils.timing import Timer

if TYPE_CHECKING:
    from app.rerankers.base import BaseReranker
    from app.retrievers.base import BaseRetriever

logger = get_logger(__name__)


class AdvancedRetrievalPipeline(BasePipeline[str, dict[str, Any]]):
    """
    Retrieval pipeline featuring Vector Search followed by Cross-Encoder Reranking.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker | None = None,
    ) -> None:
        """
        Initialize the retrieval pipeline.

        Args:
            retriever: The primary retrieval component (e.g., VectorRetriever).
            reranker: Optional reranking component (e.g., CrossEncoderReranker).
        """
        self.retriever = retriever
        self.reranker = reranker

    async def run(
        self,
        input: str,
        context: PipelineContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute the retrieval pipeline.

        Args:
            input: The user's search query.
            context: Optional pipeline context.
            **kwargs: Extra arguments.
                - initial_k: How many chunks to retrieve initially (default 50).
                - final_k: How many chunks to return after reranking (default 5).
                - filters: Optional metadata filters for the retriever.

        Returns:
            Dictionary containing the ranked chunks and telemetry.
        """
        ctx = context or PipelineContext()
        query = input

        initial_k = kwargs.get("initial_k", 50)
        final_k = kwargs.get("final_k", 5)
        filters = kwargs.get("filters")

        try:
            # 1. Primary Retrieval
            with Timer("retrieval_ms") as t:
                retrieval_res = await self.retriever.retrieve(
                    query=query,
                    top_k=initial_k if self.reranker else final_k,
                    filters=filters,
                )
            ctx.timings["retrieval_ms"] = t.elapsed_ms

            chunks = retrieval_res.chunks
            logger.info(
                "Primary retrieval complete",
                query=query,
                retrieved_count=len(chunks)
            )

            if not chunks:
                return {
                    "query": query,
                    "results": [],
                    "total_candidates": 0,
                    "timings": ctx.timings,
                }

            # 2. Reranking (if configured)
            if self.reranker:
                with Timer("rerank_ms") as t:
                    ranked_chunks = await self.reranker.rerank(
                        query=query,
                        chunks=chunks,
                        top_k=final_k,
                    )
                ctx.timings["rerank_ms"] = t.elapsed_ms

                # Format output from reranker
                results = [
                    {
                        "chunk_id": rc.chunk_id,
                        "document_id": rc.document_id,
                        "content": rc.content,
                        "score": rc.rerank_score,
                        "original_score": rc.original_score,
                        "metadata": rc.metadata,
                    }
                    for rc in ranked_chunks
                ]
            else:
                # No reranker, just format the retrieved chunks
                results = [
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "content": c.content,
                        "score": c.score,
                        "original_score": c.score,
                        "metadata": c.metadata,
                    }
                    for c in chunks[:final_k]
                ]

            return {
                "query": query,
                "results": results,
                "total_candidates": len(chunks),
                "timings": ctx.timings,
            }

        except Exception as e:
            logger.error("Advanced retrieval pipeline failed", error=str(e))
            raise PipelineError(f"Retrieval pipeline failed: {e}") from e

    async def health_check(self) -> bool:
        """Check health of retriever."""
        return await self.retriever.health_check()

    @property
    def name(self) -> str:
        return "advanced_retrieval_pipeline"

    @property
    def stages(self) -> list[str]:
        stages = ["retrieve"]
        if self.reranker:
            stages.append("rerank")
        return stages
