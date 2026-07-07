"""
Enterprise RAG OS — Vector Retriever
=======================================

Purpose:
    Retrieves documents from a vector store using dense embeddings.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from app.core.exceptions import EnterpriseRAGError
from app.logging.logger import get_logger
from app.retrievers.base import BaseRetriever, RetrievalResult, RetrievedChunk

if TYPE_CHECKING:
    from app.embeddings.base import BaseEmbeddingProvider
    from app.vectorstores.base import BaseVectorStore

logger = get_logger(__name__)


class VectorRetriever(BaseRetriever):
    """
    Retrieves relevant documents using dense vector similarity.
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        vectorstore: BaseVectorStore,
    ) -> None:
        """
        Initialize the vector retriever.

        Args:
            embedder: Provider used to generate query embeddings.
            vectorstore: The vector store backend to search.
        """
        self.embedder = embedder
        self.vectorstore = vectorstore

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve chunks matching the query.

        Args:
            query: User's search query.
            top_k: Number of results to retrieve.
            filters: Optional metadata filters.
            **kwargs: Additional args.

        Returns:
            RetrievalResult containing ranked chunks.
        """
        start_time = time.perf_counter()

        try:
            # 1. Embed the query
            query_embedding = await self.embedder.embed_query(query)

            # 2. Search the vector store
            search_results = await self.vectorstore.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                **kwargs,
            )

            # 3. Map results to RetrievedChunk objects
            chunks = []
            for hit in search_results:
                metadata = hit.metadata.copy()
                doc_id = metadata.get("document_id", "unknown")
                page = metadata.get("page_number")
                section = metadata.get("section")

                chunks.append(
                    RetrievedChunk(
                        chunk_id=hit.id,
                        document_id=doc_id,
                        content=hit.content,
                        score=hit.score,
                        metadata=metadata,
                        page_number=page,
                        section=section,
                    )
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "Vector retrieval complete",
                query=query,
                top_k=top_k,
                results=len(chunks),
                latency_ms=round(latency_ms, 2),
            )

            return RetrievalResult(
                chunks=chunks,
                query=query,
                strategy=self.name,
                latency_ms=round(latency_ms, 2),
                total_candidates=len(chunks),
            )

        except Exception as e:
            logger.error("Vector retrieval failed", query=query, error=str(e))
            raise EnterpriseRAGError(
                detail=f"Retrieval failed: {e}",
                error_code="RETRIEVAL_FAILED",
                status_code=500,
            ) from e

    async def health_check(self) -> bool:
        """Check if both embedder and vector store are healthy."""
        emb_health = await self.embedder.health_check()
        vs_health = await self.vectorstore.health_check()
        return emb_health and vs_health

    @property
    def name(self) -> str:
        return f"vector_{self.vectorstore.name}_{self.embedder.model_name}"
