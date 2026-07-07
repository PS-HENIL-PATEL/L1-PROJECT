"""
Tests — Phase 3 Retrieval Components
=======================================

Tests for the vector retriever, cross-encoder reranker, and retrieval pipeline.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.embeddings.base import BaseEmbeddingProvider, EmbeddingResult
from app.pipelines.retrieval import AdvancedRetrievalPipeline
from app.retrievers.vector import VectorRetriever
from app.vectorstores.base import BaseVectorStore, VectorSearchResult


class MockEmbedder(BaseEmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "mock"

    async def embed(self, texts: list[str], **kwargs: Any) -> EmbeddingResult:
        return EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3] for _ in texts],
            model=self.model_name,
            dimension=self.dimension,
        )

    async def embed_query(self, query: str, **kwargs: Any) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def health_check(self) -> bool:
        return True


class MockVectorStore(BaseVectorStore):
    @property
    def name(self) -> str:
        return "mock_store"

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> None:
        pass

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                id="1", score=0.9, content="Result 1", metadata={"document_id": "doc1"}
            ),
            VectorSearchResult(
                id="2", score=0.8, content="Result 2", metadata={"document_id": "doc2"}
            ),
        ]

    async def delete(self, ids: list[str], **_kwargs: Any) -> None:
        pass

    async def health_check(self) -> bool:
        return True

    async def count(self) -> int:
        return 2

    async def list_sources(self) -> list[dict[str, Any]]:
        return [{"source": "doc1", "count": 1}, {"source": "doc2", "count": 1}]


class TestVectorRetriever:
    """Test VectorRetriever functionality."""

    @pytest.mark.asyncio
    async def test_retrieve(self) -> None:
        retriever = VectorRetriever(embedder=MockEmbedder(), vectorstore=MockVectorStore())
        result = await retriever.retrieve("test query")

        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_id == "1"
        assert result.chunks[0].document_id == "doc1"
        assert result.chunks[0].score == 0.9
        assert result.query == "test query"


class TestAdvancedRetrievalPipeline:
    """Test AdvancedRetrievalPipeline functionality."""

    @pytest.mark.asyncio
    async def test_pipeline_without_reranker(self) -> None:
        retriever = VectorRetriever(embedder=MockEmbedder(), vectorstore=MockVectorStore())
        pipeline = AdvancedRetrievalPipeline(retriever=retriever)

        result = await pipeline.run("test query")

        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert result["results"][0]["chunk_id"] == "1"
        assert result["results"][0]["score"] == 0.9
        assert "timings" in result
