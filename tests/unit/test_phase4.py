"""
Tests — Phase 4: LLM, Prompt Templates, and RAG Pipeline
============================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest

from app.models.llm import BaseLLM, LLMResponse
from app.pipelines.rag import RAGPipeline
from app.pipelines.retrieval import AdvancedRetrievalPipeline
from app.prompts.base import PromptInput
from app.prompts.rag import RAGPromptTemplate
from app.retrievers.base import BaseRetriever, RetrievalResult, RetrievedChunk

# ── Mock LLM ─────────────────────────────────────────────────────────────────


class MockLLM(BaseLLM):
    """A mock LLM that returns a fixed response."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            text="This is a mock answer based on the context.",
            model="mock-model",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            latency_ms=50.0,
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **_kwargs: Any,
    ) -> AsyncIterator[str]:
        for token in ["This ", "is ", "streaming."]:
            yield token

    async def health_check(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "mock-model"


# ── Mock Retriever ────────────────────────────────────────────────────────────


class MockRetriever(BaseRetriever):
    """A mock retriever returning fixed chunks."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> RetrievalResult:
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="RAG combines retrieval with generation.",
                score=0.95,
                metadata={"file_name": "rag_intro.txt"},
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                document_id="doc-2",
                content="Vector databases store embeddings.",
                score=0.85,
                metadata={"file_name": "vectors.txt"},
            ),
        ]
        return RetrievalResult(
            chunks=chunks[:top_k],
            query=query,
            strategy="mock",
            latency_ms=10.0,
            total_candidates=2,
        )

    async def health_check(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "mock_retriever"


# ── Test RAG Prompt Template ─────────────────────────────────────────────────


class TestRAGPromptTemplate:
    """Test the RAG prompt template rendering."""

    def test_render_messages_with_context(self) -> None:
        template = RAGPromptTemplate()
        prompt_input = PromptInput(
            query="What is RAG?",
            context=[
                "RAG is retrieval-augmented generation.",
                "It improves LLM accuracy.",
            ],
            metadata={
                "chunks": [
                    {"file_name": "doc1.txt"},
                    {"file_name": "doc2.txt"},
                ]
            },
        )
        messages = template.render_messages(prompt_input)

        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is RAG?"
        assert "Document 1" in messages[0]["content"]
        assert "doc1.txt" in messages[0]["content"]

    def test_render_messages_no_context(self) -> None:
        template = RAGPromptTemplate()
        prompt_input = PromptInput(query="What is RAG?")
        messages = template.render_messages(prompt_input)

        assert len(messages) == 2
        assert "No relevant documents" in messages[0]["content"]

    def test_render_messages_with_history(self) -> None:
        template = RAGPromptTemplate()
        prompt_input = PromptInput(
            query="Tell me more.",
            context=["Some context."],
            chat_history=[
                {"role": "user", "content": "What is RAG?"},
                {"role": "assistant", "content": "RAG is..."},
            ],
        )
        messages = template.render_messages(prompt_input)

        # system + 2 history + user
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["content"] == "Tell me more."

    def test_render_single_string(self) -> None:
        template = RAGPromptTemplate()
        prompt_input = PromptInput(
            query="What is RAG?",
            context=["Some context."],
        )
        rendered = template.render(prompt_input)

        assert "[SYSTEM]" in rendered
        assert "[USER]" in rendered
        assert "What is RAG?" in rendered

    def test_name_and_max_tokens(self) -> None:
        template = RAGPromptTemplate(max_context_tokens_limit=5000)
        assert template.name == "rag_prompt_v1"
        assert template.max_context_tokens == 5000


# ── Test RAG Pipeline ────────────────────────────────────────────────────────


class TestRAGPipeline:
    """Test the complete RAG pipeline with mocked components."""

    @pytest.mark.asyncio
    async def test_rag_pipeline_full_run(self) -> None:
        retriever = MockRetriever()
        retrieval_pipeline = AdvancedRetrievalPipeline(
            retriever=retriever,
            reranker=None,
        )
        llm = MockLLM()
        pipeline = RAGPipeline(
            retrieval_pipeline=retrieval_pipeline,
            llm=llm,
        )

        from app.schemas.queries import QueryRequest

        query = QueryRequest(query="What is RAG?", top_k=2)
        response = await pipeline.run(input=query)

        assert response.query == "What is RAG?"
        assert response.answer == "This is a mock answer based on the context."
        assert response.confidence > 0.0
        assert len(response.sources) == 2
        assert response.sources[0].filename == "rag_intro.txt"
        assert response.explainability is not None
        assert response.explainability.llm_model == "mock-model"
        assert response.explainability.total_tokens == 120

    @pytest.mark.asyncio
    async def test_rag_pipeline_no_sources(self) -> None:
        retrieval_pipeline = AdvancedRetrievalPipeline(
            retriever=MockRetriever(),
            reranker=None,
        )
        llm = MockLLM()
        pipeline = RAGPipeline(
            retrieval_pipeline=retrieval_pipeline,
            llm=llm,
        )

        from app.schemas.queries import QueryRequest

        query = QueryRequest(
            query="What is RAG?",
            top_k=2,
            include_sources=False,
        )
        response = await pipeline.run(input=query)

        assert response.answer != ""
        assert len(response.sources) == 0

    @pytest.mark.asyncio
    async def test_rag_pipeline_health_check(self) -> None:
        retrieval_pipeline = AdvancedRetrievalPipeline(
            retriever=MockRetriever(),
            reranker=None,
        )
        llm = MockLLM()
        pipeline = RAGPipeline(
            retrieval_pipeline=retrieval_pipeline,
            llm=llm,
        )

        assert await pipeline.health_check() is True

    def test_pipeline_properties(self) -> None:
        retrieval_pipeline = AdvancedRetrievalPipeline(
            retriever=MockRetriever(),
            reranker=None,
        )
        llm = MockLLM()
        pipeline = RAGPipeline(
            retrieval_pipeline=retrieval_pipeline,
            llm=llm,
        )

        assert pipeline.name == "rag_pipeline"
        assert "retrieve" in pipeline.stages
        assert "generate" in pipeline.stages


# ── Test Mock LLM ────────────────────────────────────────────────────────────


class TestMockLLM:
    """Verify the mock LLM works for testing purposes."""

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        llm = MockLLM()
        response = await llm.generate(
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response.text != ""
        assert response.model == "mock-model"
        assert response.total_tokens == 120

    @pytest.mark.asyncio
    async def test_generate_stream(self) -> None:
        llm = MockLLM()
        tokens: list[str] = []
        async for token in llm.generate_stream(
            messages=[{"role": "user", "content": "Hello"}]
        ):
            tokens.append(token)
        assert len(tokens) == 3
        assert "".join(tokens) == "This is streaming."
