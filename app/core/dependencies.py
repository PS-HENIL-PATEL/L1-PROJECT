"""
Enterprise RAG OS — FastAPI Dependencies
==========================================

Purpose:
    Dependency injection functions for FastAPI route handlers.
    Dependencies are resolved per-request and provide access to
    shared resources (settings, services, current user, etc.).

Why dependency injection?
    - Testability: Replace real dependencies with mocks in tests.
    - Decoupling: Route handlers don't create their own dependencies.
    - Lifecycle: Dependencies can manage per-request resources.
    - Consistency: Common validation/auth logic lives in one place.

Design Decision — Lazy Imports:
    Heavy ML libraries (sentence-transformers, torch, cross-encoder)
    are imported inside function bodies, not at module level. This
    prevents import-time crashes when torch DLLs are missing and
    speeds up application startup for lightweight operations.

Usage:
    from app.core.dependencies import get_current_settings

    @router.get("/query")
    async def query(settings: Settings = Depends(get_current_settings)):
        ...
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.config.settings import Settings, get_settings

if TYPE_CHECKING:
    from fastapi import Request

    from app.models.llm import BaseLLM
    from app.pipelines.rag import RAGPipeline
    from app.pipelines.retrieval import AdvancedRetrievalPipeline
    from app.prompts.rag import RAGPromptTemplate


def get_current_settings() -> Settings:
    """
    Dependency that provides application settings.

    Uses the cached singleton from get_settings().
    """
    return get_settings()


def get_request_id(request: Request) -> str:
    """
    Dependency that extracts the request ID set by RequestIDMiddleware.

    Returns:
        The request correlation ID.
    """
    return getattr(request.state, "request_id", "unknown")


from app.vectorstores.in_memory import InMemoryVectorStore
from app.vectorstores.qdrant import QdrantVectorStore
from app.embeddings.sentence_transformers import SentenceTransformerEmbedding
from app.rerankers.cross_encoder import CrossEncoderReranker

@lru_cache
def get_vector_store() -> Any:
    """Dependency that provides the vector store singleton based on settings."""
    settings = get_settings()
    
    if settings.vectorstore.provider.lower() == "qdrant":
        return QdrantVectorStore(
            collection_name=settings.vectorstore.collection_name,
            dimension=settings.embedding.dimension,
            host=settings.vectorstore.qdrant_host,
            port=settings.vectorstore.qdrant_port,
        )
    
    # Default fallback
    return InMemoryVectorStore(
        collection_name=settings.vectorstore.collection_name,
        dimension=settings.embedding.dimension,
    )


@lru_cache
def get_embedder() -> Any:
    """Dependency that provides the embedding model singleton."""
    return SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")


@lru_cache
def get_reranker() -> Any:
    """Dependency that provides the cross-encoder model singleton."""
    return CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@lru_cache
def get_retriever() -> Any:
    """Dependency that provides the vector retriever."""
    from app.retrievers.vector import VectorRetriever

    return VectorRetriever(
        embedder=get_embedder(),
        vectorstore=get_vector_store(),
    )


@lru_cache
def get_retrieval_pipeline() -> AdvancedRetrievalPipeline:
    """Dependency that provides the complete advanced retrieval pipeline."""
    from app.pipelines.retrieval import AdvancedRetrievalPipeline

    return AdvancedRetrievalPipeline(
        retriever=get_retriever(),
        reranker=get_reranker(),
    )


@lru_cache
def get_llm() -> BaseLLM:
    """Dependency that provides the LLM singleton."""
    from app.models.openai_llm import OpenAICompatibleLLM

    settings = get_settings()
    
    # Determine the base URL
    base_url = None
    if settings.llm.openai_base_url:
        base_url = settings.llm.openai_base_url
    elif settings.llm.default_provider == "ollama":
        base_url = settings.llm.ollama_base_url + "/v1"

    return OpenAICompatibleLLM(
        api_key=settings.llm.openai_api_key or "no-key",
        base_url=base_url,
        model=settings.llm.default_model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )


@lru_cache
def get_prompt_template() -> RAGPromptTemplate:
    """Dependency that provides the RAG prompt template."""
    from app.prompts.rag import RAGPromptTemplate

    return RAGPromptTemplate()


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    """Dependency that provides the complete RAG pipeline."""
    from app.pipelines.rag import RAGPipeline

    return RAGPipeline(
        retrieval_pipeline=get_retrieval_pipeline(),
        llm=get_llm(),
        prompt_template=get_prompt_template(),
    )


@lru_cache
def get_faithfulness_evaluator() -> Any:
    """Dependency that provides the Faithfulness Evaluator."""
    from app.evaluators.llm_judge import FaithfulnessEvaluator

    return FaithfulnessEvaluator(llm=get_llm())


@lru_cache
def get_relevance_evaluator() -> Any:
    """Dependency that provides the Answer Relevance Evaluator."""
    from app.evaluators.llm_judge import RelevanceEvaluator

    return RelevanceEvaluator(llm=get_llm())


@lru_cache
def get_ingestion_pipeline() -> Any:
    """Dependency that provides the Document Ingestion Pipeline."""
    from app.pipelines.ingestion import DocumentIngestionPipeline
    from app.loaders.local import LocalDirectoryLoader
    from app.parsers.pdf import PyMuPDFParser
    from app.parsers.text import TextParser
    from app.parsers.vision import VisionImageParser
    from app.chunking.recursive import RecursiveCharacterChunker
    
    loader = LocalDirectoryLoader()
    parsers = {
        "pdf": PyMuPDFParser(),
        "txt": TextParser(),
        "md": TextParser(),
        "csv": TextParser(),
        "json": TextParser(),
        "png": VisionImageParser(),
        "jpg": VisionImageParser(),
        "jpeg": VisionImageParser(),
    }
    chunker = RecursiveCharacterChunker(
        chunk_size=get_settings().chunking.chunk_size,
        chunk_overlap=get_settings().chunking.chunk_overlap,
    )
    
    return DocumentIngestionPipeline(
        loader=loader,
        parsers=parsers,
        chunker=chunker,
        embedder=get_embedder(),
        vectorstore=get_vector_store(),
    )
