"""
Enterprise RAG OS — Query and Response Schemas
================================================

Purpose:
    Pydantic models for the RAG query/response cycle. Defines what the
    client sends (query + options) and what they receive (answer + sources
    + explainability data + confidence).

Architecture:
    These schemas embody the core value proposition of the system:
    answers are NEVER just text. They always include source citations,
    confidence scores, and explainability metadata — making the system
    trustworthy and auditable.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema, TimestampedSchema


class RetrievalStrategy(enum.StrEnum):
    """Available retrieval strategies."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class QueryRequest(BaseSchema):
    """
    Incoming query from the user.

    The query includes the question text plus optional configuration
    that overrides the system defaults for this specific query.
    """

    query: str = Field(
        min_length=1,
        max_length=10000,
        description="The user's question or search query",
    )
    collection: str | None = Field(
        default=None,
        description="Target collection/knowledge base to search",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return",
    )
    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Retrieval strategy to use",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters (e.g., {'author': 'John', 'year': 2024})",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response",
    )
    include_explainability: bool = Field(
        default=True,
        description="Include explainability data",
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation memory",
    )
    model: str | None = Field(
        default=None,
        description="Override the default LLM model",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override the default temperature",
    )


class SourceCitation(BaseSchema):
    """
    A source document that contributed to the answer.

    Citations provide traceability: users can verify the answer
    against the original document and specific page/section.
    """

    document_id: str = Field(description="Source document ID")
    chunk_id: str = Field(description="Specific chunk ID")
    filename: str = Field(description="Source filename")
    content: str = Field(description="Relevant text excerpt")
    page_number: int | None = Field(default=None, description="Page number")
    section: str | None = Field(default=None, description="Document section")
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity to query",
    )
    rerank_score: float | None = Field(
        default=None,
        description="Score after reranking",
    )


class ExplainabilityData(BaseSchema):
    """
    Explainability information for the answer.

    This is what makes the system trustworthy: users (and auditors)
    can see exactly HOW the answer was produced, what was retrieved,
    what was discarded, and why.
    """

    retrieved_chunks: int = Field(description="Number of chunks retrieved")
    reranked_chunks: int = Field(description="Number of chunks after reranking")
    discarded_chunks: int = Field(description="Number of chunks filtered out")
    retrieval_strategy: str = Field(description="Strategy used for retrieval")
    embedding_model: str = Field(description="Embedding model used")
    llm_model: str = Field(description="LLM used for generation")
    prompt_tokens: int = Field(default=0, description="Tokens in prompt")
    completion_tokens: int = Field(default=0, description="Tokens in completion")
    total_tokens: int = Field(default=0, description="Total tokens used")
    latency_ms: LatencyBreakdown | None = Field(
        default=None,
        description="Latency breakdown by pipeline stage",
    )
    reasoning_summary: str | None = Field(
        default=None,
        description="Brief summary of the reasoning process",
    )


class LatencyBreakdown(BaseSchema):
    """Timing breakdown for each pipeline stage."""

    total_ms: float = Field(description="Total end-to-end latency")
    query_understanding_ms: float = Field(default=0.0)
    embedding_ms: float = Field(default=0.0)
    retrieval_ms: float = Field(default=0.0)
    reranking_ms: float = Field(default=0.0)
    context_building_ms: float = Field(default=0.0)
    generation_ms: float = Field(default=0.0)


class QueryResponse(TimestampedSchema):
    """
    Complete response to a user query.

    This is the core output of the RAG system. It includes:
    - The answer text
    - Confidence score (how sure the system is)
    - Source citations (where the information came from)
    - Explainability data (how the answer was produced)
    - Limitations and unknowns (what the system couldn't answer)
    """

    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 = no confidence, 1.0 = fully confident)",
    )
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Source documents used to generate the answer",
    )
    explainability: ExplainabilityData | None = Field(
        default=None,
        description="Explainability metadata",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of this answer",
    )
    unknowns: list[str] = Field(
        default_factory=list,
        description="Information the system could not find",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions",
    )


class SearchResponse(BaseSchema):
    """
    Response for a retrieval-only search request.
    Useful for testing retrieval independently of LLM generation.
    """

    query: str = Field(description="Original query")
    results: list[SourceCitation] = Field(
        default_factory=list,
        description="Ranked search results",
    )
    total_candidates: int = Field(description="Number of chunks retrieved before reranking")
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="Latency metrics for retrieval and reranking",
    )
