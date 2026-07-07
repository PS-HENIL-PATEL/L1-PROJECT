"""
Enterprise RAG OS — RAG Pipeline (End-to-End)
===============================================

Purpose:
    The complete Retrieval-Augmented Generation pipeline. This is the
    heart of the system — it orchestrates every stage from query to answer:

    Query → Retrieve → Rerank → Build Prompt → Generate → Format Response

Design Decisions:
    - Delegates retrieval to AdvancedRetrievalPipeline (Phase 3).
    - Builds the prompt using RAGPromptTemplate with retrieved context.
    - Calls the LLM via BaseLLM.generate() for the final answer.
    - Populates ExplainabilityData so every answer is fully traceable.
    - Handles "no results" gracefully with a transparent message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.exceptions import PipelineError
from app.logging.logger import get_logger
from app.pipelines.base import BasePipeline, PipelineContext
from app.prompts.base import PromptInput
from app.prompts.rag import RAGPromptTemplate
from app.schemas.queries import (
    ExplainabilityData,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)
from app.utils.timing import Timer

if TYPE_CHECKING:
    from app.models.llm import BaseLLM
    from app.pipelines.retrieval import AdvancedRetrievalPipeline

logger = get_logger(__name__)


class RAGPipeline(BasePipeline[QueryRequest, QueryResponse]):
    """
    Complete RAG pipeline: Retrieve → Rerank → Prompt → Generate → Respond.
    """

    def __init__(
        self,
        retrieval_pipeline: AdvancedRetrievalPipeline,
        llm: BaseLLM,
        prompt_template: RAGPromptTemplate | None = None,
    ) -> None:
        self._retrieval = retrieval_pipeline
        self._llm = llm
        self._prompt = prompt_template or RAGPromptTemplate()

    async def run(
        self,
        input: QueryRequest,
        context: PipelineContext | None = None,
        **_kwargs: Any,
    ) -> QueryResponse:
        """
        Execute the full RAG pipeline.

        Args:
            input: The user's QueryRequest.
            context: Optional pipeline context.

        Returns:
            QueryResponse with answer, sources, and explainability.
        """
        ctx = context or PipelineContext()

        try:
            # ── Stage 1: Retrieve & Rerank ────────────────────────────
            with Timer("retrieval_total_ms") as t_ret:
                retrieval_result = await self._retrieval.run(
                    input=input.query,
                    initial_k=50,
                    final_k=input.top_k,
                    filters=input.filters or None,
                )
            ctx.timings["retrieval_total_ms"] = t_ret.elapsed_ms

            # Merge sub-timings from retrieval pipeline
            for key, val in retrieval_result.get("timings", {}).items():
                ctx.timings[key] = val

            results = retrieval_result.get("results", [])
            total_candidates = retrieval_result.get("total_candidates", 0)

            # ── Stage 2: Build Prompt ─────────────────────────────────
            with Timer("prompt_build_ms") as t_prompt:
                context_texts = [r["content"] for r in results]
                chunk_metadata = [r.get("metadata", {}) for r in results]

                prompt_input = PromptInput(
                    query=input.query,
                    context=context_texts,
                    metadata={"chunks": chunk_metadata},
                )

                messages = self._prompt.render_messages(prompt_input)
            ctx.timings["prompt_build_ms"] = t_prompt.elapsed_ms

            # ── Stage 3: LLM Generation ───────────────────────────────
            with Timer("generation_ms") as t_gen:
                llm_response = await self._llm.generate(
                    messages=messages,
                    temperature=input.temperature,
                    max_tokens=None,  # use model default
                )
            ctx.timings["generation_ms"] = t_gen.elapsed_ms

            # ── Stage 4: Build Response ───────────────────────────────
            sources: list[SourceCitation] = []
            if input.include_sources:
                for r in results:
                    meta = r.get("metadata", {})
                    # Use the original cosine similarity score (0 to 1) for confidence metrics
                    original_score = r.get("original_score", 0.0)
                    sources.append(
                        SourceCitation(
                            document_id=r.get("document_id", "unknown"),
                            chunk_id=r.get("chunk_id", "unknown"),
                            filename=meta.get("file_name", "unknown"),
                            content=r.get("content", "")[:500],
                            page_number=meta.get("page_number"),
                            section=meta.get("section"),
                            similarity_score=max(0.0, min(1.0, original_score)),
                            rerank_score=r.get("score"),
                        )
                    )

            # Compute confidence as average of original cosine similarities
            confidence = 0.0
            if results:
                scores = [r.get("original_score", 0.0) for r in results]
                confidence = sum(scores) / len(scores)
                confidence = max(0.0, min(1.0, confidence))

            # Build explainability data
            explainability = None
            if input.include_explainability:
                total_ms = sum(ctx.timings.values())
                explainability = ExplainabilityData(
                    retrieved_chunks=total_candidates,
                    reranked_chunks=len(results),
                    discarded_chunks=max(
                        total_candidates - len(results), 0
                    ),
                    retrieval_strategy=input.retrieval_strategy.value,
                    embedding_model="all-MiniLM-L6-v2",
                    llm_model=llm_response.model,
                    prompt_tokens=llm_response.prompt_tokens,
                    completion_tokens=llm_response.completion_tokens,
                    total_tokens=llm_response.total_tokens,
                    latency_ms=LatencyBreakdown(
                        total_ms=round(total_ms, 2),
                        retrieval_ms=ctx.timings.get(
                            "retrieval_ms", 0.0
                        ),
                        reranking_ms=ctx.timings.get(
                            "rerank_ms", 0.0
                        ),
                        generation_ms=ctx.timings.get(
                            "generation_ms", 0.0
                        ),
                    ),
                    reasoning_summary=(
                        f"Retrieved {total_candidates} candidates, "
                        f"reranked to {len(results)}, "
                        f"generated answer using {llm_response.model}."
                    ),
                )

            # Detect limitations
            limitations: list[str] = []
            if not results:
                limitations.append(
                    "No relevant documents found in the knowledge base."
                )
            if llm_response.finish_reason == "length":
                limitations.append(
                    "The answer was truncated due to token limits."
                )

            logger.info(
                "RAG pipeline complete",
                query=input.query,
                sources=len(sources),
                tokens=llm_response.total_tokens,
                latency_ms=round(sum(ctx.timings.values()), 2),
            )

            return QueryResponse(
                query=input.query,
                answer=llm_response.text,
                confidence=confidence,
                sources=sources,
                explainability=explainability,
                limitations=limitations,
                session_id=input.session_id,
            )

        except PipelineError:
            raise
        except Exception as e:
            logger.error("RAG pipeline failed", error=str(e))
            raise PipelineError(
                detail=f"RAG pipeline failed: {e}"
            ) from e

    async def health_check(self) -> bool:
        """Check health of all sub-components."""
        retrieval_ok = await self._retrieval.health_check()
        llm_ok = await self._llm.health_check()
        return retrieval_ok and llm_ok

    @property
    def name(self) -> str:
        return "rag_pipeline"

    @property
    def stages(self) -> list[str]:
        return [
            "retrieve",
            "rerank",
            "build_prompt",
            "generate",
            "format_response",
        ]
