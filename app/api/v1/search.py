"""
Enterprise RAG OS — Search API Endpoints
==========================================

Endpoints for testing the retrieval pipeline directly,
without invoking the LLM generation step.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_retrieval_pipeline
from app.logging.logger import get_logger
from app.schemas.queries import QueryRequest, SearchResponse, SourceCitation

if TYPE_CHECKING:
    from app.pipelines.retrieval import AdvancedRetrievalPipeline

router = APIRouter(prefix="/search", tags=["Retrieval"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=SearchResponse,
    summary="Retrieve and rerank chunks",
    description=(
        "Executes the advanced retrieval pipeline to fetch the most relevant "
        "chunks from the vector store and rerank them using a cross-encoder."
    ),
)
async def search_documents(
    query: QueryRequest,
    pipeline: Annotated[AdvancedRetrievalPipeline, Depends(get_retrieval_pipeline)],
) -> SearchResponse:
    """Execute the retrieval pipeline for a given query."""
    start_time = time.perf_counter()
    logger.info("Received search request", query=query.query, filters=query.filters)

    try:
        # We pass initial_k=50 for the VectorStore to get a broad candidate set,
        # then final_k=top_k for the CrossEncoder to narrow it down.
        # This gives high recall + high precision.
        pipeline_res = await pipeline.run(
            input=query.query,
            initial_k=50,
            final_k=query.top_k,
            filters=query.filters,
        )

        results = []
        for r in pipeline_res["results"]:
            # Map pipeline output dicts to SourceCitation models
            results.append(
                SourceCitation(
                    document_id=r.get("document_id", "unknown"),
                    chunk_id=r.get("chunk_id", "unknown"),
                    filename=r.get("metadata", {}).get("file_name", "unknown"),
                    content=r.get("content", ""),
                    page_number=r.get("metadata", {}).get("page_number"),
                    section=r.get("metadata", {}).get("section"),
                    similarity_score=0.0,  # we overwrite with rerank score below
                    rerank_score=r.get("score", 0.0),
                )
            )

        total_ms = (time.perf_counter() - start_time) * 1000
        timings = pipeline_res["timings"]
        timings["total_ms"] = total_ms

        logger.info(
            "Search completed",
            query=query.query,
            results_returned=len(results),
            latency_ms=round(total_ms, 2)
        )

        return SearchResponse(
            query=query.query,
            results=results,
            total_candidates=pipeline_res["total_candidates"],
            timings=timings,
        )

    except Exception as e:
        logger.error("Search request failed", error=str(e), query=query.query)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute search pipeline: {e!s}",
        ) from e
