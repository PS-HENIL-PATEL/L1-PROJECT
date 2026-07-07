"""
Enterprise RAG OS — Query API Endpoint
========================================

The core API endpoint for the RAG system. Accepts a user query,
runs the full RAG pipeline (retrieve → rerank → generate), and
returns a structured response with answer, sources, and explainability.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_rag_pipeline
from app.logging.logger import get_logger
from app.schemas.queries import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["RAG Query"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question",
    description=(
        "Submit a question to the RAG pipeline. The system retrieves "
        "relevant documents, reranks them, and generates an answer "
        "using an LLM grounded on the retrieved context."
    ),
)
async def query_rag(
    query: QueryRequest,
    pipeline: Annotated[Any, Depends(get_rag_pipeline)],
) -> QueryResponse:
    """Execute the full RAG pipeline for a user query."""
    logger.info(
        "Received RAG query",
        query=query.query,
        top_k=query.top_k,
        strategy=query.retrieval_strategy,
    )

    try:
        response = await pipeline.run(input=query)

        logger.info(
            "RAG query completed",
            query=query.query,
            confidence=response.confidence,
            sources=len(response.sources),
        )

        return response

    except Exception as e:
        logger.error("RAG query failed", error=str(e), query=query.query)
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline execution failed: {e}",
        ) from e
