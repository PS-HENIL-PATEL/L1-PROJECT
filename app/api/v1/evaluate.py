"""
Enterprise RAG OS — Evaluate API Endpoint
===========================================

Purpose:
    Exposes the LLM-as-a-judge evaluation system via a REST endpoint.
    This allows internal observability tools or async workers to grade
    RAG responses in the background without slowing down the user-facing
    query endpoint.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import (
    get_faithfulness_evaluator,
    get_relevance_evaluator,
)
from app.logging.logger import get_logger
from app.observability.metrics import RAGMetricsLogger
from app.schemas.eval import EvaluateRequest, EvaluateResponse, EvaluationMetric

if TYPE_CHECKING:
    from app.evaluators.base import BaseEvaluator

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=EvaluateResponse,
    summary="Evaluate a RAG response",
    description="Scores a generated answer for Faithfulness and Relevance using an LLM Judge.",
)
async def evaluate_rag_response(
    request: EvaluateRequest,
    faithfulness: Annotated[BaseEvaluator, Depends(get_faithfulness_evaluator)],
    relevance: Annotated[BaseEvaluator, Depends(get_relevance_evaluator)],
) -> EvaluateResponse:
    """Evaluate a RAG generation."""
    logger.info("Running RAG evaluation", query=request.query)

    try:
        # Run evaluators concurrently
        results = await asyncio.gather(
            faithfulness.evaluate(
                query=request.query,
                answer=request.answer,
                context=request.context,
            ),
            relevance.evaluate(
                query=request.query,
                answer=request.answer,
                context=request.context,
            ),
        )

        metrics: list[EvaluationMetric] = []
        total_score = 0.0

        for res in results:
            # Log telemetry for each metric
            RAGMetricsLogger.log_evaluation(
                query=request.query,
                metric_name=res.metric_name,
                score=res.score,
                reasoning=res.reasoning,
                metadata=res.metadata,
            )

            metrics.append(
                EvaluationMetric(
                    metric_name=res.metric_name,
                    score=res.score,
                    reasoning=res.reasoning,
                    metadata=res.metadata or {},
                )
            )
            total_score += res.score

        overall = total_score / len(results) if results else 0.0

        return EvaluateResponse(
            metrics=metrics,
            overall_score=round(overall, 2),
        )

    except Exception as e:
        logger.error("RAG evaluation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {e}",
        ) from e
