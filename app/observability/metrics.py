"""
Enterprise RAG OS — Observability & Telemetry
================================================

Purpose:
    Provides a standardized way to log RAG telemetry (latency, token usage,
    retrieval metrics, evaluation scores).

    By emitting this data as structured JSON logs, operators can easily build
    dashboards in Datadog, ELK, or Grafana without needing a heavy APM agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.logging.logger import get_logger

if TYPE_CHECKING:
    from app.schemas.queries import QueryResponse

logger = get_logger(__name__)


class RAGMetricsLogger:
    """
    Structured logger for RAG telemetry.
    """

    @staticmethod
    def log_query_execution(
        response: QueryResponse,
        **extra_tags: Any,
    ) -> None:
        """
        Log the execution of a RAG query including latency, tokens, and retrieval stats.
        """
        explain = response.explainability

        if explain:
            # We structure the payload specifically so log aggregators can parse
            # the metrics easily (e.g. `metrics.latency.total_ms`).
            payload = {
                "telemetry_type": "rag_query_execution",
                "metrics": {
                    "latency": {
                        "total_ms": explain.latency_ms.total_ms,
                        "retrieval_ms": explain.latency_ms.retrieval_ms,
                        "reranking_ms": explain.latency_ms.reranking_ms,
                        "generation_ms": explain.latency_ms.generation_ms,
                    },
                    "tokens": {
                        "prompt": explain.prompt_tokens,
                        "completion": explain.completion_tokens,
                        "total": explain.total_tokens,
                    },
                    "retrieval": {
                        "retrieved_chunks": explain.retrieved_chunks,
                        "reranked_chunks": explain.reranked_chunks,
                        "discarded_chunks": explain.discarded_chunks,
                        "final_sources": len(response.sources),
                    },
                    "confidence_score": response.confidence,
                },
                "tags": {
                    "strategy": explain.retrieval_strategy,
                    "embedding_model": explain.embedding_model,
                    "llm_model": explain.llm_model,
                    "session_id": response.session_id,
                    **extra_tags,
                },
            }
            logger.info("RAG Telemetry", **payload)
        else:
            logger.warning(
                "Cannot log RAG metrics: Explainability data is missing",
                session_id=response.session_id,
            )

    @staticmethod
    def log_evaluation(
        query: str,
        metric_name: str,
        score: float,
        reasoning: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log the result of an LLM-as-a-judge evaluation.
        """
        payload = {
            "telemetry_type": "rag_evaluation",
            "metrics": {
                "score": score,
            },
            "tags": {
                "metric_name": metric_name,
            },
            "evaluation": {
                "query": query,
                "reasoning": reasoning,
            },
            "metadata": metadata or {},
        }
        logger.info("RAG Evaluation Result", **payload)
