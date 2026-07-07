"""
Enterprise RAG OS — Evaluation Schemas
========================================
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema


class EvaluateRequest(BaseSchema):
    """
    Request payload for the /api/v1/evaluate endpoint.
    Requires the original query, the generated answer, and the retrieved context.
    """

    query: str = Field(description="The original user query.")
    answer: str = Field(description="The generated answer to evaluate.")
    context: list[str] = Field(description="List of retrieved text chunks.")


class EvaluationMetric(BaseSchema):
    """A single evaluation metric result."""

    metric_name: str = Field(description="Name of the metric (e.g., faithfulness).")
    score: float = Field(description="Score between 0.0 and 1.0.")
    reasoning: str = Field(description="LLM judge reasoning for the score.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata.")


class EvaluateResponse(BaseSchema):
    """
    Response payload for the /api/v1/evaluate endpoint.
    Contains the aggregated results of all configured evaluators.
    """

    metrics: list[EvaluationMetric] = Field(description="List of evaluation results.")
    overall_score: float = Field(description="Average score across all metrics.")
