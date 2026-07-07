"""
Enterprise RAG OS — Base Evaluator Interface
===============================================

Purpose:
    Abstract base class for RAG evaluation metrics. Evaluators measure
    the quality of the system's outputs across dimensions like faithfulness,
    relevancy, and hallucination rate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Result from an evaluation run."""

    metric_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate a query-answer pair.

        Args:
            query: The original query.
            answer: The generated answer.
            contexts: Retrieved context chunks.
            ground_truth: Optional ground truth answer for comparison.

        Returns:
            EvaluationResult with score and details.
        """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Name of the metric this evaluator measures."""
