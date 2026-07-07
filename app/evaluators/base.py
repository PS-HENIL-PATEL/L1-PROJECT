"""
Enterprise RAG OS — Evaluators Base
=====================================

Purpose:
    Defines the standard interface and data structures for RAG evaluation metrics.
    Every evaluator (Faithfulness, Answer Relevance, Context Precision, etc.)
    must conform to this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """
    Standard output format for all evaluators.
    """

    metric_name: str
    score: float  # Expected to be strictly bounded [0.0, 1.0]
    reasoning: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Ensure score is bounded."""
        self.score = max(0.0, min(self.score, 1.0))


class BaseEvaluator(ABC):
    """
    Abstract base class for all RAG evaluation metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the evaluation metric."""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        answer: str,
        context: list[str],
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG generation.

        Args:
            query: The original user question.
            answer: The generated answer from the LLM.
            context: The retrieved text chunks used for generation.
            **kwargs: Extra parameters (e.g. ground truth).

        Returns:
            An EvaluationResult containing the score [0.0, 1.0] and reasoning.
        """
