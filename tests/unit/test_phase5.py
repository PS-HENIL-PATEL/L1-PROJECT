"""
Tests — Phase 5: Evaluation & Observability
=============================================
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.api.v1.evaluate import evaluate_rag_response
from app.evaluators.llm_judge import FaithfulnessEvaluator, RelevanceEvaluator
from app.models.llm import BaseLLM, LLMResponse
from app.schemas.eval import EvaluateRequest


class MockJudgeLLM(BaseLLM):
    """A mock LLM that returns a predictable JSON response for the judge."""

    def __init__(self, score: float = 1.0, reasoning: str = "Looks good."):
        self._score = score
        self._reasoning = reasoning

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        content = json.dumps({
            "score": self._score,
            "reasoning": self._reasoning
        })
        return LLMResponse(
            text=content,
            model="mock-judge-model",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=10.0,
        )

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: Any):
        yield ""

    async def health_check(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "mock-judge-model"


class TestLLMJudgeEvaluators:
    """Test the native LLM-as-a-judge evaluators."""

    @pytest.mark.asyncio
    async def test_faithfulness_evaluator_success(self) -> None:
        llm = MockJudgeLLM(score=1.0, reasoning="All claims are supported.")
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate(
            query="What is X?",
            answer="X is Y.",
            context=["X is known as Y in the industry."],
        )

        assert result.metric_name == "faithfulness"
        assert result.score == 1.0
        assert "supported" in result.reasoning
        assert result.metadata is not None
        assert result.metadata["judge_model"] == "mock-judge-model"

    @pytest.mark.asyncio
    async def test_faithfulness_evaluator_no_context(self) -> None:
        llm = MockJudgeLLM(score=0.0, reasoning="No context.")
        evaluator = FaithfulnessEvaluator(llm=llm)

        result = await evaluator.evaluate(
            query="What is X?",
            answer="X is Y.",
            context=[],
        )

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_relevance_evaluator_success(self) -> None:
        llm = MockJudgeLLM(score=0.8, reasoning="Mostly relevant.")
        evaluator = RelevanceEvaluator(llm=llm)

        result = await evaluator.evaluate(
            query="What is X?",
            answer="X is Y.",
            context=["Some context"],
        )

        assert result.metric_name == "answer_relevance"
        assert result.score == 0.8
        assert "relevant" in result.reasoning


class TestEvaluateAPI:
    """Test the /api/v1/evaluate endpoint."""

    @pytest.mark.asyncio
    async def test_evaluate_endpoint(self) -> None:
        faithfulness = FaithfulnessEvaluator(llm=MockJudgeLLM(score=1.0))
        relevance = RelevanceEvaluator(llm=MockJudgeLLM(score=0.5))

        req = EvaluateRequest(
            query="What is the capital of France?",
            answer="Paris is the capital.",
            context=["Paris is the capital of France."],
        )

        response = await evaluate_rag_response(
            request=req,
            faithfulness=faithfulness,
            relevance=relevance,
        )

        assert len(response.metrics) == 2

        scores = {m.metric_name: m.score for m in response.metrics}
        assert scores["faithfulness"] == 1.0
        assert scores["answer_relevance"] == 0.5
        assert response.overall_score == 0.75
