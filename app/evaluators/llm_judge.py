"""
Enterprise RAG OS — LLM as a Judge Evaluators
===============================================

Purpose:
    Concrete evaluators that use an LLM to score RAG outputs.
    This avoids heavy dependencies (like Ragas or LangChain) while giving
    us complete control over the evaluation prompts and JSON parsing.

Metrics implemented:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevance: Does the answer address the question?
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.evaluators.base import BaseEvaluator, EvaluationResult
from app.evaluators.prompts import (
    FAITHFULNESS_SYSTEM_PROMPT,
    FAITHFULNESS_USER_PROMPT,
    RELEVANCE_SYSTEM_PROMPT,
    RELEVANCE_USER_PROMPT,
)
from app.logging.logger import get_logger
from app.utils.timing import Timer

if TYPE_CHECKING:
    from app.models.llm import BaseLLM

logger = get_logger(__name__)


class LLMJudgeEvaluator(BaseEvaluator):
    """Base class for evaluators powered by an LLM."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def _evaluate_with_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        metric_name: str,
    ) -> EvaluationResult:
        """
        Helper method to run the LLM evaluation and parse the JSON response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            with Timer(f"eval_{metric_name}_ms") as t:
                # We enforce JSON mode via the prompt, but we can also
                # use provider-specific kwargs (e.g. response_format) if needed.
                is_openai = "openai" in type(self._llm).__name__.lower()
                response_format = {"type": "json_object"} if is_openai else None
                response = await self._llm.generate(
                    messages=messages,
                    temperature=0.0,
                    response_format=response_format,
                )

            # Parse JSON
            # The LLM might wrap the JSON in markdown blocks (```json ... ```)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.removeprefix("```json").removesuffix("```").strip()
            elif text.startswith("```"):
                text = text.removeprefix("```").removesuffix("```").strip()

            data = json.loads(text)

            # Extract fields
            score = float(data.get("score", 0.0))
            reasoning = str(data.get("reasoning", "No reasoning provided."))

            return EvaluationResult(
                metric_name=metric_name,
                score=score,
                reasoning=reasoning,
                metadata={
                    "judge_model": response.model,
                    "latency_ms": t.elapsed_ms,
                    "prompt_tokens": response.prompt_tokens,
                },
            )

        except json.JSONDecodeError as e:
            logger.error(
                "LLM Judge returned invalid JSON",
                metric=metric_name,
                raw_response=response.text if 'response' in locals() else "unknown",
            )
            return EvaluationResult(
                metric_name=metric_name,
                score=0.0,
                reasoning=f"Failed to parse LLM evaluation: {e}",
            )
        except Exception as e:
            logger.error("LLM Judge evaluation failed", metric=metric_name, error=str(e))
            return EvaluationResult(
                metric_name=metric_name,
                score=0.0,
                reasoning=f"Evaluation execution failed: {e}",
            )


class FaithfulnessEvaluator(LLMJudgeEvaluator):
    """Evaluates if the answer is grounded in the retrieved context."""

    @property
    def name(self) -> str:
        return "faithfulness"

    async def evaluate(
        self,
        query: str,  # noqa: ARG002
        answer: str,
        context: list[str],
        **_kwargs: Any,
    ) -> EvaluationResult:
        if not context:
            # If there's no context, the answer can't be faithful to it.
            # But if the answer explicitly says "I don't know", that's faithful behavior.
            # We let the LLM judge decide based on the empty context string.
            context_text = "NONE (No context was provided to the assistant)."
        else:
            context_text = "\n\n".join(
                f"[Document {i+1}]\n{text}" for i, text in enumerate(context)
            )

        user_prompt = FAITHFULNESS_USER_PROMPT.format(
            context=context_text,
            answer=answer,
        )

        return await self._evaluate_with_llm(
            system_prompt=FAITHFULNESS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metric_name=self.name,
        )


class RelevanceEvaluator(LLMJudgeEvaluator):
    """Evaluates if the answer directly addresses the user's query."""

    @property
    def name(self) -> str:
        return "answer_relevance"

    async def evaluate(
        self,
        query: str,
        answer: str,
        context: list[str],  # noqa: ARG002
        **_kwargs: Any,
    ) -> EvaluationResult:
        user_prompt = RELEVANCE_USER_PROMPT.format(
            query=query,
            answer=answer,
        )

        return await self._evaluate_with_llm(
            system_prompt=RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            metric_name=self.name,
        )
