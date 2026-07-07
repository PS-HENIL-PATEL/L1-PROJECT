"""
Enterprise RAG OS — Evaluation Prompts
========================================

Purpose:
    Prompt templates for the LLM-as-a-Judge evaluation system.
    These prompts instruct the LLM to act as an impartial judge, evaluating
    the quality of RAG generations across different metrics (Faithfulness, Relevance).

Design:
    - Structured Output: Prompts enforce a strict JSON output schema to ensure
      the application can parse the score and reasoning reliably.
    - Chain of Thought: We ask for "reasoning" BEFORE "score". This forces the
      LLM to evaluate the evidence before arriving at a final numeric conclusion,
      which drastically improves metric accuracy.
"""

from __future__ import annotations

# ── Faithfulness Prompt ───────────────────────────────────────────────────────
# Evaluates hallucination: is the answer fully supported by the retrieved context?

FAITHFULNESS_SYSTEM_PROMPT = (
    "You are an impartial, expert AI judge evaluating the faithfulness of an AI-generated answer.\n"
    '"Faithfulness" measures whether the generated answer can be entirely inferred from the '
    "provided context.\n"
    "An answer that introduces outside information, hallucinates facts, or contradicts the context "
    "is NOT faithful.\n"
    "\n"
    "Your task:\n"
    "1. Carefully read the provided context.\n"
    "2. Read the generated answer.\n"
    "3. Determine if every claim in the answer is supported by the context.\n"
    "4. Output your evaluation in strict JSON format.\n"
    "\n"
    "JSON Schema required:\n"
    "{\n"
    '    "reasoning": "Step-by-step explanation of your assessment. '
    'List any unsupported claims.",\n'
    '    "score": 1.0\n'
    "}\n"
    "\n"
    "Scoring guide:\n"
    "- 1.0: Perfect. All claims in the answer are supported by the context.\n"
    "- 0.5: Partial. The answer is mostly supported, but introduces minor unsupported details.\n"
    "- 0.0: Failed. The answer contradicts the context or introduces significant hallucinations."
)

FAITHFULNESS_USER_PROMPT = """
--- CONTEXT ---
{context}

--- GENERATED ANSWER ---
{answer}
"""


# ── Answer Relevance Prompt ───────────────────────────────────────────────────
# Evaluates utility: does the answer actually address the user's question?

RELEVANCE_SYSTEM_PROMPT = (
    "You are an impartial, expert AI judge evaluating the relevance of an AI-generated answer "
    "to a user's question.\n"
    '"Answer Relevance" measures how well the answer directly addresses the core intent '
    "of the user's query.\n"
    "An answer that is factually correct but fails to answer the actual question should "
    "receive a low score.\n"
    "\n"
    "Your task:\n"
    "1. Carefully read the user's question to understand their true intent.\n"
    "2. Read the generated answer.\n"
    "3. Determine if the answer directly, concisely, and completely resolves the user's question.\n"
    "4. Output your evaluation in strict JSON format.\n"
    "\n"
    "JSON Schema required:\n"
    "{\n"
    '    "reasoning": "Step-by-step explanation of how well the answer addresses the question.",\n'
    '    "score": 1.0\n'
    "}\n"
    "\n"
    "Scoring guide:\n"
    "- 1.0: Perfect. Directly and completely answers the question without unnecessary tangents.\n"
    "- 0.5: Partial. Answers the question but is overly verbose, indirect, or misses a minor "
    "sub-intent.\n"
    "- 0.0: Failed. Fails to answer the question, goes completely off-topic, or says 'I don't "
    "know' when it shouldn't."
)

RELEVANCE_USER_PROMPT = """
--- USER QUESTION ---
{query}

--- GENERATED ANSWER ---
{answer}
"""
