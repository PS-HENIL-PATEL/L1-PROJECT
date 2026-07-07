"""
Enterprise RAG OS — Base Agent Interface
==========================================

Purpose:
    Abstract base class for AI agents. Agents are autonomous components
    that can plan, execute, and reflect on tasks. Each agent has a single
    responsibility (retrieval, reranking, evaluation, etc.) and can be
    orchestrated by a Coordinator Agent.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class AgentStatus(enum.StrEnum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentResult:
    """Result from an agent execution."""

    output: Any
    status: AgentStatus = AgentStatus.COMPLETED
    reasoning: str = ""
    confidence: float = 1.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    async def execute(
        self,
        task: Any,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Execute the agent's task.

        Args:
            task: The task input (query, chunks, etc.).
            context: Shared execution context.

        Returns:
            AgentResult with output and metadata.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this agent."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of this agent's responsibility."""
