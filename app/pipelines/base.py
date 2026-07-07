"""
Enterprise RAG OS — Base Pipeline Interface
==============================================

Purpose:
    Abstract base class for processing pipelines. A pipeline is an ordered
    sequence of stages that transforms input into output. The RAG pipeline
    orchestrates: Query → Retrieve → Rerank → Build Context → Generate → Respond.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")


@dataclass
class PipelineContext:
    """Shared context passed through pipeline stages."""

    request_id: str = ""
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class BasePipeline[T_Input, T_Output](ABC):
    """Abstract base class for all pipelines."""

    @abstractmethod
    async def run(
        self,
        input: T_Input,
        context: PipelineContext | None = None,
        **kwargs: Any,
    ) -> T_Output:
        """Execute the pipeline."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if all pipeline components are healthy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this pipeline."""

    @property
    @abstractmethod
    def stages(self) -> list[str]:
        """Ordered list of stage names in this pipeline."""
