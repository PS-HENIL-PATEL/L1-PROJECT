"""
Enterprise RAG OS — Base Service
==================================

Purpose:
    Base class for service-layer components. Services contain business logic
    and orchestrate interactions between infrastructure components (vector
    stores, LLMs, etc.) and the API layer.
"""

from __future__ import annotations

from app.logging.logger import get_logger


class BaseService:
    """
    Base service class.

    All services share:
    - A logger scoped to the service class name
    - Initialization pattern for dependency injection
    """

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__qualname__)
