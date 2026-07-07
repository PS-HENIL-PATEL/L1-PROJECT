"""
Enterprise RAG OS — Base Domain Model
========================================

Purpose:
    Base class for internal domain models (distinct from API schemas).
    Domain models represent the business entities and rules, while schemas
    handle serialization and API contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.utils.ids import generate_id


@dataclass
class BaseDomainModel:
    """
    Base for all domain model entities.

    Domain models are the internal representation of business concepts.
    They may contain business logic (validation rules, state transitions)
    that doesn't belong in API schemas or infrastructure code.
    """

    id: str = field(default_factory=generate_id)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None
