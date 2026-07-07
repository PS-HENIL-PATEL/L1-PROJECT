"""
Enterprise RAG OS — Base Schema
=================================

Purpose:
    Base Pydantic model that all API schemas inherit from. Provides
    common fields (id, created_at, updated_at) and shared configuration.

Why a base schema?
    - Consistency: Every entity has a unique ID and timestamps.
    - DRY: Timestamp logic defined once, not in every model.
    - Configuration: All models share the same serialization settings.
    - Extensibility: Add audit fields (created_by, version) in one place.

Usage:
    from app.schemas.base import BaseSchema, TimestampedSchema

    class Document(TimestampedSchema):
        title: str
        content: str
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.utils.ids import generate_id


class BaseSchema(BaseModel):
    """
    Root schema for all API models.

    Configuration:
        - from_attributes=True: Allows creating from ORM/dataclass objects.
        - populate_by_name=True: Allows using both field name and alias.
        - str_strip_whitespace=True: Strips leading/trailing whitespace from strings.
        - json_schema_extra: Adds metadata to OpenAPI docs.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={"example": {}},
    )


class TimestampedSchema(BaseSchema):
    """
    Schema with automatic ID and timestamps.

    Every persisted entity should use this as its base. The ID is generated
    automatically on creation, and timestamps track when the entity was
    created and last modified.
    """

    id: str = Field(
        default_factory=generate_id,
        description="Unique identifier (UUID v4)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp (UTC)",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Last update timestamp (UTC)",
    )


class ErrorResponse(BaseSchema):
    """
    Standard error response schema.

    All error responses from the API follow this structure, ensuring
    consistent error handling on the client side.
    """

    error: ErrorDetail


class ErrorDetail(BaseSchema):
    """Error detail within an error response."""

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    status: int = Field(description="HTTP status code")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error context",
    )


class PaginatedResponse(BaseSchema):
    """
    Paginated response wrapper.

    Used for list endpoints that may return large result sets.
    Includes total count and pagination metadata.
    """

    items: list[Any] = Field(default_factory=list, description="Result items")
    total: int = Field(default=0, description="Total number of items")
    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    has_next: bool = Field(default=False, description="Whether more pages exist")
    has_previous: bool = Field(default=False, description="Whether previous pages exist")
