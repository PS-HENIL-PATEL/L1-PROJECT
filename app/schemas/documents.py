"""
Enterprise RAG OS — Document Schemas
======================================

Purpose:
    Pydantic models for document-related API operations: upload, metadata,
    chunk representation, and document status tracking.

Architecture:
    These schemas define the data contract between the API layer and the
    document ingestion pipeline. They are NOT the internal domain models
    (those live in app/models/) — schemas are focused on serialization
    and validation for API consumers.
"""

from __future__ import annotations

import enum

from pydantic import Field

from app.schemas.base import BaseSchema, TimestampedSchema


class DocumentStatus(enum.StrEnum):
    """Document processing status lifecycle."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentFormat(enum.StrEnum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    CSV = "csv"
    HTML = "html"


class DocumentMetadata(BaseSchema):
    """
    Metadata extracted from or assigned to a document.

    This metadata powers the metadata filtering system (Phase 3),
    allowing queries like "search only in HR documents from 2024".
    """

    filename: str = Field(description="Original filename")
    format: DocumentFormat = Field(description="Document format")
    size_bytes: int = Field(ge=0, description="File size in bytes")
    page_count: int | None = Field(default=None, ge=0, description="Number of pages")
    author: str | None = Field(default=None, description="Document author")
    title: str | None = Field(default=None, description="Document title")
    language: str | None = Field(default=None, description="ISO 639-1 language code")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    department: str | None = Field(default=None, description="Organizational department")
    source: str | None = Field(default=None, description="Document source/origin")
    custom: dict[str, str] = Field(
        default_factory=dict,
        description="Custom key-value metadata",
    )


class DocumentResponse(TimestampedSchema):
    """API response for a single document."""

    filename: str
    format: DocumentFormat
    status: DocumentStatus = DocumentStatus.PENDING
    size_bytes: int = 0
    content_hash: str | None = None
    chunk_count: int = 0
    metadata: DocumentMetadata | None = None
    error_message: str | None = None


class DocumentListResponse(BaseSchema):
    """API response for listing documents."""

    documents: list[DocumentResponse] = Field(default_factory=list)
    total: int = 0


class ChunkSchema(TimestampedSchema):
    """
    Representation of a text chunk.

    A chunk is the fundamental unit of retrieval. Documents are split into
    chunks, each is embedded and stored in the vector database.
    """

    document_id: str = Field(description="Parent document ID")
    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(ge=0, description="Position within the document")
    start_char: int | None = Field(default=None, ge=0, description="Start character offset")
    end_char: int | None = Field(default=None, ge=0, description="End character offset")
    page_number: int | None = Field(default=None, ge=1, description="Source page number")
    section: str | None = Field(default=None, description="Document section/heading")
    token_count: int | None = Field(default=None, ge=0, description="Token count")
    embedding: list[float] | None = Field(
        default=None,
        description="Embedding vector (excluded from API responses by default)",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Chunk-level metadata",
    )
