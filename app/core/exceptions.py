"""
Enterprise RAG OS — Exception Hierarchy
========================================

Purpose:
    Typed exception tree for the entire application. Every error that can occur
    in the system has a specific exception class, enabling:
    - Precise error handling (catch exactly what you can handle)
    - Consistent HTTP status code mapping (global exception handler)
    - Granular alerting (page on LLMConnectionError, log ValidationError)
    - Clear error messages for API consumers

Architecture:
    All exceptions inherit from EnterpriseRAGError. The global exception handler
    in core/middleware.py maps each exception type to an HTTP status code and
    a structured JSON error response.

    Exception Tree:
    ┌─ EnterpriseRAGError (base, 500)
    │  ├─ ConfigurationError (500)
    │  ├─ DocumentError (400)
    │  │  ├─ DocumentNotFoundError (404)
    │  │  ├─ DocumentParsingError (422)
    │  │  └─ UnsupportedFormatError (415)
    │  ├─ EmbeddingError (500)
    │  ├─ VectorStoreError (500)
    │  ├─ RetrievalError (500)
    │  ├─ LLMError (502)
    │  │  ├─ LLMConnectionError (503)
    │  │  ├─ LLMRateLimitError (429)
    │  │  └─ LLMResponseError (502)
    │  ├─ AuthenticationError (401)
    │  ├─ AuthorizationError (403)
    │  ├─ RateLimitError (429)
    │  ├─ ValidationError (422)
    │  └─ PipelineError (500)

Design Decisions:
    - Each exception carries a `status_code`, `error_code` (machine-readable string),
      and `detail` (human-readable message). This triple gives API consumers
      everything they need to handle errors programmatically and display them to users.
    - The `context` dict allows attaching arbitrary metadata (document ID,
      model name, etc.) without subclassing for every scenario.

Usage:
    from app.core.exceptions import DocumentNotFoundError
    raise DocumentNotFoundError(
        detail="Document 'report.pdf' not found in collection.",
        context={"document_id": "abc-123", "collection": "default"},
    )
"""

from __future__ import annotations

from typing import Any


class EnterpriseRAGError(Exception):
    """
    Base exception for all Enterprise RAG OS errors.

    Every custom exception in the system inherits from this class.
    This allows catching all application errors with a single except clause
    when needed, while still supporting granular handling.

    Attributes:
        detail: Human-readable error message.
        error_code: Machine-readable error identifier (e.g., "DOCUMENT_NOT_FOUND").
        status_code: HTTP status code to return to the client.
        context: Additional context for debugging/logging.
    """

    def __init__(
        self,
        detail: str = "An unexpected error occurred.",
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code
        self.context = context or {}
        super().__init__(self.detail)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON API responses."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.detail,
                "status": self.status_code,
                "context": self.context,
            }
        }


# ── Configuration ─────────────────────────────────────────────────────────────


class ConfigurationError(EnterpriseRAGError):
    """Raised when application configuration is invalid or missing."""

    def __init__(self, detail: str = "Invalid configuration.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            **kwargs,
        )


# ── Document Errors ──────────────────────────────────────────────────────────


class DocumentError(EnterpriseRAGError):
    """Base exception for document-related errors."""

    def __init__(self, detail: str = "Document processing error.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="DOCUMENT_ERROR",
            status_code=400,
            **kwargs,
        )


class DocumentNotFoundError(DocumentError):
    """Raised when a requested document does not exist."""

    def __init__(self, detail: str = "Document not found.", **kwargs: Any) -> None:
        super().__init__(detail=detail, **kwargs)
        self.error_code = "DOCUMENT_NOT_FOUND"
        self.status_code = 404


class DocumentParsingError(DocumentError):
    """Raised when a document cannot be parsed (corrupt, encrypted, etc.)."""

    def __init__(self, detail: str = "Failed to parse document.", **kwargs: Any) -> None:
        super().__init__(detail=detail, **kwargs)
        self.error_code = "DOCUMENT_PARSING_ERROR"
        self.status_code = 422


class UnsupportedFormatError(DocumentError):
    """Raised when a document format is not supported."""

    def __init__(self, detail: str = "Unsupported document format.", **kwargs: Any) -> None:
        super().__init__(detail=detail, **kwargs)
        self.error_code = "UNSUPPORTED_FORMAT"
        self.status_code = 415


# ── Embedding Errors ──────────────────────────────────────────────────────────


class EmbeddingError(EnterpriseRAGError):
    """Raised when embedding generation fails."""

    def __init__(self, detail: str = "Embedding generation failed.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="EMBEDDING_ERROR",
            status_code=500,
            **kwargs,
        )


# ── Vector Store Errors ──────────────────────────────────────────────────────


class VectorStoreError(EnterpriseRAGError):
    """Raised when vector store operations fail."""

    def __init__(self, detail: str = "Vector store operation failed.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="VECTORSTORE_ERROR",
            status_code=500,
            **kwargs,
        )


# ── Retrieval Errors ─────────────────────────────────────────────────────────


class RetrievalError(EnterpriseRAGError):
    """Raised when document retrieval fails."""

    def __init__(self, detail: str = "Retrieval failed.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="RETRIEVAL_ERROR",
            status_code=500,
            **kwargs,
        )


# ── LLM Errors ───────────────────────────────────────────────────────────────


class LLMError(EnterpriseRAGError):
    """Base exception for LLM-related errors."""

    def __init__(self, detail: str = "LLM operation failed.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="LLM_ERROR",
            status_code=502,
            **kwargs,
        )


class LLMConnectionError(LLMError):
    """Raised when the LLM provider is unreachable."""

    def __init__(self, detail: str = "Cannot connect to LLM provider.", **kwargs: Any) -> None:
        super().__init__(detail=detail, **kwargs)
        self.error_code = "LLM_CONNECTION_ERROR"
        self.status_code = 503


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate limit is exceeded."""

    def __init__(self, detail: str = "LLM rate limit exceeded.", **kwargs: Any) -> None:
        super().__init__(detail=detail, **kwargs)
        self.error_code = "LLM_RATE_LIMIT"
        self.status_code = 429


class LLMResponseError(LLMError):
    """Raised when the LLM returns an invalid or unexpected response."""

    def __init__(self, detail: str = "Invalid LLM response.", **kwargs: Any) -> None:
        super().__init__(detail=detail, **kwargs)
        self.error_code = "LLM_RESPONSE_ERROR"
        self.status_code = 502


# ── Auth Errors ───────────────────────────────────────────────────────────────


class AuthenticationError(EnterpriseRAGError):
    """Raised when authentication fails (invalid/missing credentials)."""

    def __init__(self, detail: str = "Authentication failed.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            **kwargs,
        )


class AuthorizationError(EnterpriseRAGError):
    """Raised when the user lacks permission for the requested action."""

    def __init__(self, detail: str = "Insufficient permissions.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            **kwargs,
        )


# ── Rate Limiting ─────────────────────────────────────────────────────────────


class RateLimitError(EnterpriseRAGError):
    """Raised when the client exceeds the API rate limit."""

    def __init__(self, detail: str = "Rate limit exceeded.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            **kwargs,
        )


# ── Validation ────────────────────────────────────────────────────────────────


class InputValidationError(EnterpriseRAGError):
    """Raised when user input fails validation."""

    def __init__(self, detail: str = "Input validation failed.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="VALIDATION_ERROR",
            status_code=422,
            **kwargs,
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────


class PipelineError(EnterpriseRAGError):
    """Raised when a pipeline stage fails."""

    def __init__(self, detail: str = "Pipeline execution failed.", **kwargs: Any) -> None:
        super().__init__(
            detail=detail,
            error_code="PIPELINE_ERROR",
            status_code=500,
            **kwargs,
        )
