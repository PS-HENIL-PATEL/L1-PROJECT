"""
Tests — Exception Hierarchy
==============================

Tests for the custom exception classes and their serialization
to HTTP-compatible error responses.
"""

from __future__ import annotations

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DocumentNotFoundError,
    DocumentParsingError,
    EmbeddingError,
    EnterpriseRAGError,
    InputValidationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    PipelineError,
    RateLimitError,
    RetrievalError,
    UnsupportedFormatError,
    VectorStoreError,
)


class TestEnterpriseRAGError:
    """Test the base exception class."""

    def test_default_values(self) -> None:
        err = EnterpriseRAGError()
        assert err.detail == "An unexpected error occurred."
        assert err.error_code == "INTERNAL_ERROR"
        assert err.status_code == 500
        assert err.context == {}

    def test_custom_values(self) -> None:
        err = EnterpriseRAGError(
            detail="Custom error",
            error_code="CUSTOM",
            status_code=418,
            context={"key": "value"},
        )
        assert err.detail == "Custom error"
        assert err.error_code == "CUSTOM"
        assert err.status_code == 418
        assert err.context == {"key": "value"}

    def test_to_dict(self) -> None:
        err = EnterpriseRAGError(detail="Test", error_code="TEST", status_code=400)
        d = err.to_dict()
        assert d["error"]["code"] == "TEST"
        assert d["error"]["message"] == "Test"
        assert d["error"]["status"] == 400

    def test_string_representation(self) -> None:
        err = EnterpriseRAGError(detail="Something broke")
        assert str(err) == "Something broke"


class TestDocumentErrors:
    """Test document-related exceptions."""

    def test_document_not_found(self) -> None:
        err = DocumentNotFoundError()
        assert err.status_code == 404
        assert err.error_code == "DOCUMENT_NOT_FOUND"

    def test_document_parsing_error(self) -> None:
        err = DocumentParsingError(detail="Corrupt PDF")
        assert err.status_code == 422
        assert err.detail == "Corrupt PDF"

    def test_unsupported_format(self) -> None:
        err = UnsupportedFormatError()
        assert err.status_code == 415


class TestLLMErrors:
    """Test LLM-related exceptions."""

    def test_llm_connection_error(self) -> None:
        err = LLMConnectionError()
        assert err.status_code == 503

    def test_llm_rate_limit(self) -> None:
        err = LLMRateLimitError()
        assert err.status_code == 429

    def test_llm_response_error(self) -> None:
        err = LLMResponseError()
        assert err.status_code == 502


class TestAuthErrors:
    """Test authentication/authorization exceptions."""

    def test_authentication_error(self) -> None:
        err = AuthenticationError()
        assert err.status_code == 401

    def test_authorization_error(self) -> None:
        err = AuthorizationError()
        assert err.status_code == 403


class TestOtherErrors:
    """Test remaining exception types."""

    def test_configuration_error(self) -> None:
        assert ConfigurationError().status_code == 500

    def test_embedding_error(self) -> None:
        assert EmbeddingError().status_code == 500

    def test_vectorstore_error(self) -> None:
        assert VectorStoreError().status_code == 500

    def test_retrieval_error(self) -> None:
        assert RetrievalError().status_code == 500

    def test_rate_limit_error(self) -> None:
        assert RateLimitError().status_code == 429

    def test_validation_error(self) -> None:
        assert InputValidationError().status_code == 422

    def test_pipeline_error(self) -> None:
        assert PipelineError().status_code == 500


class TestExceptionInheritance:
    """Test that all exceptions inherit from EnterpriseRAGError."""

    def test_all_inherit_from_base(self) -> None:
        exceptions = [
            ConfigurationError(),
            DocumentNotFoundError(),
            DocumentParsingError(),
            UnsupportedFormatError(),
            EmbeddingError(),
            VectorStoreError(),
            RetrievalError(),
            LLMConnectionError(),
            LLMRateLimitError(),
            LLMResponseError(),
            AuthenticationError(),
            AuthorizationError(),
            RateLimitError(),
            InputValidationError(),
            PipelineError(),
        ]
        for exc in exceptions:
            assert isinstance(exc, EnterpriseRAGError), (
                f"{type(exc).__name__} does not inherit from EnterpriseRAGError"
            )
