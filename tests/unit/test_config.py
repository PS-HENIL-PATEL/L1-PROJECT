"""
Tests — Configuration System
===============================

Tests for the Pydantic Settings configuration system.
Verifies that settings load correctly, validate values, and
handle environment overrides.
"""

from __future__ import annotations

from app.config.settings import (
    ChunkingSettings,
    Environment,
    LogFormat,
    LogLevel,
    Settings,
    get_settings,
)


class TestSettingsDefaults:
    """Test that settings have sensible defaults."""

    def test_default_app_name(self, test_settings: Settings) -> None:
        assert test_settings.app_name == "Intellex"

    def test_default_environment(self, test_settings: Settings) -> None:
        assert test_settings.environment == Environment.DEVELOPMENT

    def test_default_debug(self, test_settings: Settings) -> None:
        assert test_settings.debug is True

    def test_default_server_port(self, test_settings: Settings) -> None:
        assert test_settings.server.port == 8000

    def test_default_log_level(self, test_settings: Settings) -> None:
        assert test_settings.logging.level == LogLevel.DEBUG

    def test_default_log_format(self, test_settings: Settings) -> None:
        assert test_settings.logging.format == LogFormat.JSON

    def test_default_embedding_provider(self, test_settings: Settings) -> None:
        assert test_settings.embedding.provider == "sentence-transformers"

    def test_default_embedding_dimension(self, test_settings: Settings) -> None:
        assert test_settings.embedding.dimension == 384

    def test_default_vectorstore_provider(self, test_settings: Settings) -> None:
        assert test_settings.vectorstore.provider == "qdrant"

    def test_default_chunking_strategy(self, test_settings: Settings) -> None:
        assert test_settings.chunking.strategy == "recursive"

    def test_default_retrieval_strategy(self, test_settings: Settings) -> None:
        assert test_settings.retrieval.strategy == "hybrid"


class TestSettingsProperties:
    """Test computed properties on settings."""

    def test_is_development(self, test_settings: Settings) -> None:
        assert test_settings.is_development is True

    def test_is_not_production(self, test_settings: Settings) -> None:
        assert test_settings.is_production is False

    def test_project_root_exists(self, test_settings: Settings) -> None:
        assert test_settings.project_root.exists()


class TestSettingsValidation:
    """Test configuration validation."""

    def test_chunk_overlap_less_than_size(self) -> None:
        """Chunk overlap must be less than chunk size."""
        import pytest

        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkingSettings(chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_greater_than_size_fails(self) -> None:
        """Chunk overlap greater than size should fail."""
        import pytest

        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkingSettings(chunk_size=100, chunk_overlap=200)

    def test_valid_chunk_settings(self) -> None:
        """Valid chunk settings should work."""
        settings = ChunkingSettings(chunk_size=512, chunk_overlap=50)
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50


class TestGetSettings:
    """Test the settings singleton pattern."""

    def test_get_settings_returns_settings(self) -> None:
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)
        get_settings.cache_clear()

    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()
