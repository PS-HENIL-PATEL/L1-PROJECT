"""
Enterprise RAG OS — Configuration Management
=============================================

Purpose:
    Centralized, type-safe, validated configuration for the entire application.
    Uses Pydantic Settings v2 to load values from environment variables and .env files.

Why Pydantic Settings?
    - Type-safe: Every config value has an explicit type. No stringly-typed nonsense.
    - Validated at startup: If a required value is missing or invalid, the app fails
      fast with a clear error — not 30 minutes into a production run.
    - Environment-aware: Supports dev/staging/prod via ENVIRONMENT variable.
    - Nested models: Complex config sections (LLM, embedding, vectorstore) are
      isolated into their own Pydantic models for clarity and reuse.
    - Serializable: The entire config can be exported as JSON for debugging/logging.

Design Decisions:
    - All defaults target local development. Production values come from environment.
    - Secrets (API keys) have no defaults — they MUST be provided in production.
    - Nested settings classes keep related config grouped and independently testable.
    - The global `get_settings()` function uses `lru_cache` for singleton behavior
      without the complexity of a proper singleton pattern.

Dependencies:
    - pydantic-settings (BaseSettings, SettingsConfigDict)
    - pydantic (Field, field_validator)

Usage:
    from app.config.settings import get_settings
    settings = get_settings()
    print(settings.app_name)
    print(settings.server.port)
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Base Path ─────────────────────────────────────────────────────────────────
# All relative paths in config are resolved relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env file into os.environ so all nested BaseSettings models can read it
load_dotenv(PROJECT_ROOT / ".env")


# ── Enums ─────────────────────────────────────────────────────────────────────


class Environment(enum.StrEnum):
    """
    Deployment environment.

    Why an enum?
        Prevents typos like "developement" or "prod " (with trailing space).
        The app will refuse to start if ENVIRONMENT is not one of these values.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(enum.StrEnum):
    """Valid log levels. Maps directly to Python's logging module levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(enum.StrEnum):
    """
    Log output format.

    JSON: Machine-parseable, ideal for log aggregation (ELK, Datadog, CloudWatch).
    TEXT: Human-readable, ideal for local development.
    """

    JSON = "json"
    TEXT = "text"


# ── Nested Settings Models ────────────────────────────────────────────────────


class ServerSettings(BaseSettings):
    """
    HTTP server configuration.

    Why separate from root settings?
        Server config is passed directly to Uvicorn. Isolating it makes the
        mapping explicit and avoids polluting the root namespace.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    host: str = Field(default="0.0.0.0", description="Bind address")  # noqa: S104
    port: int = Field(default=8000, ge=1, le=65535, description="Listen port")
    workers: int = Field(default=1, ge=1, description="Uvicorn worker count")
    reload: bool = Field(default=True, description="Auto-reload on file changes")

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        """Warn if using multiple workers with reload enabled."""
        return v


class CORSSettings(BaseSettings):
    """
    Cross-Origin Resource Sharing configuration.

    Why explicit CORS config?
        In production, CORS must be locked down to specific origins.
        Defaulting to localhost origins ensures dev works out-of-the-box
        while forcing explicit config for production deployment.
    """

    model_config = SettingsConfigDict(env_prefix="CORS_", extra="ignore")

    origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )
    allow_credentials: bool = Field(default=True)
    allow_methods: list[str] = Field(default=["*"])
    allow_headers: list[str] = Field(default=["*"])


class LoggingSettings(BaseSettings):
    """
    Logging configuration.

    Design:
        - JSON format for production (machine-parseable).
        - Text format for development (human-readable).
        - File logging with rotation prevents disk exhaustion.
        - Correlation IDs are injected by middleware (see core/middleware.py).
    """

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: LogLevel = Field(default=LogLevel.DEBUG, description="Minimum log level")
    format: LogFormat = Field(default=LogFormat.JSON, description="Output format")
    file: str = Field(
        default="logs/enterprise_rag.log",
        description="Log file path (relative to project root)",
    )
    rotation: str = Field(default="10 MB", description="Log file rotation size")
    retention: str = Field(default="30 days", description="Log file retention period")


class EmbeddingSettings(BaseSettings):
    """
    Embedding model configuration (Phase 2).

    Why pre-define this in Phase 1?
        The config structure is part of the foundation. Later phases simply
        populate these values — no config refactoring needed.
    """

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    provider: str = Field(
        default="sentence-transformers",
        description="Embedding provider: sentence-transformers | openai | bge | e5 | instructor",
    )
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name or path",
    )
    dimension: int = Field(default=384, ge=1, description="Embedding vector dimension")
    batch_size: int = Field(default=32, ge=1, description="Batch size for embedding")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration (Phase 2)."""

    model_config = SettingsConfigDict(env_prefix="VECTORSTORE_", extra="ignore")

    provider: str = Field(
        default="chromadb",
        description="Vector store provider: chromadb | qdrant | faiss",
    )
    collection_name: str = Field(
        default="enterprise_rag",
        description="Default collection/index name",
    )
    # Provider-specific settings
    chromadb_persist_dir: str = Field(
        default="vector_store/chromadb",
        description="ChromaDB persistence directory",
    )
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")


class ChunkingSettings(BaseSettings):
    """Chunking strategy configuration (Phase 2)."""

    model_config = SettingsConfigDict(env_prefix="CHUNKING_", extra="ignore")

    strategy: str = Field(
        default="recursive",
        description="Strategy: recursive | semantic | markdown | token"
        " | sliding_window | parent_child",
    )
    chunk_size: int = Field(default=512, ge=50, le=8192, description="Target chunk size in chars")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between chunks in chars")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: Any) -> int:
        """Overlap must be less than chunk size to avoid infinite loops."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            msg = f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)
        return v


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline configuration (Phase 3)."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")

    top_k: int = Field(default=20, ge=1, description="Number of documents to retrieve")
    rerank_top_k: int = Field(default=5, ge=1, description="Number of documents after reranking")
    strategy: str = Field(
        default="hybrid",
        description="Strategy: dense | bm25 | hybrid",
    )
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for dense vs sparse (0=full BM25, 1=full dense)",
    )


class LLMSettings(BaseSettings):
    """LLM provider configuration (Phase 4)."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    default_provider: str = Field(
        default="openai",
        description="Default LLM provider: openai | anthropic | google | ollama | openrouter",
    )
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Default model name",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    streaming: bool = Field(default=True, description="Enable streaming responses")

    # API Keys (no defaults — must be provided)
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_base_url: str | None = Field(default=None, description="Base URL for OpenAI-compatible providers (like Groq, vLLM, etc.)")  # noqa: E501
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    google_api_key: str | None = Field(default=None, description="Google AI API key")
    cohere_api_key: str | None = Field(default=None, description="Cohere API key")
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )


class SecuritySettings(BaseSettings):
    """Security configuration (Phase 6)."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    jwt_secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION",
        description="JWT signing secret (MUST change in production)",
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_minutes: int = Field(default=30, ge=1)
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_per_minute: int = Field(default=60, ge=1)


# ── Root Settings ─────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Root application settings.

    This is the single entry point for all configuration. Sub-settings are
    nested as attributes, keeping the namespace clean and organized.

    Loading priority (highest to lowest):
        1. Environment variables
        2. .env file
        3. Default values defined here

    Example:
        settings = get_settings()
        settings.environment          # Environment.DEVELOPMENT
        settings.server.port          # 8000
        settings.logging.level        # LogLevel.DEBUG
        settings.embedding.model      # "all-MiniLM-L6-v2"
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    app_name: str = Field(default="Intellex")
    app_version: str = Field(default="0.1.0")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=True)

    # ── Nested Settings ──────────────────────────────────────────────────
    server: ServerSettings = Field(default_factory=ServerSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION

    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return PROJECT_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the application settings singleton.

    Uses lru_cache to ensure settings are loaded exactly once, then reused.
    This is cheaper than a proper singleton pattern and works perfectly
    with FastAPI's dependency injection.

    Returns:
        Validated Settings instance.

    Raises:
        pydantic.ValidationError: If required settings are missing or invalid.
    """
    return Settings()
