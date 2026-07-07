"""
Enterprise RAG OS — Test Configuration & Shared Fixtures
==========================================================

Purpose:
    Shared pytest fixtures available to all test modules.
    Provides a test client, test settings, and common test utilities.

Architecture:
    Uses pytest-asyncio for async test support and httpx for the
    async test client (required by FastAPI's async test pattern).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config.settings import Settings, get_settings
from app.main import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Provide test-specific settings.

    Uses defaults which target the development environment.
    Override specific values for test scenarios.
    """
    # Clear the cached settings to allow test overrides
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


@pytest.fixture(scope="session")
def app():
    """Create a test application instance."""
    get_settings.cache_clear()
    application = create_app()
    yield application
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP test client.

    Uses httpx.AsyncClient with ASGITransport to make requests
    directly to the FastAPI app without starting a real server.
    This is faster and more isolated than running a test server.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
