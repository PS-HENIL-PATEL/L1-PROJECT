"""
Tests — Application Startup (Integration)
============================================

Integration tests that verify the full application starts correctly,
middleware is active, and endpoints are reachable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from httpx import AsyncClient


@pytest.mark.integration
class TestAppStartup:
    """Test that the application starts and responds correctly."""

    @pytest.mark.asyncio
    async def test_app_starts_successfully(self, client: AsyncClient) -> None:
        """The app should start without errors."""
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_docs_accessible(self, client: AsyncClient) -> None:
        """OpenAPI docs should be available."""
        response = await client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_json_accessible(self, client: AsyncClient) -> None:
        """OpenAPI JSON schema should be available."""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "Enterprise RAG OS" in data["info"]["title"]

    @pytest.mark.asyncio
    async def test_request_id_header_present(self, client: AsyncClient) -> None:
        """Every response should include X-Request-ID header."""
        response = await client.get("/health")
        assert "X-Request-ID" in response.headers

    @pytest.mark.asyncio
    async def test_process_time_header_present(self, client: AsyncClient) -> None:
        """Every response should include X-Process-Time header."""
        response = await client.get("/health")
        assert "X-Process-Time" in response.headers

    @pytest.mark.asyncio
    async def test_custom_request_id_propagated(self, client: AsyncClient) -> None:
        """If client sends X-Request-ID, it should be echoed back."""
        response = await client.get(
            "/health",
            headers={"X-Request-ID": "test-123"},
        )
        assert response.headers["X-Request-ID"] == "test-123"

    @pytest.mark.asyncio
    async def test_404_for_unknown_route(self, client: AsyncClient) -> None:
        """Unknown routes should return 404."""
        response = await client.get("/nonexistent")
        assert response.status_code == 404
