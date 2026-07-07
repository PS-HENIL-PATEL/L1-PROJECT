"""
Tests — Health Check Endpoints
=================================

Tests for the health and readiness endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from httpx import AsyncClient


class TestHealthEndpoint:
    """Test the liveness health check."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_status(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_includes_version(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_health_includes_environment(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        data = response.json()
        assert "environment" in data

    @pytest.mark.asyncio
    async def test_health_includes_uptime(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))


class TestReadinessEndpoint:
    """Test the readiness check."""

    @pytest.mark.asyncio
    async def test_readiness_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health/ready")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_readiness_includes_components(self, client: AsyncClient) -> None:
        response = await client.get("/health/ready")
        data = response.json()
        assert "components" in data
        assert isinstance(data["components"], list)
        assert len(data["components"]) > 0

    @pytest.mark.asyncio
    async def test_readiness_has_configuration_component(self, client: AsyncClient) -> None:
        response = await client.get("/health/ready")
        data = response.json()
        names = [c["name"] for c in data["components"]]
        assert "configuration" in names

    @pytest.mark.asyncio
    async def test_readiness_config_is_healthy(self, client: AsyncClient) -> None:
        response = await client.get("/health/ready")
        data = response.json()
        config = next(c for c in data["components"] if c["name"] == "configuration")
        assert config["status"] == "healthy"


class TestAPIv1Root:
    """Test the API v1 root endpoint."""

    @pytest.mark.asyncio
    async def test_api_v1_root(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_v1_returns_version(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/")
        data = response.json()
        assert data["version"] == "v1"
        assert data["status"] == "active"
