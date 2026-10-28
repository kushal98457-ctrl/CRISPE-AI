"""Tests for health and liveness probe endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_root_endpoint(async_client: AsyncClient):
    """Verify /health returns 200 OK with expected schema and telemetry headers."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "crispe-ai-backend-test"
    assert "version" in data
    assert data["environment"] == "testing"

    # Verify telemetry headers
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"].startswith("req-")
    assert "X-Process-Time" in response.headers


@pytest.mark.asyncio
async def test_health_versioned_endpoint(async_client: AsyncClient):
    """Verify /api/v1/health returns 200 OK identical to root probe."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "crispe-ai-backend-test"
