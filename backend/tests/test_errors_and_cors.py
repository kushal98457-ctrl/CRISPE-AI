"""Tests for error handling, CORS headers, and request tracking."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_not_found_standardized_error(async_client: AsyncClient):
    """Verify 404 on nonexistent route returns standardized ErrorResponse."""
    response = await async_client.get("/nonexistent-endpoint")
    assert response.status_code == 404

    data = response.json()
    assert data["status_code"] == 404
    assert data["error_code"] == "NOT_FOUND"
    assert "Not Found" in data["message"]
    assert "request_id" in data
    assert response.headers["X-Request-ID"] == data["request_id"]


@pytest.mark.asyncio
async def test_method_not_allowed_error(async_client: AsyncClient):
    """Verify 405 on disallowed HTTP method returns standardized ErrorResponse."""
    # /health only accepts GET
    response = await async_client.post("/health")
    assert response.status_code == 405

    data = response.json()
    assert data["status_code"] == 405
    assert data["error_code"] == "METHOD_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_custom_request_id_propagation(async_client: AsyncClient):
    """Verify client-supplied X-Request-ID is preserved in response headers."""
    client_req_id = "custom-client-trace-id-12345"
    response = await async_client.get("/health", headers={"X-Request-ID": client_req_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == client_req_id


@pytest.mark.asyncio
async def test_cors_headers_allowed_origin(async_client: AsyncClient):
    """Verify allowed frontend origin receives Access-Control-Allow-Origin header."""
    response = await async_client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "GET" in response.headers.get("access-control-allow-methods", "")


@pytest.mark.asyncio
async def test_cors_headers_disallowed_origin(async_client: AsyncClient):
    """Verify disallowed origin does not receive access-control-allow-origin."""
    response = await async_client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://malicious-site.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") is None
