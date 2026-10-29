"""Automated tests for the CRISPE-AI MCP Server Foundation (Step 29).

Covers:
1. Server construction without import-time side effects (no ports, no DB migrations, no Gemini calls).
2. Server factory configuration (defaults, custom overrides, and Pydantic validation).
3. MCP protocol initialization handshake and capability discovery via official Client.
4. Minimal smoke-test tool execution ('ping' and 'server_info').
5. Error sanitization and credential/path redaction.
6. Stdio logging isolation (logs strictly directed to stderr).
7. Dependency context lazy resolution.
8. Application isolation and stability (FastAPI application remains unaffected).
"""

import io
import logging
import sys
from typing import Optional
import pytest
from pydantic import ValidationError

from mcp.client import Client
from mcp.server.mcpserver import MCPServer

from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_application
from app.mcp import (
    MCPServerConfig,
    MCPAppContext,
    MCPError,
    MCPInternalError,
    MCPInvalidInputError,
    MCPResourceNotFoundError,
    create_mcp_server,
    sanitize_error_message,
)
from app.mcp.server import configure_stdio_logging


# =============================================================================
# 1. Server Construction & Factory Tests
# =============================================================================

def test_server_factory_default_construction():
    """Verify factory constructs valid server with project settings defaults."""
    settings = get_settings()
    server = create_mcp_server()

    assert isinstance(server, MCPServer)
    assert server.name == "crispe-ai-mcp"
    assert server.version == settings.APP_VERSION
    assert server.title == "CRISPE-AI Model Context Protocol Server"
    assert "CRISPE prompt generation" in server.description
    assert hasattr(server, "app_context")
    assert hasattr(server, "server_config")


def test_server_factory_custom_configuration():
    """Verify factory respects explicit configuration overrides."""
    custom_cfg = MCPServerConfig(
        server_name="custom-crispe-mcp",
        server_version="1.2.3",
        title="Custom Title",
        description="Custom Description",
        instructions="Custom instructions",
        enable_smoke_tools=False,
    )
    server = create_mcp_server(config=custom_cfg)

    assert server.name == "custom-crispe-mcp"
    assert server.version == "1.2.3"
    assert server.title == "Custom Title"
    assert server.description == "Custom Description"
    assert server.instructions == "Custom instructions"


def test_server_config_validation():
    """Verify invalid configuration bounds raise clear validation errors."""
    with pytest.raises(ValidationError):
        # Empty server name rejected
        MCPServerConfig(server_name="")

    with pytest.raises(ValidationError):
        # Timeout below minimum bound rejected
        MCPServerConfig(request_timeout_seconds=0.1)


def test_server_construction_has_no_side_effects():
    """Importing and creating server does not open network connections or mutate globals."""
    from app.db.session import _engine_instance
    initial_engine = _engine_instance

    server = create_mcp_server()
    assert server is not None

    # Database engine should not have been initialized if it was previously None
    from app.db.session import _engine_instance as after_engine
    assert initial_engine == after_engine


# =============================================================================
# 2. MCP Protocol Initialization & Tool Discovery
# =============================================================================

@pytest.mark.asyncio
async def test_protocol_initialization_and_tool_discovery():
    """Verify official MCP Client connects in-memory and discovers smoke tools."""
    server = create_mcp_server()
    client = Client(server)

    async with client:
        result = await client.list_tools()
        assert result is not None
        tool_names = [tool.name for tool in result.tools]
        assert "ping" in tool_names
        assert "server_info" in tool_names

        # Verify ping tool metadata
        ping_tool = next(t for t in result.tools if t.name == "ping")
        assert "smoke-test health check" in ping_tool.description.lower()

        # Verify server_info tool metadata
        info_tool = next(t for t in result.tools if t.name == "server_info")
        assert "authoritative crispe-ai mcp server metadata" in info_tool.description.lower()


@pytest.mark.asyncio
async def test_smoke_tools_disabled():
    """When enable_smoke_tools=False, no smoke tools are registered."""
    cfg = MCPServerConfig(enable_smoke_tools=False)
    server = create_mcp_server(config=cfg)
    client = Client(server)

    async with client:
        result = await client.list_tools()
        tool_names = [tool.name for tool in result.tools]
        assert "ping" not in tool_names
        assert "server_info" not in tool_names


# =============================================================================
# 3. Smoke Test Capability Execution
# =============================================================================

@pytest.mark.asyncio
async def test_ping_capability_execution():
    """Verify ping tool execution returns valid status without external dependencies."""
    server = create_mcp_server()
    client = Client(server)

    async with client:
        result = await client.call_tool("ping")
        assert result.is_error is False
        assert result.structured_content is not None
        assert result.structured_content.get("status") == "ok"
        assert "timestamp" in result.structured_content


@pytest.mark.asyncio
async def test_server_info_capability_execution():
    """Verify server_info returns static metadata without exposing environment secrets."""
    server = create_mcp_server()
    client = Client(server)

    async with client:
        result = await client.call_tool("server_info")
        assert result.is_error is False
        content = result.structured_content
        assert content is not None
        assert content["server_name"] == "crispe-ai-mcp"
        assert content["server_version"] == "0.1.0"
        assert content["protocol_status"] == "ready"
        assert "tools" in content["supported_capabilities"]

        # Security check: must NOT leak environment keys or system paths
        raw_text = str(content)
        assert "AIza" not in raw_text
        assert "postgresql://" not in raw_text
        assert "Users" not in raw_text


# =============================================================================
# 4. Security, Error Redaction, and Logging Tests
# =============================================================================

def test_sanitize_error_message():
    """Verify sensitive connection strings, tokens, and local paths are scrubbed."""
    raw_db_error = "Connection failed to postgresql://admin:secret123@db.prod.internal:5432/crispe_ai"
    sanitized_db = sanitize_error_message(raw_db_error)
    assert "secret123" not in sanitized_db
    assert "<REDACTED_DB_URI>" in sanitized_db

    raw_key_error = "Authentication failed for API key AIzaSyD1234567890abcdef1234567890abcdef"
    sanitized_key = sanitize_error_message(raw_key_error)
    assert "AIzaSyD" not in sanitized_key
    assert "<REDACTED_CREDENTIAL>" in sanitized_key

    raw_path_error = r"File missing at C:\Users\operator\private\config.json"
    sanitized_path = sanitize_error_message(raw_path_error)
    assert r"C:\Users\operator\private" not in sanitized_path
    assert "<REDACTED_PATH>" in sanitized_path

    raw_unix_path = "Cannot read /home/ubuntu/secrets/key.pem"
    sanitized_unix = sanitize_error_message(raw_unix_path)
    assert "/home/ubuntu/secrets" not in sanitized_unix
    assert "<REDACTED_PATH>" in sanitized_unix


def test_structured_mcp_exceptions():
    """Verify MCPError and subclasses format to safe dictionaries."""
    inv_err = MCPInvalidInputError("Missing idea parameter", details={"field": "idea"})
    data = inv_err.to_dict()
    assert data["error"] is True
    assert data["code"] == "INVALID_INPUT"
    assert data["message"] == "Missing idea parameter"
    assert data["details"] == {"field": "idea"}

    int_err = MCPInternalError("DB crash at postgresql://user:pass@localhost:5432/db")
    int_data = int_err.to_dict()
    assert int_data["code"] == "INTERNAL_ERROR"
    assert "pass" not in int_data["message"]
    assert "<REDACTED_DB_URI>" in int_data["message"]


def test_stdio_logging_redirection(monkeypatch):
    """Verify stdio logging sends log records strictly to stderr, never stdout."""
    stderr_capture = io.StringIO()
    stdout_capture = io.StringIO()

    monkeypatch.setattr(sys, "stderr", stderr_capture)
    monkeypatch.setattr(sys, "stdout", stdout_capture)

    configure_stdio_logging(log_level="DEBUG")
    test_logger = logging.getLogger("test_mcp_stdio")
    test_logger.info("Stdio diagnostic message")

    # Stderr must contain the log record
    assert "Stdio diagnostic message" in stderr_capture.getvalue()
    # Stdout must remain completely empty to avoid corrupting MCP JSON-RPC
    assert stdout_capture.getvalue() == ""


# =============================================================================
# 5. Dependency Injection & Application Isolation Tests
# =============================================================================

def test_app_context_lazy_resolution():
    """Verify application context defers service instantiation until accessed."""
    called = False

    def dummy_gen_factory():
        nonlocal called
        called = True
        return "mock_generation_service"

    ctx = MCPAppContext(generation_service_factory=dummy_gen_factory)
    assert not called

    service = ctx.get_generation_service()
    assert called
    assert service == "mock_generation_service"


@pytest.mark.asyncio
async def test_fastapi_application_remains_functional():
    """Verify existing FastAPI application factory and endpoints are unaffected."""
    app = create_application()
    assert app is not None
    assert app.title == "CRISPE-AI Backend"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json().get("status") in {"ok", "healthy"}
