"""Minimal, safe smoke-test capabilities for CRISPE-AI MCP Server Foundation (Step 29).

Enforces:
1. Pure static response verification for MCP client initialization and tool discovery.
2. Absolute isolation from external APIs (no Gemini API calls, no database queries, no model costs).
3. Zero credential or internal infrastructure leakage.
"""

from typing import TYPE_CHECKING
from app.mcp.schemas import PingResponse, ServerMetadataResponse

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer
    from app.mcp.config import MCPServerConfig
    from app.mcp.services.context import MCPAppContext


def register_smoke_tools(
    server: "MCPServer",
    config: "MCPServerConfig",
    context: "MCPAppContext",
) -> None:
    """Register safe smoke-test capabilities on the provided MCPServer instance."""

    @server.tool(
        name="ping",
        description="Minimal smoke-test health check verifying that the CRISPE-AI MCP server is operational.",
    )
    def ping() -> PingResponse:
        """Return a simple successful health check response."""
        return PingResponse(status="ok")

    @server.tool(
        name="server_info",
        description="Retrieve authoritative CRISPE-AI MCP server metadata, protocol status, and capabilities.",
    )
    def server_info() -> ServerMetadataResponse:
        """Return static server identity and capability status without leaking sensitive environment details."""
        cfg = context.get_settings()
        return ServerMetadataResponse(
            server_name=config.server_name,
            server_version=config.server_version,
            title=config.title,
            description=config.description,
            protocol_status="ready",
            supported_capabilities=["tools"],
            environment=cfg.ENVIRONMENT,
        )
