"""Model Context Protocol (MCP) server foundation for CRISPE-AI (Step 29).

Exposes prompt generation, prompt memory, and self-evolving optimization capabilities
to compatible AI assistants through the standard Model Context Protocol.
"""

from app.mcp.config import MCPServerConfig
from app.mcp.errors import (
    MCPError,
    MCPInternalError,
    MCPInvalidInputError,
    MCPResourceNotFoundError,
    sanitize_error_message,
)
from app.mcp.schemas import PingResponse, ServerMetadataResponse
from app.mcp.server import create_mcp_server, run_stdio
from app.mcp.services.context import MCPAppContext

__all__ = [
    "create_mcp_server",
    "run_stdio",
    "MCPServerConfig",
    "MCPAppContext",
    "MCPError",
    "MCPInvalidInputError",
    "MCPResourceNotFoundError",
    "MCPInternalError",
    "sanitize_error_message",
    "PingResponse",
    "ServerMetadataResponse",
]
