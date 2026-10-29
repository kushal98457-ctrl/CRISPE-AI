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
    MCPUnauthorizedError,
    sanitize_error_message,
)
from app.mcp.schemas import (
    GenerateCrispePromptInput,
    GenerateCrispePromptOutput,
    PingResponse,
    PromptMemoryMatchItem,
    SavePromptMemoryInput,
    SavePromptMemoryOutput,
    SearchPromptMemoryInput,
    SearchPromptMemoryOutput,
    ServerMetadataResponse,
)
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
    "MCPUnauthorizedError",
    "MCPInternalError",
    "sanitize_error_message",
    "PingResponse",
    "ServerMetadataResponse",
    "GenerateCrispePromptInput",
    "GenerateCrispePromptOutput",
    "SearchPromptMemoryInput",
    "PromptMemoryMatchItem",
    "SearchPromptMemoryOutput",
    "SavePromptMemoryInput",
    "SavePromptMemoryOutput",
]
