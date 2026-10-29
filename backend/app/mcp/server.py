"""CRISPE-AI Model Context Protocol (MCP) Server Factory and Lifecycle (Step 29).

Enforces:
1. Reusable, testable server construction without import-time side effects.
2. Clean capability registration and metadata publishing.
3. Strict stdio isolation (all logs directed to stderr to preserve stdio JSON-RPC streams).
4. Dependency injection context for application domain service access.
"""

import logging
import sys
from typing import Optional

from mcp.server.mcpserver import MCPServer

from app.core.logging import get_logger
from app.mcp.config import MCPServerConfig
from app.mcp.services.context import MCPAppContext
from app.mcp.tools import (
    register_generation_tools,
    register_memory_tools,
    register_smoke_tools,
)

logger = get_logger("mcp_server")


def configure_stdio_logging(log_level: str = "INFO") -> None:
    """Configure Python logging to emit solely to stderr during stdio MCP transport.

    CRITICAL: Writing any log message, banner, or debug output to stdout corrupts
    the MCP JSON-RPC protocol messages and breaks client communication.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Reconfigure root logger with a stderr-targeted StreamHandler
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers to prevent stdout leaks
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [mcp] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(numeric_level)
    root_logger.addHandler(stderr_handler)


def create_mcp_server(
    config: Optional[MCPServerConfig] = None,
    context: Optional[MCPAppContext] = None,
) -> MCPServer:
    """Construct and configure the authoritative CRISPE-AI MCP server instance.

    Pure factory function: does NOT open network ports, connect to databases,
    or dispatch external API calls during construction.
    """
    cfg = config or MCPServerConfig.from_settings()
    ctx = context or MCPAppContext()

    server = MCPServer(
        name=cfg.server_name,
        title=cfg.title,
        description=cfg.description,
        instructions=cfg.instructions,
        version=cfg.server_version,
    )

    # Attach application context to server for tool access
    setattr(server, "app_context", ctx)
    setattr(server, "server_config", cfg)

    # Register initial foundation capabilities
    if cfg.enable_smoke_tools:
        register_smoke_tools(server=server, config=cfg, context=ctx)

    # Register CRISPE prompt generation tools (Step 30)
    if cfg.enable_prompt_tools:
        register_generation_tools(server=server, config=cfg, context=ctx)

    # Register semantic prompt memory tools (Step 30)
    if cfg.enable_memory_tools:
        register_memory_tools(server=server, config=cfg, context=ctx)

    logger.debug(
        "Constructed MCP server '%s' v%s (smoke_tools=%s, prompt_tools=%s, memory_tools=%s)",
        cfg.server_name,
        cfg.server_version,
        cfg.enable_smoke_tools,
        cfg.enable_prompt_tools,
        cfg.enable_memory_tools,
    )
    return server


async def run_stdio(
    config: Optional[MCPServerConfig] = None,
    context: Optional[MCPAppContext] = None,
) -> None:
    """Execute the CRISPE-AI MCP server over stdio transport."""
    cfg = config or MCPServerConfig.from_settings()
    configure_stdio_logging(cfg.log_level)

    server = create_mcp_server(config=cfg, context=context)
    logger.info("CRISPE-AI MCP server initialized on stdio transport (name: %s)", cfg.server_name)

    await server.run_stdio_async()


def main() -> None:
    """Main CLI entrypoint for running the CRISPE-AI MCP server via standard I/O."""
    import asyncio
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
