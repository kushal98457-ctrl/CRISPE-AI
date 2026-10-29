"""Configuration settings for the CRISPE-AI MCP Server Foundation (Step 29).

Provides explicit, validated configuration for server identification, capabilities,
and transport behavior without hidden import-time side effects.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings, get_settings


class MCPServerConfig(BaseModel):
    """Configuration contract governing MCP server construction and capability exposure."""

    model_config = ConfigDict(extra="ignore")

    server_name: str = Field(
        default="crispe-ai-mcp",
        min_length=1,
        max_length=100,
        description="Public protocol name advertised to MCP clients.",
    )
    server_version: str = Field(
        default="0.1.0",
        min_length=1,
        max_length=50,
        description="Semantic version string for the MCP server.",
    )
    title: str = Field(
        default="CRISPE-AI Model Context Protocol Server",
        description="Human-readable title displayed in client directories.",
    )
    description: str = Field(
        default="Enterprise CRISPE prompt generation, semantic prompt memory, and self-evolving engine",
        description="Server capability description published to connecting clients.",
    )
    instructions: Optional[str] = Field(
        default=(
            "CRISPE-AI provides AI-assisted prompt engineering, semantic prompt memory, "
            "and self-evolving prompt optimization tools adhering to the CRISPE framework."
        ),
        description="System prompt guidance supplied to MCP client reasoning models.",
    )
    enable_smoke_tools: bool = Field(
        default=True,
        description="Whether to register minimal smoke test capabilities (ping, server_info).",
    )
    enable_prompt_tools: bool = Field(
        default=True,
        description="Whether to register CRISPE prompt generation tools (generate_crispe_prompt).",
    )
    enable_memory_tools: bool = Field(
        default=True,
        description="Whether to register semantic prompt memory tools (search_prompt_memory, save_prompt_memory).",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for stdio transport diagnostics (emitted to stderr).",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Default timeout ceiling for bounded MCP tool operations.",
    )

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "MCPServerConfig":
        """Construct server configuration from global application settings safely."""
        cfg = settings or get_settings()
        return cls(
            server_name=cfg.MCP_SERVER_NAME,
            server_version=cfg.APP_VERSION,
            title=cfg.MCP_SERVER_TITLE,
            description=cfg.MCP_SERVER_DESCRIPTION,
            enable_smoke_tools=cfg.MCP_ENABLE_SMOKE_TOOLS,
            log_level=cfg.MCP_STDIO_LOG_LEVEL,
        )
