"""Data contracts and response schemas for CRISPE-AI MCP Server Foundation (Step 29).

Enforces strictly typed payloads for protocol metadata, tool status, and diagnostic results.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PingResponse(BaseModel):
    """Payload returned by the minimal smoke-test ping capability."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok", description="Server status indicator ('ok').")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the ping response.",
    )


class ServerMetadataResponse(BaseModel):
    """Authoritative server information and supported capabilities for MCP clients."""

    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(..., description="Unique advertised MCP server name.")
    server_version: str = Field(..., description="Server build version.")
    title: str = Field(..., description="Human-readable title.")
    description: str = Field(..., description="Brief summary of server purpose.")
    protocol_status: str = Field(
        default="ready",
        description="Protocol lifecycle readiness ('ready', 'initializing', 'maintenance').",
    )
    supported_capabilities: List[str] = Field(
        default_factory=lambda: ["tools"],
        description="Active MCP feature domains supported by this server.",
    )
    environment: str = Field(default="production", description="Server environment tier.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC generation timestamp.",
    )


class ErrorEnvelope(BaseModel):
    """Structured error envelope returned when a tool invocation fails safely."""

    model_config = ConfigDict(extra="forbid")

    error: bool = Field(default=True, description="Always true for error envelopes.")
    code: str = Field(..., description="Machine-readable error category code.")
    message: str = Field(..., description="Sanitized, client-safe error description.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Non-sensitive error diagnostics.")
