"""Data contracts and response schemas for CRISPE-AI MCP Server (Steps 29, 30).

Enforces strictly typed payloads for protocol metadata, CRISPE prompt generation,
semantic prompt memory retrieval, and persistent memory storage.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.crispe import CrispeSections, ModelUsageMetadata


# =============================================================================
# Foundation & Smoke Schemas (Step 29)
# =============================================================================

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


# =============================================================================
# CRISPE Prompt Generation Schemas (Step 30)
# =============================================================================

class GenerateCrispePromptInput(BaseModel):
    """Input payload for generating a structured CRISPE framework prompt."""

    model_config = ConfigDict(extra="forbid")

    idea: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="Raw prompt idea or user task description to transform into CRISPE framework.",
    )
    context: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Optional background context or domain environment.",
    )
    role: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional persona, professional perspective, or tone.",
    )
    instruction: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Optional step-by-step commands or primary task instructions.",
    )
    specification: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Optional formatting rules, constraints, and negative boundaries.",
    )
    performance: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional success benchmarks and evaluation standards.",
    )
    example: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Optional few-shot demonstrations or expected output pattern.",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional generation model override (e.g. gemma-4-26b-a4b-it, gemini-2.5-flash).",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional model sampling temperature override.",
    )
    thinking_budget: Optional[int] = Field(
        default=None,
        ge=0,
        le=32768,
        description="Optional reasoning token budget override.",
    )

    @model_validator(mode="after")
    def validate_content_provided(self) -> "GenerateCrispePromptInput":
        """Ensure that at least an idea or instruction is supplied."""
        has_idea = bool(self.idea and self.idea.strip())
        has_instruction = bool(self.instruction and self.instruction.strip())
        has_any_section = any([
            self.context and self.context.strip(),
            self.role and self.role.strip(),
            self.specification and self.specification.strip(),
        ])
        if not has_idea and not has_instruction and not has_any_section:
            raise ValueError("At least an 'idea' or CRISPE sections must be provided.")
        return self


class GenerateCrispePromptOutput(BaseModel):
    """Structured response containing synthesized CRISPE prompt and metadata."""

    model_config = ConfigDict(extra="ignore")

    success: bool = Field(default=True, description="Generation success status.")
    idea: str = Field(..., description="Original user prompt idea.")
    sections: CrispeSections = Field(..., description="Six canonical CRISPE sections.")
    formatted_prompt: str = Field(..., description="Complete Markdown-rendered CRISPE prompt.")
    model_id: str = Field(..., description="Model checkpoint used for generation.")
    usage: Optional[ModelUsageMetadata] = Field(default=None, description="Token consumption metrics.")
    finish_reason: Optional[str] = Field(default=None, description="Completion finish reason.")
    created_at: str = Field(..., description="UTC ISO timestamp of synthesis.")


# =============================================================================
# Semantic Prompt Memory Schemas (Step 30)
# =============================================================================

class SearchPromptMemoryInput(BaseModel):
    """Input payload for retrieving semantically relevant prompts from memory."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language search query to retrieve semantically related saved prompts.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of nearest neighbor prompts to retrieve (1 to 10).",
    )
    task_type: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional category or domain classification filter.",
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold (0.0 to 1.0).",
    )


class PromptMemoryMatchItem(BaseModel):
    """Reference representation of a single matching prompt record."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str = Field(..., description="Unique UUID identifier of the stored prompt.")
    original_idea: str = Field(..., description="Original idea behind the prompt.")
    formatted_prompt: str = Field(..., description="Canonical rendered Markdown prompt.")
    sections: CrispeSections = Field(..., description="Six CRISPE sections.")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0).")
    score_metric: str = Field(
        default="cosine_similarity",
        description="Metric used for similarity evaluation.",
    )
    task_type: Optional[str] = Field(default=None, description="Domain or task category tag.")
    model_id: str = Field(..., description="Model used when the prompt was synthesized.")
    created_at: str = Field(..., description="UTC timestamp when the prompt was saved.")


class SearchPromptMemoryOutput(BaseModel):
    """Response containing ordered matching prompt records from semantic memory."""

    model_config = ConfigDict(extra="ignore")

    success: bool = Field(default=True, description="Query execution status.")
    query: str = Field(..., description="Search query executed.")
    total_matches: int = Field(..., ge=0, description="Total matching prompt records returned.")
    results: List[PromptMemoryMatchItem] = Field(
        default_factory=list,
        description="Ordered list of nearest matching prompt records.",
    )
    owner_id: str = Field(..., description="Security principal memory scope queried.")


class SavePromptMemoryInput(BaseModel):
    """Input payload to explicitly store a CRISPE prompt and generate its embedding."""

    model_config = ConfigDict(extra="forbid")

    idea: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The task description or original idea for the prompt.",
    )
    context: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Context section: Operational background and environment.",
    )
    role: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Role section: Precise AI persona and authority.",
    )
    instruction: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Instruction section: Explicit task commands and actions.",
    )
    specification: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Specification section: Rules, constraints, and negative boundaries.",
    )
    performance: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Performance section: Quality benchmarks and success criteria.",
    )
    example: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Example section: Demonstration and illustrative sample.",
    )
    task_type: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional category or domain classification tag.",
    )
    model_id: Optional[str] = Field(
        default="gemma-4-26b-a4b-it",
        max_length=128,
        description="Model identifier that generated the prompt.",
    )


class SavePromptMemoryOutput(BaseModel):
    """Result of saving a prompt to persistent semantic memory."""

    model_config = ConfigDict(extra="ignore")

    success: bool = Field(default=True, description="Persistence status.")
    prompt_id: str = Field(..., description="Unique UUID identifier assigned to the saved prompt.")
    status: str = Field(..., description="Persistence outcome ('created' or 'already_exists').")
    owner_id: str = Field(..., description="Security principal that owns this prompt record.")
    task_type: Optional[str] = Field(default=None, description="Assigned task category.")
    created_at: str = Field(..., description="UTC timestamp of persistence.")
