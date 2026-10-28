"""CRISPE prompt generation schemas and data contracts."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CrispeSections(BaseModel):
    """The six structured sections of a CRISPE framework prompt."""

    model_config = ConfigDict(extra="ignore")

    context: str = Field(
        ...,
        description=(
            "Context: The background situation, domain circumstances, user environment, "
            "and business context necessary for the AI to understand the setting."
        ),
    )
    role: str = Field(
        ...,
        description=(
            "Role: The persona, authority, perspective, professional identity, and tone "
            "that the AI must embody."
        ),
    )
    instruction: str = Field(
        ...,
        description=(
            "Instruction: The primary task, explicit actions, step-by-step commands, "
            "and direct instructions the AI is directed to carry out."
        ),
    )
    specification: str = Field(
        ...,
        description=(
            "Specification: Constraints, formatting rules, length guidelines, structural requirements, "
            "and negative boundaries (what to avoid doing)."
        ),
    )
    performance: str = Field(
        ...,
        description=(
            "Performance: Evaluation criteria, success benchmarks, qualitative standards, "
            "and output quality metrics."
        ),
    )
    example: str = Field(
        ...,
        description=(
            "Example: Concrete, illustrative sample input/output pairs, demonstrations, "
            "or reference examples showing the desired execution pattern."
        ),
    )

    @field_validator("context", "role", "instruction", "specification", "performance", "example", mode="before")
    @classmethod
    def clean_section(cls, value: str) -> str:
        """Ensure section content is cleanly trimmed and non-empty."""
        if not isinstance(value, str):
            return str(value)
        return value.strip()

    def to_markdown(self) -> str:
        """Render the structured sections into the canonical CRISPE Markdown prompt format.
        
        Matches the headings used by the existing CRISPE-AI user interface:
        # C - Context
        # R - Role
        # I - Instruction
        # S - Specification
        # P - Performance
        # E - Example
        """
        return (
            f"# C - Context\n{self.context}\n\n"
            f"# R - Role\n{self.role}\n\n"
            f"# I - Instruction\n{self.instruction}\n\n"
            f"# S - Specification\n{self.specification}\n\n"
            f"# P - Performance\n{self.performance}\n\n"
            f"# E - Example\n{self.example}"
        )


class ModelUsageMetadata(BaseModel):
    """Token consumption and usage metadata from provider response."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0, description="Tokens used by the prompt and system instructions.")
    candidates_tokens: int = Field(default=0, ge=0, description="Tokens generated in candidate output.")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed.")
    thoughts_tokens: Optional[int] = Field(default=None, ge=0, description="Tokens used for reasoning/thinking.")


class CrispeGenerationRequest(BaseModel):
    """Request payload for generating a structured CRISPE prompt."""

    model_config = ConfigDict(extra="ignore")

    idea: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The raw prompt or initial user idea to transform into CRISPE framework.",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional model override (e.g. gemma-4-26b-a4b-it or gemini-2.5-flash).",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional sampling temperature override.",
    )
    thinking_budget: Optional[int] = Field(
        default=None,
        ge=0,
        le=32768,
        description="Optional thinking token budget override.",
    )

    @field_validator("idea", mode="before")
    @classmethod
    def validate_idea(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Prompt idea cannot be empty or whitespace.")
        return v.strip()


class CrispeGenerationResponse(BaseModel):
    """Response payload containing the structured CRISPE prompt and metadata."""

    model_config = ConfigDict(extra="ignore")

    idea: str = Field(..., description="Original user idea.")
    sections: CrispeSections = Field(..., description="Six structured sections of the CRISPE framework.")
    formatted_prompt: str = Field(..., description="Rendered Markdown CRISPE prompt ready for use.")
    model_id: str = Field(..., description="Model identifier used for generation.")
    usage: Optional[ModelUsageMetadata] = Field(default=None, description="Token usage metadata.")
    finish_reason: Optional[str] = Field(default=None, description="Completion finish reason.")
    request_id: Optional[str] = Field(default=None, description="Request correlation identifier.")
    schema_version: str = Field(default="1.0", description="API contract schema version.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of generation.",
    )
