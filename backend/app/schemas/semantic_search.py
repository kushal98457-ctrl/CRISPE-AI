"""Data transfer objects and request/response contracts for Semantic Prompt Memory search."""

from datetime import datetime
from typing import List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.exceptions import EmptyPromptError, PromptValidationError
from app.schemas.crispe import CrispeSections

# Maximum allowable characters for semantic search query text
MAX_SEARCH_QUERY_LENGTH = 1000


class SemanticSearchQuery(BaseModel):
    """Query parameters for semantic prompt similarity search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SEARCH_QUERY_LENGTH,
        description="Search text or task description to match against saved prompt embeddings.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of nearest neighbor results to return (1-50).",
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum derived cosine similarity threshold (1.0 - distance).",
    )
    task_type: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional task classification or domain category filter.",
    )
    config_version: str = Field(
        default="v1",
        max_length=32,
        description="Target embedding configuration version distinguishing incompatible vector spaces.",
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_and_clean_query(cls, value: str) -> str:
        """Validate query text is non-empty after whitespace trimming."""
        if not isinstance(value, str):
            raise PromptValidationError(f"Query must be a string, got {type(value).__name__}.")
        cleaned = value.strip()
        if not cleaned:
            raise EmptyPromptError("Search query cannot be empty or whitespace only.")
        if len(cleaned) > MAX_SEARCH_QUERY_LENGTH:
            raise PromptValidationError(
                f"Search query exceeds maximum permitted length of {MAX_SEARCH_QUERY_LENGTH} characters."
            )
        return cleaned


class SemanticSearchItem(BaseModel):
    """Structured representation of a single matching prompt retrieved via semantic search.

    Raw vector embeddings are strictly excluded from search items to prevent unnecessary
    bandwidth overhead and avoid leaking internal vector representations.
    """

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    prompt_id: uuid.UUID = Field(
        ...,
        description="Unique identifier of the matching persistent prompt record.",
    )
    original_idea: str = Field(
        ...,
        description="The original user idea or task description of the saved prompt.",
    )
    sections: CrispeSections = Field(
        ...,
        description="The structured six-section CRISPE prompt definition.",
    )
    formatted_prompt: str = Field(
        ...,
        description="Rendered canonical Markdown prompt matching UI headers.",
    )
    task_type: Optional[str] = Field(
        default=None,
        description="Optional classification or domain category tag.",
    )
    model_id: str = Field(
        ...,
        description="Embedding model checkpoint identifier associated with the matched vector.",
    )
    distance: float = Field(
        ...,
        ge=0.0,
        description="Calculated vector distance using the configured metric (0.0 = identical).",
    )
    similarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Derived cosine similarity score (1.0 - distance).",
    )
    config_version: str = Field(
        ...,
        description="Embedding configuration version of the matched vector.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the parent prompt record was created.",
    )


class SemanticSearchResponse(BaseModel):
    """Complete response payload returned by the Semantic Search service."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        description="The normalized search query text executed.",
    )
    total_results: int = Field(
        ...,
        ge=0,
        description="Total number of matching prompt records returned.",
    )
    items: List[SemanticSearchItem] = Field(
        default_factory=list,
        description="List of nearest neighbor prompt matches ordered by relevance.",
    )
    model_id: str = Field(
        ...,
        description="Embedding model checkpoint identifier used for query embedding.",
    )
    config_version: str = Field(
        ...,
        description="Configuration version used for semantic retrieval.",
    )
    search_metric: str = Field(
        default="cosine_distance",
        description="Underlying vector distance operator applied in PostgreSQL.",
    )
