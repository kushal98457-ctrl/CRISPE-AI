"""Pydantic schemas and typed data contracts for Semantic Prompt Memory embeddings."""

from enum import Enum
import math
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingTaskType(str, Enum):
    """Supported embedding task types aligned with Google GenAI text-embedding-004."""

    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"


class EmbeddingRequest(BaseModel):
    """Input payload contract for generating a single text embedding vector."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(
        ...,
        min_length=1,
        max_length=16000,
        description="Text content to embed (engineering prompt, query, or few-shot example).",
    )
    task_type: EmbeddingTaskType = Field(
        default=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        description="Task type guiding the embedding optimization (document indexing vs query retrieval).",
    )
    title: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional title parameter for document embeddings in asymmetric retrieval.",
    )
    output_dimensionality: Optional[int] = Field(
        default=None,
        description="Optional Matryoshka output dimension truncation (128, 256, 512, 768).",
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        """Reject empty or whitespace-only inputs."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Embedding input text cannot be empty or whitespace only.")
        return stripped


class EmbeddingBatchRequest(BaseModel):
    """Input payload contract for generating batch text embeddings."""

    model_config = ConfigDict(extra="ignore")

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text strings to embed in a single batch operation (1-100 items).",
    )
    task_type: EmbeddingTaskType = Field(
        default=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        description="Task type for all items in the batch.",
    )
    output_dimensionality: Optional[int] = Field(
        default=None,
        description="Optional Matryoshka output dimension truncation.",
    )

    @field_validator("texts")
    @classmethod
    def validate_batch_texts(cls, values: List[str]) -> List[str]:
        """Validate each item in the batch is non-empty."""
        cleaned = []
        for i, text in enumerate(values):
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"Batch item at index {i} cannot be empty or whitespace only.")
            cleaned.append(stripped)
        return cleaned


class EmbeddingUsageMetadata(BaseModel):
    """Token consumption telemetry for embedding requests."""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class EmbeddingResult(BaseModel):
    """Typed result contract for a generated text embedding vector."""

    model_config = ConfigDict(extra="ignore")

    vector: List[float] = Field(..., description="High-dimensional floating-point embedding vector.")
    dimension: int = Field(..., ge=1, description="Length of the embedding vector.")
    model_id: str = Field(..., description="Model checkpoint that produced the vector (e.g. text-embedding-004).")
    config_version: str = Field(..., description="Configuration version tag distinguishing incompatible vector spaces.")
    usage: Optional[EmbeddingUsageMetadata] = Field(default=None, description="Token usage telemetry if available.")

    @field_validator("vector")
    @classmethod
    def validate_vector_values(cls, values: List[float]) -> List[float]:
        """Verify vector is non-empty and contains only finite real numbers."""
        if not values:
            raise ValueError("Embedding vector cannot be empty.")
        for i, val in enumerate(values):
            if not math.isfinite(val):
                raise ValueError(f"Embedding vector element at index {i} is not a finite number: {val}")
        return values

    @field_validator("dimension")
    @classmethod
    def validate_dimension_match(cls, value: int, info: Any) -> int:
        """Verify dimension matches vector length."""
        # Will be validated post-init if vector is present
        return value

    def model_post_init(self, __context: Any) -> None:
        """Enforce strict equality between declared dimension and vector length."""
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Declared dimension ({self.dimension}) does not match vector length ({len(self.vector)})."
            )


class EmbeddingBatchResult(BaseModel):
    """Typed result contract for a batch embedding operation."""

    model_config = ConfigDict(extra="ignore")

    embeddings: List[EmbeddingResult] = Field(..., description="List of generated embedding results.")
    model_id: str = Field(..., description="Model identifier used for the batch.")
    config_version: str = Field(..., description="Configuration version tag.")
    total_tokens: Optional[int] = Field(default=None, ge=0, description="Total tokens consumed across the batch.")
