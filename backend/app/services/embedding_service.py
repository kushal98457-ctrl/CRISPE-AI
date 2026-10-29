"""Semantic Memory embedding generation service and vector validation pipeline.

Provides high-level text embedding generation for CRISPE-AI Semantic Prompt Memory,
handling input validation, Unicode normalization, provider delegation, and strict
vector dimension and mathematical invariant validation.
"""

import math
from typing import List, Optional

from app.adapters.embedding_adapter import (
    BaseEmbeddingAdapter,
    get_embedding_adapter,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    EmptyPromptError,
    InvalidProviderResponseError,
    PromptValidationError,
)
from app.core.logging import get_logger
from app.schemas.embedding import (
    EmbeddingBatchResult,
    EmbeddingResult,
    EmbeddingTaskType,
)

logger = get_logger("embedding_service")

# Documented context limit for text-embedding-004 is 2048 tokens.
# At ~4-8 characters per token, 16000 characters is a safe boundary.
MAX_EMBEDDING_TEXT_CHARS = 16000


class EmbeddingService:
    """Service orchestrating text embedding generation and vector validation for Semantic Memory."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        adapter: Optional[BaseEmbeddingAdapter] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._adapter = adapter

    @property
    def adapter(self) -> BaseEmbeddingAdapter:
        """Lazily initialize the configured embedding adapter."""
        if self._adapter is None:
            self._adapter = get_embedding_adapter(self.settings)
        return self._adapter

    def validate_and_normalize_input(self, text: str) -> str:
        """Validate input text type, length, and content, returning a clean normalized string."""
        if not isinstance(text, str):
            raise PromptValidationError(
                f"Embedding input must be a string, got {type(text).__name__}."
            )

        stripped = text.strip()
        if not stripped:
            raise EmptyPromptError("Embedding input text cannot be empty or whitespace only.")

        if len(stripped) > MAX_EMBEDDING_TEXT_CHARS:
            raise PromptValidationError(
                f"Embedding input text exceeds maximum length limit of "
                f"{MAX_EMBEDDING_TEXT_CHARS} characters (received {len(stripped)})."
            )

        return stripped

    def validate_vector(self, result: EmbeddingResult) -> None:
        """Strictly validate vector invariants, dimensions, and mathematical finiteness."""
        expected_dim = self.settings.EMBEDDING_DIMENSIONS
        if result.dimension != expected_dim or len(result.vector) != expected_dim:
            raise InvalidProviderResponseError(
                f"Vector dimension mismatch: expected {expected_dim}, got {result.dimension}."
            )

        if not result.vector:
            raise InvalidProviderResponseError("Vector cannot be empty.")

        for i, val in enumerate(result.vector):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise InvalidProviderResponseError(
                    f"Vector contains invalid or non-finite value at index {i}: {val}."
                )

        if result.model_id != self.settings.EMBEDDING_MODEL_ID:
            raise InvalidProviderResponseError(
                f"Vector model mismatch: expected '{self.settings.EMBEDDING_MODEL_ID}', "
                f"got '{result.model_id}'."
            )

    async def embed_text(
        self,
        text: str,
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        title: Optional[str] = None,
    ) -> EmbeddingResult:
        """Validate, normalize, and generate an embedding vector for a single text."""
        normalized_text = self.validate_and_normalize_input(text)

        result = await self.adapter.embed_content(
            normalized_text,
            task_type=task_type,
            title=title,
            output_dimensionality=self.settings.EMBEDDING_DIMENSIONS,
        )

        self.validate_vector(result)
        logger.info(
            "Successfully embedded text (%d chars) into %d-dim vector [model: %s, v: %s]",
            len(normalized_text),
            result.dimension,
            result.model_id,
            result.config_version,
        )
        return result

    async def embed_batch(
        self,
        texts: List[str],
        *,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingBatchResult:
        """Validate, normalize, and generate embedding vectors for a batch of texts."""
        if not texts or not isinstance(texts, list):
            raise PromptValidationError("Batch embedding requires a non-empty list of text strings.")

        max_batch = self.settings.EMBEDDING_BATCH_SIZE
        if len(texts) > max_batch:
            raise PromptValidationError(
                f"Batch size of {len(texts)} exceeds maximum permitted batch size of {max_batch}."
            )

        normalized_texts = [self.validate_and_normalize_input(t) for t in texts]

        batch_result = await self.adapter.embed_batch(
            normalized_texts,
            task_type=task_type,
            output_dimensionality=self.settings.EMBEDDING_DIMENSIONS,
        )

        for res in batch_result.embeddings:
            self.validate_vector(res)

        logger.info(
            "Successfully embedded batch of %d texts into %d-dim vectors [model: %s]",
            len(normalized_texts),
            self.settings.EMBEDDING_DIMENSIONS,
            batch_result.model_id,
        )
        return batch_result

    async def close(self) -> None:
        """Release underlying adapter resources."""
        if self._adapter is not None:
            await self._adapter.close()


def get_embedding_service(
    settings: Optional[Settings] = None,
    adapter: Optional[BaseEmbeddingAdapter] = None,
) -> EmbeddingService:
    """Factory creating an EmbeddingService instance."""
    return EmbeddingService(settings=settings, adapter=adapter)


def get_embedding_service_dependency() -> EmbeddingService:
    """FastAPI parameterless dependency provider for EmbeddingService."""
    return get_embedding_service()
