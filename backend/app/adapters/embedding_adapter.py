"""Google AI Studio / GenAI embedding adapter for CRISPE-AI Semantic Memory.

Provides an asynchronous, resilient, and typed interface to the Google GenAI SDK
for generating text embedding vectors using official models like text-embedding-004.
"""

from abc import ABC, abstractmethod
import asyncio
import math
import random
from typing import Any, List, Optional
import httpx

import google.genai as genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    InvalidProviderResponseError,
    MissingProviderCredentialError,
    ModelNotFoundError,
    ProviderAPIError,
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderError,
    ProviderPermissionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.core.logging import get_logger
from app.schemas.embedding import (
    EmbeddingBatchResult,
    EmbeddingResult,
    EmbeddingTaskType,
    EmbeddingUsageMetadata,
)

logger = get_logger("embedding_adapter")


class BaseEmbeddingAdapter(ABC):
    """Abstract interface for text embedding provider adapters."""

    @abstractmethod
    async def embed_content(
        self,
        text: str,
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbeddingResult:
        """Generate an embedding vector for a single text."""
        raise NotImplementedError

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbeddingBatchResult:
        """Generate embedding vectors for a batch of texts."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release underlying client resources and network connections."""
        raise NotImplementedError


class GoogleGenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """Production embedding adapter using the official Google GenAI Python SDK."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[genai.Client] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    def _get_or_create_client(self) -> genai.Client:
        """Lazily initialize the official Google GenAI Client."""
        if self._client is not None:
            return self._client

        try:
            raw_key = self.settings.get_embedding_api_key()
        except MissingProviderCredentialError as exc:
            logger.error("Embedding adapter cannot initialize client: missing credential.")
            raise ProviderConfigError(
                "Gemini/Google API credential is not configured for embeddings. "
                "Please set GEMINI_API_KEY (or EMBEDDING_API_KEY) in the environment.",
                original_error=exc,
            ) from exc

        try:
            self._client = genai.Client(api_key=raw_key)
            return self._client
        except Exception as exc:
            logger.error("Failed to construct google-genai Client for embeddings: %s", str(exc))
            raise ProviderConfigError(
                f"Failed to instantiate Google GenAI client for embeddings: {str(exc)}",
                original_error=exc,
            ) from exc

    def _map_provider_error(self, exc: Exception) -> ProviderError:
        """Translate upstream Google GenAI and HTTP exceptions into domain ProviderErrors."""
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            message = getattr(exc, "message", str(exc))

            if code == 401 or "API_KEY_INVALID" in message or "UNAUTHENTICATED" in message:
                return ProviderAuthenticationError(
                    f"Gemini embedding authentication failed (HTTP 401): {message}",
                    original_error=exc,
                )
            if code == 403 or "PERMISSION_DENIED" in message:
                return ProviderPermissionError(
                    f"Gemini embedding permission denied (HTTP 403): {message}",
                    original_error=exc,
                )
            if code == 404 or "NOT_FOUND" in message:
                return ModelNotFoundError(
                    f"Specified embedding model was not found (HTTP 404): {message}",
                    original_error=exc,
                )
            if code == 429 or "RESOURCE_EXHAUSTED" in message or "RATE_LIMIT_EXCEEDED" in message:
                return ProviderRateLimitError(
                    f"Gemini embedding quota or rate limit exceeded (HTTP 429): {message}",
                    original_error=exc,
                )
            if code in (408, 504) or "DEADLINE_EXCEEDED" in message:
                return ProviderTimeoutError(
                    f"Gemini embedding request timed out (HTTP {code}): {message}",
                    original_error=exc,
                )
            if code in (500, 502, 503) or isinstance(exc, genai_errors.ServerError):
                return ProviderUnavailableError(
                    f"Gemini embedding service temporarily unavailable (HTTP {code}): {message}",
                    original_error=exc,
                )
            return ProviderAPIError(
                f"Gemini embedding API error (HTTP {code}): {message}",
                status_code=code or 502,
                original_error=exc,
            )

        if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
            return ProviderTimeoutError(
                f"Gemini embedding request timed out: {str(exc)}",
                original_error=exc,
            )
        if isinstance(exc, httpx.NetworkError):
            return ProviderUnavailableError(
                f"Network failure connecting to Gemini embedding provider: {str(exc)}",
                original_error=exc,
            )
        if isinstance(exc, ProviderError):
            return exc

        return ProviderAPIError(
            f"Unexpected embedding provider exception: {str(exc)}",
            status_code=500,
            original_error=exc,
        )

    async def _execute_with_retry(self, coroutine_func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute an asynchronous embedding provider call with exponential backoff on transient errors."""
        max_retries = self.settings.GEMINI_MAX_RETRIES
        base_delay = 0.5
        attempt = 0

        while True:
            attempt += 1
            try:
                timeout_sec = self.settings.EMBEDDING_TIMEOUT_SECONDS
                return await asyncio.wait_for(
                    coroutine_func(*args, **kwargs),
                    timeout=timeout_sec,
                )
            except Exception as raw_exc:
                mapped_error = self._map_provider_error(raw_exc)

                if mapped_error.is_retryable and attempt <= max_retries:
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.05, 0.25)
                    logger.warning(
                        "Transient embedding error on attempt %d/%d (%s). Retrying in %.2fs...",
                        attempt,
                        max_retries,
                        mapped_error.error_code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if attempt > max_retries and mapped_error.is_retryable:
                    logger.error(
                        "Embedding request exhausted %d retry attempts. Final error: %s",
                        max_retries,
                        mapped_error.message,
                    )
                raise mapped_error from raw_exc

    def _validate_vector_data(
        self,
        vector: Any,
        expected_dim: int,
    ) -> List[float]:
        """Verify vector data validity and ensure it contains finite real numbers."""
        if not vector or not isinstance(vector, (list, tuple)):
            raise InvalidProviderResponseError("Embedding provider returned empty or non-sequence vector.")

        actual_len = len(vector)
        if actual_len != expected_dim:
            raise InvalidProviderResponseError(
                f"Embedding vector dimension mismatch: expected {expected_dim}, got {actual_len}."
            )

        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise InvalidProviderResponseError(
                    f"Embedding vector contains invalid or non-finite value at index {i}: {val}."
                )

        return [float(v) for v in vector]

    async def embed_content(
        self,
        text: str,
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbeddingResult:
        """Generate an embedding vector for a single text string."""
        client = self._get_or_create_client()
        target_dim = output_dimensionality or self.settings.EMBEDDING_DIMENSIONS
        effective_task = task_type.value if task_type else "RETRIEVAL_DOCUMENT"

        config = genai_types.EmbedContentConfig(
            task_type=effective_task,
            title=title,
            output_dimensionality=target_dim,
        )

        model_id = self.settings.EMBEDDING_MODEL_ID
        logger.debug(
            "Requesting embedding from model '%s' (dim: %d, task: %s)",
            model_id,
            target_dim,
            effective_task,
        )

        response = await self._execute_with_retry(
            client.aio.models.embed_content,
            model=model_id,
            contents=text,
            config=config,
        )

        if not response or not hasattr(response, "embeddings") or not response.embeddings:
            raise InvalidProviderResponseError(
                "Gemini embedding provider returned empty response without embeddings."
            )

        first_embedding = response.embeddings[0]
        raw_values = getattr(first_embedding, "values", None)
        validated_vector = self._validate_vector_data(raw_values, target_dim)

        # Extract usage telemetry if available
        usage_metadata = None
        if hasattr(response, "metadata") and response.metadata:
            billable_tokens = getattr(response.metadata, "billable_character_count", 0)
            usage_metadata = EmbeddingUsageMetadata(
                prompt_tokens=billable_tokens or len(text) // 4,
                total_tokens=billable_tokens or len(text) // 4,
            )

        return EmbeddingResult(
            vector=validated_vector,
            dimension=target_dim,
            model_id=model_id,
            config_version=self.settings.EMBEDDING_CONFIG_VERSION,
            usage=usage_metadata,
        )

    async def embed_batch(
        self,
        texts: List[str],
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbeddingBatchResult:
        """Generate embedding vectors for a batch of text strings."""
        if not texts:
            raise InvalidProviderResponseError("Batch texts list cannot be empty.")

        client = self._get_or_create_client()
        target_dim = output_dimensionality or self.settings.EMBEDDING_DIMENSIONS
        effective_task = task_type.value if task_type else "RETRIEVAL_DOCUMENT"

        config = genai_types.EmbedContentConfig(
            task_type=effective_task,
            output_dimensionality=target_dim,
        )

        model_id = self.settings.EMBEDDING_MODEL_ID
        logger.debug(
            "Requesting batch embeddings (%d texts) from model '%s' (dim: %d)",
            len(texts),
            model_id,
            target_dim,
        )

        response = await self._execute_with_retry(
            client.aio.models.embed_content,
            model=model_id,
            contents=texts,
            config=config,
        )

        if not response or not hasattr(response, "embeddings") or not response.embeddings:
            raise InvalidProviderResponseError(
                "Gemini batch embedding provider returned empty response without embeddings."
            )

        if len(response.embeddings) != len(texts):
            raise InvalidProviderResponseError(
                f"Batch embedding count mismatch: sent {len(texts)} texts, "
                f"received {len(response.embeddings)} embeddings."
            )

        results: List[EmbeddingResult] = []
        for i, item in enumerate(response.embeddings):
            raw_values = getattr(item, "values", None)
            valid_vec = self._validate_vector_data(raw_values, target_dim)
            results.append(
                EmbeddingResult(
                    vector=valid_vec,
                    dimension=target_dim,
                    model_id=model_id,
                    config_version=self.settings.EMBEDDING_CONFIG_VERSION,
                )
            )

        return EmbeddingBatchResult(
            embeddings=results,
            model_id=model_id,
            config_version=self.settings.EMBEDDING_CONFIG_VERSION,
        )

    async def close(self) -> None:
        """Close underlying client connections."""
        if self._client is not None:
            # google-genai Client lifecycle
            self._client = None
            logger.info("Closed embedding provider adapter client.")


def get_embedding_adapter(settings: Optional[Settings] = None) -> GoogleGenAIEmbeddingAdapter:
    """Factory creating an embedding provider adapter configured from settings."""
    return GoogleGenAIEmbeddingAdapter(settings=settings)
