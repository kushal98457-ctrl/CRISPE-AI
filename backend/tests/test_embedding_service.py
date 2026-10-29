"""Comprehensive automated tests for EmbeddingService and GoogleGenAIEmbeddingAdapter (Step 13)."""

import math
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pydantic import SecretStr

from google.genai import errors as genai_errors

from app.adapters.embedding_adapter import (
    BaseEmbeddingAdapter,
    GoogleGenAIEmbeddingAdapter,
)
from app.core.config import Settings
from app.core.exceptions import (
    EmptyPromptError,
    InvalidProviderResponseError,
    ModelNotFoundError,
    PromptValidationError,
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.schemas.embedding import (
    EmbeddingBatchResult,
    EmbeddingResult,
    EmbeddingTaskType,
    EmbeddingUsageMetadata,
)
from app.services.embedding_service import EmbeddingService


# -----------------------------------------------------------------------------
# Test Fixtures & Fakes
# -----------------------------------------------------------------------------

class FakeEmbeddingAdapter(BaseEmbeddingAdapter):
    """Deterministic in-memory embedding adapter for fast unit testing."""

    def __init__(
        self,
        dimension: int = 768,
        model_id: str = "text-embedding-004",
        config_version: str = "v1",
        simulate_error: Optional[Exception] = None,
        return_invalid_dim: Optional[int] = None,
        return_nan: bool = False,
        return_inf: bool = False,
    ) -> None:
        self.dimension = dimension
        self.model_id = model_id
        self.config_version = config_version
        self.simulate_error = simulate_error
        self.return_invalid_dim = return_invalid_dim
        self.return_nan = return_nan
        self.return_inf = return_inf
        self.closed = False
        self.calls: List[dict] = []

    async def embed_content(
        self,
        text: str,
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbeddingResult:
        self.calls.append({"text": text, "task_type": task_type, "title": title})

        if self.simulate_error:
            raise self.simulate_error

        dim = self.return_invalid_dim or self.dimension
        vec = [0.05 * (i % 10) for i in range(dim)]

        if self.return_nan and len(vec) > 0:
            vec[0] = float("nan")
        if self.return_inf and len(vec) > 0:
            vec[0] = float("inf")

        return EmbeddingResult(
            vector=vec,
            dimension=dim,
            model_id=self.model_id,
            config_version=self.config_version,
            usage=EmbeddingUsageMetadata(prompt_tokens=10, total_tokens=10),
        )

    async def embed_batch(
        self,
        texts: List[str],
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbeddingBatchResult:
        self.calls.append({"texts": texts, "task_type": task_type})

        if self.simulate_error:
            raise self.simulate_error

        embeddings = []
        for text in texts:
            dim = self.return_invalid_dim or self.dimension
            vec = [0.05 * (i % 10) for i in range(dim)]
            if self.return_nan and len(vec) > 0:
                vec[0] = float("nan")
            embeddings.append(
                EmbeddingResult(
                    vector=vec,
                    dimension=dim,
                    model_id=self.model_id,
                    config_version=self.config_version,
                )
            )

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model_id=self.model_id,
            config_version=self.config_version,
            total_tokens=len(texts) * 10,
        )

    async def close(self) -> None:
        self.closed = True


# -----------------------------------------------------------------------------
# A. Successful Embedding Service Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_single_text_success() -> None:
    """EmbeddingService successfully generates a validated 768-dim EmbeddingResult."""
    adapter = FakeEmbeddingAdapter(dimension=768)
    service = EmbeddingService(adapter=adapter)

    result = await service.embed_text(
        "Design a high-throughput OAuth 2.1 authentication gateway",
        task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
    )

    assert isinstance(result, EmbeddingResult)
    assert result.dimension == 768
    assert len(result.vector) == 768
    assert result.model_id == "text-embedding-004"
    assert result.config_version == "v1"
    assert all(math.isfinite(v) for v in result.vector)
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["task_type"] == EmbeddingTaskType.RETRIEVAL_DOCUMENT


@pytest.mark.asyncio
async def test_embed_batch_texts_success() -> None:
    """EmbeddingService generates validated batch embeddings matching input size."""
    adapter = FakeEmbeddingAdapter(dimension=768)
    service = EmbeddingService(adapter=adapter)

    batch_texts = [
        "First prompt for microservice security",
        "Second prompt for distributed cache invalidation",
        "Third prompt for pgvector similarity indexing",
    ]

    result = await service.embed_batch(
        batch_texts,
        task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
    )

    assert isinstance(result, EmbeddingBatchResult)
    assert len(result.embeddings) == 3
    assert all(r.dimension == 768 for r in result.embeddings)
    assert all(len(r.vector) == 768 for r in result.embeddings)
    assert result.total_tokens == 30


@pytest.mark.asyncio
async def test_unicode_and_technical_text_preserved() -> None:
    """Unicode, emojis, math equations, code snippets, and JSON are preserved."""
    adapter = FakeEmbeddingAdapter(dimension=768)
    service = EmbeddingService(adapter=adapter)

    technical_text = """
    # Architectural Formula: E = mc² 🚀
    def calculate_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    // JSON Schema
    {"type": "object", "properties": {"token": "∑(x_i)"}}
    """

    result = await service.embed_text(technical_text)
    assert result.dimension == 768
    assert adapter.calls[0]["text"] == technical_text.strip()


# -----------------------------------------------------------------------------
# B. Invalid Input Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_input_rejected() -> None:
    """Empty string raises EmptyPromptError before adapter invocation."""
    adapter = FakeEmbeddingAdapter()
    service = EmbeddingService(adapter=adapter)

    with pytest.raises(EmptyPromptError):
        await service.embed_text("")

    assert len(adapter.calls) == 0


@pytest.mark.asyncio
async def test_whitespace_only_input_rejected() -> None:
    """Whitespace-only input raises EmptyPromptError."""
    adapter = FakeEmbeddingAdapter()
    service = EmbeddingService(adapter=adapter)

    with pytest.raises(EmptyPromptError):
        await service.embed_text("   \n\t  ")

    assert len(adapter.calls) == 0


@pytest.mark.asyncio
async def test_non_string_input_rejected() -> None:
    """Non-string input raises PromptValidationError."""
    adapter = FakeEmbeddingAdapter()
    service = EmbeddingService(adapter=adapter)

    with pytest.raises(PromptValidationError):
        await service.embed_text(12345)  # type: ignore

    with pytest.raises(PromptValidationError):
        await service.embed_text(None)  # type: ignore


@pytest.mark.asyncio
async def test_oversized_input_rejected() -> None:
    """Text exceeding 16000 characters raises PromptValidationError."""
    adapter = FakeEmbeddingAdapter()
    service = EmbeddingService(adapter=adapter)

    oversized = "a" * 16001
    with pytest.raises(PromptValidationError) as exc_info:
        await service.embed_text(oversized)

    assert "exceeds maximum length limit" in str(exc_info.value)
    assert len(adapter.calls) == 0


@pytest.mark.asyncio
async def test_empty_batch_rejected() -> None:
    """Empty batch list raises PromptValidationError."""
    adapter = FakeEmbeddingAdapter()
    service = EmbeddingService(adapter=adapter)

    with pytest.raises(PromptValidationError):
        await service.embed_batch([])


@pytest.mark.asyncio
async def test_batch_exceeding_max_size_rejected() -> None:
    """Batch exceeding EMBEDDING_BATCH_SIZE (32) raises PromptValidationError."""
    adapter = FakeEmbeddingAdapter()
    service = EmbeddingService(adapter=adapter)

    large_batch = [f"Text item {i}" for i in range(35)]
    with pytest.raises(PromptValidationError) as exc_info:
        await service.embed_batch(large_batch)

    assert "exceeds maximum permitted batch size" in str(exc_info.value)


# -----------------------------------------------------------------------------
# C. Invalid Vector Pipeline Validation Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vector_dimension_mismatch_rejected() -> None:
    """Adapter returning unexpected vector dimension raises InvalidProviderResponseError."""
    # Adapter returns 512, but service settings expect 768
    adapter = FakeEmbeddingAdapter(dimension=768, return_invalid_dim=512)
    service = EmbeddingService(adapter=adapter)

    # Note: Pydantic schema will reject post-init or service will validate
    with pytest.raises((InvalidProviderResponseError, ValueError)):
        await service.embed_text("Test prompt")


@pytest.mark.asyncio
async def test_vector_with_nan_rejected() -> None:
    """Vector containing NaN values raises InvalidProviderResponseError."""
    adapter = FakeEmbeddingAdapter(dimension=768, return_nan=True)
    service = EmbeddingService(adapter=adapter)

    with pytest.raises((InvalidProviderResponseError, ValueError)):
        await service.embed_text("Test prompt")


@pytest.mark.asyncio
async def test_vector_with_infinity_rejected() -> None:
    """Vector containing infinity raises InvalidProviderResponseError."""
    adapter = FakeEmbeddingAdapter(dimension=768, return_inf=True)
    service = EmbeddingService(adapter=adapter)

    with pytest.raises((InvalidProviderResponseError, ValueError)):
        await service.embed_text("Test prompt")


@pytest.mark.asyncio
async def test_vector_model_mismatch_rejected() -> None:
    """Vector reporting an unexpected model ID is rejected."""
    adapter = FakeEmbeddingAdapter(dimension=768, model_id="unexpected-embedding-model")
    service = EmbeddingService(adapter=adapter)

    with pytest.raises(InvalidProviderResponseError):
        await service.embed_text("Test prompt")


# -----------------------------------------------------------------------------
# D. Provider Adapter & Error Mapping Tests
# -----------------------------------------------------------------------------

def test_adapter_missing_credentials_raises_config_error() -> None:
    """GoogleGenAIEmbeddingAdapter raises ProviderConfigError when credentials are not configured."""
    settings = Settings(GEMINI_API_KEY=None, EMBEDDING_API_KEY=None)
    adapter = GoogleGenAIEmbeddingAdapter(settings=settings)

    with pytest.raises(ProviderConfigError):
        adapter._get_or_create_client()


@pytest.mark.asyncio
async def test_adapter_auth_error_is_not_retried() -> None:
    """HTTP 401 unauthenticated error is mapped to ProviderAuthenticationError without retry."""
    mock_client = MagicMock()
    auth_err = genai_errors.ClientError(401, {"error": {"message": "API_KEY_INVALID"}})
    mock_client.aio.models.embed_content = AsyncMock(side_effect=auth_err)

    settings = Settings(
        GEMINI_API_KEY=SecretStr("invalid-key"),
        GEMINI_MAX_RETRIES=3,
    )
    adapter = GoogleGenAIEmbeddingAdapter(settings=settings, client=mock_client)

    with pytest.raises(ProviderAuthenticationError):
        await adapter.embed_content("Some prompt text")

    # Auth error is non-retryable: exactly 1 attempt
    assert mock_client.aio.models.embed_content.call_count == 1


@pytest.mark.asyncio
async def test_adapter_rate_limit_retries_and_exhausts() -> None:
    """HTTP 429 rate limit retries with backoff and raises ProviderRateLimitError upon exhaustion."""
    mock_client = MagicMock()
    rate_err = genai_errors.ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
    mock_client.aio.models.embed_content = AsyncMock(side_effect=rate_err)

    settings = Settings(
        GEMINI_API_KEY=SecretStr("valid-key"),
        GEMINI_MAX_RETRIES=2,
    )
    adapter = GoogleGenAIEmbeddingAdapter(settings=settings, client=mock_client)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(ProviderRateLimitError):
            await adapter.embed_content("Some prompt text")

        # Initial attempt + 2 retries = 3 calls
        assert mock_client.aio.models.embed_content.call_count == 3
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_adapter_transient_error_succeeds_on_retry() -> None:
    """Transient 503 error succeeds when retry succeeds."""
    mock_client = MagicMock()
    transient_err = genai_errors.ServerError(503, {"error": {"message": "Service Unavailable"}})

    # Mock response object
    mock_resp = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_resp.embeddings = [mock_embedding]
    mock_resp.metadata = None

    # First call fails, second call succeeds
    mock_client.aio.models.embed_content = AsyncMock(side_effect=[transient_err, mock_resp])

    settings = Settings(
        GEMINI_API_KEY=SecretStr("valid-key"),
        GEMINI_MAX_RETRIES=2,
    )
    adapter = GoogleGenAIEmbeddingAdapter(settings=settings, client=mock_client)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter.embed_content("Prompt text")
        assert result.dimension == 768
        assert mock_client.aio.models.embed_content.call_count == 2
        assert mock_sleep.call_count == 1


@pytest.mark.asyncio
async def test_adapter_empty_embeddings_raises_invalid_response() -> None:
    """Provider returning response with empty embeddings list raises InvalidProviderResponseError."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embeddings = []

    mock_client.aio.models.embed_content = AsyncMock(return_value=mock_resp)

    settings = Settings(GEMINI_API_KEY=SecretStr("valid-key"))
    adapter = GoogleGenAIEmbeddingAdapter(settings=settings, client=mock_client)

    with pytest.raises(InvalidProviderResponseError):
        await adapter.embed_content("Prompt text")


@pytest.mark.asyncio
async def test_adapter_close_releases_client() -> None:
    """Calling close() releases client reference."""
    mock_client = MagicMock()
    adapter = GoogleGenAIEmbeddingAdapter(client=mock_client)
    assert adapter._client is not None

    await adapter.close()
    assert adapter._client is None
