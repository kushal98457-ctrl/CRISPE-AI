"""Unit tests for the Google GenAI provider adapter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from pydantic import BaseModel, SecretStr

import google.genai.errors as genai_errors
from app.adapters.gemini_adapter import (
    GeminiModelAdapter,
    ModelGenerationRequest,
    ModelGenerationResponse,
)
from app.core.config import Settings
from app.core.exceptions import (
    InvalidProviderResponseError,
    ModelNotFoundError,
    ProviderAPIError,
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderPermissionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.schemas.crispe import CrispeSections


class SampleSchema(BaseModel):
    title: str
    summary: str


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing settings with a test Gemini API key."""
    return Settings(
        GEMINI_API_KEY=SecretStr("AIzaSyFakeTestKeyForUnitTestingOnly12345"),
        GEMINI_MODEL_ID="gemma-4-26b-a4b-it",
        GEMINI_TIMEOUT_SECONDS=10.0,
        GEMINI_MAX_RETRIES=2,
    )


@pytest.fixture
def mock_empty_settings() -> Settings:
    """Fixture providing settings without an API key."""
    return Settings(
        GEMINI_API_KEY=None,
        GOOGLE_API_KEY=None,
    )


@pytest.mark.asyncio
async def test_adapter_missing_credentials_raises_provider_config_error(mock_empty_settings: Settings) -> None:
    """Adapter must raise ProviderConfigError on generate when API key is missing."""
    adapter = GeminiModelAdapter(settings=mock_empty_settings)
    req = ModelGenerationRequest(prompt="Hello CRISPE")
    with pytest.raises(ProviderConfigError) as exc_info:
        await adapter.generate_content(req)
    assert "Gemini API credential is not configured" in exc_info.value.message


@pytest.mark.asyncio
async def test_adapter_successful_text_generation(mock_settings: Settings) -> None:
    """Adapter must correctly extract text, finish reason, and usage from successful provider response."""
    mock_client = MagicMock()
    mock_raw_response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="Generated prompt response")]),
                finish_reason="STOP",
            )
        ],
        parsed=None,
        response_id="resp-12345",
        usage_metadata=SimpleNamespace(
            prompt_token_count=15,
            candidates_token_count=45,
            total_token_count=60,
            thoughts_token_count=0,
        ),
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_raw_response)

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(
        prompt="Create a Python backend prompt",
        temperature=0.5,
    )

    response = await adapter.generate_content(req)

    assert isinstance(response, ModelGenerationResponse)
    assert response.content == "Generated prompt response"
    assert response.model_id == "gemma-4-26b-a4b-it"
    assert response.response_id == "resp-12345"
    assert response.finish_reason == "STOP"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 15
    assert response.usage.candidates_tokens == 45
    assert response.usage.total_tokens == 60


@pytest.mark.asyncio
async def test_adapter_successful_structured_generation(mock_settings: Settings) -> None:
    """Adapter must preserve parsed Pydantic schema returned by provider."""
    mock_client = MagicMock()
    parsed_crispe = CrispeSections(
        context="Web SaaS domain",
        role="Principal engineer",
        instruction="Design API",
        specification="REST standard",
        performance="100% test coverage",
        example="Sample route definition",
    )
    mock_raw_response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text='{"context": "Web SaaS domain"}')]),
                finish_reason="STOP",
            )
        ],
        parsed=parsed_crispe,
        response_id="resp-structured-99",
        usage_metadata=SimpleNamespace(
            prompt_token_count=30,
            candidates_token_count=120,
            total_token_count=150,
            thoughts_token_count=20,
        ),
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_raw_response)

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(
        prompt="Convert idea",
        response_schema=CrispeSections,
    )

    response = await adapter.generate_content(req)
    assert response.parsed == parsed_crispe
    assert response.usage is not None
    assert response.usage.thoughts_tokens == 20


@pytest.mark.asyncio
async def test_adapter_empty_candidates_raises_invalid_response(mock_settings: Settings) -> None:
    """Adapter must raise InvalidProviderResponseError when response contains no candidates."""
    mock_client = MagicMock()
    mock_raw_response = SimpleNamespace(candidates=[], parsed=None, response_id="resp-empty")
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_raw_response)

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(prompt="Test prompt")

    with pytest.raises(InvalidProviderResponseError) as exc_info:
        await adapter.generate_content(req)
    assert "no candidate completions" in exc_info.value.message


@pytest.mark.asyncio
async def test_adapter_auth_error_is_not_retried(mock_settings: Settings) -> None:
    """Authentication failures (HTTP 401) must raise ProviderAuthenticationError without retry."""
    mock_client = MagicMock()
    auth_err = genai_errors.ClientError(401, {"error": {"message": "API_KEY_INVALID"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=auth_err)

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(prompt="Test auth")

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await adapter.generate_content(req)
    assert exc_info.value.status_code == 401
    assert not exc_info.value.is_retryable
    # Must only be called once — NO retries for auth failure!
    assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_adapter_permission_error_is_not_retried(mock_settings: Settings) -> None:
    """Permission denied (HTTP 403) must raise ProviderPermissionError without retry."""
    mock_client = MagicMock()
    perm_err = genai_errors.ClientError(403, {"error": {"message": "PERMISSION_DENIED"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=perm_err)

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(prompt="Test permission")

    with pytest.raises(ProviderPermissionError) as exc_info:
        await adapter.generate_content(req)
    assert exc_info.value.status_code == 403
    assert not exc_info.value.is_retryable
    assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_adapter_model_not_found_is_not_retried(mock_settings: Settings) -> None:
    """Model not found (HTTP 404) must raise ModelNotFoundError without retry."""
    mock_client = MagicMock()
    not_found_err = genai_errors.ClientError(404, {"error": {"message": "models/unknown-model NOT_FOUND"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=not_found_err)

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(prompt="Test 404", model_id="unknown-model")

    with pytest.raises(ModelNotFoundError) as exc_info:
        await adapter.generate_content(req)
    assert exc_info.value.status_code == 404
    assert not exc_info.value.is_retryable
    assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_adapter_rate_limit_retries_and_exhausts(mock_settings: Settings) -> None:
    """Rate limit (HTTP 429) must be retried up to GEMINI_MAX_RETRIES then raise ProviderRateLimitError."""
    mock_client = MagicMock()
    rate_err = genai_errors.ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=rate_err)

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(prompt="Test rate limit")

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(ProviderRateLimitError) as exc_info:
            await adapter.generate_content(req)
        assert exc_info.value.status_code == 429
        # Initial attempt + 2 retries = 3 calls
        assert mock_client.aio.models.generate_content.call_count == 3
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_adapter_transient_server_error_succeeds_on_retry(mock_settings: Settings) -> None:
    """Transient server error (HTTP 503) succeeds when second attempt succeeds."""
    mock_client = MagicMock()
    server_err = genai_errors.ServerError(503, {"error": {"message": "Service Unavailable"}})
    successful_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="Recovered output")]),
                finish_reason="STOP",
            )
        ],
        parsed=None,
        response_id="resp-recov",
        usage_metadata=None,
    )
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[server_err, successful_resp])

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(prompt="Test retry recovery")

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        resp = await adapter.generate_content(req)
        assert resp.content == "Recovered output"
        assert mock_client.aio.models.generate_content.call_count == 2
        assert mock_sleep.call_count == 1


@pytest.mark.asyncio
async def test_adapter_timeout_translates_and_retries(mock_settings: Settings) -> None:
    """Async timeout (asyncio.TimeoutError) is translated to ProviderTimeoutError."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=asyncio.TimeoutError("Deadline exceeded"))

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    req = ModelGenerationRequest(prompt="Test timeout")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await adapter.generate_content(req)
        assert exc_info.value.status_code == 504
        assert exc_info.value.error_code == "PROVIDER_TIMEOUT"
        assert mock_client.aio.models.generate_content.call_count == 3


@pytest.mark.asyncio
async def test_adapter_close_releases_resources(mock_settings: Settings) -> None:
    """Adapter close() invokes aclose() on client.aio without throwing."""
    mock_client = MagicMock()
    mock_client.aio.aclose = AsyncMock()

    adapter = GeminiModelAdapter(settings=mock_settings, client=mock_client)
    await adapter.close()

    mock_client.aio.aclose.assert_awaited_once()
    assert adapter._client is None
