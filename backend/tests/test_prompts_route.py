"""Unit and integration tests for the CRISPE prompt generation API route."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes.prompts import get_generation_service_dependency
from app.core.exceptions import (
    InvalidProviderResponseError,
    ModelNotFoundError,
    PromptValidationError,
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderPermissionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.main import app
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeGenerationResponse,
    CrispeSections,
    ModelUsageMetadata,
)
from app.services.generation_service import CrispeGenerationService


@pytest.fixture
def mock_crispe_sections() -> CrispeSections:
    """Fixture providing valid 6-section CRISPE data."""
    return CrispeSections(
        context="Distributed event-driven e-commerce order management.",
        role="Principal Cloud Architect and Kafka specialist.",
        instruction="Design a dead-letter queue and idempotency strategy for payment events.",
        specification="Use Markdown tables. Strict zero-message-loss requirement.",
        performance="Deduplication latency under 5ms at 50k events/sec.",
        example="Input: Duplicate payment event. Output: Idempotent Redis lock match.",
    )


@pytest.fixture
def fake_generation_service(mock_crispe_sections: CrispeSections) -> CrispeGenerationService:
    """Fixture creating a fake CrispeGenerationService for dependency override."""
    service = MagicMock(spec=CrispeGenerationService)

    async def fake_generate(request: CrispeGenerationRequest) -> CrispeGenerationResponse:
        return CrispeGenerationResponse(
            idea=request.idea,
            sections=mock_crispe_sections,
            formatted_prompt=mock_crispe_sections.to_markdown(),
            model_id=request.model_id or "gemma-4-26b-a4b-it",
            usage=ModelUsageMetadata(prompt_tokens=45, candidates_tokens=150, total_tokens=195),
            finish_reason="STOP",
            schema_version="1.0",
        )

    service.generate_crispe_prompt = AsyncMock(side_effect=fake_generate)
    return service


@pytest.fixture
async def client_with_fake_service(fake_generation_service: CrispeGenerationService):
    """Async test client with generation service dependency overridden."""
    app.dependency_overrides[get_generation_service_dependency] = lambda: fake_generation_service
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        yield client
    app.dependency_overrides.clear()


# ==============================================================================
# Category A: Successful Generation
# ==============================================================================

@pytest.mark.asyncio
async def test_generate_prompt_successful_200(client_with_fake_service: AsyncClient, mock_crispe_sections: CrispeSections) -> None:
    """POST /api/v1/prompts/generate returns 200 OK with fully validated CRISPE response."""
    payload = {
        "idea": "Design an idempotent payment event processor",
        "temperature": 0.4,
    }

    response = await client_with_fake_service.post("/api/v1/prompts/generate", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Validate high-level fields
    assert data["idea"] == "Design an idempotent payment event processor"
    assert data["model_id"] == "gemma-4-26b-a4b-it"
    assert data["schema_version"] == "1.0"
    assert data["request_id"] is not None

    # Validate all 6 CRISPE sections exist and match
    sections = data["sections"]
    assert sections["context"] == mock_crispe_sections.context
    assert sections["role"] == mock_crispe_sections.role
    assert sections["instruction"] == mock_crispe_sections.instruction
    assert sections["specification"] == mock_crispe_sections.specification
    assert sections["performance"] == mock_crispe_sections.performance
    assert sections["example"] == mock_crispe_sections.example

    # Validate canonical Markdown headers in formatted prompt
    formatted = data["formatted_prompt"]
    assert "# C - Context" in formatted
    assert "# R - Role" in formatted
    assert "# I - Instruction" in formatted
    assert "# S - Specification" in formatted
    assert "# P - Performance" in formatted
    assert "# E - Example" in formatted

    # Validate usage metadata
    assert data["usage"]["total_tokens"] == 195

    # Validate HTTP telemetry headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers


@pytest.mark.asyncio
async def test_generate_prompt_passes_model_id_and_options(
    client_with_fake_service: AsyncClient, fake_generation_service: CrispeGenerationService
) -> None:
    """Endpoint forwards model_id, temperature, and thinking budget to generation service."""
    payload = {
        "idea": "Build a secure webhook handler",
        "model_id": "gemini-2.5-flash",
        "temperature": 0.7,
        "thinking_budget": 2048,
    }

    response = await client_with_fake_service.post("/api/v1/prompts/generate", json=payload)
    assert response.status_code == 200

    # Verify service was invoked with the exact parsed request
    call_args = fake_generation_service.generate_crispe_prompt.call_args[0][0]
    assert isinstance(call_args, CrispeGenerationRequest)
    assert call_args.idea == "Build a secure webhook handler"
    assert call_args.model_id == "gemini-2.5-flash"
    assert call_args.temperature == 0.7
    assert call_args.thinking_budget == 2048


# ==============================================================================
# Category B: Request Validation
# ==============================================================================

@pytest.mark.asyncio
async def test_generate_prompt_missing_idea_returns_422(client_with_fake_service: AsyncClient) -> None:
    """Missing required 'idea' field returns 422 Unprocessable Content."""
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert any("idea" in err["loc"] for err in data["details"])


@pytest.mark.asyncio
async def test_generate_prompt_empty_idea_returns_422(client_with_fake_service: AsyncClient) -> None:
    """Empty or whitespace-only 'idea' returns 422 Unprocessable Content."""
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "   "})
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_generate_prompt_excessive_length_returns_422(client_with_fake_service: AsyncClient) -> None:
    """Idea exceeding 10,000 characters returns 422 Unprocessable Content."""
    long_idea = "A" * 10001
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": long_idea})
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_generate_prompt_invalid_temperature_returns_422(client_with_fake_service: AsyncClient) -> None:
    """Temperature outside 0.0–2.0 boundary returns 422."""
    response = await client_with_fake_service.post(
        "/api/v1/prompts/generate",
        json={"idea": "Valid idea", "temperature": 3.5},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_generate_prompt_invalid_json_body_returns_422(client_with_fake_service: AsyncClient) -> None:
    """Malformed non-JSON payload returns 422."""
    response = await client_with_fake_service.post(
        "/api/v1/prompts/generate",
        content="not-a-valid-json-string",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


# ==============================================================================
# Category C: Service Failures and Error Mapping
# ==============================================================================

@pytest.mark.asyncio
async def test_provider_authentication_error_maps_to_401(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """ProviderAuthenticationError maps to HTTP 401 with structured ErrorResponse."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=ProviderAuthenticationError("Invalid Google GenAI API key")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 401
    data = response.json()
    assert data["error_code"] == "PROVIDER_AUTH_ERROR"
    assert "Invalid Google GenAI API key" in data["message"]
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_provider_permission_error_maps_to_403(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """ProviderPermissionError maps to HTTP 403."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=ProviderPermissionError("Access forbidden to model checkpoint")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 403
    data = response.json()
    assert data["error_code"] == "PROVIDER_PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_model_not_found_error_maps_to_404(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """ModelNotFoundError maps to HTTP 404."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=ModelNotFoundError("Specified model checkpoint does not exist")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 404
    data = response.json()
    assert data["error_code"] == "MODEL_NOT_FOUND"


@pytest.mark.asyncio
async def test_provider_rate_limit_error_maps_to_429(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """ProviderRateLimitError maps to HTTP 429."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=ProviderRateLimitError("Quota limit reached")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 429
    data = response.json()
    assert data["error_code"] == "PROVIDER_RATE_LIMIT"


@pytest.mark.asyncio
async def test_provider_timeout_error_maps_to_504(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """ProviderTimeoutError maps to HTTP 504 Gateway Timeout."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=ProviderTimeoutError("Upstream model request timed out")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 504
    data = response.json()
    assert data["error_code"] == "PROVIDER_TIMEOUT"


@pytest.mark.asyncio
async def test_provider_unavailable_error_maps_to_503(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """ProviderUnavailableError maps to HTTP 503 Service Unavailable."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=ProviderUnavailableError("Gemini servers temporarily offline")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 503
    data = response.json()
    assert data["error_code"] == "PROVIDER_UNAVAILABLE"


@pytest.mark.asyncio
async def test_invalid_provider_response_maps_to_502(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """InvalidProviderResponseError maps to HTTP 502 Bad Gateway."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=InvalidProviderResponseError("Provider returned empty candidate text")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 502
    data = response.json()
    assert data["error_code"] == "INVALID_PROVIDER_RESPONSE"


@pytest.mark.asyncio
async def test_prompt_validation_error_maps_to_422(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """PromptValidationError maps to HTTP 422."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=PromptValidationError("Model output missing 'example' section")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "PROMPT_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_unhandled_exception_maps_to_500_without_stack_trace(client_with_fake_service: AsyncClient, fake_generation_service: MagicMock) -> None:
    """Unhandled internal exceptions map to 500 without leaking stack traces or credentials."""
    fake_generation_service.generate_crispe_prompt = AsyncMock(
        side_effect=RuntimeError("Unexpected internal memory corruption with AIzaSySecretKey123")
    )
    response = await client_with_fake_service.post("/api/v1/prompts/generate", json={"idea": "Valid idea"})
    assert response.status_code == 500
    data = response.json()
    assert data["error_code"] == "INTERNAL_SERVER_ERROR"
    # Never leak internal messages or secrets to client
    assert "AIzaSy" not in response.text
    assert "memory corruption" not in response.text


# ==============================================================================
# Category E: Regression Health Probes
# ==============================================================================

@pytest.mark.asyncio
async def test_health_routes_remain_healthy_regression(client_with_fake_service: AsyncClient) -> None:
    """Health endpoints remain 200 OK after registering prompts route."""
    res_root = await client_with_fake_service.get("/health")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "ok"

    res_v1 = await client_with_fake_service.get("/api/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "ok"
