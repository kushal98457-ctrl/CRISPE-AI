"""Unit tests for the CRISPE prompt generation service."""

import json
from unittest.mock import AsyncMock
import pytest

from app.adapters.gemini_adapter import (
    BaseModelAdapter,
    ModelGenerationRequest,
    ModelGenerationResponse,
)
from app.core.exceptions import (
    EmptyPromptError,
    PromptValidationError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
)
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeSections,
    ModelUsageMetadata,
)
from app.services.generation_service import (
    CrispeGenerationService,
    get_generation_service,
)


class FakeModelAdapter(BaseModelAdapter):
    """Test fake for BaseModelAdapter."""

    def __init__(self, response: ModelGenerationResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.last_request: ModelGenerationRequest | None = None
        self.closed = False

    async def generate_content(self, request: ModelGenerationRequest) -> ModelGenerationResponse:
        self.last_request = request
        if self.error:
            raise self.error
        if self.response:
            return self.response
        raise RuntimeError("No response or error configured in FakeModelAdapter")

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def sample_sections() -> CrispeSections:
    """Fixture providing valid 6-section CRISPE structure."""
    return CrispeSections(
        context="Modern cloud-native payment microservice architecture.",
        role="Principal Security Architect specializing in zero-trust PCI-DSS compliance.",
        instruction="Conduct a threat model analysis of tokenized credit card processing endpoints.",
        specification="Produce a STRIDE threat matrix in Markdown table format. No preamble.",
        performance="Every identified threat must include mitigation strategy and DREAD score.",
        example="Input: POST /tokenize endpoint. Output: Spoofing - Mitigate via mTLS & JWT.",
    )


@pytest.mark.asyncio
async def test_generation_service_successful_transformation(sample_sections: CrispeSections) -> None:
    """Service successfully transforms raw idea into structured CRISPE response and formatted Markdown."""
    fake_response = ModelGenerationResponse(
        content='{"valid": "json"}',
        parsed=sample_sections,
        model_id="gemma-4-26b-a4b-it",
        usage=ModelUsageMetadata(prompt_tokens=40, candidates_tokens=180, total_tokens=220),
    )
    fake_adapter = FakeModelAdapter(response=fake_response)
    service = CrispeGenerationService(adapter=fake_adapter)

    req = CrispeGenerationRequest(idea="Secure my payment API")
    result = await service.generate_crispe_prompt(req)

    assert result.idea == "Secure my payment API"
    assert result.model_id == "gemma-4-26b-a4b-it"
    assert result.sections == sample_sections
    assert result.usage is not None
    assert result.usage.total_tokens == 220

    # Verify Markdown formatting contains all 6 canonical CRISPE headings
    assert "# C - Context" in result.formatted_prompt
    assert "# R - Role" in result.formatted_prompt
    assert "# I - Instruction" in result.formatted_prompt
    assert "# S - Specification" in result.formatted_prompt
    assert "# P - Performance" in result.formatted_prompt
    assert "# E - Example" in result.formatted_prompt
    assert sample_sections.context in result.formatted_prompt
    assert sample_sections.role in result.formatted_prompt


@pytest.mark.asyncio
async def test_generation_service_markdown_json_fallback_parsing(sample_sections: CrispeSections) -> None:
    """Service parses JSON cleanly when model wraps output in a Markdown code fence."""
    raw_json_in_markdown = f"""```json
{sample_sections.model_dump_json(indent=2)}
```"""
    fake_response = ModelGenerationResponse(
        content=raw_json_in_markdown,
        parsed=None,  # parsed is None; service must fall back to parsing content
        model_id="gemini-2.5-flash",
        usage=ModelUsageMetadata(prompt_tokens=50, candidates_tokens=200, total_tokens=250),
    )
    fake_adapter = FakeModelAdapter(response=fake_response)
    service = CrispeGenerationService(adapter=fake_adapter)

    req = CrispeGenerationRequest(idea="Build a recommendation engine")
    result = await service.generate_crispe_prompt(req)

    assert result.sections.context == sample_sections.context
    assert result.sections.role == sample_sections.role
    assert result.sections.instruction == sample_sections.instruction
    assert result.model_id == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_generation_service_rejects_missing_sections() -> None:
    """Service raises PromptValidationError when model output is missing required CRISPE sections."""
    incomplete_json = json.dumps({
        "context": "Cloud architecture",
        "role": "DevOps Engineer",
        # Missing instruction, specification, performance, example!
    })
    fake_response = ModelGenerationResponse(
        content=incomplete_json,
        parsed=None,
        model_id="gemma-4-26b-a4b-it",
    )
    fake_adapter = FakeModelAdapter(response=fake_response)
    service = CrispeGenerationService(adapter=fake_adapter)

    req = CrispeGenerationRequest(idea="Help configure Kubernetes")
    with pytest.raises(PromptValidationError) as exc_info:
        await service.generate_crispe_prompt(req)
    assert "does not satisfy CRISPE schema" in exc_info.value.message


@pytest.mark.asyncio
async def test_generation_service_rejects_malformed_json() -> None:
    """Service raises PromptValidationError when model output is not valid JSON."""
    fake_response = ModelGenerationResponse(
        content="This is not valid JSON content at all.",
        parsed=None,
        model_id="gemma-4-26b-a4b-it",
    )
    fake_adapter = FakeModelAdapter(response=fake_response)
    service = CrispeGenerationService(adapter=fake_adapter)

    req = CrispeGenerationRequest(idea="Write a novel")
    with pytest.raises(PromptValidationError) as exc_info:
        await service.generate_crispe_prompt(req)
    assert "not valid JSON" in exc_info.value.message


@pytest.mark.asyncio
async def test_generation_service_rejects_empty_prompt() -> None:
    """Empty or whitespace idea raises EmptyPromptError or validation error."""
    fake_adapter = FakeModelAdapter()
    service = CrispeGenerationService(adapter=fake_adapter)

    with pytest.raises(Exception):
        # CrispeGenerationRequest validation blocks whitespace
        req = CrispeGenerationRequest(idea="   ")
        await service.generate_crispe_prompt(req)


@pytest.mark.asyncio
async def test_generation_service_propagates_provider_errors_without_swallowing() -> None:
    """Provider authentication and rate-limit errors must propagate directly to caller."""
    auth_error = ProviderAuthenticationError("Invalid API key")
    fake_adapter = FakeModelAdapter(error=auth_error)
    service = CrispeGenerationService(adapter=fake_adapter)

    req = CrispeGenerationRequest(idea="Create marketing prompt")
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await service.generate_crispe_prompt(req)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_generation_service_factory() -> None:
    """Factory function creates CrispeGenerationService instance."""
    fake_adapter = FakeModelAdapter()
    svc = get_generation_service(adapter=fake_adapter)
    assert isinstance(svc, CrispeGenerationService)
    assert svc.adapter is fake_adapter
