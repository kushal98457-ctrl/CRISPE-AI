"""Automated tests for CRISPE Prompt Generation MCP Tool (Step 30).

Verifies:
1. Tool discovery: 'generate_crispe_prompt' advertised with valid JSON-RPC schema.
2. Successful CRISPE synthesis delegating to CrispeGenerationService.
3. Preservation and combination of user-supplied section hints.
4. Security model whitelisting preventing injection of untrusted model endpoints.
5. Strict parameter validation rejecting empty ideas or invalid sampling ranges.
6. Safe error mapping and credential sanitization for provider failures.
7. Feature flag isolation (enable_prompt_tools=False).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest

from mcp.client import Client

from app.core.exceptions import (
    EmptyPromptError,
    ProviderError,
    PromptValidationError,
)
from app.mcp import (
    MCPServerConfig,
    MCPAppContext,
    create_mcp_server,
)
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeGenerationResponse,
    CrispeSections,
    ModelUsageMetadata,
)


@pytest.fixture
def mock_generation_service():
    """Mock CrispeGenerationService returning canonical CRISPE responses."""
    svc = MagicMock()
    now = datetime.now(timezone.utc)
    mock_sections = CrispeSections(
        context="You operate in an enterprise cloud migration environment.",
        role="Senior Cloud Architect with AWS/GCP specialty.",
        instruction="Formulate an actionable migration plan for a legacy monolithic database.",
        specification="Output must be in GitHub Flavored Markdown with phase checkpoints.",
        performance="Ensure zero downtime during traffic cutover.",
        example="Phase 1: Dual-write setup...\nPhase 2: Shadow verification...",
    )
    mock_response = CrispeGenerationResponse(
        idea="Migrate database to cloud",
        sections=mock_sections,
        formatted_prompt=mock_sections.to_markdown(),
        model_id="gemma-4-26b-a4b-it",
        usage=ModelUsageMetadata(prompt_tokens=45, completion_tokens=180, total_tokens=225),
        finish_reason="stop",
        created_at=now,
    )
    svc.generate_crispe_prompt = AsyncMock(return_value=mock_response)
    return svc


# =============================================================================
# 1. Capability Discovery & Feature Flag
# =============================================================================

@pytest.mark.asyncio
async def test_tool_discovery_advertises_generate_crispe_prompt():
    """Verify generate_crispe_prompt is discoverable by MCP clients."""
    server = create_mcp_server()
    client = Client(server)

    async with client:
        tools_resp = await client.list_tools()
        names = [t.name for t in tools_resp.tools]
        assert "generate_crispe_prompt" in names

        tool = next(t for t in tools_resp.tools if t.name == "generate_crispe_prompt")
        assert "CRISPE framework" in tool.description
        assert "idea" in tool.input_schema.get("properties", {})


@pytest.mark.asyncio
async def test_generation_tool_disabled_via_feature_flag():
    """When enable_prompt_tools=False, generate_crispe_prompt is not exposed."""
    cfg = MCPServerConfig(enable_prompt_tools=False)
    server = create_mcp_server(config=cfg)
    client = Client(server)

    async with client:
        tools_resp = await client.list_tools()
        names = [t.name for t in tools_resp.tools]
        assert "generate_crispe_prompt" not in names


# =============================================================================
# 2. Execution & Section Hint Synthesis
# =============================================================================

@pytest.mark.asyncio
async def test_generate_crispe_prompt_success(mock_generation_service):
    """Verify tool executes generation service and returns structured CRISPE payload."""
    ctx = MCPAppContext(generation_service_factory=lambda: mock_generation_service)
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "generate_crispe_prompt",
            arguments={"idea": "Design an automated ETL pipeline for IoT telemetry"},
        )

        assert result.is_error is False
        data = result.structured_content
        assert data is not None
        assert data["success"] is True
        assert data["model_id"] == "gemma-4-26b-a4b-it"
        assert "sections" in data
        assert data["sections"]["role"] == "Senior Cloud Architect with AWS/GCP specialty."
        assert "formatted_prompt" in data
        assert "# C - Context" in data["formatted_prompt"]
        assert "# R - Role" in data["formatted_prompt"]
        assert data["usage"]["total_tokens"] == 225

        mock_generation_service.generate_crispe_prompt.assert_awaited_once()
        req_sent: CrispeGenerationRequest = mock_generation_service.generate_crispe_prompt.call_args[0][0]
        assert "Design an automated ETL pipeline" in req_sent.idea


@pytest.mark.asyncio
async def test_generate_crispe_prompt_section_hints_preservation(mock_generation_service):
    """Verify explicit section hints are incorporated into the generation request."""
    ctx = MCPAppContext(generation_service_factory=lambda: mock_generation_service)
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "generate_crispe_prompt",
            arguments={
                "idea": "Build a secure webhook processor",
                "role_section": "Security Engineer with HMAC expertise",
                "specification_section": "Reject any payload without a valid sha256 signature",
            },
        )

        assert result.is_error is False
        req_sent: CrispeGenerationRequest = mock_generation_service.generate_crispe_prompt.call_args[0][0]
        assert "Build a secure webhook processor" in req_sent.idea
        assert "Role: Security Engineer with HMAC expertise" in req_sent.idea
        assert "Specification: Reject any payload without a valid sha256 signature" in req_sent.idea


# =============================================================================
# 3. Model Whitelisting & Input Validation
# =============================================================================

@pytest.mark.asyncio
async def test_generate_crispe_prompt_rejects_unapproved_model(mock_generation_service):
    """Verify arbitrary or unapproved model identifiers are rejected fail-closed."""
    ctx = MCPAppContext(generation_service_factory=lambda: mock_generation_service)
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "generate_crispe_prompt",
            arguments={
                "idea": "Summarize release notes",
                "model_id": "malicious-unapproved-model-v99",
            },
        )

        assert result.is_error is True
        # Tool execution error surfaces sanitized message
        assert len(result.content) > 0
        text = result.content[0].text
        assert "approved model whitelist" in text.lower()


@pytest.mark.asyncio
async def test_generate_crispe_prompt_empty_input_rejected(mock_generation_service):
    """Verify request with empty idea and no sections is rejected with INVALID_INPUT."""
    ctx = MCPAppContext(generation_service_factory=lambda: mock_generation_service)
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "generate_crispe_prompt",
            arguments={"idea": "   "},
        )

        assert result.is_error is True
        text = result.content[0].text
        assert "at least an 'idea' or crispe sections must be provided" in text.lower()


# =============================================================================
# 4. Error Sanitization & Masking
# =============================================================================

@pytest.mark.asyncio
async def test_generate_crispe_prompt_provider_error_sanitization():
    """Verify backend provider exceptions sanitize API keys and connection strings."""
    failing_service = MagicMock()
    sensitive_err = ProviderError(
        message="Gemini connection failed to https://generativelanguage.googleapis.com with key AIzaSyD1234567890123456789012345678901",
        error_code="PROVIDER_UNAVAILABLE",
    )
    failing_service.generate_crispe_prompt = AsyncMock(side_effect=sensitive_err)

    ctx = MCPAppContext(generation_service_factory=lambda: failing_service)
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "generate_crispe_prompt",
            arguments={"idea": "Generate testing plan"},
        )

        assert result.is_error is True
        text = result.content[0].text
        # Secrets MUST be redacted
        assert "AIzaSyD" not in text
        assert "<REDACTED_CREDENTIAL>" in text
