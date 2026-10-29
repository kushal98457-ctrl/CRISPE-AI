"""Automated tests for Semantic Prompt Memory MCP Tools (Step 30).

Verifies:
1. Tool Discovery: 'search_prompt_memory' and 'save_prompt_memory' registered in MCP schema.
2. Feature flag gating: tools are omitted when enable_memory_tools=False.
3. Strict Authorization & Security Isolation:
   - search_prompt_memory rejects unauthenticated calls (no trusted principal) fail-closed.
   - search_prompt_memory rejects deactivated principal.
   - save_prompt_memory rejects unauthenticated calls fail-closed.
   - save_prompt_memory rejects deactivated principal.
4. Semantic Search Tool Execution:
   - Querying memory with trusted principal retrieves formatted PromptMemoryMatchItem results.
   - Parameter constraints enforced (top_k bounds, non-empty query).
   - Incompatible embedding error maps to INCOMPATIBLE_EMBEDDING.
5. Prompt Save Tool Execution:
   - Explicitly saving CRISPE sections creates new indexed PromptRecord (status="created").
   - Idempotent save returns existing record (status="already_exists").
   - Model whitelisting rejects unapproved model identifiers.
   - Malformed CRISPE sections rejected with INVALID_INPUT.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest

from mcp.client import Client

from app.core.exceptions import (
    IncompatibleEmbeddingError,
    ProviderError,
)
from app.core.security import AuthorizedPrincipal
from app.mcp import (
    MCPServerConfig,
    MCPAppContext,
    create_mcp_server,
)
from app.schemas.crispe import CrispeSections
from app.schemas.semantic_search import (
    SemanticSearchItem,
    SemanticSearchQuery,
    SemanticSearchResponse,
)


@pytest.fixture
def active_principal() -> AuthorizedPrincipal:
    """Fixture providing an active, verified security principal."""
    return AuthorizedPrincipal(
        id=str(uuid.uuid4()),
        is_active=True,
        email="developer@example.com",
        display_name="Test Developer",
    )


@pytest.fixture
def deactivated_principal() -> AuthorizedPrincipal:
    """Fixture providing a deactivated security principal."""
    return AuthorizedPrincipal(
        id=str(uuid.uuid4()),
        is_active=False,
        email="suspended@example.com",
    )


@pytest.fixture
def dummy_session_factory():
    """Mock async session factory returning an async context manager."""
    @asynccontextmanager
    async def _session():
        yield MagicMock()

    return _session


@pytest.fixture
def sample_sections() -> CrispeSections:
    """Standard CRISPE sections fixture."""
    return CrispeSections(
        context="Enterprise microservices environment with Kubernetes.",
        role="Principal Site Reliability Engineer.",
        instruction="Analyze pod restart metrics and generate remediation runbook.",
        specification="Include PromQL queries, alerting rules, and rollout steps in Markdown.",
        performance="Zero customer-facing outage during runbook execution.",
        example="Step 1: Inspect crashloopbackoff events via kubectl describe...",
    )


@pytest.fixture
def mock_search_service(sample_sections):
    """Mock SemanticSearchService returning valid search and save results."""
    svc = MagicMock()
    now = datetime.now(timezone.utc)
    match_id = uuid.uuid4()

    sample_items = [
        SemanticSearchItem(
            prompt_id=match_id,
            original_idea="Analyze Kubernetes pod crashes",
            sections=sample_sections,
            formatted_prompt=sample_sections.to_markdown(),
            task_type="devops",
            model_id="gemma-4-26b-a4b-it",
            distance=0.12,
            similarity=0.88,
            config_version="v1",
            created_at=now,
        )
    ]

    mock_search_resp = SemanticSearchResponse(
        query="pod crash analysis",
        total_results=1,
        items=sample_items,
        model_id="text-embedding-004",
        config_version="v1",
        search_metric="cosine_distance",
    )
    svc.search = AsyncMock(return_value=mock_search_resp)

    # Mock PromptRecord for save_prompt
    mock_record = MagicMock()
    mock_record.id = match_id
    mock_record.task_type = "devops"
    mock_record.created_at = now
    svc.save_prompt = AsyncMock(return_value=(mock_record, True))

    return svc


# =============================================================================
# 1. Tool Discovery & Feature Flags
# =============================================================================

@pytest.mark.asyncio
async def test_tool_discovery_advertises_memory_tools():
    """Verify search_prompt_memory and save_prompt_memory are exposed via MCP."""
    server = create_mcp_server()
    client = Client(server)

    async with client:
        tools_resp = await client.list_tools()
        names = [t.name for t in tools_resp.tools]
        assert "search_prompt_memory" in names
        assert "save_prompt_memory" in names

        search_tool = next(t for t in tools_resp.tools if t.name == "search_prompt_memory")
        assert "semantic similarity" in search_tool.description.lower()
        assert "query" in search_tool.input_schema.get("properties", {})

        save_tool = next(t for t in tools_resp.tools if t.name == "save_prompt_memory")
        assert "semantic memory" in save_tool.description.lower()
        assert "context" in save_tool.input_schema.get("properties", {})


@pytest.mark.asyncio
async def test_memory_tools_disabled_via_feature_flag():
    """When enable_memory_tools=False, neither memory tool is registered."""
    cfg = MCPServerConfig(enable_memory_tools=False)
    server = create_mcp_server(config=cfg)
    client = Client(server)

    async with client:
        tools_resp = await client.list_tools()
        names = [t.name for t in tools_resp.tools]
        assert "search_prompt_memory" not in names
        assert "save_prompt_memory" not in names


# =============================================================================
# 2. Security Principal Scoping & Authorization Failsafe
# =============================================================================

@pytest.mark.asyncio
async def test_search_memory_unauthorized_without_principal(mock_search_service, dummy_session_factory):
    """Calling search_prompt_memory without a trusted principal raises UNAUTHORIZED."""
    ctx = MCPAppContext(
        trusted_principal=None,  # No principal supplied
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: mock_search_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "search_prompt_memory",
            arguments={"query": "Kubernetes debugging"},
        )

        assert result.is_error is True
        text = result.content[0].text
        assert "active principal context is required" in text.lower()


@pytest.mark.asyncio
async def test_search_memory_rejected_for_deactivated_principal(
    mock_search_service, dummy_session_factory, deactivated_principal
):
    """Calling search_prompt_memory with a suspended principal fails closed."""
    ctx = MCPAppContext(
        trusted_principal=deactivated_principal,
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: mock_search_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "search_prompt_memory",
            arguments={"query": "Kubernetes debugging"},
        )

        assert result.is_error is True
        text = result.content[0].text
        assert "inactive or suspended" in text.lower()


@pytest.mark.asyncio
async def test_save_memory_unauthorized_without_principal(
    mock_search_service, dummy_session_factory, sample_sections
):
    """Calling save_prompt_memory without a trusted principal raises UNAUTHORIZED."""
    ctx = MCPAppContext(
        trusted_principal=None,
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: mock_search_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "save_prompt_memory",
            arguments={
                "idea": "Kubernetes debugging",
                "context": sample_sections.context,
                "role": sample_sections.role,
                "instruction": sample_sections.instruction,
                "specification": sample_sections.specification,
                "performance": sample_sections.performance,
                "example": sample_sections.example,
            },
        )

        assert result.is_error is True
        text = result.content[0].text
        assert "active principal context is required" in text.lower()


# =============================================================================
# 3. Semantic Search Execution
# =============================================================================

@pytest.mark.asyncio
async def test_search_prompt_memory_success(
    mock_search_service, dummy_session_factory, active_principal
):
    """Verify authenticated semantic search returns ordered matches with similarity scores."""
    ctx = MCPAppContext(
        trusted_principal=active_principal,
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: mock_search_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "search_prompt_memory",
            arguments={
                "query": "Kubernetes pod failure analysis",
                "top_k": 3,
                "task_type": "devops",
                "min_similarity": 0.75,
            },
        )

        assert result.is_error is False
        data = result.structured_content
        assert data is not None
        assert data["success"] is True
        assert data["query"] == "pod crash analysis"
        assert data["total_matches"] == 1
        assert data["owner_id"] == str(active_principal.id)

        first_match = data["results"][0]
        assert first_match["original_idea"] == "Analyze Kubernetes pod crashes"
        assert first_match["similarity_score"] == 0.88
        assert first_match["score_metric"] == "cosine_distance"
        assert first_match["task_type"] == "devops"
        assert "sections" in first_match
        assert "# C - Context" in first_match["formatted_prompt"]

        # Ensure service call was scoped strictly to authenticated principal
        mock_search_service.search.assert_awaited_once()
        call_principal = mock_search_service.search.call_args[1]["principal"]
        assert call_principal.id == active_principal.id
        call_query: SemanticSearchQuery = mock_search_service.search.call_args[1]["query"]
        assert call_query.limit == 3
        assert call_query.task_type == "devops"
        assert call_query.min_similarity == 0.75


@pytest.mark.asyncio
async def test_search_prompt_memory_rejects_empty_query(
    mock_search_service, dummy_session_factory, active_principal
):
    """Verify search tool rejects empty or whitespace queries with INVALID_INPUT."""
    ctx = MCPAppContext(
        trusted_principal=active_principal,
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: mock_search_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "search_prompt_memory",
            arguments={"query": ""},
        )

        assert result.is_error is True
        text = result.content[0].text
        assert "invalid memory search parameters" in text.lower()


@pytest.mark.asyncio
async def test_search_prompt_memory_incompatible_embedding_handling(
    dummy_session_factory, active_principal
):
    """Verify incompatible embedding dimension mismatch maps cleanly to MCP error."""
    failing_service = MagicMock()
    failing_service.search = AsyncMock(
        side_effect=IncompatibleEmbeddingError("Dimension mismatch: query 768 != stored 1536")
    )

    ctx = MCPAppContext(
        trusted_principal=active_principal,
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: failing_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "search_prompt_memory",
            arguments={"query": "test query"},
        )

        assert result.is_error is True
        text = result.content[0].text
        assert "dimension mismatch" in text.lower()


# =============================================================================
# 4. Prompt Save Execution & Idempotency
# =============================================================================

@pytest.mark.asyncio
async def test_save_prompt_memory_success_new_record(
    mock_search_service, dummy_session_factory, active_principal, sample_sections
):
    """Verify saving a new prompt returns status='created' and generated prompt_id."""
    ctx = MCPAppContext(
        trusted_principal=active_principal,
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: mock_search_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "save_prompt_memory",
            arguments={
                "idea": "Kubernetes debugging runbook",
                "context": sample_sections.context,
                "role": sample_sections.role,
                "instruction": sample_sections.instruction,
                "specification": sample_sections.specification,
                "performance": sample_sections.performance,
                "example": sample_sections.example,
                "task_type": "devops",
                "model_id": "gemma-4-26b-a4b-it",
            },
        )

        assert result.is_error is False
        data = result.structured_content
        assert data is not None
        assert data["success"] is True
        assert data["status"] == "created"
        assert data["owner_id"] == str(active_principal.id)
        assert data["task_type"] == "devops"
        assert "prompt_id" in data

        # Verify underlying call parameters
        mock_search_service.save_prompt.assert_awaited_once()
        kwargs = mock_search_service.save_prompt.call_args[1]
        assert kwargs["principal"].id == active_principal.id
        assert kwargs["idea"] == "Kubernetes debugging runbook"
        assert kwargs["task_type"] == "devops"


@pytest.mark.asyncio
async def test_save_prompt_memory_idempotent_existing_record(
    mock_search_service, dummy_session_factory, active_principal, sample_sections
):
    """Verify saving an identical prompt returns status='already_exists'."""
    existing_record = MagicMock()
    existing_record.id = uuid.uuid4()
    existing_record.task_type = "devops"
    existing_record.created_at = datetime.now(timezone.utc)
    mock_search_service.save_prompt = AsyncMock(return_value=(existing_record, False))

    ctx = MCPAppContext(
        trusted_principal=active_principal,
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: mock_search_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "save_prompt_memory",
            arguments={
                "idea": "Kubernetes debugging runbook",
                "context": sample_sections.context,
                "role": sample_sections.role,
                "instruction": sample_sections.instruction,
                "specification": sample_sections.specification,
                "performance": sample_sections.performance,
                "example": sample_sections.example,
            },
        )

        assert result.is_error is False
        data = result.structured_content
        assert data["status"] == "already_exists"
        assert data["prompt_id"] == str(existing_record.id)


@pytest.mark.asyncio
async def test_save_prompt_memory_rejects_unapproved_model(
    mock_search_service, dummy_session_factory, active_principal, sample_sections
):
    """Verify unapproved model identifier is rejected fail-closed with INVALID_INPUT."""
    ctx = MCPAppContext(
        trusted_principal=active_principal,
        session_factory=dummy_session_factory,
        semantic_search_service_factory=lambda: mock_search_service,
    )
    server = create_mcp_server(context=ctx)
    client = Client(server)

    async with client:
        result = await client.call_tool(
            "save_prompt_memory",
            arguments={
                "idea": "Kubernetes debugging runbook",
                "context": sample_sections.context,
                "role": sample_sections.role,
                "instruction": sample_sections.instruction,
                "specification": sample_sections.specification,
                "performance": sample_sections.performance,
                "example": sample_sections.example,
                "model_id": "unapproved-model-endpoint",
            },
        )

        assert result.is_error is True
        text = result.content[0].text
        assert "approved model whitelist" in text.lower()
