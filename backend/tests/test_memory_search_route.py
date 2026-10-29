"""Automated contract, validation, error mapping, and authorization tests for POST /api/v1/memory/search (Step 16)."""

from datetime import datetime, timezone
import math
from typing import AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    EmptyPromptError,
    IncompatibleEmbeddingError,
    InactivePrincipalError,
    InvalidProviderResponseError,
    OwnershipViolationError,
    PromptValidationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.core.security import AuthorizedPrincipal, get_current_principal
from app.db.session import get_db_session
from app.main import app, create_application
from app.schemas.crispe import CrispeSections
from app.schemas.semantic_search import (
    SemanticMemorySearchRequest,
    SemanticMemorySearchResultItem,
    SemanticMemorySearchResponse,
    SemanticSearchItem,
    SemanticSearchResponse,
)
from app.services.semantic_search import (
    SemanticSearchService,
    get_semantic_search_service_dependency,
)


# ==============================================================================
# Fixtures and Test Doubles
# ==============================================================================

@pytest.fixture
def mock_crispe_sections() -> CrispeSections:
    """Fixture providing valid 6-section CRISPE prompt data."""
    return CrispeSections(
        context="Distributed OAuth 2.1 authentication microservice.",
        role="Principal Security Architect.",
        instruction="Design a token refresh interceptor with replay attack detection.",
        specification="Use Markdown tables. Zero sensitive secrets logged.",
        performance="Token refresh latency under 5ms at 99.9th percentile.",
        example="Input: Expired JWT. Output: Refreshed token pair with rotated refresh token.",
    )


@pytest.fixture
def test_principal() -> AuthorizedPrincipal:
    """Standard authenticated principal for route tests."""
    return AuthorizedPrincipal(
        id="usr_route_test_001",
        is_active=True,
        email="test@crispe.internal",
        display_name="Route Tester",
    )


def make_service_search_item(
    mock_sections: CrispeSections,
    prompt_id: Optional[uuid.UUID] = None,
    idea: str = "Design a token refresh interceptor",
    distance: float = 0.05,
    task_type: Optional[str] = "security",
) -> SemanticSearchItem:
    """Helper constructing an internal SemanticSearchItem."""
    pid = prompt_id or uuid.uuid4()
    return SemanticSearchItem(
        prompt_id=pid,
        original_idea=idea,
        sections=mock_sections,
        formatted_prompt=mock_sections.to_markdown(),
        task_type=task_type,
        model_id="text-embedding-004",
        distance=distance,
        similarity=round(1.0 - distance, 6),
        config_version="v1",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def fake_search_service(mock_crispe_sections: CrispeSections) -> SemanticSearchService:
    """Fixture creating a mock SemanticSearchService."""
    service = MagicMock(spec=SemanticSearchService)

    async def fake_search(
        session,
        *,
        principal: AuthorizedPrincipal,
        query: str,
        limit: int = 10,
        min_similarity: Optional[float] = None,
        task_type: Optional[str] = None,
        config_version: Optional[str] = None,
    ) -> SemanticSearchResponse:
        item = make_service_search_item(
            mock_crispe_sections,
            idea=f"Saved prompt matching '{query}'",
            distance=0.08,
            task_type=task_type or "backend",
        )
        return SemanticSearchResponse(
            query=query,
            total_results=1,
            items=[item],
            model_id="text-embedding-004",
            config_version="v1",
            search_metric="cosine_distance",
        )

    service.search = AsyncMock(side_effect=fake_search)
    return service


@pytest.fixture
async def authenticated_client(
    test_principal: AuthorizedPrincipal,
    fake_search_service: SemanticSearchService,
) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with authenticated principal and fake search service overridden."""
    mock_session = AsyncMock()

    app.dependency_overrides[get_current_principal] = lambda: test_principal
    app.dependency_overrides[get_semantic_search_service_dependency] = lambda: fake_search_service
    app.dependency_overrides[get_db_session] = lambda: mock_session

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
async def unauthenticated_client() -> AsyncGenerator[AsyncClient, None]:
    """Async test client without principal override (testing unauthenticated requests)."""
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        yield client

    app.dependency_overrides.clear()


# ==============================================================================
# 1. Successful Search (Category A)
# ==============================================================================

@pytest.mark.asyncio
async def test_search_memory_success_200(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
    mock_crispe_sections: CrispeSections,
) -> None:
    """POST /api/v1/memory/search returns 200 OK with validated SemanticMemorySearchResponse."""
    payload = {
        "query": "React token refresh and silent authentication",
        "limit": 5,
        "task_type": "frontend",
        "min_similarity": 0.75,
    }

    response = await authenticated_client.post("/api/v1/memory/search", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "React token refresh and silent authentication"
    assert data["result_count"] == 1
    assert len(data["results"]) == 1

    result_item = data["results"][0]
    assert "prompt_id" in result_item
    assert uuid.UUID(result_item["prompt_id"])
    assert result_item["original_idea"] == "Saved prompt matching 'React token refresh and silent authentication'"
    assert result_item["task_type"] == "frontend"
    assert result_item["distance"] == 0.08
    assert result_item["similarity"] == 0.92

    # Verify structured CRISPE sections
    sections = result_item["generated_prompt"]
    assert sections["context"] == mock_crispe_sections.context
    assert sections["role"] == mock_crispe_sections.role
    assert sections["instruction"] == mock_crispe_sections.instruction
    assert sections["specification"] == mock_crispe_sections.specification
    assert sections["performance"] == mock_crispe_sections.performance
    assert sections["example"] == mock_crispe_sections.example

    # Verify canonical formatted Markdown headers
    formatted = result_item["formatted_prompt"]
    assert "# C - Context" in formatted
    assert "# R - Role" in formatted

    # Verify raw vector is strictly omitted
    assert "vector" not in result_item
    assert "embedding" not in result_item

    # Verify telemetry headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers

    # Verify service received exact parameters
    call_kwargs = fake_search_service.search.call_args.kwargs
    assert call_kwargs["query"] == "React token refresh and silent authentication"
    assert call_kwargs["limit"] == 5
    assert call_kwargs["task_type"] == "frontend"
    assert call_kwargs["min_similarity"] == 0.75


@pytest.mark.asyncio
async def test_search_memory_preserves_ordering_and_distance(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
    mock_crispe_sections: CrispeSections,
) -> None:
    """Search response preserves exact distance ordering from service."""
    p1 = make_service_search_item(mock_crispe_sections, idea="Closest match", distance=0.02)
    p2 = make_service_search_item(mock_crispe_sections, idea="Second closest match", distance=0.15)
    p3 = make_service_search_item(mock_crispe_sections, idea="Third match", distance=0.45)

    async def multi_match_search(*args, **kwargs) -> SemanticSearchResponse:
        return SemanticSearchResponse(
            query=kwargs["query"],
            total_results=3,
            items=[p1, p2, p3],
            model_id="text-embedding-004",
            config_version="v1",
            search_metric="cosine_distance",
        )

    fake_search_service.search = AsyncMock(side_effect=multi_match_search)

    response = await authenticated_client.post(
        "/api/v1/memory/search",
        json={"query": "database indexing"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["result_count"] == 3
    results = data["results"]
    assert results[0]["distance"] == 0.02
    assert results[1]["distance"] == 0.15
    assert results[2]["distance"] == 0.45
    assert results[0]["similarity"] > results[1]["similarity"] > results[2]["similarity"]


# ==============================================================================
# 2. Request Validation (Category B)
# ==============================================================================

@pytest.mark.asyncio
async def test_search_memory_missing_query_returns_422(
    authenticated_client: AsyncClient,
) -> None:
    """Missing required 'query' field returns HTTP 422 Unprocessable Content."""
    response = await authenticated_client.post("/api/v1/memory/search", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert any("query" in err["loc"] for err in data["details"])


@pytest.mark.asyncio
async def test_search_memory_empty_query_returns_422(
    authenticated_client: AsyncClient,
) -> None:
    """Empty or whitespace-only 'query' returns HTTP 422."""
    for empty_val in ["", "   ", "\t\n  "]:
        response = await authenticated_client.post(
            "/api/v1/memory/search", json={"query": empty_val}
        )
        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] in ["VALIDATION_ERROR", "EMPTY_PROMPT_ERROR"]


@pytest.mark.asyncio
async def test_search_memory_query_too_long_returns_422(
    authenticated_client: AsyncClient,
) -> None:
    """Query exceeding 1,000 characters returns HTTP 422."""
    long_query = "x" * 1001
    response = await authenticated_client.post(
        "/api/v1/memory/search", json={"query": long_query}
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] in ["VALIDATION_ERROR", "PROMPT_VALIDATION_ERROR"]


@pytest.mark.asyncio
async def test_search_memory_invalid_limit_returns_422(
    authenticated_client: AsyncClient,
) -> None:
    """Limit < 1 or > 50 returns HTTP 422."""
    for invalid_limit in [0, -1, -10, 51, 100]:
        response = await authenticated_client.post(
            "/api/v1/memory/search",
            json={"query": "valid query", "limit": invalid_limit},
        )
        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_search_memory_invalid_similarity_threshold_returns_422(
    authenticated_client: AsyncClient,
) -> None:
    """min_similarity outside [-1.0, 1.0] returns HTTP 422."""
    for invalid_sim in [-1.5, 1.05, 2.0, 100.0]:
        response = await authenticated_client.post(
            "/api/v1/memory/search",
            json={"query": "valid query", "min_similarity": invalid_sim},
        )
        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_search_memory_extra_fields_forbidden_422(
    authenticated_client: AsyncClient,
) -> None:
    """Unapproved extra fields (e.g. user_id, config_version, model_id) are rejected with 422."""
    forbidden_payloads = [
        {"query": "valid query", "user_id": "malicious_user"},
        {"query": "valid query", "owner_id": "malicious_user"},
        {"query": "valid query", "config_version": "v999"},
        {"query": "valid query", "model_id": "arbitrary_model"},
        {"query": "valid query", "sql": "DROP TABLE prompt_embeddings;"},
    ]
    for payload in forbidden_payloads:
        response = await authenticated_client.post("/api/v1/memory/search", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["error_code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_search_memory_malformed_json_returns_422(
    authenticated_client: AsyncClient,
) -> None:
    """Non-JSON or malformed payload returns HTTP 422."""
    response = await authenticated_client.post(
        "/api/v1/memory/search",
        content="this is not a valid json object",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"


# ==============================================================================
# 3. Empty Results Behaviour (Category C)
# ==============================================================================

@pytest.mark.asyncio
async def test_search_memory_empty_results_returns_200(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
) -> None:
    """A search that matches 0 saved prompts returns 200 OK with result_count=0 and results=[]."""
    async def empty_search(*args, **kwargs) -> SemanticSearchResponse:
        return SemanticSearchResponse(
            query=kwargs["query"],
            total_results=0,
            items=[],
            model_id="text-embedding-004",
            config_version="v1",
            search_metric="cosine_distance",
        )

    fake_search_service.search = AsyncMock(side_effect=empty_search)

    response = await authenticated_client.post(
        "/api/v1/memory/search",
        json={"query": "query with no matches in memory"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "query with no matches in memory"
    assert data["result_count"] == 0
    assert data["results"] == []


# ==============================================================================
# 4. Service Failures and Error Mapping (Category D)
# ==============================================================================

@pytest.mark.asyncio
async def test_search_memory_unauthenticated_returns_401(
    unauthenticated_client: AsyncClient,
) -> None:
    """Requests without trusted authenticated principal return HTTP 401."""
    response = await unauthenticated_client.post(
        "/api/v1/memory/search",
        json={"query": "test query"},
    )
    assert response.status_code == 401
    data = response.json()
    assert data["error_code"] == "AUTHENTICATION_REQUIRED"
    assert "authentication is required" in data["message"].lower()


@pytest.mark.asyncio
async def test_search_memory_inactive_principal_returns_403(
    fake_search_service: SemanticSearchService,
) -> None:
    """Inactive or suspended principal returns HTTP 403 INACTIVE_PRINCIPAL."""
    inactive_principal = AuthorizedPrincipal(id="usr_inactive_99", is_active=False)

    app.dependency_overrides[get_current_principal] = lambda: inactive_principal
    app.dependency_overrides[get_semantic_search_service_dependency] = lambda: fake_search_service
    app.dependency_overrides[get_db_session] = lambda: AsyncMock()

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/api/v1/memory/search", json={"query": "test query"})
        assert response.status_code == 403
        data = response.json()
        assert data["error_code"] == "INACTIVE_PRINCIPAL"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_memory_incompatible_embedding_returns_400(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
) -> None:
    """Incompatible embedding dimensions or config version returns HTTP 400."""
    fake_search_service.search = AsyncMock(
        side_effect=IncompatibleEmbeddingError("Query embedding dimension mismatch: expected 768, got 512.")
    )

    response = await authenticated_client.post(
        "/api/v1/memory/search",
        json={"query": "incompatible vector test"},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "INCOMPATIBLE_EMBEDDING_ERROR"
    assert "dimension mismatch" in data["message"].lower()


@pytest.mark.asyncio
async def test_search_memory_provider_timeout_returns_504(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
) -> None:
    """Upstream embedding provider timeout returns HTTP 504."""
    fake_search_service.search = AsyncMock(
        side_effect=ProviderTimeoutError("Upstream Google GenAI embedding request timed out.")
    )

    response = await authenticated_client.post(
        "/api/v1/memory/search",
        json={"query": "timeout test"},
    )
    assert response.status_code == 504
    data = response.json()
    assert data["error_code"] == "PROVIDER_TIMEOUT"


@pytest.mark.asyncio
async def test_search_memory_provider_rate_limit_returns_429(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
) -> None:
    """Upstream embedding provider rate limit exhaustion returns HTTP 429."""
    fake_search_service.search = AsyncMock(
        side_effect=ProviderRateLimitError("Upstream embedding quota exceeded. Retry later.")
    )

    response = await authenticated_client.post(
        "/api/v1/memory/search",
        json={"query": "rate limit test"},
    )
    assert response.status_code == 429
    data = response.json()
    assert data["error_code"] == "PROVIDER_RATE_LIMIT"


@pytest.mark.asyncio
async def test_search_memory_invalid_provider_response_returns_502(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
) -> None:
    """Malformed provider embedding output returns HTTP 502."""
    fake_search_service.search = AsyncMock(
        side_effect=InvalidProviderResponseError("Provider returned malformed vector with NaN values.")
    )

    response = await authenticated_client.post(
        "/api/v1/memory/search",
        json={"query": "invalid response test"},
    )
    assert response.status_code == 502
    data = response.json()
    assert data["error_code"] == "INVALID_PROVIDER_RESPONSE"


@pytest.mark.asyncio
async def test_search_memory_database_failure_returns_500_or_503(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
) -> None:
    """PostgreSQL connectivity or query failure returns controlled 503 or 500 error."""
    # 1. Connection failure -> 503
    fake_search_service.search = AsyncMock(
        side_effect=DatabaseConnectionError("Cannot connect to PostgreSQL host on port 5432.")
    )
    res_503 = await authenticated_client.post("/api/v1/memory/search", json={"query": "db failure"})
    assert res_503.status_code == 503
    assert res_503.json()["error_code"] == "DATABASE_CONNECTION_ERROR"

    # 2. General database error -> 500
    fake_search_service.search = AsyncMock(
        side_effect=DatabaseError("Failed to execute semantic similarity query: table locked")
    )
    res_500 = await authenticated_client.post("/api/v1/memory/search", json={"query": "db failure"})
    assert res_500.status_code == 500
    assert res_500.json()["error_code"] == "DATABASE_ERROR"


@pytest.mark.asyncio
async def test_search_memory_unhandled_exception_returns_500_masked(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
) -> None:
    """Unhandled internal exception returns masked 500 error without exposing stack trace or credentials."""
    fake_search_service.search = AsyncMock(side_effect=RuntimeError("Unforeseen system memory exception"))

    response = await authenticated_client.post("/api/v1/memory/search", json={"query": "unhandled"})
    assert response.status_code == 500
    data = response.json()
    assert data["error_code"] == "INTERNAL_SERVER_ERROR"
    assert "unexpected server error" in data["message"].lower()
    # Ensure raw exception text is NOT exposed in response
    assert "Unforeseen system memory exception" not in data["message"]


# ==============================================================================
# 5. Authorization and Tenant Isolation (Category E)
# ==============================================================================

@pytest.mark.asyncio
async def test_search_memory_delegates_to_authenticated_principal(
    authenticated_client: AsyncClient,
    fake_search_service: SemanticSearchService,
    test_principal: AuthorizedPrincipal,
) -> None:
    """The endpoint strictly forwards the authenticated principal context to the search service."""
    await authenticated_client.post("/api/v1/memory/search", json={"query": "tenant test"})

    assert fake_search_service.search.called
    call_principal = fake_search_service.search.call_args.kwargs["principal"]
    assert call_principal.id == test_principal.id
    assert call_principal.is_active is True


# ==============================================================================
# 6. Regression: Health and Prompt Generation (Category F)
# ==============================================================================

@pytest.mark.asyncio
async def test_health_and_prompts_routes_remain_functional(
    authenticated_client: AsyncClient,
) -> None:
    """Adding /api/v1/memory/search does not regress health or prompt generation routes."""
    # 1. Health liveness
    health_res = await authenticated_client.get("/api/v1/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "ok"

    # 2. Prompts generate validation regression
    gen_res = await authenticated_client.post("/api/v1/prompts/generate", json={})
    assert gen_res.status_code == 422
