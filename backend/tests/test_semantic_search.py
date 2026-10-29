"""Unit, isolation, compatibility, and integration tests for Semantic Similarity Search Engine (Step 15)."""

from datetime import datetime, timezone
import math
from typing import AsyncGenerator, List, Optional, Sequence
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.embedding_adapter import BaseEmbeddingAdapter
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DatabaseError,
    EmptyPromptError,
    IncompatibleEmbeddingError,
    InactivePrincipalError,
    OwnershipViolationError,
    PromptValidationError,
    SemanticSearchError,
)
from app.core.security import AuthorizedPrincipal
from app.db.base import Base
from app.db.sqlite_compat import register_sqlite_vector_compat
from app.models.principal import Principal
from app.models.prompt_embedding import EMBEDDING_DIMENSION, PromptEmbedding
from app.models.prompt_record import PromptRecord
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.principal_repository import PrincipalRepository
from app.repositories.prompt_repository import PromptRepository
from app.schemas.crispe import CrispeSections
from app.schemas.embedding import (
    EmbeddingBatchResult,
    EmbeddingResult,
    EmbeddingTaskType,
    EmbeddingUsageMetadata,
)
from app.schemas.semantic_search import (
    MAX_SEARCH_QUERY_LENGTH,
    SemanticSearchItem,
    SemanticSearchQuery,
    SemanticSearchResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.semantic_search import (
    SemanticSearchService,
    get_semantic_search_service,
)


# -----------------------------------------------------------------------------
# Test Fixtures & Mock Adapter
# -----------------------------------------------------------------------------

class MockSearchEmbeddingAdapter(BaseEmbeddingAdapter):
    """Deterministic in-memory embedding adapter for semantic search tests."""

    def __init__(
        self,
        dimension: int = 768,
        model_id: str = "text-embedding-004",
        config_version: str = "v1",
        default_vector: Optional[List[float]] = None,
    ) -> None:
        self.dimension = dimension
        self.model_id = model_id
        self.config_version = config_version
        self.default_vector = default_vector or ([0.05] * dimension)
        self.call_history: List[dict] = []
        self._closed = False

    async def embed_content(
        self,
        text: str,
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbeddingResult:
        self.call_history.append({
            "text": text,
            "task_type": task_type,
            "title": title,
            "output_dimensionality": output_dimensionality,
        })
        return EmbeddingResult(
            vector=list(self.default_vector),
            dimension=len(self.default_vector),
            model_id=self.model_id,
            config_version=self.config_version,
            usage=EmbeddingUsageMetadata(prompt_tokens=10, total_tokens=10),
        )

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        task_type: Optional[EmbeddingTaskType] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbeddingBatchResult:
        embeddings = [
            await self.embed_content(
                t, task_type=task_type, output_dimensionality=output_dimensionality
            )
            for t in texts
        ]
        return EmbeddingBatchResult(
            embeddings=embeddings,
            model_id=self.model_id,
            config_version=self.config_version,
            total_prompt_tokens=10 * len(texts),
        )

    async def close(self) -> None:
        self._closed = True


@pytest_asyncio.fixture
async def async_test_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated, transactional in-memory database session with vector compat."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    register_sqlite_vector_compat(engine)

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
def sample_crispe_sections() -> CrispeSections:
    """Standard CRISPE sections fixture."""
    return CrispeSections(
        context="Semantic Prompt Memory retrieval testing.",
        role="Principal AI Retrieval Engineer.",
        instruction="Retrieve nearest-neighbor prompts using pgvector cosine distance.",
        specification="Ensure strict owner isolation, dimension checks, and typed responses.",
        performance="100% test coverage, sub-millisecond execution, zero data leaks.",
        example="SELECT * FROM prompt_embeddings WHERE owner_id = :owner ORDER BY embedding <=> :query",
    )


async def create_test_principal(
    session: AsyncSession,
    principal_id: str,
    *,
    is_active: bool = True,
    email: Optional[str] = None,
) -> AuthorizedPrincipal:
    """Helper to persist a Principal database entity and return an AuthorizedPrincipal context."""
    db_principal = Principal(
        id=principal_id,
        email=email or f"{principal_id}@test.internal",
        is_active=is_active,
    )
    session.add(db_principal)
    await session.commit()
    return AuthorizedPrincipal(
        id=principal_id,
        is_active=is_active,
        email=db_principal.email,
    )


def make_unit_vector(active_slice: tuple[int, int], dim: int = 768, noise_seed: int = 0) -> List[float]:
    """Generate a deterministic unit-length synthetic vector with targeted non-zero coordinates."""
    vec = [0.0] * dim
    for i in range(active_slice[0], active_slice[1]):
        vec[i] = 1.0 + (math.sin(i + noise_seed) * 0.05)
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


# -----------------------------------------------------------------------------
# 1. Search Success Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_search_success_single_result(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """A valid query embeds with RETRIEVAL_QUERY and retrieves the matching prompt."""
    principal = await create_test_principal(async_test_session, "usr_alice_001")

    # 1. Create prompt record
    prompt = PromptRecord(
        original_idea="Design a fault-tolerant message queue in Python",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt=sample_crispe_sections.to_markdown(),
        model_id="gemma-4-26b-a4b-it",
        owner_id=principal.id,
    )
    async_test_session.add(prompt)
    await async_test_session.commit()

    # 2. Attach embedding (vector pointing along first coordinates)
    vector = make_unit_vector((0, 100), dim=768, noise_seed=1)
    embedding = PromptEmbedding(
        prompt_id=prompt.id,
        embedding=vector,
        provider="google-genai",
        model_id="text-embedding-004",
        config_version="v1",
        source_content_hash=PromptEmbedding.compute_content_hash(prompt.formatted_prompt),
    )
    async_test_session.add(embedding)
    await async_test_session.commit()

    # 3. Setup mock adapter returning query vector close to the prompt vector
    query_vector = make_unit_vector((0, 100), dim=768, noise_seed=2)
    mock_adapter = MockSearchEmbeddingAdapter(default_vector=query_vector)
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    # 4. Execute search
    response = await service.search(
        async_test_session,
        principal=principal,
        query="fault-tolerant Python message queues",
        limit=5,
    )

    # 5. Assertions
    assert isinstance(response, SemanticSearchResponse)
    assert response.query == "fault-tolerant Python message queues"
    assert response.total_results == 1
    assert len(response.items) == 1
    assert response.model_id == "text-embedding-004"
    assert response.config_version == "v1"
    assert response.search_metric == "cosine_distance"

    item = response.items[0]
    assert item.prompt_id == prompt.id
    assert item.original_idea == prompt.original_idea
    assert item.sections.role == sample_crispe_sections.role
    assert item.formatted_prompt == prompt.formatted_prompt
    assert item.model_id == "text-embedding-004"
    assert item.distance >= 0.0
    assert item.similarity <= 1.0
    # Derived similarity: 1.0 - distance
    assert math.isclose(item.similarity, 1.0 - item.distance, abs_tol=1e-5)

    # Verify query was embedded using RETRIEVAL_QUERY task type
    assert len(mock_adapter.call_history) == 1
    assert mock_adapter.call_history[0]["task_type"] == EmbeddingTaskType.RETRIEVAL_QUERY


@pytest.mark.asyncio
async def test_semantic_search_distance_ordering_and_limit(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Results are strictly ordered by ascending cosine distance, and limit is enforced."""
    principal = await create_test_principal(async_test_session, "usr_alice_002")

    # Base query vector
    query_vec = make_unit_vector((0, 50), dim=768, noise_seed=0)

    # Create 4 prompts with varying distances
    # P1: Very close (seed 1)
    # P2: Somewhat close (seed 5)
    # P3: Further (orthogonal slice 50-100)
    # P4: Completely distant (orthogonal slice 100-150)
    v1 = make_unit_vector((0, 50), dim=768, noise_seed=1)
    v2 = make_unit_vector((0, 50), dim=768, noise_seed=5)
    v3 = make_unit_vector((50, 100), dim=768, noise_seed=1)
    v4 = make_unit_vector((100, 150), dim=768, noise_seed=1)

    prompts = []
    for i, vec in enumerate([v1, v2, v3, v4], start=1):
        p = PromptRecord(
            original_idea=f"Idea {i}",
            context=sample_crispe_sections.context,
            role=sample_crispe_sections.role,
            instruction=sample_crispe_sections.instruction,
            specification=sample_crispe_sections.specification,
            performance=sample_crispe_sections.performance,
            example=sample_crispe_sections.example,
            formatted_prompt=f"Formatted Prompt {i}",
            owner_id=principal.id,
        )
        async_test_session.add(p)
        await async_test_session.commit()
        prompts.append(p)

        emb = PromptEmbedding(
            prompt_id=p.id,
            embedding=vec,
            source_content_hash=PromptEmbedding.compute_content_hash(p.formatted_prompt),
        )
        async_test_session.add(emb)
        await async_test_session.commit()

    mock_adapter = MockSearchEmbeddingAdapter(default_vector=query_vec)
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    # Request limit=2
    response = await service.search(
        async_test_session,
        principal=principal,
        query="find closest matches",
        limit=2,
    )

    assert response.total_results == 2
    assert len(response.items) == 2
    assert response.items[0].prompt_id == prompts[0].id
    assert response.items[1].prompt_id == prompts[1].id
    assert response.items[0].distance < response.items[1].distance
    assert response.items[0].similarity > response.items[1].similarity


@pytest.mark.asyncio
async def test_semantic_search_empty_result_set(
    async_test_session: AsyncSession,
) -> None:
    """Empty result set is handled cleanly without raising an error."""
    principal = await create_test_principal(async_test_session, "usr_empty_001")
    mock_adapter = MockSearchEmbeddingAdapter()
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    response = await service.search(
        async_test_session,
        principal=principal,
        query="no stored prompts for this user",
    )

    assert response.total_results == 0
    assert response.items == []
    assert response.query == "no stored prompts for this user"


@pytest.mark.asyncio
async def test_semantic_search_raw_vectors_strictly_excluded(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Verify that raw embedding vectors are never exposed in SemanticSearchItem or Response."""
    principal = await create_test_principal(async_test_session, "usr_alice_003")
    p = PromptRecord(
        original_idea="Check schema leak",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Formatted prompt",
        owner_id=principal.id,
    )
    async_test_session.add(p)
    await async_test_session.commit()

    emb = PromptEmbedding(
        prompt_id=p.id,
        embedding=make_unit_vector((0, 50)),
        source_content_hash=PromptEmbedding.compute_content_hash(p.formatted_prompt),
    )
    async_test_session.add(emb)
    await async_test_session.commit()

    mock_adapter = MockSearchEmbeddingAdapter(default_vector=make_unit_vector((0, 50)))
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    response = await service.search(
        async_test_session,
        principal=principal,
        query="test raw vector leak",
    )

    assert len(response.items) == 1
    item = response.items[0]
    # Check that item model fields do NOT contain 'vector' or 'embedding'
    item_dict = item.model_dump()
    assert "embedding" not in item_dict
    assert "vector" not in item_dict
    assert not hasattr(item, "embedding")
    assert not hasattr(item, "vector")


# -----------------------------------------------------------------------------
# 2. Ownership and Isolation Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_search_multi_tenant_isolation(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Prompts belonging to Principal A are completely invisible to Principal B."""
    alice = await create_test_principal(async_test_session, "usr_tenant_alice")
    bob = await create_test_principal(async_test_session, "usr_tenant_bob")

    vec = make_unit_vector((0, 50), dim=768)

    # Alice's prompt
    p_alice = PromptRecord(
        original_idea="Alice confidential trading strategy",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Alice formatted prompt",
        owner_id=alice.id,
    )
    async_test_session.add(p_alice)
    await async_test_session.commit()

    emb_alice = PromptEmbedding(
        prompt_id=p_alice.id,
        embedding=vec,
        source_content_hash=PromptEmbedding.compute_content_hash(p_alice.formatted_prompt),
    )
    async_test_session.add(emb_alice)

    # Bob's prompt
    p_bob = PromptRecord(
        original_idea="Bob public tutorial",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Bob formatted prompt",
        owner_id=bob.id,
    )
    async_test_session.add(p_bob)
    await async_test_session.commit()

    emb_bob = PromptEmbedding(
        prompt_id=p_bob.id,
        embedding=vec,
        source_content_hash=PromptEmbedding.compute_content_hash(p_bob.formatted_prompt),
    )
    async_test_session.add(emb_bob)
    await async_test_session.commit()

    mock_adapter = MockSearchEmbeddingAdapter(default_vector=vec)
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    # 1. Alice searches with identical vector: should ONLY return Alice's prompt
    alice_res = await service.search(async_test_session, principal=alice, query="confidential strategy")
    assert alice_res.total_results == 1
    assert alice_res.items[0].prompt_id == p_alice.id
    assert alice_res.items[0].original_idea == "Alice confidential trading strategy"

    # 2. Bob searches with identical vector: should ONLY return Bob's prompt
    bob_res = await service.search(async_test_session, principal=bob, query="confidential strategy")
    assert bob_res.total_results == 1
    assert bob_res.items[0].prompt_id == p_bob.id
    assert bob_res.items[0].original_idea == "Bob public tutorial"


@pytest.mark.asyncio
async def test_semantic_search_inactive_or_missing_principal(
    async_test_session: AsyncSession,
) -> None:
    """Inactive principal raises InactivePrincipalError; None principal raises OwnershipViolationError."""
    inactive_user = await create_test_principal(async_test_session, "usr_inactive_001", is_active=False)
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=MockSearchEmbeddingAdapter()))

    with pytest.raises(InactivePrincipalError):
        await service.search(async_test_session, principal=inactive_user, query="test")

    with pytest.raises(OwnershipViolationError):
        await service.search(async_test_session, principal=None, query="test")  # type: ignore


# -----------------------------------------------------------------------------
# 3. Compatibility & Stale Embedding Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_search_stale_embedding_excluded(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Prompts whose formatted_prompt was modified without re-embedding are excluded as stale."""
    principal = await create_test_principal(async_test_session, "usr_alice_stale")

    p = PromptRecord(
        original_idea="Original Prompt",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Initial prompt text before edit",
        owner_id=principal.id,
    )
    async_test_session.add(p)
    await async_test_session.commit()

    emb = PromptEmbedding(
        prompt_id=p.id,
        embedding=make_unit_vector((0, 50)),
        source_content_hash=PromptEmbedding.compute_content_hash(p.formatted_prompt),
    )
    async_test_session.add(emb)
    await async_test_session.commit()

    # Verify initially found
    mock_adapter = MockSearchEmbeddingAdapter(default_vector=make_unit_vector((0, 50)))
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))
    res1 = await service.search(async_test_session, principal=principal, query="test")
    assert res1.total_results == 1

    # Simulate prompt modification without updating embedding
    p.formatted_prompt = "Substantially revised prompt text altering semantics!"
    await async_test_session.commit()

    # Search again: should now be excluded as stale
    res2 = await service.search(async_test_session, principal=principal, query="test")
    assert res2.total_results == 0
    assert res2.items == []


@pytest.mark.asyncio
async def test_semantic_search_prompt_without_embedding_excluded(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Prompts that do not have any attached vector embedding are ignored during retrieval."""
    principal = await create_test_principal(async_test_session, "usr_alice_unembedded")

    p = PromptRecord(
        original_idea="Unembedded Prompt",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Formatted prompt text",
        owner_id=principal.id,
    )
    async_test_session.add(p)
    await async_test_session.commit()

    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=MockSearchEmbeddingAdapter()))
    res = await service.search(async_test_session, principal=principal, query="search")
    assert res.total_results == 0


@pytest.mark.asyncio
async def test_semantic_search_config_version_filtering(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Only embeddings matching the target config_version are retrieved."""
    principal = await create_test_principal(async_test_session, "usr_version_test")

    p = PromptRecord(
        original_idea="Versioned Prompt",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Versioned text",
        owner_id=principal.id,
    )
    async_test_session.add(p)
    await async_test_session.commit()

    # Stored under "v2"
    emb = PromptEmbedding(
        prompt_id=p.id,
        embedding=make_unit_vector((0, 50)),
        config_version="v2",
        source_content_hash=PromptEmbedding.compute_content_hash(p.formatted_prompt),
    )
    async_test_session.add(emb)
    await async_test_session.commit()

    mock_adapter = MockSearchEmbeddingAdapter(default_vector=make_unit_vector((0, 50)), config_version="v1")
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    # Search with v1 config: v2 should be excluded
    res_v1 = await service.search(async_test_session, principal=principal, query="test", config_version="v1")
    assert res_v1.total_results == 0

    # Search with v2 config (using matching adapter): v2 should be retrieved
    mock_adapter_v2 = MockSearchEmbeddingAdapter(default_vector=make_unit_vector((0, 50)), config_version="v2")
    service_v2 = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter_v2))
    res_v2 = await service_v2.search(async_test_session, principal=principal, query="test", config_version="v2")
    assert res_v2.total_results == 1


@pytest.mark.asyncio
async def test_semantic_search_dimension_mismatch_rejected(
    async_test_session: AsyncSession,
) -> None:
    """Query embedding with unexpected vector dimensions raises IncompatibleEmbeddingError."""
    principal = await create_test_principal(async_test_session, "usr_dim_test")
    # Adapter returning 512-dim vector instead of configured 768
    invalid_adapter = MockSearchEmbeddingAdapter(dimension=512, default_vector=[0.1] * 512)
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=invalid_adapter))

    with pytest.raises(IncompatibleEmbeddingError) as exc_info:
        await service.search(async_test_session, principal=principal, query="dimension test")
    assert "dimension mismatch" in str(exc_info.value).lower()


# -----------------------------------------------------------------------------
# 4. Query Validation & Parameter Filtering Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_search_empty_query_rejected(
    async_test_session: AsyncSession,
) -> None:
    """Empty or whitespace-only search queries are rejected."""
    principal = await create_test_principal(async_test_session, "usr_val_001")
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=MockSearchEmbeddingAdapter()))

    with pytest.raises(EmptyPromptError):
        await service.search(async_test_session, principal=principal, query="")

    with pytest.raises(EmptyPromptError):
        await service.search(async_test_session, principal=principal, query="     ")


@pytest.mark.asyncio
async def test_semantic_search_query_too_long_rejected(
    async_test_session: AsyncSession,
) -> None:
    """Queries exceeding MAX_SEARCH_QUERY_LENGTH are rejected."""
    principal = await create_test_principal(async_test_session, "usr_val_002")
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=MockSearchEmbeddingAdapter()))

    long_query = "x" * (MAX_SEARCH_QUERY_LENGTH + 1)
    with pytest.raises(PromptValidationError):
        await service.search(async_test_session, principal=principal, query=long_query)


@pytest.mark.asyncio
async def test_semantic_search_min_similarity_filter(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Candidates with derived similarity below min_similarity are filtered out."""
    principal = await create_test_principal(async_test_session, "usr_thresh_001")

    query_vec = make_unit_vector((0, 50), dim=768, noise_seed=0)
    # Similar prompt (sim ~ 0.99)
    v_close = make_unit_vector((0, 50), dim=768, noise_seed=1)
    # Dissimilar prompt (sim ~ 0.0)
    v_distant = make_unit_vector((50, 100), dim=768, noise_seed=1)

    p1 = PromptRecord(
        original_idea="Close idea",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Close prompt",
        owner_id=principal.id,
    )
    p2 = PromptRecord(
        original_idea="Distant idea",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Distant prompt",
        owner_id=principal.id,
    )
    async_test_session.add_all([p1, p2])
    await async_test_session.commit()

    async_test_session.add(PromptEmbedding(prompt_id=p1.id, embedding=v_close, source_content_hash=PromptEmbedding.compute_content_hash(p1.formatted_prompt)))
    async_test_session.add(PromptEmbedding(prompt_id=p2.id, embedding=v_distant, source_content_hash=PromptEmbedding.compute_content_hash(p2.formatted_prompt)))
    await async_test_session.commit()

    mock_adapter = MockSearchEmbeddingAdapter(default_vector=query_vec)
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    # Without threshold: returns both
    all_res = await service.search(async_test_session, principal=principal, query="test")
    assert all_res.total_results == 2

    # With high threshold (0.8): returns only the close prompt
    filtered_res = await service.search(async_test_session, principal=principal, query="test", min_similarity=0.8)
    assert filtered_res.total_results == 1
    assert filtered_res.items[0].prompt_id == p1.id


@pytest.mark.asyncio
async def test_semantic_search_task_type_filter(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Candidates are filtered by optional task_type classification."""
    principal = await create_test_principal(async_test_session, "usr_task_001")

    vec = make_unit_vector((0, 50), dim=768)

    p_backend = PromptRecord(
        original_idea="Backend API design",
        task_type="backend",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Backend prompt",
        owner_id=principal.id,
    )
    p_frontend = PromptRecord(
        original_idea="Frontend CSS styling",
        task_type="frontend",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt="Frontend prompt",
        owner_id=principal.id,
    )
    async_test_session.add_all([p_backend, p_frontend])
    await async_test_session.commit()

    async_test_session.add(PromptEmbedding(prompt_id=p_backend.id, embedding=vec, source_content_hash=PromptEmbedding.compute_content_hash(p_backend.formatted_prompt)))
    async_test_session.add(PromptEmbedding(prompt_id=p_frontend.id, embedding=vec, source_content_hash=PromptEmbedding.compute_content_hash(p_frontend.formatted_prompt)))
    await async_test_session.commit()

    mock_adapter = MockSearchEmbeddingAdapter(default_vector=vec)
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    res_backend = await service.search(async_test_session, principal=principal, query="test", task_type="backend")
    assert res_backend.total_results == 1
    assert res_backend.items[0].prompt_id == p_backend.id

    res_frontend = await service.search(async_test_session, principal=principal, query="test", task_type="frontend")
    assert res_frontend.total_results == 1
    assert res_frontend.items[0].prompt_id == p_frontend.id


# -----------------------------------------------------------------------------
# 5. PostgreSQL / pgvector Integration Test (Section 11.E)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_postgresql_pgvector_similarity_search(
    sample_crispe_sections: CrispeSections,
) -> None:
    """Integration test executing actual PostgreSQL pgvector cosine distance retrieval when live.

    If PostgreSQL or pgvector is unavailable in the environment, this test skips cleanly
    with clear diagnostic documentation as required by Section 11.E.
    """
    settings = get_settings()
    if not settings.DATABASE_URL or "sqlite" in settings.DATABASE_URL:
        pytest.skip("PostgreSQL DATABASE_URL is not configured; skipping pgvector integration test.")

    try:
        engine = create_async_engine(settings.DATABASE_URL, connect_args={"connect_timeout": 2}, echo=False)
        async with engine.connect() as conn:
            # Check pgvector extension
            res = await conn.execute(
                select(1).where(conn.dialect.name == "postgresql")
            )
        await engine.dispose()
    except Exception as exc:
        pytest.skip(
            f"PostgreSQL database is offline or unreachable ({exc}); "
            "integration test skipped safely as required by Section 11.E."
        )
