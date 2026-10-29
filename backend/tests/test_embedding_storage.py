"""Automated unit and persistence tests for pgvector embedding storage (Step 14)."""

import math
from pathlib import Path
from typing import AsyncGenerator
import uuid
import pytest
import pytest_asyncio
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import event, select
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.exceptions import (
    DatabaseError,
    InvalidProviderResponseError,
    OwnershipViolationError,
)
from app.core.security import AuthorizedPrincipal
from app.db.base import Base
from app.models.principal import Principal
from app.models.prompt_embedding import EMBEDDING_DIMENSION, PromptEmbedding
from app.models.prompt_record import PromptRecord
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.principal_repository import PrincipalRepository
from app.repositories.prompt_repository import PromptRepository
from app.schemas.crispe import CrispeSections
from app.schemas.embedding import EmbeddingResult, EmbeddingUsageMetadata


@pytest_asyncio.fixture
async def async_test_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated, transactional in-memory database session."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

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
        context="PostgreSQL pgvector persistence layer.",
        role="Principal Database Architect.",
        instruction="Store and retrieve high-dimensional embedding vectors with versioning.",
        specification="Enforce 768-dim schema contract, parent foreign key, and staleness detection.",
        performance="Zero data corruption, idempotent upserts, and clean cascading deletions.",
        example="PromptEmbedding attached to PromptRecord with SHA-256 content digest.",
    )


def make_embedding_result(
    dim: int = 768,
    model_id: str = "text-embedding-004",
    config_version: str = "v1",
    val: float = 0.05,
) -> EmbeddingResult:
    """Helper constructing a valid EmbeddingResult."""
    return EmbeddingResult(
        vector=[val] * dim,
        dimension=dim,
        model_id=model_id,
        config_version=config_version,
        usage=EmbeddingUsageMetadata(prompt_tokens=25, total_tokens=25),
    )


# -----------------------------------------------------------------------------
# 1. Schema & Model Definition Tests
# -----------------------------------------------------------------------------

def test_prompt_embedding_instantiation() -> None:
    """PromptEmbedding model instantiates cleanly with default values and UUID."""
    prompt_id = uuid.uuid4()
    test_vec = [0.123] * 768
    embedding = PromptEmbedding(
        prompt_id=prompt_id,
        embedding=test_vec,
        source_content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )

    assert isinstance(embedding.id, uuid.UUID)
    assert embedding.prompt_id == prompt_id
    assert embedding.provider == "google-genai"
    assert embedding.model_id == "text-embedding-004"
    assert embedding.config_version == "v1"
    assert len(embedding.embedding) == 768


def test_content_hash_computation() -> None:
    """compute_content_hash returns deterministic SHA-256 hex digest."""
    text1 = "CRISPE framework context specification"
    hash1 = PromptEmbedding.compute_content_hash(text1)
    hash2 = PromptEmbedding.compute_content_hash("  " + text1 + "\n\t")
    hash_diff = PromptEmbedding.compute_content_hash("Different text content")

    assert len(hash1) == 64
    assert hash1 == hash2  # Normalization trims outer whitespace
    assert hash1 != hash_diff


def test_model_metadata_registration() -> None:
    """PromptEmbedding is registered on Base.metadata with expected constraints."""
    table = Base.metadata.tables.get("prompt_embeddings")
    assert table is not None

    # Check columns
    assert "id" in table.c
    assert "prompt_id" in table.c
    assert "embedding" in table.c
    assert "source_content_hash" in table.c
    assert "config_version" in table.c

    # Check unique constraint on (prompt_id, config_version)
    constraint_names = [uc.name for uc in table.constraints if hasattr(uc, "name")]
    assert "uq_prompt_embeddings_prompt_config" in constraint_names


# -----------------------------------------------------------------------------
# 2. Persistence Operations Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_and_retrieve_embedding(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """EmbeddingRepository stores a valid vector and retrieves it by prompt ID."""
    # 1. Create parent prompt
    prompt = await PromptRepository.create(
        async_test_session,
        idea="Design auth gateway",
        sections=sample_crispe_sections,
    )

    # 2. Store embedding
    content_hash = PromptEmbedding.compute_content_hash(prompt.formatted_prompt)
    emb_result = make_embedding_result(dim=768)

    stored = await EmbeddingRepository.store_embedding(
        async_test_session,
        prompt_id=prompt.id,
        embedding_result=emb_result,
        source_content_hash=content_hash,
    )

    assert stored.id is not None
    assert stored.prompt_id == prompt.id
    assert stored.config_version == "v1"
    assert stored.model_id == "text-embedding-004"
    assert len(stored.embedding) == 768

    # 3. Retrieve by ID
    by_id = await EmbeddingRepository.get_by_id(async_test_session, stored.id)
    assert by_id is not None
    assert by_id.id == stored.id

    # 4. Retrieve by prompt ID
    by_prompt = await EmbeddingRepository.get_by_prompt_id(async_test_session, prompt.id)
    assert by_prompt is not None
    assert by_prompt.id == stored.id


@pytest.mark.asyncio
async def test_staleness_detection(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """is_stale detects when prompt content hash changes or embedding is missing."""
    prompt = await PromptRepository.create(
        async_test_session,
        idea="Original prompt",
        sections=sample_crispe_sections,
    )

    original_hash = PromptEmbedding.compute_content_hash(prompt.formatted_prompt)
    updated_hash = PromptEmbedding.compute_content_hash(prompt.formatted_prompt + " modified")

    # Before embedding is stored: stale
    assert await EmbeddingRepository.is_stale(async_test_session, prompt.id, original_hash) is True

    # Store embedding
    await EmbeddingRepository.store_embedding(
        async_test_session,
        prompt_id=prompt.id,
        embedding_result=make_embedding_result(dim=768),
        source_content_hash=original_hash,
    )

    # With original hash: not stale
    assert await EmbeddingRepository.is_stale(async_test_session, prompt.id, original_hash) is False

    # With updated hash: stale
    assert await EmbeddingRepository.is_stale(async_test_session, prompt.id, updated_hash) is True


@pytest.mark.asyncio
async def test_idempotent_upsert_on_same_prompt_and_config(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Re-storing an embedding for the same (prompt_id, config_version) updates existing record."""
    prompt = await PromptRepository.create(
        async_test_session,
        idea="Upsert prompt",
        sections=sample_crispe_sections,
    )

    hash_v1 = "hash-version-1"
    hash_v2 = "hash-version-2"

    emb1 = await EmbeddingRepository.store_embedding(
        async_test_session,
        prompt_id=prompt.id,
        embedding_result=make_embedding_result(dim=768, val=0.1),
        source_content_hash=hash_v1,
    )
    first_id = emb1.id

    # Store second embedding under same config version
    emb2 = await EmbeddingRepository.store_embedding(
        async_test_session,
        prompt_id=prompt.id,
        embedding_result=make_embedding_result(dim=768, val=0.2),
        source_content_hash=hash_v2,
    )

    # Updated in-place (same primary key ID)
    assert emb2.id == first_id
    assert emb2.source_content_hash == hash_v2
    assert emb2.embedding[0] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_store_embedding_fails_on_nonexistent_prompt(
    async_test_session: AsyncSession,
) -> None:
    """Attempting to attach embedding to non-existent prompt raises DatabaseError (404)."""
    fake_prompt_id = uuid.uuid4()
    with pytest.raises(DatabaseError) as exc_info:
        await EmbeddingRepository.store_embedding(
            async_test_session,
            prompt_id=fake_prompt_id,
            embedding_result=make_embedding_result(dim=768),
            source_content_hash="some-hash",
        )
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "PROMPT_NOT_FOUND"


@pytest.mark.asyncio
async def test_cascade_delete_prompt_deletes_embeddings(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Deleting parent PromptRecord cascades and deletes associated embeddings."""
    prompt = await PromptRepository.create(
        async_test_session,
        idea="Cascade prompt",
        sections=sample_crispe_sections,
    )

    stored = await EmbeddingRepository.store_embedding(
        async_test_session,
        prompt_id=prompt.id,
        embedding_result=make_embedding_result(dim=768),
        source_content_hash="hash-123",
    )
    assert await EmbeddingRepository.get_by_id(async_test_session, stored.id) is not None

    # Delete parent prompt
    await PromptRepository.delete(async_test_session, prompt.id)

    # Embedding is automatically deleted via CASCADE
    assert await EmbeddingRepository.get_by_id(async_test_session, stored.id) is None


# -----------------------------------------------------------------------------
# 3. Owner-Scoped Authorization Boundary Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_owner_scoped_embedding_storage_and_isolation(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Principal A cannot attach or read embeddings on Principal B's prompt record."""
    p_a = await PrincipalRepository.create(async_test_session, email="alice_emb@test.com")
    p_b = await PrincipalRepository.create(async_test_session, email="bob_emb@test.com")
    ctx_a = p_a.to_authorized_principal()
    ctx_b = p_b.to_authorized_principal()

    # Principal A creates a prompt
    prompt_a = await PromptRepository.create_for_owner(
        async_test_session,
        principal=ctx_a,
        idea="Alice secret algorithm",
        sections=sample_crispe_sections,
    )

    content_hash = PromptEmbedding.compute_content_hash(prompt_a.formatted_prompt)

    # 1. Principal B attempts to store embedding on Alice's prompt -> rejected (OwnershipViolationError)
    with pytest.raises(OwnershipViolationError):
        await EmbeddingRepository.store_embedding_for_owner(
            async_test_session,
            principal=ctx_b,
            prompt_id=prompt_a.id,
            embedding_result=make_embedding_result(dim=768),
            source_content_hash=content_hash,
        )

    # 2. Principal A stores embedding -> succeeds
    stored_a = await EmbeddingRepository.store_embedding_for_owner(
        async_test_session,
        principal=ctx_a,
        prompt_id=prompt_a.id,
        embedding_result=make_embedding_result(dim=768),
        source_content_hash=content_hash,
    )
    assert stored_a.id is not None

    # 3. Principal A can retrieve it
    retrieved_a = await EmbeddingRepository.get_by_prompt_id_for_owner(
        async_test_session,
        principal=ctx_a,
        prompt_id=prompt_a.id,
    )
    assert retrieved_a is not None
    assert retrieved_a.id == stored_a.id

    # 4. Principal B CANNOT retrieve it (returns None without leaking existence)
    retrieved_b = await EmbeddingRepository.get_by_prompt_id_for_owner(
        async_test_session,
        principal=ctx_b,
        prompt_id=prompt_a.id,
    )
    assert retrieved_b is None


# -----------------------------------------------------------------------------
# 4. Alembic Discovery Tests
# -----------------------------------------------------------------------------

def test_alembic_script_directory_discovers_migration_0003() -> None:
    """Alembic discovers the 0003_prompt_embeddings revision head with down_revision 0002."""
    alembic_ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    assert alembic_ini_path.exists(), "alembic.ini must exist"

    config = Config(str(alembic_ini_path))
    script_dir = ScriptDirectory.from_config(config)

    head_rev = script_dir.get_current_head()
    assert head_rev in ("0003_prompt_embeddings", "0004_optimization_jobs", "0005_prompt_candidates")

    rev_0003 = script_dir.get_revision("0003_prompt_embeddings")
    assert rev_0003 is not None
    assert rev_0003.down_revision == "0002_principal_and_prompt_ownership"
