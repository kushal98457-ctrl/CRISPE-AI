"""Tests for PromptRecord ORM model, repository CRUD operations, and Alembic migration."""

import uuid
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from alembic.config import Config
from alembic.script import ScriptDirectory
from app.db.base import Base
from app.models.prompt_record import PromptRecord
from app.repositories.prompt_repository import PromptRepository
from app.schemas.crispe import CrispeSections


@pytest.fixture
def sample_crispe_sections() -> CrispeSections:
    """Fixture providing valid 6-section CRISPE data."""
    return CrispeSections(
        context="High-traffic banking authentication gateway.",
        role="Principal Security Architect.",
        instruction="Design zero-trust OAuth 2.1 PKCE token exchange.",
        specification="Produce Markdown specification table with strict negative rules.",
        performance="Token issuance latency under 10ms with replay attack protection.",
        example="Input: /token authorization_code. Output: Encrypted JWT pair.",
    )


@pytest.fixture
async def async_sqlite_session():
    """Async session fixture bound to in-memory SQLite database with initialized schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ==============================================================================
# Model Tests
# ==============================================================================

def test_prompt_record_instantiation(sample_crispe_sections: CrispeSections) -> None:
    """PromptRecord instantiates cleanly and sets UUID and field defaults."""
    record = PromptRecord(
        original_idea="Secure my banking OAuth flow",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt=sample_crispe_sections.to_markdown(),
        model_id="gemma-4-26b-a4b-it",
        prompt_tokens=50,
        candidates_tokens=150,
        total_tokens=200,
    )

    assert isinstance(record.id, uuid.UUID)
    assert record.original_idea == "Secure my banking OAuth flow"
    assert record.model_id == "gemma-4-26b-a4b-it"


def test_prompt_record_to_sections_roundtrip(sample_crispe_sections: CrispeSections) -> None:
    """to_sections() returns a validated CrispeSections matching the stored fields."""
    record = PromptRecord(
        original_idea="Sample idea",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt=sample_crispe_sections.to_markdown(),
    )

    reconstituted = record.to_sections()
    assert isinstance(reconstituted, CrispeSections)
    assert reconstituted.context == sample_crispe_sections.context
    assert reconstituted.role == sample_crispe_sections.role
    assert reconstituted.instruction == sample_crispe_sections.instruction
    assert reconstituted.specification == sample_crispe_sections.specification
    assert reconstituted.performance == sample_crispe_sections.performance
    assert reconstituted.example == sample_crispe_sections.example


def test_prompt_record_to_response(sample_crispe_sections: CrispeSections) -> None:
    """to_response() formats the record into CrispeGenerationResponse."""
    record = PromptRecord(
        original_idea="Test idea",
        context=sample_crispe_sections.context,
        role=sample_crispe_sections.role,
        instruction=sample_crispe_sections.instruction,
        specification=sample_crispe_sections.specification,
        performance=sample_crispe_sections.performance,
        example=sample_crispe_sections.example,
        formatted_prompt=sample_crispe_sections.to_markdown(),
        model_id="gemma-4-26b-a4b-it",
        prompt_tokens=40,
        candidates_tokens=160,
        total_tokens=200,
    )

    resp = record.to_response(request_id="req-999")
    assert resp.idea == "Test idea"
    assert resp.model_id == "gemma-4-26b-a4b-it"
    assert resp.request_id == "req-999"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 200
    assert "# C - Context" in resp.formatted_prompt


# ==============================================================================
# Repository CRUD Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_repository_create_and_get_by_id(
    async_sqlite_session: AsyncSession, sample_crispe_sections: CrispeSections
) -> None:
    """Repository creates record and retrieves it by UUID and UUID string."""
    created = await PromptRepository.create(
        async_sqlite_session,
        idea="Build a resilient payment queue",
        sections=sample_crispe_sections,
        model_id="gemma-4-26b-a4b-it",
        task_type="backend",
        owner_id="user-123",
        prompt_tokens=30,
        candidates_tokens=120,
        total_tokens=150,
    )

    assert created.id is not None

    # Retrieve by UUID object
    fetched = await PromptRepository.get_by_id(async_sqlite_session, created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.original_idea == "Build a resilient payment queue"
    assert fetched.task_type == "backend"
    assert fetched.owner_id == "user-123"

    # Retrieve by string representation
    fetched_str = await PromptRepository.get_by_id(async_sqlite_session, str(created.id))
    assert fetched_str is not None
    assert fetched_str.id == created.id


@pytest.mark.asyncio
async def test_repository_get_by_nonexistent_id(async_sqlite_session: AsyncSession) -> None:
    """Repository returns None for invalid or nonexistent record IDs."""
    # Nonexistent UUID
    result = await PromptRepository.get_by_id(async_sqlite_session, uuid.uuid4())
    assert result is None

    # Invalid string UUID
    invalid_result = await PromptRepository.get_by_id(async_sqlite_session, "not-a-valid-uuid")
    assert invalid_result is None


@pytest.mark.asyncio
async def test_repository_list_and_pagination(
    async_sqlite_session: AsyncSession, sample_crispe_sections: CrispeSections
) -> None:
    """Repository lists prompt records with pagination and filtering."""
    # Create 3 records
    for i in range(3):
        await PromptRepository.create(
            async_sqlite_session,
            idea=f"Idea {i}",
            sections=sample_crispe_sections,
            task_type="security" if i < 2 else "devops",
            owner_id="user-alice" if i == 0 else "user-bob",
        )

    # List all
    all_records = await PromptRepository.list(async_sqlite_session)
    assert len(all_records) == 3

    # Limit to 2
    paged = await PromptRepository.list(async_sqlite_session, limit=2)
    assert len(paged) == 2

    # Filter by task_type
    security_records = await PromptRepository.list(async_sqlite_session, task_type="security")
    assert len(security_records) == 2

    # Filter by owner_id
    alice_records = await PromptRepository.list(async_sqlite_session, owner_id="user-alice")
    assert len(alice_records) == 1
    assert alice_records[0].original_idea == "Idea 0"


@pytest.mark.asyncio
async def test_repository_delete(
    async_sqlite_session: AsyncSession, sample_crispe_sections: CrispeSections
) -> None:
    """Repository deletes existing record and returns False for nonexistent."""
    created = await PromptRepository.create(
        async_sqlite_session,
        idea="Ephemeral prompt",
        sections=sample_crispe_sections,
    )

    deleted = await PromptRepository.delete(async_sqlite_session, created.id)
    assert deleted is True

    # Confirm deletion
    lookup = await PromptRepository.get_by_id(async_sqlite_session, created.id)
    assert lookup is None

    # Delete non-existent record
    deleted_again = await PromptRepository.delete(async_sqlite_session, created.id)
    assert deleted_again is False


# ==============================================================================
# Alembic Migration Verification
# ==============================================================================

def test_alembic_script_directory_discovers_migration() -> None:
    """Alembic ScriptDirectory detects the 0001_initial_prompt_records migration."""
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    rev = script.get_revision("0001_prompt_records")
    assert rev is not None
    assert rev.down_revision is None
    assert "prompt_records" in rev.doc
