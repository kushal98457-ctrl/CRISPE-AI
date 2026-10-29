"""Automated tests for prompt ownership, principal identity, and tenant isolation (Step 11)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator
import uuid
import pytest
import pytest_asyncio
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import (
    AuthenticationRequiredError,
    InactivePrincipalError,
    SecurityError,
)
from app.core.security import AuthorizedPrincipal, get_current_principal
from app.db.base import Base
from app.models.principal import Principal
from app.models.prompt_record import PromptRecord
from app.repositories.principal_repository import PrincipalRepository
from app.repositories.prompt_repository import PromptRepository
from app.schemas.crispe import CrispeSections


@pytest_asyncio.fixture
async def async_test_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated, transactional in-memory SQLite database session."""
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
        context="Enterprise multi-tenant database infrastructure.",
        role="Principal Security Architect.",
        instruction="Enforce strict tenant data isolation across all persistence layers.",
        specification="Reject unauthenticated access, prevent IDOR, and cascade deletions.",
        performance="Zero data leakage across distinct security principals.",
        example="Principal A cannot read or delete Principal B prompt records.",
    )


# -----------------------------------------------------------------------------
# Principal Identity Model & Repository Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_principal_instantiation_and_defaults() -> None:
    """Principal entity initializes with valid UUID-based ID, active status, and timestamps."""
    principal = Principal(
        email="architect@crispe.ai",
        display_name="Security Lead",
    )

    assert principal.id.startswith("principal_")
    assert principal.email == "architect@crispe.ai"
    assert principal.display_name == "Security Lead"
    assert principal.is_active is True
    assert principal.is_superuser is False

    auth_ctx = principal.to_authorized_principal()
    assert isinstance(auth_ctx, AuthorizedPrincipal)
    assert auth_ctx.id == principal.id
    assert auth_ctx.email == "architect@crispe.ai"
    assert auth_ctx.is_active is True


@pytest.mark.asyncio
async def test_principal_repository_crud(async_test_session: AsyncSession) -> None:
    """PrincipalRepository creates, retrieves, deactivates, and queries by email."""
    # 1. Create
    p = await PrincipalRepository.create(
        async_test_session,
        email="developer@crispe.ai",
        display_name="Core Dev",
    )
    assert p.id is not None
    assert p.is_active is True

    # 2. Lookup by ID and Email
    by_id = await PrincipalRepository.get_by_id(async_test_session, p.id)
    assert by_id is not None
    assert by_id.id == p.id

    by_email = await PrincipalRepository.get_by_email(async_test_session, "developer@crispe.ai")
    assert by_email is not None
    assert by_email.id == p.id

    # 3. Deactivate
    deactivated = await PrincipalRepository.deactivate(async_test_session, p.id)
    assert deactivated is not None
    assert deactivated.is_active is False

    # 4. Inactive principal raises on ensure_active()
    auth_ctx = deactivated.to_authorized_principal()
    with pytest.raises(InactivePrincipalError):
        auth_ctx.ensure_active()


# -----------------------------------------------------------------------------
# Record Ownership & Cross-Tenant Data Isolation Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_owner_scoped_creation_and_retrieval_isolation(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Principal A cannot retrieve Principal B's prompt record via owner-scoped lookup."""
    p_a = await PrincipalRepository.create(async_test_session, email="user_a@test.com")
    p_b = await PrincipalRepository.create(async_test_session, email="user_b@test.com")

    ctx_a = p_a.to_authorized_principal()
    ctx_b = p_b.to_authorized_principal()

    # Principal A creates a prompt
    record_a = await PromptRepository.create_for_owner(
        async_test_session,
        principal=ctx_a,
        idea="Confidential audit instructions for User A",
        sections=sample_crispe_sections,
        task_type="security",
    )

    # 1. Principal A can retrieve their own prompt
    retrieved_a = await PromptRepository.get_by_id_and_owner(
        async_test_session,
        record_a.id,
        principal=ctx_a,
    )
    assert retrieved_a is not None
    assert retrieved_a.id == record_a.id
    assert retrieved_a.owner_id == ctx_a.id

    # 2. Principal B CANNOT retrieve Principal A's prompt (isolated / IDOR protected)
    retrieved_b = await PromptRepository.get_by_id_and_owner(
        async_test_session,
        record_a.id,
        principal=ctx_b,
    )
    assert retrieved_b is None

    # 3. Verify ownership confirmation helper
    assert await PromptRepository.is_owner(async_test_session, record_a.id, principal=ctx_a) is True
    assert await PromptRepository.is_owner(async_test_session, record_a.id, principal=ctx_b) is False


@pytest.mark.asyncio
async def test_owner_scoped_list_and_pagination_isolation(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Listing prompts strictly filters by the requesting owner and excludes other owners' records."""
    p_a = await PrincipalRepository.create(async_test_session, email="alice@test.com")
    p_b = await PrincipalRepository.create(async_test_session, email="bob@test.com")

    ctx_a = p_a.to_authorized_principal()
    ctx_b = p_b.to_authorized_principal()

    # Create 3 records for Alice
    for i in range(3):
        await PromptRepository.create_for_owner(
            async_test_session,
            principal=ctx_a,
            idea=f"Alice prompt {i}",
            sections=sample_crispe_sections,
            task_type="coding" if i % 2 == 0 else "writing",
        )

    # Create 2 records for Bob
    for i in range(2):
        await PromptRepository.create_for_owner(
            async_test_session,
            principal=ctx_b,
            idea=f"Bob prompt {i}",
            sections=sample_crispe_sections,
            task_type="security",
        )

    # Alice's list has exactly 3 records
    alice_list = await PromptRepository.list_for_owner(async_test_session, principal=ctx_a)
    assert len(alice_list) == 3
    assert all(r.owner_id == ctx_a.id for r in alice_list)

    # Bob's list has exactly 2 records
    bob_list = await PromptRepository.list_for_owner(async_test_session, principal=ctx_b)
    assert len(bob_list) == 2
    assert all(r.owner_id == ctx_b.id for r in bob_list)

    # Filtering by task_type operates within owner boundary
    alice_coding = await PromptRepository.list_for_owner(
        async_test_session,
        principal=ctx_a,
        task_type="coding",
    )
    assert len(alice_coding) == 2
    assert all(r.task_type == "coding" for r in alice_coding)


@pytest.mark.asyncio
async def test_owner_scoped_update_and_immutability(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """One owner cannot update another owner's prompt, and record ownership is immutable."""
    p_a = await PrincipalRepository.create(async_test_session, email="owner1@test.com")
    p_b = await PrincipalRepository.create(async_test_session, email="owner2@test.com")

    ctx_a = p_a.to_authorized_principal()
    ctx_b = p_b.to_authorized_principal()

    record = await PromptRepository.create_for_owner(
        async_test_session,
        principal=ctx_a,
        idea="Initial prompt for owner 1",
        sections=sample_crispe_sections,
        task_type="initial",
    )

    # 1. Owner B attempts to update Owner A's prompt -> returns None
    update_by_b = await PromptRepository.update_for_owner(
        async_test_session,
        record.id,
        principal=ctx_b,
        task_type="malicious_takeover",
    )
    assert update_by_b is None

    # Verify record was not modified
    fresh = await PromptRepository.get_by_id_and_owner(async_test_session, record.id, principal=ctx_a)
    assert fresh is not None
    assert fresh.task_type == "initial"
    assert fresh.owner_id == ctx_a.id

    # 2. Owner A updates their own prompt -> succeeds
    update_by_a = await PromptRepository.update_for_owner(
        async_test_session,
        record.id,
        principal=ctx_a,
        task_type="updated_type",
        formatted_prompt="## Updated Prompt Text",
    )
    assert update_by_a is not None
    assert update_by_a.task_type == "updated_type"
    assert update_by_a.formatted_prompt == "## Updated Prompt Text"
    assert update_by_a.owner_id == ctx_a.id  # Owner unchanged


@pytest.mark.asyncio
async def test_owner_scoped_delete_isolation(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """One owner cannot delete another owner's prompt record."""
    p_a = await PrincipalRepository.create(async_test_session, email="alice_del@test.com")
    p_b = await PrincipalRepository.create(async_test_session, email="bob_del@test.com")

    ctx_a = p_a.to_authorized_principal()
    ctx_b = p_b.to_authorized_principal()

    record = await PromptRepository.create_for_owner(
        async_test_session,
        principal=ctx_a,
        idea="Alice's sacred record",
        sections=sample_crispe_sections,
    )

    # 1. Bob attempts to delete Alice's record -> rejected (returns False)
    deleted_by_bob = await PromptRepository.delete_for_owner(
        async_test_session,
        record.id,
        principal=ctx_b,
    )
    assert deleted_by_bob is False

    # Alice's record is still intact
    assert await PromptRepository.get_by_id_and_owner(async_test_session, record.id, principal=ctx_a) is not None

    # 2. Alice deletes her own record -> succeeds (returns True)
    deleted_by_alice = await PromptRepository.delete_for_owner(
        async_test_session,
        record.id,
        principal=ctx_a,
    )
    assert deleted_by_alice is True

    # Now it is gone
    assert await PromptRepository.get_by_id_and_owner(async_test_session, record.id, principal=ctx_a) is None


@pytest.mark.asyncio
async def test_unowned_legacy_records_are_isolated(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Legacy/unowned records (owner_id IS NULL) are never exposed via owner-scoped operations."""
    p = await PrincipalRepository.create(async_test_session, email="tenant@test.com")
    ctx = p.to_authorized_principal()

    # Create unowned prompt (e.g. from prior unauthenticated generation)
    unowned_record = await PromptRepository.create(
        async_test_session,
        idea="Legacy unowned idea",
        sections=sample_crispe_sections,
        owner_id=None,
    )
    assert unowned_record.owner_id is None

    # Authenticated user cannot access it via owner-scoped lookup
    retrieved = await PromptRepository.get_by_id_and_owner(
        async_test_session,
        unowned_record.id,
        principal=ctx,
    )
    assert retrieved is None

    # Authenticated user's listing does not include the legacy record
    records = await PromptRepository.list_for_owner(async_test_session, principal=ctx)
    assert len(records) == 0


@pytest.mark.asyncio
async def test_inactive_principal_is_rejected_on_all_operations(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Inactive/suspended principals are forbidden from all repository operations."""
    p = await PrincipalRepository.create(async_test_session, email="inactive@test.com")
    await PrincipalRepository.deactivate(async_test_session, p.id)

    # Reload deactivated principal context
    deactivated = await PrincipalRepository.get_by_id(async_test_session, p.id)
    assert deactivated is not None
    ctx = deactivated.to_authorized_principal()

    with pytest.raises(InactivePrincipalError):
        await PromptRepository.create_for_owner(
            async_test_session,
            principal=ctx,
            idea="Should fail",
            sections=sample_crispe_sections,
        )

    with pytest.raises(InactivePrincipalError):
        await PromptRepository.list_for_owner(async_test_session, principal=ctx)

    with pytest.raises(InactivePrincipalError):
        await PromptRepository.get_by_id_and_owner(
            async_test_session,
            uuid.uuid4(),
            principal=ctx,
        )


@pytest.mark.asyncio
async def test_cascade_delete_principal_purges_prompts(
    async_test_session: AsyncSession,
    sample_crispe_sections: CrispeSections,
) -> None:
    """Deleting a principal cascades and deletes all owned prompt records (GDPR compliance)."""
    p = await PrincipalRepository.create(async_test_session, email="purge_me@test.com")
    ctx = p.to_authorized_principal()

    rec1 = await PromptRepository.create_for_owner(
        async_test_session,
        principal=ctx,
        idea="Prompt 1 to purge",
        sections=sample_crispe_sections,
    )
    rec2 = await PromptRepository.create_for_owner(
        async_test_session,
        principal=ctx,
        idea="Prompt 2 to purge",
        sections=sample_crispe_sections,
    )

    # Verify both records exist
    assert await PromptRepository.get_by_id(async_test_session, rec1.id) is not None
    assert await PromptRepository.get_by_id(async_test_session, rec2.id) is not None

    # Delete principal
    await PrincipalRepository.delete(async_test_session, p.id)

    # Verify both records were cascaded
    assert await PromptRepository.get_by_id(async_test_session, rec1.id) is None
    assert await PromptRepository.get_by_id(async_test_session, rec2.id) is None


# -----------------------------------------------------------------------------
# Security Boundary & Migration Discovery Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_security_dependency_fails_closed() -> None:
    """get_current_principal dependency fails closed with AuthenticationRequiredError."""
    with pytest.raises(AuthenticationRequiredError):
        await get_current_principal()


def test_alembic_script_directory_discovers_migration_0002() -> None:
    """Alembic discovers the 0002_principal_and_prompt_ownership revision head."""
    alembic_ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    assert alembic_ini_path.exists(), "alembic.ini must exist"

    config = Config(str(alembic_ini_path))
    script_dir = ScriptDirectory.from_config(config)

    rev_0002 = script_dir.get_revision("0002_principal_and_prompt_ownership")
    assert rev_0002 is not None
    assert rev_0002.down_revision == "0001_prompt_records"
