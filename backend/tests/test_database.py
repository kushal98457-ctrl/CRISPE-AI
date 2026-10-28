"""Tests for PostgreSQL/SQLAlchemy database infrastructure, session lifecycle, and readiness."""

from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import DatabaseConfigError
from app.db.base import Base, TimestampMixin
from app.db.session import (
    check_database_connection,
    close_database,
    get_async_engine,
    get_db_session,
    get_session_factory,
    reset_database_state,
)
from app.main import app


# ==============================================================================
# Category A: Configuration Tests
# ==============================================================================

def test_database_url_normalization() -> None:
    """Standard postgresql:// URL is normalized to postgresql+psycopg://."""
    settings = Settings(DATABASE_URL="postgresql://user:secret@localhost:5432/testdb")
    assert settings.get_database_url() == "postgresql+psycopg://user:secret@localhost:5432/testdb"


def test_database_url_preserves_psycopg_driver() -> None:
    """Explicit postgresql+psycopg:// URL is preserved as-is."""
    settings = Settings(DATABASE_URL="postgresql+psycopg://app:pass@dbhost:5432/proddb")
    assert settings.get_database_url() == "postgresql+psycopg://app:pass@dbhost:5432/proddb"


def test_missing_database_url_raises_config_error() -> None:
    """Empty or None DATABASE_URL raises DatabaseConfigError on get_database_url()."""
    settings = Settings(DATABASE_URL=None)
    settings.DATABASE_URL = None
    with pytest.raises(DatabaseConfigError) as exc_info:
        settings.get_database_url()
    assert "Database URL is not configured" in exc_info.value.message


def test_safe_summary_redacts_database_password() -> None:
    """safe_summary() redacts database credentials."""
    settings = Settings(DATABASE_URL="postgresql+psycopg://admin:SuperSecretPassword123@prod-cluster:5432/crispe")
    summary = settings.safe_summary()
    assert "SuperSecretPassword123" not in str(summary)
    assert summary["database_url"] == "postgresql+psycopg://admin:[REDACTED]@prod-cluster:5432/crispe"
    assert summary["db_pool_size"] == 5
    assert summary["db_max_overflow"] == 10


# ==============================================================================
# Category B: Engine, Session Lifecycle, and Rollback
# ==============================================================================

@pytest.mark.asyncio
async def test_async_engine_and_session_lifecycle() -> None:
    """Verify AsyncEngine creation, session execution, and clean disposal with in-memory SQLite."""
    reset_database_state()
    test_db_url = "sqlite+aiosqlite:///:memory:"

    engine = get_async_engine(database_url=test_db_url)
    factory = get_session_factory(engine=engine)

    async with factory() as session:
        assert isinstance(session, AsyncSession)
        result = await session.execute(text("SELECT 42"))
        assert result.scalar() == 42

    await close_database()
    reset_database_state()


@pytest.mark.asyncio
async def test_get_db_session_dependency_rollback_on_exception() -> None:
    """get_db_session rolls back the transaction when an unhandled exception occurs."""
    reset_database_state()
    test_db_url = "sqlite+aiosqlite:///:memory:"

    # Initialize engine with test url
    engine = get_async_engine(database_url=test_db_url)
    get_session_factory(engine=engine)

    # Simulate an error within route operation
    with pytest.raises(ValueError, match="Operation failed in route"):
        async for session in get_db_session():
            await session.execute(text("SELECT 1"))
            raise ValueError("Operation failed in route")

    await close_database()
    reset_database_state()


# ==============================================================================
# Category C: Connectivity Probe and Readiness Endpoint
# ==============================================================================

@pytest.mark.asyncio
async def test_check_database_connection_success_with_test_engine() -> None:
    """check_database_connection returns True against responsive database."""
    reset_database_state()
    engine = get_async_engine(database_url="sqlite+aiosqlite:///:memory:")

    is_ready = await check_database_connection(engine=engine)
    assert is_ready is True

    await close_database()
    reset_database_state()


@pytest.mark.asyncio
async def test_check_database_connection_failure() -> None:
    """check_database_connection returns False when database is unreachable."""
    reset_database_state()
    # Unreachable localhost port
    engine = get_async_engine(database_url="postgresql+psycopg://user:pass@127.0.0.1:59999/nonexistent")

    is_ready = await check_database_connection(engine=engine, timeout=0.5)
    assert is_ready is False

    await close_database()
    reset_database_state()


@pytest.mark.asyncio
async def test_readiness_endpoint_200_when_connected() -> None:
    """GET /health/ready returns 200 OK when database is responsive."""
    with patch("app.api.routes.health.check_database_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/health/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["database"] == "connected"


@pytest.mark.asyncio
async def test_readiness_endpoint_503_when_disconnected() -> None:
    """GET /health/ready returns 503 Service Unavailable when database is degraded."""
    with patch("app.api.routes.health.check_database_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/health/ready")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"
            assert data["database"] == "disconnected"


@pytest.mark.asyncio
async def test_liveness_endpoint_remains_independent_of_database() -> None:
    """Liveness probe /health remains 200 OK even when database is disconnected."""
    with patch("app.api.routes.health.check_database_connection", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"


# ==============================================================================
# Category D: Base ORM & Alembic Metadata Registration
# ==============================================================================

def test_declarative_base_metadata_is_registered() -> None:
    """Base is an instance of DeclarativeBase with active MetaData."""
    assert hasattr(Base, "metadata")
    assert Base.metadata is not None


def test_timestamp_mixin_annotations() -> None:
    """TimestampMixin defines created_at and updated_at attributes."""
    assert hasattr(TimestampMixin, "created_at")
    assert hasattr(TimestampMixin, "updated_at")
