"""SQLAlchemy 2.x asynchronous database engine and session lifecycle management."""

import asyncio
import sys
from typing import AsyncGenerator, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.exceptions import DatabaseConfigError
from app.core.logging import get_logger

# On Windows, psycopg async requires a SelectorEventLoop
if sys.platform == "win32":
    try:
        if isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

logger = get_logger("db_session")

# Singleton cached engine and session factory
_engine_instance: Optional[AsyncEngine] = None
_session_factory_instance: Optional[async_sessionmaker[AsyncSession]] = None


def get_async_engine(
    settings: Optional[Settings] = None,
    database_url: Optional[str] = None,
) -> AsyncEngine:
    """Retrieve or create the shared AsyncEngine singleton.
    
    Supports PostgreSQL (production) and in-memory SQLite (testing).
    """
    global _engine_instance
    if _engine_instance is not None and database_url is None:
        return _engine_instance

    cfg = settings or get_settings()
    url = database_url or cfg.get_database_url()

    engine_kwargs: dict = {
        "echo": cfg.DB_ECHO_SQL,
    }

    # Configure connection pool for PostgreSQL; SQLite in-memory doesn't use queue pools
    if "sqlite" not in url:
        engine_kwargs.update({
            "pool_size": cfg.DB_POOL_SIZE,
            "max_overflow": cfg.DB_MAX_OVERFLOW,
            "pool_timeout": cfg.DB_POOL_TIMEOUT,
            "pool_recycle": cfg.DB_POOL_RECYCLE,
        })

    logger.info("Initializing SQLAlchemy async engine for target: %s", url.split("@")[-1] if "@" in url else "local")
    engine = create_async_engine(url, **engine_kwargs)
    _engine_instance = engine

    return engine


def set_async_engine(engine: Optional[AsyncEngine]) -> None:
    """Explicitly override or clear the shared async engine (for testing)."""
    global _engine_instance, _session_factory_instance
    _engine_instance = engine
    _session_factory_instance = None


def get_session_factory(
    engine: Optional[AsyncEngine] = None,
) -> async_sessionmaker[AsyncSession]:
    """Retrieve or create the shared async_sessionmaker factory."""
    global _session_factory_instance
    if _session_factory_instance is not None and engine is None:
        return _session_factory_instance

    target_engine = engine or get_async_engine()
    factory = async_sessionmaker(
        bind=target_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    _session_factory_instance = factory
    return factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an isolated, transaction-managed AsyncSession.
    
    Ensures safe automatic rollback on unhandled exceptions and guaranteed cleanup.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception as exc:
            logger.warning("Database session encountered error, rolling back: %s", str(exc))
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_connection(
    engine: Optional[AsyncEngine] = None,
    timeout: float = 3.0,
) -> bool:
    """Execute a lightweight SELECT 1 connectivity probe against the database.
    
    Returns True if reachable and responsive, False otherwise. Never leaks credentials.
    """
    async def _probe() -> bool:
        target_engine = engine or get_async_engine()
        async with target_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            return True

    try:
        return await asyncio.wait_for(_probe(), timeout=timeout)
    except DatabaseConfigError:
        logger.warning("Database connectivity check failed: DATABASE_URL not configured.")
        return False
    except Exception as exc:
        logger.warning("Database connectivity probe failed: %s", type(exc).__name__)
        return False


async def close_database() -> None:
    """Gracefully terminate engine connection pool on application shutdown."""
    global _engine_instance, _session_factory_instance
    if _engine_instance is not None:
        try:
            logger.info("Disposing SQLAlchemy async database connection pool.")
            await _engine_instance.dispose()
        except Exception as exc:
            logger.error("Error disposing database engine: %s", str(exc))
        finally:
            _engine_instance = None
            _session_factory_instance = None


def reset_database_state() -> None:
    """Reset cached singleton instances (primarily for testing fixtures)."""
    global _engine_instance, _session_factory_instance
    _engine_instance = None
    _session_factory_instance = None
