"""Repository for security principal identity management and lookup operations."""

from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.principal import Principal

logger = get_logger("principal_repository")


class PrincipalRepository:
    """Data access repository managing Principal identity anchors and lifecycles."""

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        principal_id: Optional[str] = None,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        is_active: bool = True,
        is_superuser: bool = False,
        auto_commit: bool = True,
    ) -> Principal:
        """Create and persist a new security principal."""
        principal = Principal(
            id=principal_id,
            email=email,
            display_name=display_name,
            is_active=is_active,
            is_superuser=is_superuser,
        )

        session.add(principal)
        if auto_commit:
            await session.commit()
            await session.refresh(principal)
        else:
            await session.flush()

        logger.info("Created security principal with ID: %s (email: %s)", principal.id, email)
        return principal

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        principal_id: str,
    ) -> Optional[Principal]:
        """Retrieve a security principal by its unique identifier."""
        stmt = select(Principal).where(Principal.id == principal_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_email(
        session: AsyncSession,
        email: str,
    ) -> Optional[Principal]:
        """Retrieve a security principal by its verified email address."""
        stmt = select(Principal).where(Principal.email == email)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def deactivate(
        session: AsyncSession,
        principal_id: str,
        *,
        auto_commit: bool = True,
    ) -> Optional[Principal]:
        """Deactivate a principal, revoking its ability to access owned resources."""
        principal = await PrincipalRepository.get_by_id(session, principal_id)
        if not principal:
            return None

        principal.is_active = False
        if auto_commit:
            await session.commit()
            await session.refresh(principal)
        else:
            await session.flush()

        logger.info("Deactivated security principal with ID: %s", principal_id)
        return principal

    @staticmethod
    async def delete(
        session: AsyncSession,
        principal_id: str,
        *,
        auto_commit: bool = True,
    ) -> bool:
        """Delete a principal, triggering cascade deletion of all owned records."""
        principal = await PrincipalRepository.get_by_id(session, principal_id)
        if not principal:
            return False

        await session.delete(principal)
        if auto_commit:
            await session.commit()

        logger.info("Deleted security principal and cascaded owned data for ID: %s", principal_id)
        return True
