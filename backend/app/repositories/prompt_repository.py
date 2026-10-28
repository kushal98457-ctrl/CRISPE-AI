"""Repository for persistence and owner-scoped operations on CRISPE prompt records."""

from typing import List, Optional, Union
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    InactivePrincipalError,
    OwnerImmutableError,
)
from app.core.logging import get_logger
from app.core.security import AuthorizedPrincipal
from app.models.prompt_record import PromptRecord
from app.schemas.crispe import CrispeSections

logger = get_logger("prompt_repository")


class PromptRepository:
    """Data access repository managing prompt record database transactions and data isolation."""

    # -------------------------------------------------------------------------
    # Owner-Scoped Authorization Boundary Operations (Step 11)
    # -------------------------------------------------------------------------

    @staticmethod
    async def create_for_owner(
        session: AsyncSession,
        *,
        principal: AuthorizedPrincipal,
        idea: str,
        sections: CrispeSections,
        model_id: str = "gemma-4-26b-a4b-it",
        formatted_prompt: Optional[str] = None,
        task_type: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        candidates_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        auto_commit: bool = True,
    ) -> PromptRecord:
        """Create and persist a prompt record strictly owned by the authorized principal.

        Guarantees:
        - Rejects deactivated principals via principal.ensure_active().
        - Enforces owner_id = principal.id (untrusted client-supplied owner_id is ignored).
        - Links referential integrity with the principals table.
        """
        principal.ensure_active()

        return await PromptRepository.create(
            session,
            idea=idea,
            sections=sections,
            model_id=model_id,
            formatted_prompt=formatted_prompt,
            task_type=task_type,
            owner_id=principal.id,
            prompt_tokens=prompt_tokens,
            candidates_tokens=candidates_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            auto_commit=auto_commit,
        )

    @staticmethod
    async def get_by_id_and_owner(
        session: AsyncSession,
        record_id: Union[uuid.UUID, str],
        *,
        principal: AuthorizedPrincipal,
    ) -> Optional[PromptRecord]:
        """Retrieve a prompt record by UUID only if owned by the authorized principal.

        Guarantees:
        - Atomic SQL filter: WHERE id = :id AND owner_id = :principal_id.
        - Never leaks existence of records owned by another principal (returns None).
        - Unowned legacy records (owner_id IS NULL) are never returned to any principal.
        """
        principal.ensure_active()

        if isinstance(record_id, str):
            try:
                record_id = uuid.UUID(record_id)
            except ValueError:
                return None

        stmt = select(PromptRecord).where(
            PromptRecord.id == record_id,
            PromptRecord.owner_id == principal.id,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_owner(
        session: AsyncSession,
        *,
        principal: AuthorizedPrincipal,
        skip: int = 0,
        limit: int = 50,
        task_type: Optional[str] = None,
    ) -> List[PromptRecord]:
        """List prompt records owned by the authorized principal with bounded pagination.

        Guarantees:
        - Enforces WHERE owner_id = :principal_id on all queries.
        - Leverages composite index (owner_id, created_at) for efficient reverse chronological sorting.
        - Excludes records belonging to other users or unowned legacy records.
        """
        principal.ensure_active()

        safe_limit = max(1, min(limit, 100))
        safe_skip = max(0, skip)

        stmt = (
            select(PromptRecord)
            .where(PromptRecord.owner_id == principal.id)
            .order_by(PromptRecord.created_at.desc())
        )

        if task_type:
            stmt = stmt.where(PromptRecord.task_type == task_type)

        stmt = stmt.offset(safe_skip).limit(safe_limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def update_for_owner(
        session: AsyncSession,
        record_id: Union[uuid.UUID, str],
        *,
        principal: AuthorizedPrincipal,
        formatted_prompt: Optional[str] = None,
        task_type: Optional[str] = None,
        auto_commit: bool = True,
    ) -> Optional[PromptRecord]:
        """Update mutable prompt fields only if the record is owned by the authorized principal.

        Guarantees:
        - Rejects update if record does not exist or is owned by another principal.
        - Strict immutability: record.owner_id cannot be transferred or changed.
        """
        record = await PromptRepository.get_by_id_and_owner(
            session,
            record_id,
            principal=principal,
        )
        if not record:
            return None

        if formatted_prompt is not None:
            record.formatted_prompt = formatted_prompt
        if task_type is not None:
            record.task_type = task_type

        if auto_commit:
            await session.commit()
            await session.refresh(record)
        else:
            await session.flush()

        logger.info(
            "Updated prompt record %s for authorized principal: %s",
            record.id,
            principal.id,
        )
        return record

    @staticmethod
    async def delete_for_owner(
        session: AsyncSession,
        record_id: Union[uuid.UUID, str],
        *,
        principal: AuthorizedPrincipal,
        auto_commit: bool = True,
    ) -> bool:
        """Delete a prompt record only if owned by the authorized principal.

        Guarantees:
        - Cannot delete records belonging to another principal or unowned legacy records.
        """
        record = await PromptRepository.get_by_id_and_owner(
            session,
            record_id,
            principal=principal,
        )
        if not record:
            return False

        await session.delete(record)
        if auto_commit:
            await session.commit()

        logger.info(
            "Deleted prompt record %s for authorized principal: %s",
            record_id,
            principal.id,
        )
        return True

    @staticmethod
    async def is_owner(
        session: AsyncSession,
        record_id: Union[uuid.UUID, str],
        *,
        principal: AuthorizedPrincipal,
    ) -> bool:
        """Check whether a specific prompt record is owned by the authorized principal."""
        principal.ensure_active()
        record = await PromptRepository.get_by_id_and_owner(
            session,
            record_id,
            principal=principal,
        )
        return record is not None

    # -------------------------------------------------------------------------
    # Low-Level Persistence Primitives (Step 10 Compatibility)
    # -------------------------------------------------------------------------

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        idea: str,
        sections: CrispeSections,
        model_id: str = "gemma-4-26b-a4b-it",
        formatted_prompt: Optional[str] = None,
        task_type: Optional[str] = None,
        owner_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        candidates_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        auto_commit: bool = True,
    ) -> PromptRecord:
        """Persist a new CRISPE prompt record into the database."""
        rendered_prompt = formatted_prompt or sections.to_markdown()

        record = PromptRecord(
            original_idea=idea,
            context=sections.context,
            role=sections.role,
            instruction=sections.instruction,
            specification=sections.specification,
            performance=sections.performance,
            example=sections.example,
            formatted_prompt=rendered_prompt,
            model_id=model_id,
            task_type=task_type,
            owner_id=owner_id,
            prompt_tokens=prompt_tokens,
            candidates_tokens=candidates_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

        session.add(record)
        if auto_commit:
            await session.commit()
            await session.refresh(record)
        else:
            await session.flush()

        logger.info("Created prompt record with ID: %s (model: %s)", record.id, model_id)
        return record

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        record_id: Union[uuid.UUID, str],
    ) -> Optional[PromptRecord]:
        """Retrieve a single prompt record by its primary key UUID."""
        if isinstance(record_id, str):
            try:
                record_id = uuid.UUID(record_id)
            except ValueError:
                return None

        stmt = select(PromptRecord).where(PromptRecord.id == record_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        session: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 50,
        owner_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[PromptRecord]:
        """List prompt records with pagination and optional filtering."""
        safe_limit = max(1, min(limit, 100))
        safe_skip = max(0, skip)

        stmt = select(PromptRecord).order_by(PromptRecord.created_at.desc())

        if owner_id:
            stmt = stmt.where(PromptRecord.owner_id == owner_id)
        if task_type:
            stmt = stmt.where(PromptRecord.task_type == task_type)

        stmt = stmt.offset(safe_skip).limit(safe_limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def delete(
        session: AsyncSession,
        record_id: Union[uuid.UUID, str],
        *,
        auto_commit: bool = True,
    ) -> bool:
        """Delete a prompt record by its primary key UUID."""
        record = await PromptRepository.get_by_id(session, record_id)
        if not record:
            return False

        await session.delete(record)
        if auto_commit:
            await session.commit()

        logger.info("Deleted prompt record with ID: %s", record_id)
        return True
