"""Repository for CRUD persistence operations on CRISPE prompt records."""

from typing import List, Optional, Union
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.prompt_record import PromptRecord
from app.schemas.crispe import CrispeSections

logger = get_logger("prompt_repository")


class PromptRepository:
    """Data access repository managing prompt record database transactions."""

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
