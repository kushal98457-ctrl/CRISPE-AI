"""Repository for version-aware prompt embedding persistence and lifecycle operations."""

from datetime import datetime, timezone
import math
from typing import List, Optional, Union
import uuid
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import (
    DatabaseError,
    InvalidProviderResponseError,
    OwnershipViolationError,
    PromptValidationError,
)
from app.core.logging import get_logger
from app.core.security import AuthorizedPrincipal
from app.models.prompt_embedding import EMBEDDING_DIMENSION, PromptEmbedding
from app.models.prompt_record import PromptRecord
from app.repositories.prompt_repository import PromptRepository
from app.schemas.embedding import EmbeddingResult

logger = get_logger("embedding_repository")


class EmbeddingRepository:
    """Data access repository managing persistent vector embeddings and version tracking."""

    @staticmethod
    def _coerce_uuid(val: Union[uuid.UUID, str]) -> uuid.UUID:
        """Coerce string or UUID to uuid.UUID, raising ValueError if invalid."""
        if isinstance(val, uuid.UUID):
            return val
        return uuid.UUID(str(val))

    @staticmethod
    def _validate_vector_invariants(result: EmbeddingResult, expected_dim: int = EMBEDDING_DIMENSION) -> None:
        """Verify vector matches schema dimensions and contains only finite real numbers."""
        if not result.vector or len(result.vector) != expected_dim:
            raise InvalidProviderResponseError(
                f"Embedding vector dimension mismatch: expected {expected_dim}, "
                f"got {len(result.vector) if result.vector else 0}."
            )
        for i, val in enumerate(result.vector):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise InvalidProviderResponseError(
                    f"Embedding vector contains invalid or non-finite value at index {i}: {val}."
                )

    # -------------------------------------------------------------------------
    # Core Persistence Primitives
    # -------------------------------------------------------------------------

    @staticmethod
    async def store_embedding(
        session: AsyncSession,
        *,
        prompt_id: Union[uuid.UUID, str],
        embedding_result: EmbeddingResult,
        source_content_hash: str,
        auto_commit: bool = True,
    ) -> PromptEmbedding:
        """Persist or update an embedding record for a parent prompt record.

        Idempotent Upsert Policy:
        - If an embedding already exists for (prompt_id, config_version), it is updated
          with the new vector, hash, model_id, and updated_at timestamp.
        - If no embedding exists, a new PromptEmbedding record is inserted.
        """
        prompt_uuid = EmbeddingRepository._coerce_uuid(prompt_id)
        EmbeddingRepository._validate_vector_invariants(embedding_result)

        # 1. Verify parent prompt record exists
        prompt = await session.get(PromptRecord, prompt_uuid)
        if not prompt:
            raise DatabaseError(
                f"Cannot attach embedding: parent PromptRecord '{prompt_uuid}' does not exist.",
                status_code=404,
                error_code="PROMPT_NOT_FOUND",
            )

        # 2. Check for existing embedding under this config_version
        existing = await EmbeddingRepository.get_by_prompt_id(
            session,
            prompt_uuid,
            config_version=embedding_result.config_version,
        )

        if existing:
            # Update existing record
            existing.embedding = list(embedding_result.vector)
            existing.source_content_hash = source_content_hash
            existing.model_id = embedding_result.model_id
            existing.provider = "google-genai"
            existing.updated_at = datetime.now(timezone.utc)
            record = existing
            logger.info(
                "Updated existing embedding for prompt %s [config: %s, model: %s]",
                prompt_uuid,
                embedding_result.config_version,
                embedding_result.model_id,
            )
        else:
            # Insert new record
            record = PromptEmbedding(
                prompt_id=prompt_uuid,
                embedding=list(embedding_result.vector),
                provider="google-genai",
                model_id=embedding_result.model_id,
                config_version=embedding_result.config_version,
                source_content_hash=source_content_hash,
            )
            session.add(record)
            logger.info(
                "Created new embedding for prompt %s [config: %s, model: %s]",
                prompt_uuid,
                embedding_result.config_version,
                embedding_result.model_id,
            )

        if auto_commit:
            await session.commit()
            await session.refresh(record)
        else:
            await session.flush()

        return record

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        embedding_id: Union[uuid.UUID, str],
    ) -> Optional[PromptEmbedding]:
        """Retrieve a specific embedding record by primary key UUID."""
        try:
            emb_uuid = EmbeddingRepository._coerce_uuid(embedding_id)
        except ValueError:
            return None
        return await session.get(PromptEmbedding, emb_uuid)

    @staticmethod
    async def get_by_prompt_id(
        session: AsyncSession,
        prompt_id: Union[uuid.UUID, str],
        config_version: str = "v1",
    ) -> Optional[PromptEmbedding]:
        """Retrieve the embedding for a given prompt record and configuration version."""
        try:
            prompt_uuid = EmbeddingRepository._coerce_uuid(prompt_id)
        except ValueError:
            return None

        stmt = select(PromptEmbedding).where(
            PromptEmbedding.prompt_id == prompt_uuid,
            PromptEmbedding.config_version == config_version,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def is_stale(
        session: AsyncSession,
        prompt_id: Union[uuid.UUID, str],
        current_content_hash: str,
        config_version: str = "v1",
    ) -> bool:
        """Check whether the stored embedding matches the current prompt source content hash."""
        embedding = await EmbeddingRepository.get_by_prompt_id(
            session, prompt_id, config_version=config_version
        )
        if not embedding:
            return True
        return embedding.source_content_hash != current_content_hash

    @staticmethod
    async def delete_by_prompt_id(
        session: AsyncSession,
        prompt_id: Union[uuid.UUID, str],
        *,
        auto_commit: bool = True,
    ) -> int:
        """Delete all embedding records associated with a prompt."""
        try:
            prompt_uuid = EmbeddingRepository._coerce_uuid(prompt_id)
        except ValueError:
            return 0

        stmt = delete(PromptEmbedding).where(PromptEmbedding.prompt_id == prompt_uuid)
        result = await session.execute(stmt)
        if auto_commit:
            await session.commit()
        return result.rowcount

    # -------------------------------------------------------------------------
    # Owner-Scoped Authorization Boundary Operations
    # -------------------------------------------------------------------------

    @staticmethod
    async def store_embedding_for_owner(
        session: AsyncSession,
        *,
        principal: AuthorizedPrincipal,
        prompt_id: Union[uuid.UUID, str],
        embedding_result: EmbeddingResult,
        source_content_hash: str,
        auto_commit: bool = True,
    ) -> PromptEmbedding:
        """Persist or update an embedding, strictly enforcing parent prompt ownership."""
        principal.ensure_active()
        prompt_uuid = EmbeddingRepository._coerce_uuid(prompt_id)

        prompt = await PromptRepository.get_by_id_and_owner(
            session,
            prompt_uuid,
            principal=principal,
        )
        if not prompt:
            raise OwnershipViolationError(
                f"Prompt record '{prompt_uuid}' was not found or is not owned by the current principal."
            )

        return await EmbeddingRepository.store_embedding(
            session,
            prompt_id=prompt_uuid,
            embedding_result=embedding_result,
            source_content_hash=source_content_hash,
            auto_commit=auto_commit,
        )

    @staticmethod
    async def get_by_prompt_id_for_owner(
        session: AsyncSession,
        *,
        principal: AuthorizedPrincipal,
        prompt_id: Union[uuid.UUID, str],
        config_version: str = "v1",
    ) -> Optional[PromptEmbedding]:
        """Retrieve an embedding only if the parent prompt belongs to the authorized principal."""
        principal.ensure_active()
        try:
            prompt_uuid = EmbeddingRepository._coerce_uuid(prompt_id)
        except ValueError:
            return None

        prompt = await PromptRepository.get_by_id_and_owner(
            session,
            prompt_uuid,
            principal=principal,
        )
        if not prompt:
            return None

        return await EmbeddingRepository.get_by_prompt_id(
            session,
            prompt_uuid,
            config_version=config_version,
        )
