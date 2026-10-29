"""Repository for version-aware prompt embedding persistence and lifecycle operations."""

from datetime import datetime, timezone
import math
from typing import List, Optional, Sequence, Tuple, Union
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
    def _validate_raw_vector(vector: Sequence[float], expected_dim: int = EMBEDDING_DIMENSION) -> None:
        """Verify raw float vector matches schema dimensions and contains only finite real numbers."""
        if not vector or len(vector) != expected_dim:
            raise InvalidProviderResponseError(
                f"Embedding vector dimension mismatch: expected {expected_dim}, "
                f"got {len(vector) if vector else 0}."
            )
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise InvalidProviderResponseError(
                    f"Embedding vector contains invalid or non-finite value at index {i}: {val}."
                )

    @staticmethod
    def _validate_vector_invariants(result: EmbeddingResult, expected_dim: int = EMBEDDING_DIMENSION) -> None:
        """Verify vector matches schema dimensions and contains only finite real numbers."""
        EmbeddingRepository._validate_raw_vector(result.vector, expected_dim)

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

    @staticmethod
    async def search_similar_prompts_for_owner(
        session: AsyncSession,
        *,
        principal: AuthorizedPrincipal,
        query_vector: Sequence[float],
        limit: int = 10,
        min_similarity: Optional[float] = None,
        task_type: Optional[str] = None,
        config_version: str = "v1",
        model_id: Optional[str] = None,
        exclude_stale: bool = True,
    ) -> List[Tuple[PromptRecord, PromptEmbedding, float, float]]:
        """Retrieve nearest-neighbor prompts for an authorized owner using pgvector cosine distance.

        Returns a list of tuples: (prompt_record, prompt_embedding, distance, derived_similarity).
        Results are ordered by cosine distance ascending, breaking ties with prompt creation timestamp descending.

        Parameters:
            session: Active SQLAlchemy AsyncSession.
            principal: The authenticated and authorized security principal context.
            query_vector: 768-dimensional float embedding vector of the search query.
            limit: Maximum number of nearest neighbor results to return (1-50).
            min_similarity: Optional derived cosine similarity threshold (1.0 - distance).
            task_type: Optional domain category or task type filter.
            config_version: Embedding configuration version (must match stored embeddings).
            model_id: Optional embedding model checkpoint filter.
            exclude_stale: Whether to filter out embeddings whose source content hash no longer
                           matches the parent prompt's formatted text.
        """
        if not principal:
            raise OwnershipViolationError("Principal context is required to execute semantic search.")
        principal.ensure_active()
        EmbeddingRepository._validate_raw_vector(query_vector)

        bounded_limit = min(max(1, limit), 50)
        vector_list = list(query_vector)

        # Distance expression via pgvector cosine distance operator <=>
        distance_expr = PromptEmbedding.embedding.cosine_distance(vector_list).label("distance")

        stmt = (
            select(PromptRecord, PromptEmbedding, distance_expr)
            .join(PromptRecord, PromptEmbedding.prompt_id == PromptRecord.id)
            .where(
                PromptRecord.owner_id == principal.id,
                PromptEmbedding.config_version == config_version,
            )
        )

        if model_id:
            stmt = stmt.where(PromptEmbedding.model_id == model_id)

        if task_type:
            stmt = stmt.where(PromptRecord.task_type == task_type)

        if min_similarity is not None:
            # Cosine distance = 1 - similarity.
            # Thus similarity >= min_similarity <=> distance <= (1.0 - min_similarity).
            max_distance = 1.0 - min_similarity
            stmt = stmt.where(distance_expr <= max_distance)

        # Order by distance ASC, secondary tie-breakers: created_at DESC, id ASC
        stmt = stmt.order_by(
            distance_expr.asc(),
            PromptRecord.created_at.desc(),
            PromptRecord.id.asc(),
        )

        # Over-fetch if stale filtering is active to account for potential invalidated records
        fetch_limit = min(max(bounded_limit * 2, 20), 100) if exclude_stale else bounded_limit
        stmt = stmt.limit(fetch_limit)

        try:
            db_result = await session.execute(stmt)
            rows = db_result.all()
        except Exception as exc:
            logger.error("Database vector query failed: %s", exc)
            raise DatabaseError(
                f"Failed to execute semantic similarity query: {exc}",
                original_error=exc,
            ) from exc

        results: List[Tuple[PromptRecord, PromptEmbedding, float, float]] = []
        for prompt, embedding, raw_distance in rows:
            # Stale embedding check: hash of prompt.formatted_prompt must match stored hash
            if exclude_stale:
                current_hash = PromptEmbedding.compute_content_hash(prompt.formatted_prompt)
                if embedding.source_content_hash != current_hash:
                    logger.warning(
                        "Excluding stale embedding for prompt %s (hash mismatch: stored=%s, current=%s)",
                        prompt.id,
                        embedding.source_content_hash,
                        current_hash,
                    )
                    continue

            # Defensively clamp distance and derived similarity
            dist = max(0.0, float(raw_distance))
            similarity = max(-1.0, min(1.0, 1.0 - dist))

            if min_similarity is not None and similarity < min_similarity:
                continue

            results.append((prompt, embedding, dist, similarity))
            if len(results) >= bounded_limit:
                break

        return results

