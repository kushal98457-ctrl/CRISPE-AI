"""Semantic Prompt Memory similarity search service for CRISPE-AI.

Orchestrates query validation, query embedding generation with task_type=RETRIEVAL_QUERY,
compatibility verification, owner-scoped pgvector similarity retrieval, and typed
response serialization.
"""

from typing import List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DatabaseError,
    EmptyPromptError,
    IncompatibleEmbeddingError,
    InvalidProviderResponseError,
    OwnershipViolationError,
    PromptValidationError,
    SemanticSearchError,
)
from app.core.logging import get_logger
from app.core.security import AuthorizedPrincipal
from app.repositories.embedding_repository import EmbeddingRepository
from app.schemas.embedding import EmbeddingResult, EmbeddingTaskType
from app.schemas.semantic_search import (
    SemanticSearchItem,
    SemanticSearchQuery,
    SemanticSearchResponse,
)
from app.services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)

logger = get_logger("semantic_search_service")


class SemanticSearchService:
    """Service managing semantic memory similarity retrieval across persistent prompt records."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or get_embedding_service(settings=self.settings)

    def _validate_embedding_compatibility(
        self,
        result: EmbeddingResult,
        expected_config_version: str,
    ) -> None:
        """Enforce strict dimensional and version compatibility for semantic search operations."""
        expected_dim = self.settings.EMBEDDING_DIMENSIONS
        if result.dimension != expected_dim or len(result.vector) != expected_dim:
            raise IncompatibleEmbeddingError(
                f"Query embedding dimension mismatch: expected {expected_dim}, got {result.dimension}."
            )

        if result.model_id != self.settings.EMBEDDING_MODEL_ID:
            raise IncompatibleEmbeddingError(
                f"Query embedding model mismatch: expected '{self.settings.EMBEDDING_MODEL_ID}', "
                f"got '{result.model_id}'."
            )

        if result.config_version != expected_config_version:
            raise IncompatibleEmbeddingError(
                f"Query embedding configuration version mismatch: expected '{expected_config_version}', "
                f"got '{result.config_version}'."
            )

    async def search(
        self,
        session: AsyncSession,
        *,
        principal: AuthorizedPrincipal,
        query: Union[str, SemanticSearchQuery],
        limit: int = 10,
        min_similarity: Optional[float] = None,
        task_type: Optional[str] = None,
        config_version: Optional[str] = None,
    ) -> SemanticSearchResponse:
        """Execute semantic prompt retrieval for an authenticated principal.

        Workflow:
        1. Validates principal identity and active status.
        2. Normalizes and validates search query and parameters.
        3. Generates 768-dim query embedding using Google text-embedding-004 (RETRIEVAL_QUERY task).
        4. Verifies embedding dimension and configuration version compatibility.
        5. Executes owner-scoped, exact pgvector cosine distance retrieval in PostgreSQL.
        6. Filters out stale or incompatible embeddings.
        7. Serializes matching prompts into SemanticSearchResponse, strictly excluding raw vectors.

        Parameters:
            session: Active database AsyncSession.
            principal: The authenticated security principal initiating the search.
            query: Raw search query string or validated SemanticSearchQuery instance.
            limit: Maximum number of nearest-neighbor results (default 10, bounded 1-50).
            min_similarity: Optional minimum cosine similarity threshold (1.0 - distance).
            task_type: Optional domain category or task type filter.
            config_version: Target embedding configuration version (defaults to active settings version).

        Returns:
            SemanticSearchResponse containing structured nearest neighbor prompt records.
        """
        # 1. Ownership boundary check
        if principal is None:
            raise OwnershipViolationError("Principal context is required to execute semantic search.")
        principal.ensure_active()

        # 2. Query parameter validation
        target_version = config_version or self.settings.EMBEDDING_CONFIG_VERSION
        if isinstance(query, str):
            query_obj = SemanticSearchQuery(
                query=query,
                limit=limit,
                min_similarity=min_similarity,
                task_type=task_type,
                config_version=target_version,
            )
        elif isinstance(query, SemanticSearchQuery):
            query_obj = query
        else:
            raise PromptValidationError(
                f"Query must be a string or SemanticSearchQuery, got {type(query).__name__}."
            )

        # 3. Generate query vector embedding via EmbeddingService (RETRIEVAL_QUERY)
        try:
            embedding_result = await self.embedding_service.embed_text(
                query_obj.query,
                task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
            )
        except (EmptyPromptError, PromptValidationError, IncompatibleEmbeddingError):
            raise
        except InvalidProviderResponseError as exc:
            if "dimension mismatch" in str(exc).lower() or "model mismatch" in str(exc).lower():
                raise IncompatibleEmbeddingError(str(exc), original_error=exc) from exc
            raise SemanticSearchError(
                f"Embedding generation failed for search query: {exc}",
                original_error=exc,
            ) from exc
        except Exception as exc:
            logger.error("Failed to generate query embedding: %s", exc)
            raise SemanticSearchError(
                f"Embedding generation failed for search query: {exc}",
                original_error=exc,
            ) from exc

        # 4. Strict compatibility check
        self._validate_embedding_compatibility(embedding_result, query_obj.config_version)

        # 5. Execute owner-scoped vector retrieval via repository
        raw_matches = await EmbeddingRepository.search_similar_prompts_for_owner(
            session=session,
            principal=principal,
            query_vector=embedding_result.vector,
            limit=query_obj.limit,
            min_similarity=query_obj.min_similarity,
            task_type=query_obj.task_type,
            config_version=query_obj.config_version,
            model_id=self.settings.EMBEDDING_MODEL_ID,
            exclude_stale=True,
        )

        # 6. Map matches to typed search items without raw vectors
        items: List[SemanticSearchItem] = []
        for prompt, emb, dist, sim in raw_matches:
            items.append(
                SemanticSearchItem(
                    prompt_id=prompt.id,
                    original_idea=prompt.original_idea,
                    sections=prompt.to_sections(),
                    formatted_prompt=prompt.formatted_prompt,
                    task_type=prompt.task_type,
                    model_id=emb.model_id,
                    distance=round(dist, 6),
                    similarity=round(sim, 6),
                    config_version=emb.config_version,
                    created_at=prompt.created_at,
                )
            )

        logger.info(
            "Semantic search executed for principal %s: query='%s', matched %d prompts (limit=%d)",
            principal.id,
            query_obj.query[:40],
            len(items),
            query_obj.limit,
        )

        return SemanticSearchResponse(
            query=query_obj.query,
            total_results=len(items),
            items=items,
            model_id=self.settings.EMBEDDING_MODEL_ID,
            config_version=query_obj.config_version,
            search_metric="cosine_distance",
        )


def get_semantic_search_service(
    embedding_service: Optional[EmbeddingService] = None,
    settings: Optional[Settings] = None,
) -> SemanticSearchService:
    """Factory creating a SemanticSearchService instance."""
    return SemanticSearchService(embedding_service=embedding_service, settings=settings)


def get_semantic_search_service_dependency() -> SemanticSearchService:
    """FastAPI parameterless dependency provider for SemanticSearchService."""
    return get_semantic_search_service()
