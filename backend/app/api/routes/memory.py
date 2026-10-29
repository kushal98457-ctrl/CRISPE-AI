"""API routes for Semantic Prompt Memory similarity retrieval."""

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.security import AuthorizedPrincipal, get_current_principal
from app.db.session import get_db_session
from app.schemas.error import ErrorResponse
from app.schemas.semantic_search import (
    SemanticMemorySearchRequest,
    SemanticMemorySearchResponse,
)
from app.services.semantic_search import (
    SemanticSearchService,
    get_semantic_search_service_dependency,
)

logger = get_logger("memory_route")

router = APIRouter(prefix="/memory", tags=["Semantic Memory"])


@router.post(
    "/search",
    response_model=SemanticMemorySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search saved CRISPE prompts using vector similarity",
    description=(
        "Retrieves semantically relevant persistent prompt records owned by the authenticated "
        "security principal using pgvector cosine distance operations and Google text-embedding-004."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Semantic search executed successfully; returns ordered matching prompts.",
            "model": SemanticMemorySearchResponse,
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Incompatible embedding model, dimension, or version configuration.",
            "model": ErrorResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication is required to query semantic prompt memory.",
            "model": ErrorResponse,
        },
        status.HTTP_403_FORBIDDEN: {
            "description": "Caller is inactive or lacks permission to access target prompt memory.",
            "model": ErrorResponse,
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid search query, limit, or threshold parameters.",
            "model": ErrorResponse,
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "Upstream embedding provider quota or rate limit exceeded.",
            "model": ErrorResponse,
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Upstream embedding provider returned an invalid response.",
            "model": ErrorResponse,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "PostgreSQL database or embedding provider is temporarily unreachable.",
            "model": ErrorResponse,
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "Upstream embedding provider or database query timed out.",
            "model": ErrorResponse,
        },
    },
)
async def search_memory(
    payload: SemanticMemorySearchRequest,
    http_request: Request,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    service: SemanticSearchService = Depends(get_semantic_search_service_dependency),
) -> SemanticMemorySearchResponse:
    """Execute owner-scoped semantic similarity search over persistent prompt records."""
    request_id = getattr(http_request.state, "request_id", None)
    principal.ensure_active()

    logger.info(
        "Semantic memory search initiated [request_id=%s, principal=%s, query_len=%d chars, limit=%d, task_type=%s, min_sim=%s]",
        request_id,
        principal.id,
        len(payload.query),
        payload.limit,
        payload.task_type or "all",
        payload.min_similarity if payload.min_similarity is not None else "none",
    )

    # Delegate to the established SemanticSearchService
    service_response = await service.search(
        session=session,
        principal=principal,
        query=payload.query,
        limit=payload.limit,
        min_similarity=payload.min_similarity,
        task_type=payload.task_type,
    )

    # Transform internal service response into client-facing API response contract
    api_response = SemanticMemorySearchResponse.from_service_response(service_response)

    logger.info(
        "Semantic memory search completed [request_id=%s, principal=%s, result_count=%d]",
        request_id,
        principal.id,
        api_response.result_count,
    )

    return api_response
