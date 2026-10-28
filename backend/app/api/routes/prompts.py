"""API routes for CRISPE prompt generation and transformation."""

from fastapi import APIRouter, Depends, Request, status

from app.core.logging import get_logger
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeGenerationResponse,
)
from app.schemas.error import ErrorResponse
from app.services.generation_service import (
    CrispeGenerationService,
    get_generation_service_dependency,
)

logger = get_logger("prompts_route")

router = APIRouter(prefix="/prompts", tags=["Prompts"])


@router.post(
    "/generate",
    response_model=CrispeGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a structured CRISPE framework prompt",
    description=(
        "Transforms a raw user idea or task instruction into a production-grade prompt "
        "structured into six canonical CRISPE sections (Context, Role, Instruction, Specification, "
        "Performance, Example)."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "CRISPE prompt successfully synthesized.",
            "model": CrispeGenerationResponse,
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Validation error on input payload or structured model output.",
            "model": ErrorResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Model provider authentication failure.",
            "model": ErrorResponse,
        },
        status.HTTP_429_TOO_MANY_REQUESTS: {
            "description": "Provider rate limit or quota exceeded.",
            "model": ErrorResponse,
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Upstream AI model provider returned an invalid response or failure.",
            "model": ErrorResponse,
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "Request to upstream AI model provider timed out.",
            "model": ErrorResponse,
        },
    },
)
async def generate_prompt(
    payload: CrispeGenerationRequest,
    http_request: Request,
    service: CrispeGenerationService = Depends(get_generation_service_dependency),
) -> CrispeGenerationResponse:
    """Generate a structured CRISPE framework prompt from a raw user prompt."""
    request_id = getattr(http_request.state, "request_id", None)

    logger.info(
        "Received CRISPE prompt generation request [request_id=%s, idea_length=%d chars, model_id=%s]",
        request_id,
        len(payload.idea),
        payload.model_id or "default",
    )

    result = await service.generate_crispe_prompt(payload)

    # Attach correlation request identifier to the returned API response
    if request_id and not result.request_id:
        result.request_id = request_id

    logger.info(
        "Completed CRISPE prompt generation [request_id=%s, model_id=%s, total_tokens=%s]",
        request_id,
        result.model_id,
        result.usage.total_tokens if result.usage else "N/A",
    )

    return result
