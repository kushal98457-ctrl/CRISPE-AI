"""Health, liveness, and database readiness probe endpoints."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.db.session import check_database_connection
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Health Check",
    description="Confirms that the FastAPI application process is alive and receiving traffic. Does not depend on database.",
)
async def get_health(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Fast process liveness probe for load balancers."""
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Health Check",
    description="Verifies database connectivity before directing persistent traffic to this instance.",
    responses={
        status.HTTP_200_OK: {
            "description": "Database is reachable and ready to accept queries.",
            "model": ReadinessResponse,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Database is unreachable or degraded.",
            "model": ReadinessResponse,
        },
    },
)
async def get_readiness(
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """Readiness probe checking PostgreSQL availability via SELECT 1."""
    is_db_ready = await check_database_connection()

    if is_db_ready:
        payload = ReadinessResponse(
            status="ready",
            service=settings.APP_NAME,
            database="connected",
            environment=settings.ENVIRONMENT,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=payload.model_dump(),
        )

    payload = ReadinessResponse(
        status="degraded",
        service=settings.APP_NAME,
        database="disconnected",
        environment=settings.ENVIRONMENT,
        detail="PostgreSQL is unreachable or connection refused.",
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )
