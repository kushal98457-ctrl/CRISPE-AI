"""Health and liveness probe endpoints."""

from fastapi import APIRouter, Depends
from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Health Check",
    description="Confirms that the FastAPI application process is alive and receiving traffic.",
)
async def get_health(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )
