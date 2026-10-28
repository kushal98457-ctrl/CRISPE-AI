"""Root API v1 router aggregator."""

from fastapi import APIRouter
from app.api.routes import health, prompts

api_router = APIRouter()

# Register health probe router
api_router.include_router(health.router, tags=["Health"])

# Register CRISPE prompt generation router
api_router.include_router(prompts.router)
