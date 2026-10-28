"""Root API v1 router aggregator."""

from fastapi import APIRouter
from app.api.routes import health

api_router = APIRouter()

# Register health probe router
api_router.include_router(health.router, tags=["Health"])
