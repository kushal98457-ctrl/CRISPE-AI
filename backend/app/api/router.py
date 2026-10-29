"""Root API v1 router aggregator."""

from fastapi import APIRouter
from app.api.routes import health, memory, optimization, prompts

api_router = APIRouter()

# Register health probe router
api_router.include_router(health.router, tags=["Health"])

# Register CRISPE prompt generation router
api_router.include_router(prompts.router)

# Register Semantic Prompt Memory search router
api_router.include_router(memory.router)

# Register Self-Evolving Optimization Jobs router
api_router.include_router(optimization.router)
