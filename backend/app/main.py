"""Main FastAPI application entry point for CRISPE-AI."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.api.routes import health
from app.core.config import Settings, get_settings
from app.core.exceptions import PromptGenerationError, ProviderError
from app.core.logging import get_logger, setup_logging
from app.middleware.request_context import RequestContextMiddleware
from app.schemas.error import ErrorDetail, ErrorResponse

logger = get_logger("crispe_app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan managing startup configuration and graceful shutdown."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "CRISPE-AI backend initialized: %s v%s [env: %s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )
    yield
    logger.info("CRISPE-AI backend shutting down gracefully.")
    try:
        from app.adapters.gemini_adapter import get_model_adapter
        adapter = get_model_adapter()
        await adapter.close()
    except Exception as exc:
        logger.warning("Error during model adapter shutdown: %s", str(exc))


def create_application(settings: Settings | None = None) -> FastAPI:
    """Application factory for FastAPI instance."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="CRISPE-AI Backend",
        description="FastAPI service for CRISPE Prompt Forge, Semantic Prompt Memory, and Self-Evolving Engine",
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # 1. Register Request Context & Telemetry Middleware (Outermost)
    app.add_middleware(RequestContextMiddleware)

    # 2. Register Cross-Origin Resource Sharing (CORS) Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # 3. Register Global Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        errors = [
            ErrorDetail(loc=list(err.get("loc", [])), msg=err.get("msg", ""), type=err.get("type", ""))
            for err in exc.errors()
        ]
        response_content = ErrorResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            error_code="VALIDATION_ERROR",
            message="Request input validation failed.",
            request_id=request_id,
            details=errors,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=response_content.model_dump(),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        error_code = "HTTP_ERROR"
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            error_code = "NOT_FOUND"
        elif exc.status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
            error_code = "METHOD_NOT_ALLOWED"

        response_content = ErrorResponse(
            status_code=exc.status_code,
            error_code=error_code,
            message=str(exc.detail),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response_content.model_dump(),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(ProviderError)
    async def provider_exception_handler(
        request: Request, exc: ProviderError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.error(
            "Provider error during request [%s]: code=%s msg=%s",
            request_id,
            exc.error_code,
            exc.message,
        )
        response_content = ErrorResponse(
            status_code=exc.status_code,
            error_code=exc.error_code,
            message=exc.message,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response_content.model_dump(),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(PromptGenerationError)
    async def prompt_generation_exception_handler(
        request: Request, exc: PromptGenerationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.warning(
            "Prompt generation error during request [%s]: code=%s msg=%s",
            request_id,
            exc.error_code,
            exc.message,
        )
        response_content = ErrorResponse(
            status_code=exc.status_code,
            error_code=exc.error_code,
            message=exc.message,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response_content.model_dump(),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("Unhandled internal exception during request: %s", str(exc))
        response_content = ErrorResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred. Please reference the request ID when reporting.",
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_content.model_dump(),
            headers={"X-Request-ID": request_id},
        )

    # 4. Register API Routers
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    # Root level health probe alias for cloud load balancers
    app.include_router(health.router, tags=["Health"])

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
