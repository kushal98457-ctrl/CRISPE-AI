"""Pytest shared fixtures for CRISPE-AI backend test suite."""

import sys
from pathlib import Path
from typing import AsyncGenerator
import pytest
from httpx import ASGITransport, AsyncClient

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import Settings, get_settings
from app.main import create_application


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated test settings without external requirements."""
    return Settings(
        APP_NAME="crispe-ai-backend-test",
        ENVIRONMENT="testing",
        DEBUG=True,
        CORS_ORIGINS=["http://localhost:5173", "http://testserver"],
        GEMINI_API_KEY=None,
    )


@pytest.fixture
async def async_client(test_settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """Asynchronous HTTP test client bound to FastAPI application."""
    app = create_application(settings=test_settings)
    app.dependency_overrides[get_settings] = lambda: test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
