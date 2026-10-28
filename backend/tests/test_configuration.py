"""Tests for application configuration parsing and validation."""

import pytest
from app.core.config import Settings


def test_default_settings():
    """Verify default non-secret values load safely."""
    settings = Settings()
    assert settings.APP_NAME == "crispe-ai-backend"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.PORT == 8000
    assert "http://localhost:5173" in settings.CORS_ORIGINS
    assert settings.PRIMARY_MODEL == "gemma-4-26b-a4b-it"
    assert settings.EMBEDDING_DIMENSION == 768


def test_cors_origins_json_parsing():
    """Verify JSON-string formatted CORS origins parse into list of strings."""
    settings = Settings(CORS_ORIGINS='["http://example.com", "https://app.domain.com"]')
    assert settings.CORS_ORIGINS == ["http://example.com", "https://app.domain.com"]


def test_cors_origins_comma_separated_parsing():
    """Verify comma-separated string CORS origins parse properly."""
    settings = Settings(CORS_ORIGINS="http://localhost:3000, https://client.app")
    assert settings.CORS_ORIGINS == ["http://localhost:3000", "https://client.app"]


def test_missing_gemini_key_does_not_break_startup():
    """Verify application settings initialize cleanly without a Gemini key."""
    settings = Settings(GEMINI_API_KEY=None)
    assert settings.GEMINI_API_KEY is None


def test_budget_constraints_validation():
    """Verify invalid budget or iteration parameters raise validation error."""
    with pytest.raises(ValueError):
        Settings(MAX_BUDGET_USD_PER_RUN=-1.0)

    with pytest.raises(ValueError):
        Settings(DEFAULT_MAX_ITERATIONS=0)
