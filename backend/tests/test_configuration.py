"""Tests for application and Gemini provider configuration parsing, security, and validation."""

import pytest
from app.core.config import MissingProviderCredentialError, Settings


def test_default_settings():
    """Verify default non-secret values load safely."""
    settings = Settings()
    assert settings.APP_NAME == "crispe-ai-backend"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.PORT == 8000
    assert "http://localhost:5173" in settings.CORS_ORIGINS
    assert settings.PROVIDER_NAME == "google-genai"
    assert settings.PRIMARY_MODEL == "gemma-4-26b-a4b-it"
    assert settings.FALLBACK_MODEL == "gemini-2.5-flash"
    assert settings.REFLECTION_MODEL == "gemini-2.5-flash"
    assert settings.GEMINI_TIMEOUT_SECONDS == 30.0
    assert settings.GEMINI_MAX_RETRIES == 3
    assert settings.GEMINI_MAX_OUTPUT_TOKENS == 4096
    assert settings.GEMINI_DEFAULT_TEMPERATURE == 0.4
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
    assert settings.has_gemini_api_key() is False


def test_get_gemini_api_key_raises_when_missing():
    """Verify accessing the key when missing raises MissingProviderCredentialError."""
    settings = Settings(GEMINI_API_KEY=None)
    with pytest.raises(MissingProviderCredentialError) as exc_info:
        settings.get_gemini_api_key()
    assert "not configured" in str(exc_info.value)


def test_get_gemini_api_key_raises_when_empty_string():
    """Verify accessing an empty or whitespace key raises an error."""
    settings = Settings(GEMINI_API_KEY="   ")
    with pytest.raises(MissingProviderCredentialError) as exc_info:
        settings.get_gemini_api_key()
    assert "configured but empty" in str(exc_info.value)


def test_get_gemini_api_key_returns_raw_string_when_valid():
    """Verify valid SecretStr safely unwraps to plain string only when explicitly called."""
    secret_key = "AIzaSy_test_secret_key_12345"
    settings = Settings(GEMINI_API_KEY=secret_key)
    assert settings.has_gemini_api_key() is True
    assert settings.get_gemini_api_key() == secret_key


def test_google_api_key_alias_support():
    """Verify GOOGLE_API_KEY environment alias properly populates GEMINI_API_KEY."""
    settings = Settings(GOOGLE_API_KEY="AIzaSy_google_alias_key")
    assert settings.has_gemini_api_key() is True
    assert settings.get_gemini_api_key() == "AIzaSy_google_alias_key"


def test_secret_str_masking_in_repr_and_str():
    """Verify SecretStr never leaks raw key value in str(), repr(), or model_dump()."""
    raw_key = "AIzaSy_secret_key_must_not_leak"
    settings = Settings(GEMINI_API_KEY=raw_key)

    # str() and repr() must mask
    assert raw_key not in str(settings.GEMINI_API_KEY)
    assert raw_key not in repr(settings.GEMINI_API_KEY)
    assert raw_key not in repr(settings)

    # model_dump should not contain the plain secret value by default
    dump = settings.model_dump()
    assert dump["GEMINI_API_KEY"] is not raw_key


def test_safe_summary_masks_sensitive_values():
    """Verify safe_summary() includes diagnostic metadata but masks secrets."""
    raw_key = "AIzaSy_super_secret_key"
    raw_db = "postgresql+psycopg://myuser:supersecretpassword@localhost:5432/mydb"
    settings = Settings(GEMINI_API_KEY=raw_key, DATABASE_URL=raw_db)

    summary = settings.safe_summary()
    assert summary["gemini_api_key_configured"] is True
    assert raw_key not in str(summary)
    assert "supersecretpassword" not in str(summary)
    assert "[REDACTED]" in summary["database_url"]


def test_timeout_and_retries_validation():
    """Verify timeout and retry limit boundaries are enforced."""
    # Valid overrides
    settings = Settings(GEMINI_TIMEOUT_SECONDS=45.0, GEMINI_MAX_RETRIES=5)
    assert settings.GEMINI_TIMEOUT_SECONDS == 45.0
    assert settings.GEMINI_MAX_RETRIES == 5

    # Out of bounds timeout (< 5.0s or > 300.0s)
    with pytest.raises(ValueError):
        Settings(GEMINI_TIMEOUT_SECONDS=2.0)
    with pytest.raises(ValueError):
        Settings(GEMINI_TIMEOUT_SECONDS=400.0)

    # Negative retries
    with pytest.raises(ValueError):
        Settings(GEMINI_MAX_RETRIES=-1)


def test_max_output_tokens_validation():
    """Verify output token range boundaries."""
    # Under minimum
    with pytest.raises(ValueError):
        Settings(GEMINI_MAX_OUTPUT_TOKENS=100)

    # Valid
    s = Settings(GEMINI_MAX_OUTPUT_TOKENS=8192)
    assert s.GEMINI_MAX_OUTPUT_TOKENS == 8192


def test_budget_constraints_validation():
    """Verify invalid budget or iteration parameters raise validation error."""
    with pytest.raises(ValueError):
        Settings(MAX_BUDGET_USD_PER_RUN=-1.0)

    with pytest.raises(ValueError):
        Settings(DEFAULT_MAX_ITERATIONS=0)
