"""Centralized application and Gemini provider configuration using Pydantic Settings."""

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Base exception for application configuration errors."""
    pass


class MissingProviderCredentialError(ConfigurationError):
    """Raised when an operation requiring Google GenAI is invoked without an API key."""
    pass


class Settings(BaseSettings):
    """Application and provider settings with environment variable support and validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application identity
    APP_NAME: str = "crispe-ai-backend"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = Field(
        default="development",
        description="Runtime environment: development, staging, or production",
    )
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    HOST: str = Field(default="127.0.0.1", description="Backend bind host")
    PORT: int = Field(default=8000, description="Backend bind port")
    API_V1_PREFIX: str = Field(
        default="/api/v1", description="Prefix for API version 1 routes"
    )

    # Logging
    LOG_LEVEL: str = Field(
        default="INFO", description="Standard logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Cross-Origin Resource Sharing (CORS)
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed frontend origin URLs for CORS",
    )

    # --------------------------------------------------------------------------
    # Gemini AI Provider Configuration (Step 6)
    # --------------------------------------------------------------------------
    PROVIDER_NAME: str = Field(
        default="google-genai",
        description="Active AI model provider interface name",
    )
    GEMINI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description="Google Gemini API key (supports GEMINI_API_KEY or GOOGLE_API_KEY)",
    )
    PRIMARY_MODEL: str = Field(
        default="gemma-4-26b-a4b-it",
        description="Primary CRISPE prompt generation model checkpoint",
    )
    FALLBACK_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Production fallback model if primary is unavailable",
    )
    REFLECTION_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Reflection model for DSPy/GEPA optimization diagnostics",
    )

    # Operational limits & timeouts
    GEMINI_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Request timeout in seconds for Gemini API operations",
    )
    GEMINI_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient network or quota errors",
    )
    GEMINI_MAX_OUTPUT_TOKENS: int = Field(
        default=4096,
        ge=256,
        le=65536,
        description="Maximum permitted output tokens per prompt generation",
    )
    GEMINI_DEFAULT_TEMPERATURE: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature for prompt generation",
    )
    GEMINI_ENABLE_THINKING: bool = Field(
        default=False,
        description="Enable model internal reasoning mode where supported (e.g. Gemini 2.5 Flash)",
    )

    # --------------------------------------------------------------------------
    # Semantic Memory & Vector Database (Feature C Placeholders)
    # --------------------------------------------------------------------------
    DATABASE_URL: Optional[str] = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/crispe_ai",
        description="PostgreSQL connection string with psycopg driver",
    )
    EMBEDDING_MODEL: str = Field(
        default="gemini-embedding-001",
        description="Google embedding model identifier",
    )
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Vector embedding dimension (standardized for pgvector)",
    )

    # --------------------------------------------------------------------------
    # Optimization Engine Defaults (Feature A Placeholders)
    # --------------------------------------------------------------------------
    DEFAULT_MAX_ITERATIONS: int = Field(
        default=5, ge=1, le=50, description="Default max iterations per optimization run"
    )
    MAX_BUDGET_USD_PER_RUN: float = Field(
        default=2.50, gt=0.0, description="Maximum budget allowed per run in USD"
    )
    MAX_CONCURRENT_RUNS: int = Field(
        default=2, ge=1, le=10, description="Maximum concurrent optimization runs"
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> List[str]:
        """Parse CORS origins from JSON string, comma-separated string, or list."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, list):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    def get_gemini_api_key(self) -> str:
        """
        Safely retrieve the plaintext Gemini API key.
        Raises MissingProviderCredentialError if the credential is not configured or empty.
        """
        if self.GEMINI_API_KEY is None:
            raise MissingProviderCredentialError(
                "Google Gemini API key is not configured. "
                "Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in backend/.env or your environment."
            )
        secret_value = self.GEMINI_API_KEY.get_secret_value().strip()
        if not secret_value:
            raise MissingProviderCredentialError(
                "Google Gemini API key is configured but empty. "
                "Please provide a valid API key in GEMINI_API_KEY."
            )
        return secret_value

    def has_gemini_api_key(self) -> bool:
        """Check if a non-empty Gemini API key is configured without exposing it."""
        if self.GEMINI_API_KEY is None:
            return False
        return bool(self.GEMINI_API_KEY.get_secret_value().strip())

    def safe_summary(self) -> Dict[str, Any]:
        """
        Return a sanitized diagnostic dictionary of configuration settings.
        Sensitive values (API keys, database passwords) are masked or omitted.
        """
        raw_db = self.DATABASE_URL or ""
        masked_db = (
            re.sub(r"://([^:]+):([^@]+)@", r"://\1:[REDACTED]@", raw_db)
            if raw_db
            else None
        )

        return {
            "app_name": self.APP_NAME,
            "app_version": self.APP_VERSION,
            "environment": self.ENVIRONMENT,
            "debug": self.DEBUG,
            "host": self.HOST,
            "port": self.PORT,
            "api_v1_prefix": self.API_V1_PREFIX,
            "log_level": self.LOG_LEVEL,
            "cors_origins": self.CORS_ORIGINS,
            "provider_name": self.PROVIDER_NAME,
            "primary_model": self.PRIMARY_MODEL,
            "fallback_model": self.FALLBACK_MODEL,
            "reflection_model": self.REFLECTION_MODEL,
            "gemini_api_key_configured": self.has_gemini_api_key(),
            "gemini_timeout_seconds": self.GEMINI_TIMEOUT_SECONDS,
            "gemini_max_retries": self.GEMINI_MAX_RETRIES,
            "gemini_max_output_tokens": self.GEMINI_MAX_OUTPUT_TOKENS,
            "gemini_default_temperature": self.GEMINI_DEFAULT_TEMPERATURE,
            "gemini_enable_thinking": self.GEMINI_ENABLE_THINKING,
            "database_url": masked_db,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "default_max_iterations": self.DEFAULT_MAX_ITERATIONS,
            "max_budget_usd_per_run": self.MAX_BUDGET_USD_PER_RUN,
            "max_concurrent_runs": self.MAX_CONCURRENT_RUNS,
        }


# Cached settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retrieve or initialize application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
