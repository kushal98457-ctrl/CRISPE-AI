"""Centralized application configuration using Pydantic Settings."""

import json
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support and validation."""

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

    # AI Model Configuration (Optional at startup; required when invoking generation)
    GEMINI_API_KEY: Optional[str] = Field(
        default=None, description="Google Gemini API key"
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

    # Database & Vector Memory Configuration (Optional at startup)
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

    # Optimization Engine Defaults (Feature A)
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


# Cached settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retrieve or initialize application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
