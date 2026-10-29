"""Centralized application and Gemini provider configuration using Pydantic Settings."""

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


from app.core.exceptions import (
    ConfigurationError,
    DatabaseConfigError,
    MissingProviderCredentialError,
)


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
        validation_alias=AliasChoices("PRIMARY_MODEL", "GEMINI_MODEL_ID"),
        description="Primary CRISPE prompt generation model checkpoint",
    )
    FALLBACK_MODEL: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("FALLBACK_MODEL", "GEMINI_FALLBACK_MODEL_ID"),
        description="Production fallback model if primary is unavailable",
    )
    REFLECTION_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Reflection model for DSPy/GEPA optimization diagnostics",
    )
    EVALUATOR_MODEL: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("EVALUATOR_MODEL", "JUDGE_MODEL_ID"),
        description="LLM-as-a-judge model identifier for model-assisted prompt evaluation",
    )
    EVALUATOR_TIMEOUT_SECONDS: float = Field(
        default=45.0,
        ge=5.0,
        le=300.0,
        description="Request timeout in seconds for evaluation judge model",
    )
    EVALUATOR_MAX_RETRIES: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts on transient network or quota errors for judge",
    )
    EVALUATOR_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for judge model (0.0 recommended for deterministic grading)",
    )
    EVALUATOR_MAX_CONCURRENCY: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent judge evaluation requests",
    )

    @property
    def GEMINI_MODEL_ID(self) -> str:
        """Alias property for PRIMARY_MODEL."""
        return self.PRIMARY_MODEL

    @property
    def GEMINI_FALLBACK_MODEL_ID(self) -> str:
        """Alias property for FALLBACK_MODEL."""
        return self.FALLBACK_MODEL

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
    GEMINI_THINKING_BUDGET: int = Field(
        default=1024,
        ge=0,
        le=32768,
        description="Reasoning token budget for models supporting thinking mode",
    )

    # --------------------------------------------------------------------------
    # Semantic Memory & Vector Database (Feature C)
    # --------------------------------------------------------------------------
    DATABASE_URL: Optional[str] = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/crispe_ai",
        description="PostgreSQL connection string with psycopg driver",
    )
    DB_POOL_SIZE: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Database connection pool size",
    )
    DB_MAX_OVERFLOW: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum overflow connections beyond pool size",
    )
    DB_POOL_TIMEOUT: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Seconds to wait before connection acquisition times out",
    )
    DB_POOL_RECYCLE: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Recycle connections after this duration in seconds",
    )
    DB_ECHO_SQL: bool = Field(
        default=False,
        description="Log executed SQL queries (development/debugging only)",
    )
    EMBEDDING_PROVIDER: str = Field(
        default="google-genai",
        description="Active embedding provider backend name (e.g. google-genai)",
    )
    EMBEDDING_MODEL_ID: str = Field(
        default="text-embedding-004",
        validation_alias=AliasChoices("EMBEDDING_MODEL_ID", "EMBEDDING_MODEL"),
        description="Official Google embedding model checkpoint identifier",
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=768,
        validation_alias=AliasChoices("EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSION"),
        description="Output vector embedding dimension (768 default for text-embedding-004)",
    )
    EMBEDDING_CONFIG_VERSION: str = Field(
        default="v1",
        description="Semantic memory configuration version distinguishing incompatible vector spaces",
    )
    EMBEDDING_SIMILARITY_METRIC: str = Field(
        default="cosine",
        description="Vector distance metric for pgvector retrieval (cosine <=> recommended)",
    )
    EMBEDDING_MAX_INPUT_TOKENS: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum permitted input tokens for embedding requests",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=100,
        description="Maximum number of texts processed in a single batch embedding call",
    )
    EMBEDDING_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Request timeout in seconds for embedding API operations",
    )
    EMBEDDING_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Optional dedicated API key for embedding calls; falls back to GEMINI_API_KEY if omitted",
    )

    @property
    def EMBEDDING_MODEL(self) -> str:
        """Alias property for EMBEDDING_MODEL_ID (backward compatibility)."""
        return self.EMBEDDING_MODEL_ID

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        """Alias property for EMBEDDING_DIMENSIONS (backward compatibility)."""
        return self.EMBEDDING_DIMENSIONS

    # --------------------------------------------------------------------------
    # Optimization Engine & AI Resource Budgeting (Steps 24-25)
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
    BUDGET_MAX_METRIC_CALLS_DEFAULT: int = Field(
        default=20, ge=1, le=100, description="Default ceiling on GEPA metric invocations"
    )
    BUDGET_MAX_PROVIDER_REQUESTS_DEFAULT: int = Field(
        default=50, ge=1, le=500, description="Default ceiling on total provider API requests"
    )
    BUDGET_MAX_ELAPSED_SECONDS_DEFAULT: float = Field(
        default=300.0, ge=5.0, le=3600.0, description="Default max elapsed execution seconds"
    )
    BUDGET_MAX_CONCURRENT_REQUESTS_DEFAULT: int = Field(
        default=2, ge=1, le=20, description="Default max concurrent model requests"
    )
    BUDGET_MAX_RETRIES_DEFAULT: int = Field(
        default=3, ge=0, le=10, description="Default max retry attempts per operation"
    )
    BUDGET_MAX_PROCESSED_EXAMPLES_DEFAULT: int = Field(
        default=20, ge=1, le=100, description="Default max dataset examples processed per run"
    )
    BUDGET_MAX_TOTAL_TOKENS_DEFAULT: Optional[int] = Field(
        default=100000, ge=100, le=10000000, description="Default aggregate token ceiling"
    )
    BUDGET_PRICING_CONFIG_VERSION: str = Field(
        default="gemini-2025-standard-v1",
        description="Pricing table version used for cost estimates",
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

    @field_validator("EMBEDDING_DIMENSIONS")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        """Validate embedding dimensions against documented constraints.

        text-embedding-004 supports Matryoshka dimensions (128, 256, 512, 768).
        """
        if value <= 0:
            raise ValueError(f"EMBEDDING_DIMENSIONS must be a positive integer, got {value}")
        if value > 768:
            raise ValueError(
                f"EMBEDDING_DIMENSIONS {value} exceeds the maximum supported dimension (768) "
                f"for text-embedding-004."
            )
        valid_dimensions = {128, 256, 512, 768}
        if value not in valid_dimensions:
            raise ValueError(
                f"EMBEDDING_DIMENSIONS {value} is not a supported dimension for text-embedding-004. "
                f"Supported dimensions are {sorted(valid_dimensions)}."
            )
        return value

    @field_validator("EMBEDDING_SIMILARITY_METRIC")
    @classmethod
    def validate_similarity_metric(cls, value: str) -> str:
        """Validate vector distance similarity metric."""
        normalized = value.strip().lower()
        if normalized not in {"cosine", "euclidean", "inner_product"}:
            raise ValueError(
                f"EMBEDDING_SIMILARITY_METRIC '{value}' is unsupported. "
                f"Must be one of: 'cosine', 'euclidean', 'inner_product'."
            )
        return normalized

    @field_validator("BUDGET_MAX_PROVIDER_REQUESTS_DEFAULT")
    @classmethod
    def validate_budget_provider_requests(cls, value: int, info) -> int:
        """Ensure provider request ceiling is at least equal to metric call ceiling."""
        metric_calls = info.data.get("BUDGET_MAX_METRIC_CALLS_DEFAULT", 20)
        if value < metric_calls:
            raise ValueError(
                f"BUDGET_MAX_PROVIDER_REQUESTS_DEFAULT ({value}) must be greater than or equal to "
                f"BUDGET_MAX_METRIC_CALLS_DEFAULT ({metric_calls})"
            )
        return value

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

    def get_embedding_api_key(self) -> str:
        """Retrieve the plaintext API key for embedding operations.

        Reuses GEMINI_API_KEY if EMBEDDING_API_KEY is not explicitly provided.
        Raises MissingProviderCredentialError if neither credential is configured.
        """
        if self.EMBEDDING_API_KEY is not None:
            secret = self.EMBEDDING_API_KEY.get_secret_value().strip()
            if secret:
                return secret
        return self.get_gemini_api_key()

    def has_embedding_api_key(self) -> bool:
        """Check if a valid embedding API key is available without exposing it."""
        if self.EMBEDDING_API_KEY is not None and self.EMBEDDING_API_KEY.get_secret_value().strip():
            return True
        return self.has_gemini_api_key()

    def get_database_url(self) -> str:
        """Retrieve and normalize the asynchronous database connection URL."""
        if not self.DATABASE_URL or not self.DATABASE_URL.strip():
            raise DatabaseConfigError(
                "Database URL is not configured. Please set DATABASE_URL in the environment."
            )
        url = self.DATABASE_URL.strip()
        # Normalize standard postgresql:// URL to postgresql+psycopg://
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        return url

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
            "db_pool_size": self.DB_POOL_SIZE,
            "db_max_overflow": self.DB_MAX_OVERFLOW,
            "db_pool_timeout": self.DB_POOL_TIMEOUT,
            "db_pool_recycle": self.DB_POOL_RECYCLE,
            "embedding_provider": self.EMBEDDING_PROVIDER,
            "embedding_model": self.EMBEDDING_MODEL_ID,
            "embedding_dimension": self.EMBEDDING_DIMENSIONS,
            "embedding_config_version": self.EMBEDDING_CONFIG_VERSION,
            "embedding_similarity_metric": self.EMBEDDING_SIMILARITY_METRIC,
            "embedding_max_input_tokens": self.EMBEDDING_MAX_INPUT_TOKENS,
            "embedding_batch_size": self.EMBEDDING_BATCH_SIZE,
            "embedding_timeout_seconds": self.EMBEDDING_TIMEOUT_SECONDS,
            "embedding_api_key_configured": self.has_embedding_api_key(),
            "default_max_iterations": self.DEFAULT_MAX_ITERATIONS,
            "max_budget_usd_per_run": self.MAX_BUDGET_USD_PER_RUN,
            "max_concurrent_runs": self.MAX_CONCURRENT_RUNS,
            "budget_max_metric_calls_default": self.BUDGET_MAX_METRIC_CALLS_DEFAULT,
            "budget_max_provider_requests_default": self.BUDGET_MAX_PROVIDER_REQUESTS_DEFAULT,
            "budget_max_elapsed_seconds_default": self.BUDGET_MAX_ELAPSED_SECONDS_DEFAULT,
            "budget_max_concurrent_requests_default": self.BUDGET_MAX_CONCURRENT_REQUESTS_DEFAULT,
            "budget_max_retries_default": self.BUDGET_MAX_RETRIES_DEFAULT,
            "budget_max_processed_examples_default": self.BUDGET_MAX_PROCESSED_EXAMPLES_DEFAULT,
            "budget_max_total_tokens_default": self.BUDGET_MAX_TOTAL_TOKENS_DEFAULT,
            "budget_pricing_config_version": self.BUDGET_PRICING_CONFIG_VERSION,
        }


# Cached settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retrieve or initialize application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
