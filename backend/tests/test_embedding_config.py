"""Automated unit and compatibility tests for Semantic Prompt Memory embedding configuration (Step 12)."""

import math
import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.core.exceptions import MissingProviderCredentialError
from app.schemas.embedding import (
    EmbeddingBatchRequest,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingTaskType,
    EmbeddingUsageMetadata,
)


# -----------------------------------------------------------------------------
# Valid Configuration Tests
# -----------------------------------------------------------------------------

def test_default_embedding_settings() -> None:
    """Default settings specify google-genai, text-embedding-004, 768 dimensions, and v1 version."""
    settings = Settings()

    assert settings.EMBEDDING_PROVIDER == "google-genai"
    assert settings.EMBEDDING_MODEL_ID == "text-embedding-004"
    assert settings.EMBEDDING_DIMENSIONS == 768
    assert settings.EMBEDDING_CONFIG_VERSION == "v1"
    assert settings.EMBEDDING_SIMILARITY_METRIC == "cosine"
    assert settings.EMBEDDING_MAX_INPUT_TOKENS == 2048
    assert settings.EMBEDDING_BATCH_SIZE == 32
    assert settings.EMBEDDING_TIMEOUT_SECONDS == 10.0

    # Backward compatibility properties
    assert settings.EMBEDDING_MODEL == "text-embedding-004"
    assert settings.EMBEDDING_DIMENSION == 768


def test_supported_matryoshka_dimensions() -> None:
    """text-embedding-004 supports Matryoshka dimensions (128, 256, 512, 768)."""
    for dim in [128, 256, 512, 768]:
        s = Settings(EMBEDDING_DIMENSIONS=dim)
        assert s.EMBEDDING_DIMENSIONS == dim
        assert s.EMBEDDING_DIMENSION == dim


def test_supported_similarity_metrics() -> None:
    """Supported distance metrics are cosine, euclidean, inner_product (case-insensitive)."""
    for metric in ["cosine", "COSINE", "euclidean", "inner_product"]:
        s = Settings(EMBEDDING_SIMILARITY_METRIC=metric)
        assert s.EMBEDDING_SIMILARITY_METRIC == metric.strip().lower()


def test_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables successfully override embedding configuration."""
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "text-embedding-004")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    monkeypatch.setenv("EMBEDDING_CONFIG_VERSION", "v2-custom")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "64")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "15.5")

    s = Settings()
    assert s.EMBEDDING_DIMENSIONS == 512
    assert s.EMBEDDING_CONFIG_VERSION == "v2-custom"
    assert s.EMBEDDING_BATCH_SIZE == 64
    assert s.EMBEDDING_TIMEOUT_SECONDS == 15.5


# -----------------------------------------------------------------------------
# Invalid Configuration Tests
# -----------------------------------------------------------------------------

def test_unsupported_dimensions_rejected() -> None:
    """Arbitrary, non-positive, or non-Matryoshka dimensions are strictly rejected."""
    for invalid_dim in [0, -1, 100, 384, 1024, 1536]:
        with pytest.raises(ValidationError):
            Settings(EMBEDDING_DIMENSIONS=invalid_dim)


def test_unsupported_similarity_metric_rejected() -> None:
    """Unrecognized distance metrics are rejected."""
    for invalid_metric in ["manhattan", "hamming", "levenshtein", ""]:
        with pytest.raises(ValidationError):
            Settings(EMBEDDING_SIMILARITY_METRIC=invalid_metric)


def test_invalid_batch_size_rejected() -> None:
    """Batch size must be between 1 and 100."""
    with pytest.raises(ValidationError):
        Settings(EMBEDDING_BATCH_SIZE=0)
    with pytest.raises(ValidationError):
        Settings(EMBEDDING_BATCH_SIZE=101)


def test_invalid_timeout_rejected() -> None:
    """Timeout must be at least 1.0 second and at most 120.0 seconds."""
    with pytest.raises(ValidationError):
        Settings(EMBEDDING_TIMEOUT_SECONDS=0.5)
    with pytest.raises(ValidationError):
        Settings(EMBEDDING_TIMEOUT_SECONDS=150.0)


# -----------------------------------------------------------------------------
# Configuration Isolation & Credential Management Tests
# -----------------------------------------------------------------------------

def test_credential_fallback_to_gemini_api_key() -> None:
    """Embedding operations reuse GEMINI_API_KEY when EMBEDDING_API_KEY is not set."""
    s = Settings(
        GEMINI_API_KEY=SecretStr("gemini-test-secret-key"),
        EMBEDDING_API_KEY=None,
    )
    assert s.has_embedding_api_key() is True
    assert s.get_embedding_api_key() == "gemini-test-secret-key"


def test_dedicated_embedding_api_key_takes_precedence() -> None:
    """Dedicated EMBEDDING_API_KEY takes precedence over GEMINI_API_KEY when set."""
    s = Settings(
        GEMINI_API_KEY=SecretStr("gemini-primary-key"),
        EMBEDDING_API_KEY=SecretStr("embedding-dedicated-key"),
    )
    assert s.has_embedding_api_key() is True
    assert s.get_embedding_api_key() == "embedding-dedicated-key"


def test_missing_credentials_raise_error() -> None:
    """Attempting to retrieve key raises MissingProviderCredentialError when unset."""
    s = Settings(GEMINI_API_KEY=None, EMBEDDING_API_KEY=None)
    assert s.has_embedding_api_key() is False
    with pytest.raises(MissingProviderCredentialError):
        s.get_embedding_api_key()


def test_embedding_settings_do_not_alter_generation_model() -> None:
    """Configuring embedding model does NOT silently change generation models."""
    s = Settings(
        EMBEDDING_MODEL_ID="text-embedding-004",
        EMBEDDING_DIMENSIONS=512,
    )
    assert s.PRIMARY_MODEL == "gemma-4-26b-a4b-it"
    assert s.GEMINI_MODEL_ID == "gemma-4-26b-a4b-it"
    assert s.FALLBACK_MODEL == "gemini-2.5-flash"


def test_safe_summary_masks_credentials_and_includes_embedding() -> None:
    """Diagnostic safe_summary() includes all embedding metadata with zero credential leakage."""
    s = Settings(
        GEMINI_API_KEY=SecretStr("secret-gemini-key"),
        EMBEDDING_API_KEY=SecretStr("secret-embedding-key"),
        EMBEDDING_DIMENSIONS=768,
    )
    summary = s.safe_summary()

    # Embedding configuration present
    assert summary["embedding_provider"] == "google-genai"
    assert summary["embedding_model"] == "text-embedding-004"
    assert summary["embedding_dimension"] == 768
    assert summary["embedding_config_version"] == "v1"
    assert summary["embedding_similarity_metric"] == "cosine"
    assert summary["embedding_api_key_configured"] is True

    # No secret leakage in serialized summary
    raw_text = str(summary)
    assert "secret-gemini-key" not in raw_text
    assert "secret-embedding-key" not in raw_text


# -----------------------------------------------------------------------------
# Service Contract & Schema Validation Invariants Tests
# -----------------------------------------------------------------------------

def test_embedding_request_validation() -> None:
    """EmbeddingRequest validates input text, task types, and rejects empty strings."""
    req = EmbeddingRequest(
        text="Architect a resilient OAuth 2.1 PKCE token exchange flow.",
        task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
    )
    assert req.text == "Architect a resilient OAuth 2.1 PKCE token exchange flow."
    assert req.task_type == EmbeddingTaskType.RETRIEVAL_DOCUMENT

    # Whitespace-only text is rejected
    with pytest.raises(ValueError):
        EmbeddingRequest(text="   \n\t  ")


def test_embedding_batch_request_validation() -> None:
    """EmbeddingBatchRequest validates list of non-empty texts."""
    batch = EmbeddingBatchRequest(
        texts=["First engineering prompt", "Second architecture note"],
        task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
    )
    assert len(batch.texts) == 2

    # Empty batch is rejected
    with pytest.raises(ValidationError):
        EmbeddingBatchRequest(texts=[])

    # Batch containing an empty string is rejected
    with pytest.raises(ValueError):
        EmbeddingBatchRequest(texts=["Valid prompt", "  "])


def test_embedding_result_invariants() -> None:
    """EmbeddingResult verifies vector length matches dimension and values are finite."""
    sample_vec = [0.123] * 768
    res = EmbeddingResult(
        vector=sample_vec,
        dimension=768,
        model_id="text-embedding-004",
        config_version="v1",
        usage=EmbeddingUsageMetadata(prompt_tokens=42, total_tokens=42),
    )
    assert len(res.vector) == 768
    assert res.dimension == 768
    assert res.usage.total_tokens == 42

    # Dimension mismatch is rejected
    with pytest.raises(ValueError):
        EmbeddingResult(
            vector=[0.1] * 512,
            dimension=768,
            model_id="text-embedding-004",
            config_version="v1",
        )

    # Non-finite values (NaN, Inf) are rejected
    with pytest.raises(ValueError):
        EmbeddingResult(
            vector=[float("nan")] * 768,
            dimension=768,
            model_id="text-embedding-004",
            config_version="v1",
        )
