"""Google AI Studio / Gemini provider adapter for CRISPE-AI.

Provides an asynchronous, resilient, and typed interface to the Google GenAI SDK.
Encapsulates client initialization, authentication, structured output schema compilation,
timeout management, exponential backoff retries, and domain error translation.
"""

import asyncio
from abc import ABC, abstractmethod
import random
import time
from typing import Any, Optional, Type
import httpx
from pydantic import BaseModel, ConfigDict, Field

import google.genai as genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    InvalidProviderResponseError,
    MissingProviderCredentialError,
    ModelNotFoundError,
    ProviderAPIError,
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderError,
    ProviderPermissionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.core.logging import get_logger
from app.schemas.crispe import ModelUsageMetadata
from app.services.resource_budget import (
    BudgetExceededError,
    OptimizationTimeoutError,
    OperationType,
    get_active_budget_manager,
)

logger = get_logger("gemini_adapter")


class ModelGenerationRequest(BaseModel):
    """Normalized request specification for model generation."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    prompt: str = Field(..., description="User prompt or instructions to submit.")
    system_instruction: Optional[str] = Field(
        default=None,
        description="High-priority system instructions directing the model persona.",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Specific model identifier override. Defaults to configured GEMINI_MODEL_ID.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature controlling randomness.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=65536,
        description="Upper token limit on generated response.",
    )
    response_schema: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic schema class enforcing structured JSON output.",
    )
    response_mime_type: Optional[str] = Field(
        default=None,
        description="Expected MIME type, e.g. application/json or text/plain.",
    )
    thinking_budget: Optional[int] = Field(
        default=None,
        ge=0,
        le=32768,
        description="Reasoning token budget for supported models.",
    )
    timeout: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=600.0,
        description="Request timeout in seconds. Defaults to configured GEMINI_TIMEOUT_SECONDS.",
    )
    budget_manager: Optional[Any] = Field(
        default=None,
        description="Optional ResourceBudgetManager to enforce pre-dispatch capacity limits.",
    )
    operation_type: Optional[OperationType] = Field(
        default=None,
        description="Logical operation category (e.g. generation, evaluation, reflection).",
    )


class ModelGenerationResponse(BaseModel):
    """Normalized response from model provider."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    content: str = Field(..., description="Raw or formatted text returned by the model.")
    parsed: Optional[Any] = Field(
        default=None,
        description="Parsed Pydantic instance when response_schema was supplied.",
    )
    model_id: str = Field(..., description="Exact model identifier that produced the response.")
    response_id: Optional[str] = Field(default=None, description="Provider correlation/response ID.")
    finish_reason: Optional[str] = Field(default=None, description="Completion finish reason.")
    usage: Optional[ModelUsageMetadata] = Field(default=None, description="Normalized token consumption.")


class BaseModelAdapter(ABC):
    """Abstract interface for AI model generation adapters."""

    @abstractmethod
    async def generate_content(
        self, request: ModelGenerationRequest
    ) -> ModelGenerationResponse:
        """Execute an asynchronous generation request against the model provider."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release underlying client resources and network connections."""
        raise NotImplementedError


class GeminiModelAdapter(BaseModelAdapter):
    """Production provider adapter for Google AI Studio / Gemini API."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[genai.Client] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    def _get_or_create_client(self) -> genai.Client:
        """Lazily initialize the official Google GenAI Client."""
        if self._client is not None:
            return self._client

        try:
            raw_key = self.settings.get_gemini_api_key()
        except MissingProviderCredentialError as exc:
            logger.error("Gemini adapter cannot initialize client: missing credential.")
            raise ProviderConfigError(
                "Gemini API credential is not configured. Please set GEMINI_API_KEY in the environment.",
                original_error=exc,
            ) from exc

        try:
            self._client = genai.Client(api_key=raw_key)
            return self._client
        except Exception as exc:
            logger.error("Failed to construct google-genai Client: %s", str(exc))
            raise ProviderConfigError(
                f"Failed to instantiate Google GenAI client: {str(exc)}",
                original_error=exc,
            ) from exc

    def _map_provider_error(self, exc: Exception) -> ProviderError:
        """Translate upstream Google GenAI and HTTP exceptions into domain ProviderErrors."""
        # 1. Google GenAI APIError and derived subclasses
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            message = getattr(exc, "message", str(exc))

            if code == 401 or "API_KEY_INVALID" in message or "UNAUTHENTICATED" in message:
                return ProviderAuthenticationError(
                    f"Gemini authentication failed (HTTP 401): {message}",
                    original_error=exc,
                )
            if code == 403 or "PERMISSION_DENIED" in message:
                return ProviderPermissionError(
                    f"Gemini permission denied (HTTP 403): {message}",
                    original_error=exc,
                )
            if code == 404 or "NOT_FOUND" in message:
                return ModelNotFoundError(
                    f"Specified model was not found (HTTP 404): {message}",
                    original_error=exc,
                )
            if code == 429 or "RESOURCE_EXHAUSTED" in message or "RATE_LIMIT_EXCEEDED" in message:
                return ProviderRateLimitError(
                    f"Gemini rate limit or quota exceeded (HTTP 429): {message}",
                    original_error=exc,
                )
            if code in (408, 504) or "DEADLINE_EXCEEDED" in message:
                return ProviderTimeoutError(
                    f"Gemini request timed out (HTTP {code}): {message}",
                    original_error=exc,
                )
            if code in (500, 502, 503) or isinstance(exc, genai_errors.ServerError):
                return ProviderUnavailableError(
                    f"Gemini service temporarily unavailable (HTTP {code}): {message}",
                    original_error=exc,
                )
            return ProviderAPIError(
                f"Gemini API error (HTTP {code}): {message}",
                status_code=code or 502,
                original_error=exc,
            )

        # 2. Async HTTP & Network Client Exceptions
        if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
            return ProviderTimeoutError(
                f"Gemini request timed out: {str(exc)}",
                original_error=exc,
            )
        if isinstance(exc, httpx.ConnectError):
            return ProviderUnavailableError(
                f"Failed to connect to Gemini API endpoint: {str(exc)}",
                original_error=exc,
            )

        # 3. Already mapped internal ProviderErrors
        if isinstance(exc, ProviderError):
            return exc

        # 4. Fallback generic provider failure
        return ProviderAPIError(
            f"Unexpected provider error: {str(exc)}",
            original_error=exc,
        )

    def _extract_usage(self, response: Any) -> Optional[ModelUsageMetadata]:
        """Safely extract token usage metadata if present in response."""
        usage_meta = getattr(response, "usage_metadata", None)
        if not usage_meta:
            return None

        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_meta, "total_token_count", 0) or 0
        thoughts_tokens = getattr(usage_meta, "thoughts_token_count", None)

        return ModelUsageMetadata(
            prompt_tokens=prompt_tokens,
            candidates_tokens=candidates_tokens,
            total_tokens=total_tokens,
            thoughts_tokens=thoughts_tokens,
        )

    async def generate_content(
        self, request: ModelGenerationRequest
    ) -> ModelGenerationResponse:
        """Submit an asynchronous generation request with retries and timeout enforcement."""
        client = self._get_or_create_client()

        model_id = request.model_id or self.settings.GEMINI_MODEL_ID
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.settings.GEMINI_DEFAULT_TEMPERATURE
        )
        max_tokens = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else self.settings.GEMINI_MAX_OUTPUT_TOKENS
        )
        timeout_seconds = (
            request.timeout
            if request.timeout is not None
            else self.settings.GEMINI_TIMEOUT_SECONDS
        )

        # Build GenerateContentConfig
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "http_options": genai_types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        }

        if request.system_instruction:
            config_kwargs["system_instruction"] = request.system_instruction

        if request.response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = request.response_schema
        elif request.response_mime_type:
            config_kwargs["response_mime_type"] = request.response_mime_type

        # Optional reasoning / thinking budget for models that support it
        thinking_budget = (
            request.thinking_budget
            if request.thinking_budget is not None
            else (self.settings.GEMINI_THINKING_BUDGET if self.settings.GEMINI_ENABLE_THINKING else None)
        )
        if thinking_budget is not None and "gemini-2.5" in model_id:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        gen_config = genai_types.GenerateContentConfig(**config_kwargs)

        # Resolve budget manager from request or ambient context
        active_bm = request.budget_manager or get_active_budget_manager()
        op_type = request.operation_type or OperationType.GENERATION

        max_retries = self.settings.GEMINI_MAX_RETRIES
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            is_retry = (attempt > 0)
            reservation_token = None
            if active_bm:
                # Pre-dispatch check and capacity reservation
                reservation_token = active_bm.reserve_provider_call(
                    operation_type=op_type,
                    model_id=model_id,
                    is_retry=is_retry,
                    attempt_number=attempt + 1,
                    acquire_concurrency=True,
                )

            call_start = time.monotonic()
            try:
                # Wrap in asyncio.wait_for as a robust client-side deadline guarantee
                raw_response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_id,
                        contents=request.prompt,
                        config=gen_config,
                    ),
                    timeout=timeout_seconds,
                )

                # Validate candidates exist
                candidates = getattr(raw_response, "candidates", None)
                if not candidates or len(candidates) == 0:
                    raise InvalidProviderResponseError(
                        "Provider returned response with no candidate completions."
                    )

                primary_candidate = candidates[0]
                finish_reason = getattr(primary_candidate, "finish_reason", None)
                if finish_reason is not None:
                    finish_reason = str(finish_reason)

                # Extract parsed schema or raw text content
                parsed_obj = getattr(raw_response, "parsed", None)
                content_text = ""
                if primary_candidate.content and primary_candidate.content.parts:
                    content_text = "".join(
                        getattr(p, "text", "") for p in primary_candidate.content.parts if getattr(p, "text", None)
                    )
                elif hasattr(raw_response, "text") and raw_response.text:
                    content_text = raw_response.text

                if not content_text and parsed_obj is None:
                    raise InvalidProviderResponseError(
                        "Provider returned empty candidate text and no parsed structure."
                    )

                usage = self._extract_usage(raw_response)
                response_id = getattr(raw_response, "response_id", None)

                if active_bm and reservation_token:
                    call_latency = (time.monotonic() - call_start) * 1000.0
                    active_bm.record_call_completion(
                        token=reservation_token,
                        usage=usage,
                        latency_ms=call_latency,
                        provider_name=self.settings.PROVIDER_NAME,
                        model_id=model_id,
                    )

                return ModelGenerationResponse(
                    content=content_text,
                    parsed=parsed_obj,
                    model_id=model_id,
                    response_id=response_id,
                    finish_reason=finish_reason,
                    usage=usage,
                )

            except Exception as exc:
                call_latency = (time.monotonic() - call_start) * 1000.0
                if active_bm and reservation_token:
                    active_bm.record_call_failure(
                        token=reservation_token,
                        error=exc,
                        latency_ms=call_latency,
                        provider_name=self.settings.PROVIDER_NAME,
                        model_id=model_id,
                    )

                if isinstance(exc, (BudgetExceededError, OptimizationTimeoutError)):
                    # Do not retry if budget limit or deadline is exhausted
                    raise exc

                domain_error = self._map_provider_error(exc)

                # If non-retryable or retries exhausted, raise immediately
                if not domain_error.is_retryable or attempt >= max_retries:
                    logger.error(
                        "Gemini generation call failed (attempt %d/%d, retryable=%s): %s",
                        attempt + 1,
                        max_retries + 1,
                        domain_error.is_retryable,
                        domain_error.message,
                    )
                    raise domain_error from exc

                # Exponential backoff with jitter for transient failures
                jitter = random.uniform(0.1, 0.5)
                delay = (base_delay * (2 ** attempt)) + jitter
                if isinstance(domain_error, ProviderRateLimitError) and domain_error.retry_after:
                    delay = max(delay, domain_error.retry_after)

                logger.warning(
                    "Transient provider error (%s). Retrying in %.2fs (attempt %d/%d)...",
                    domain_error.error_code,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)

        raise ProviderAPIError("Generation retry loop terminated unexpectedly.")

    async def close(self) -> None:
        """Close asynchronous client resources."""
        if self._client is not None and hasattr(self._client, "aio"):
            try:
                if hasattr(self._client.aio, "aclose"):
                    await self._client.aio.aclose()
                logger.info("Closed Gemini client connection pool.")
            except Exception as exc:
                logger.warning("Error closing Gemini client connection: %s", str(exc))
            finally:
                self._client = None


# Module-level cached instance
_adapter_instance: Optional[BaseModelAdapter] = None


def get_model_adapter(
    settings: Optional[Settings] = None,
    client: Optional[genai.Client] = None,
) -> BaseModelAdapter:
    """Retrieve or create the shared model adapter instance."""
    global _adapter_instance
    if _adapter_instance is None or client is not None or settings is not None:
        adapter = GeminiModelAdapter(settings=settings, client=client)
        if client is None and settings is None:
            _adapter_instance = adapter
        return adapter
    return _adapter_instance


def set_model_adapter(adapter: Optional[BaseModelAdapter]) -> None:
    """Explicitly override or reset the model adapter (useful in testing suites)."""
    global _adapter_instance
    _adapter_instance = adapter
