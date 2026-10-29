"""DSPy Language Model client integration and concurrency-safe configuration (Step 22).

Encapsulates DSPy LM initialization, provider model identifier normalization,
credential resolution, and request-scoped context isolation.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
import dspy

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    MissingProviderCredentialError,
    ProviderConfigError,
)
from app.core.logging import get_logger

logger = get_logger("dspy_model_client")


def normalize_dspy_model_name(model_id: str) -> str:
    """Normalize application model identifiers to LiteLLM / DSPy provider syntax.

    For example, 'gemini-2.5-flash' -> 'gemini/gemini-2.5-flash'.
    If a provider prefix is already present (e.g. 'gemini/', 'openai/'), preserves it.
    """
    cleaned = model_id.strip()
    if "/" in cleaned:
        return cleaned

    # Default to Google Gemini provider prefix for DSPy LiteLLM router
    return f"gemini/{cleaned}"


def create_dspy_lm(
    model_id: Optional[str] = None,
    settings: Optional[Settings] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    cache: bool = False,
    **kwargs: Any,
) -> dspy.LM:
    """Instantiate a configured DSPy LM client for Google Gemini models.

    Parameters:
        model_id: Model checkpoint override (defaults to settings.PRIMARY_MODEL).
        settings: Application Settings instance.
        api_key: Optional explicit API key override.
        temperature: Optional sampling temperature override.
        max_tokens: Upper bound on output tokens.
        timeout: Request timeout in seconds.
        cache: Whether to enable DSPy disk caching (default False for evaluation reproducibility).
        kwargs: Additional arguments forwarded to dspy.LM.

    Returns:
        dspy.LM instance.

    Raises:
        ProviderConfigError: If credentials cannot be resolved or LM initialization fails.
    """
    cfg = settings or get_settings()
    target_model = model_id or cfg.PRIMARY_MODEL
    normalized_name = normalize_dspy_model_name(target_model)

    resolved_api_key = api_key
    if not resolved_api_key:
        try:
            resolved_api_key = cfg.get_gemini_api_key()
        except MissingProviderCredentialError as exc:
            logger.error("DSPy cannot initialize LM: missing Gemini API key.")
            raise ProviderConfigError(
                "Gemini API key is required to initialize DSPy LM client.",
                original_error=exc,
            ) from exc

    resolved_temp = temperature if temperature is not None else cfg.GEMINI_DEFAULT_TEMPERATURE
    resolved_max_tokens = max_tokens or cfg.GEMINI_MAX_OUTPUT_TOKENS
    resolved_timeout = timeout or cfg.GEMINI_TIMEOUT_SECONDS

    try:
        logger.info(
            "Initializing DSPy LM client for model '%s' (timeout=%.1fs, temp=%.2f)",
            normalized_name,
            resolved_timeout,
            resolved_temp,
        )
        lm = dspy.LM(
            model=normalized_name,
            api_key=resolved_api_key,
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
            timeout=resolved_timeout,
            cache=cache,
            **kwargs,
        )
        return lm
    except Exception as exc:
        logger.error("Failed to construct DSPy LM client for '%s': %s", normalized_name, exc)
        raise ProviderConfigError(
            f"Failed to instantiate DSPy LM client: {str(exc)}",
            original_error=exc,
        ) from exc


def create_dspy_reflection_lm(
    model_id: Optional[str] = None,
    settings: Optional[Settings] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    cache: bool = False,
    **kwargs: Any,
) -> dspy.LM:
    """Instantiate a configured DSPy LM client specifically for GEPA reflection (Step 23).

    Reflection reasoning analyzes evaluation feedback and proposes candidate prompt mutations.
    Defaults to settings.REFLECTION_MODEL ('gemini-2.5-flash') at low temperature (0.0).

    Parameters:
        model_id: Optional reflection model override (defaults to settings.REFLECTION_MODEL).
        settings: Application Settings instance.
        api_key: Optional explicit API key override.
        temperature: Sampling temperature (defaults to 0.0 for structured reflection).
        max_tokens: Upper bound on output tokens.
        timeout: Request timeout in seconds.
        cache: Whether to enable DSPy disk caching.
        kwargs: Additional arguments forwarded to dspy.LM.

    Returns:
        dspy.LM instance configured for reflection operations.
    """
    cfg = settings or get_settings()
    target_reflection_model = model_id or cfg.REFLECTION_MODEL
    resolved_temperature = temperature if temperature is not None else 0.0

    logger.info(
        "Configuring DSPy reflection LM for model '%s' (temp=%.2f)",
        target_reflection_model,
        resolved_temperature,
    )
    return create_dspy_lm(
        model_id=target_reflection_model,
        settings=cfg,
        api_key=api_key,
        temperature=resolved_temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        cache=cache,
        **kwargs,
    )


@contextmanager
def dspy_isolated_context(lm: dspy.LM, **override_kwargs: Any) -> Generator[None, None, None]:
    """Execute code within an isolated request-scoped DSPy context.

    Uses dspy.context(...) to ensure that LM configuration is scoped exclusively
    to the active async task / thread, preventing concurrent cross-request pollution.
    """
    with dspy.context(lm=lm, **override_kwargs):
        yield
