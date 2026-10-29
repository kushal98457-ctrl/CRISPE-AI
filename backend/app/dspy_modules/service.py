"""DSPy generation service adapter offering functional parity with CrispeGenerationService (Step 22).

Implements the shared generation contract (CrispeGenerationRequest -> CrispeGenerationResponse)
using DSPy under isolated model contexts, without replacing the existing production service.
"""

from datetime import datetime, timezone
import time
from typing import Optional
import dspy

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    EmptyPromptError,
    PromptValidationError,
    ProviderAPIError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.core.logging import get_logger
from app.dspy_modules.model_client import create_dspy_lm
from app.dspy_modules.program import CrispeGenerationProgram, create_crispe_program
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeGenerationResponse,
    CrispeSections,
    ModelUsageMetadata,
)

logger = get_logger("dspy_generation_service")


class DspyCrispeGenerationService:
    """Generation service utilizing the DSPy CrispeGenerationProgram under request-scoped LM contexts."""

    def __init__(
        self,
        program: Optional[CrispeGenerationProgram] = None,
        lm: Optional[dspy.LM] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.program = program or create_crispe_program()
        self._lm = lm

    def _get_lm_for_request(self, request_model_id: Optional[str] = None) -> dspy.LM:
        """Resolve or instantiate the appropriate LM for this generation call."""
        if self._lm is not None and request_model_id is None:
            return self._lm

        target_model = request_model_id or self.settings.PRIMARY_MODEL
        return create_dspy_lm(
            model_id=target_model,
            settings=self.settings,
        )

    def _map_dspy_error(self, exc: Exception) -> ProviderError:
        """Translate DSPy and LiteLLM runtime errors into domain exceptions."""
        err_str = str(exc)
        lower_err = err_str.lower()

        if "auth" in lower_err or "unauthorized" in lower_err or "api_key" in lower_err:
            return ProviderAuthenticationError(f"DSPy provider authentication failed: {err_str}", original_error=exc)
        if "rate limit" in lower_err or "429" in lower_err or "quota" in lower_err:
            return ProviderRateLimitError(f"DSPy provider rate limit exceeded: {err_str}", original_error=exc)
        if "timeout" in lower_err or "timed out" in lower_err or "deadline" in lower_err:
            return ProviderTimeoutError(f"DSPy provider request timed out: {err_str}", original_error=exc)
        if isinstance(exc, (EmptyPromptError, PromptValidationError)):
            return exc

        return ProviderAPIError(f"DSPy model execution failed: {err_str}", original_error=exc)

    async def generate_crispe_prompt(
        self, request: CrispeGenerationRequest
    ) -> CrispeGenerationResponse:
        """Transform a user idea into a validated CRISPE prompt using DSPy.

        Parameters:
            request: CrispeGenerationRequest containing idea and optional parameters.

        Returns:
            CrispeGenerationResponse conforming to the canonical API schema.
        """
        lm = self._get_lm_for_request(request.model_id)
        effective_model_id = getattr(lm, "model", request.model_id or self.settings.PRIMARY_MODEL)

        start_time = time.perf_counter()
        logger.info(
            "Executing DSPy CRISPE generation (model=%s, idea_len=%d)",
            effective_model_id,
            len(request.idea),
        )

        try:
            # Execute inside isolated context to prevent concurrent setting mutation
            with dspy.context(lm=lm):
                sections, prediction = await self.program.execute_async(
                    idea=request.idea,
                )

            # Extract token usage metadata if exposed by LM history
            usage: Optional[ModelUsageMetadata] = None
            history = getattr(lm, "history", None)
            if history and len(history) > 0:
                last_call = history[-1]
                call_usage = last_call.get("usage") or last_call.get("response", {}).get("usage")
                if isinstance(call_usage, dict):
                    usage = ModelUsageMetadata(
                        prompt_tokens=call_usage.get("prompt_tokens", 0),
                        candidates_tokens=call_usage.get("completion_tokens", 0),
                        total_tokens=call_usage.get("total_tokens", 0),
                    )

            formatted_prompt = sections.to_markdown()

            return CrispeGenerationResponse(
                idea=request.idea,
                sections=sections,
                formatted_prompt=formatted_prompt,
                model_id=effective_model_id,
                usage=usage,
                finish_reason="stop",
            )

        except (EmptyPromptError, PromptValidationError):
            raise
        except Exception as exc:
            mapped_err = self._map_dspy_error(exc)
            logger.error("DSPy generation encountered error: %s", mapped_err)
            raise mapped_err from exc
