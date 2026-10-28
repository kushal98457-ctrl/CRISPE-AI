"""CRISPE prompt generation service.

Coordinates the transformation of raw user ideas into rigorous, structured CRISPE prompts
by orchestrating model adapters, structured schema enforcement, and Markdown formatting.
"""

import json
import re
from typing import Optional
from pydantic import ValidationError

from app.adapters.gemini_adapter import (
    BaseModelAdapter,
    ModelGenerationRequest,
    get_model_adapter,
)
from app.core.exceptions import (
    EmptyPromptError,
    PromptValidationError,
)
from app.core.logging import get_logger
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeGenerationResponse,
    CrispeSections,
)

logger = get_logger("generation_service")

CRISPE_SYSTEM_PROMPT = """You are an elite AI prompt engineer and systems architect specializing in the CRISPE prompt framework.
Your task is to transform any user idea or instruction into a production-grade, highly structured CRISPE prompt with the following six sections:

1. Context: The operational environment, domain background, problem scope, business constraints, and relevant context.
2. Role: The precise persona, expert credentials, professional perspective, and tone of voice the AI must adopt.
3. Instruction: The explicit, unambiguous commands, step-by-step methodologies, and actions the AI must execute.
4. Specification: Structural formatting rules, schema requirements, length parameters, and explicit negative constraints (what NOT to do).
5. Performance: Clear evaluation benchmarks, quality standards, success metrics, and criteria for an exemplary response.
6. Example: High-quality, concrete demonstration or illustrative input/output pairs demonstrating the expected style and structure.

Return a JSON object conforming strictly to the requested schema. Ensure all six fields are rich, actionable, and comprehensive."""

CRISPE_USER_PROMPT_TEMPLATE = """Transform the following raw idea into a complete, high-quality CRISPE framework prompt:

--- RAW USER IDEA ---
{idea}
---------------------

Ensure all 6 CRISPE sections (context, role, instruction, specification, performance, example) are fully elaborated and deeply relevant to this idea."""


class CrispeGenerationService:
    """Service orchestrating prompt transformation into the CRISPE framework."""

    def __init__(self, adapter: Optional[BaseModelAdapter] = None) -> None:
        self.adapter = adapter or get_model_adapter()

    def _extract_and_validate_sections(
        self, parsed_obj: Optional[object], raw_content: str
    ) -> CrispeSections:
        """Validate structured output from the provider, applying JSON cleanup fallbacks if needed."""
        # 1. Direct parsed Pydantic object
        if isinstance(parsed_obj, CrispeSections):
            return parsed_obj

        # 2. Parsed dictionary
        if isinstance(parsed_obj, dict):
            try:
                return CrispeSections.model_validate(parsed_obj)
            except ValidationError as exc:
                raise PromptValidationError(
                    f"Model returned invalid CRISPE structure: {str(exc)}"
                ) from exc

        # 3. Raw JSON text fallback parsing (e.g. if wrapped in markdown code blocks)
        cleaned_content = raw_content.strip()
        if cleaned_content.startswith("```"):
            # Strip markdown json block fences
            cleaned_content = re.sub(r"^```(?:json)?\s*", "", cleaned_content, flags=re.MULTILINE)
            cleaned_content = re.sub(r"\s*```$", "", cleaned_content, flags=re.MULTILINE).strip()

        try:
            data = json.loads(cleaned_content)
            if not isinstance(data, dict):
                raise PromptValidationError("Model output was valid JSON but not a JSON object/dictionary.")
            return CrispeSections.model_validate(data)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode model JSON output: %s", cleaned_content[:200])
            raise PromptValidationError(
                f"Model response was not valid JSON: {str(exc)}"
            ) from exc
        except ValidationError as exc:
            logger.error("Validation error for parsed CRISPE sections: %s", str(exc))
            raise PromptValidationError(
                f"Parsed JSON does not satisfy CRISPE schema: {str(exc)}"
            ) from exc

    async def generate_crispe_prompt(
        self, request: CrispeGenerationRequest
    ) -> CrispeGenerationResponse:
        """Transform a raw prompt idea into a validated CRISPE prompt response."""
        idea = request.idea.strip()
        if not idea:
            raise EmptyPromptError("Cannot generate CRISPE prompt from empty or whitespace text.")

        prompt_text = CRISPE_USER_PROMPT_TEMPLATE.format(idea=idea)

        model_request = ModelGenerationRequest(
            prompt=prompt_text,
            system_instruction=CRISPE_SYSTEM_PROMPT,
            model_id=request.model_id,
            temperature=request.temperature,
            response_schema=CrispeSections,
            response_mime_type="application/json",
            thinking_budget=request.thinking_budget,
        )

        logger.info(
            "Submitting CRISPE prompt generation request for idea (length: %d chars)",
            len(idea),
        )

        model_response = await self.adapter.generate_content(model_request)

        # Validate sections
        sections = self._extract_and_validate_sections(
            parsed_obj=model_response.parsed,
            raw_content=model_response.content,
        )

        formatted_markdown = sections.to_markdown()

        logger.info(
            "Successfully generated CRISPE prompt via model %s (total tokens: %s)",
            model_response.model_id,
            model_response.usage.total_tokens if model_response.usage else "N/A",
        )

        return CrispeGenerationResponse(
            idea=idea,
            sections=sections,
            formatted_prompt=formatted_markdown,
            model_id=model_response.model_id,
            usage=model_response.usage,
            finish_reason=model_response.finish_reason,
        )


def get_generation_service(
    adapter: Optional[BaseModelAdapter] = None,
) -> CrispeGenerationService:
    """Factory creating an instance of CrispeGenerationService with injected adapter."""
    return CrispeGenerationService(adapter=adapter)
