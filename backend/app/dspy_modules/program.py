"""DSPy CRISPE prompt generation module and schema validation (Step 22).

Implements the programmable DSPy module wrapping CrispeGenerationSignature,
handling input validation, prediction execution, and canonical CrispeSections mapping.
"""

import asyncio
from typing import Any, Dict, Optional, Tuple, Type
import dspy
from pydantic import ValidationError

from app.core.exceptions import (
    EmptyPromptError,
    PromptValidationError,
)
from app.core.logging import get_logger
from app.dspy_modules.signature import CrispeGenerationSignature
from app.schemas.crispe import CrispeSections

logger = get_logger("dspy_crispe_program")

CANONICAL_SECTIONS = ("context", "role", "instruction", "specification", "performance", "example")


class CrispeGenerationProgram(dspy.Module):
    """Programmable DSPy Module transforming user ideas into structured CRISPE prompts."""

    def __init__(
        self,
        signature: Optional[Type[dspy.Signature]] = None,
        predictor: Optional[dspy.Module] = None,
    ) -> None:
        super().__init__()
        self.signature = signature or CrispeGenerationSignature
        self.predictor = predictor or dspy.Predict(self.signature)

    def forward(
        self,
        idea: str,
        task_context: str = "",
        constraints: str = "",
        output_format_preference: str = "",
        **kwargs: Any,
    ) -> dspy.Prediction:
        """Execute the DSPy prediction step for CRISPE prompt generation."""
        cleaned_idea = (idea or "").strip()
        if not cleaned_idea:
            raise EmptyPromptError("Prompt idea cannot be empty or whitespace only.")

        prediction = self.predictor(
            idea=cleaned_idea,
            task_context=(task_context or "").strip(),
            constraints=(constraints or "").strip(),
            output_format_preference=(output_format_preference or "").strip(),
            **kwargs,
        )
        return prediction

    def validate_and_extract_sections(self, prediction: dspy.Prediction) -> CrispeSections:
        """Validate and convert the raw DSPy prediction into canonical CrispeSections.

        Parameters:
            prediction: Raw output from forward pass.

        Returns:
            Validated CrispeSections model instance.

        Raises:
            PromptValidationError: If required sections are missing, non-string, or empty.
        """
        issues = []
        raw_values: Dict[str, str] = {}

        for sec in CANONICAL_SECTIONS:
            val = getattr(prediction, sec, None)
            if val is None:
                issues.append(f"Missing required CRISPE section: '{sec}'.")
                continue
            if not isinstance(val, str):
                issues.append(f"Section '{sec}' must be a string, got {type(val).__name__}.")
                continue
            cleaned = val.strip()
            if not cleaned:
                issues.append(f"Section '{sec}' is empty or whitespace only.")
                continue
            raw_values[sec] = cleaned

        if issues:
            error_summary = "; ".join(issues)
            logger.error("DSPy prediction failed canonical CRISPE validation: %s", error_summary)
            raise PromptValidationError(
                f"DSPy generated invalid CRISPE output: {error_summary}"
            )

        try:
            return CrispeSections(
                context=raw_values["context"],
                role=raw_values["role"],
                instruction=raw_values["instruction"],
                specification=raw_values["specification"],
                performance=raw_values["performance"],
                example=raw_values["example"],
            )
        except ValidationError as exc:
            logger.error("Pydantic validation error for CrispeSections: %s", exc)
            raise PromptValidationError(
                f"Failed to instantiate canonical CrispeSections: {str(exc)}"
            ) from exc

    def execute(
        self,
        idea: str,
        task_context: str = "",
        constraints: str = "",
        output_format_preference: str = "",
        **kwargs: Any,
    ) -> Tuple[CrispeSections, dspy.Prediction]:
        """Synchronously execute the program and return validated CrispeSections."""
        prediction = self.forward(
            idea=idea,
            task_context=task_context,
            constraints=constraints,
            output_format_preference=output_format_preference,
            **kwargs,
        )
        sections = self.validate_and_extract_sections(prediction)
        return sections, prediction

    async def execute_async(
        self,
        idea: str,
        task_context: str = "",
        constraints: str = "",
        output_format_preference: str = "",
        **kwargs: Any,
    ) -> Tuple[CrispeSections, dspy.Prediction]:
        """Asynchronously execute the program in a worker thread to keep the event loop non-blocking."""
        return await asyncio.to_thread(
            self.execute,
            idea=idea,
            task_context=task_context,
            constraints=constraints,
            output_format_preference=output_format_preference,
            **kwargs,
        )


def create_crispe_program(
    signature: Optional[Type[dspy.Signature]] = None,
    predictor: Optional[dspy.Module] = None,
) -> CrispeGenerationProgram:
    """Factory creating an isolated CrispeGenerationProgram instance."""
    return CrispeGenerationProgram(signature=signature, predictor=predictor)
