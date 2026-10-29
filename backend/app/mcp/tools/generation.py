"""CRISPE prompt generation tool for the CRISPE-AI MCP Server (Step 30).

Exposes prompt synthesis conforming to the CRISPE framework (Context, Role, Instruction,
Specification, Performance, Example) via the established CrispeGenerationService.
"""

from typing import Optional, TYPE_CHECKING

from app.core.exceptions import (
    EmptyPromptError,
    PromptValidationError,
    ProviderError,
)
from app.mcp.errors import (
    MCPError,
    MCPInternalError,
    MCPInvalidInputError,
    sanitize_error_message,
)
from app.mcp.schemas import (
    GenerateCrispePromptInput,
    GenerateCrispePromptOutput,
)
from app.schemas.crispe import CrispeGenerationRequest

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer
    from app.mcp.config import MCPServerConfig
    from app.mcp.services.context import MCPAppContext

ALLOWED_MODELS = frozenset({
    "gemma-4-26b-a4b-it",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "mock-student-gemini-flash",
})


def register_generation_tools(
    server: "MCPServer",
    config: "MCPServerConfig",
    context: "MCPAppContext",
) -> None:
    """Register the generate_crispe_prompt capability on the MCP server."""

    @server.tool(
        name="generate_crispe_prompt",
        description=(
            "Transform a prompt idea or requirements into a production-grade, structured prompt "
            "conforming strictly to the CRISPE framework (Context, Role, Instruction, Specification, "
            "Performance, Example). Connects directly to the CRISPE-AI generation engine."
        ),
    )
    async def generate_crispe_prompt(
        idea: Optional[str] = None,
        context_section: Optional[str] = None,
        role_section: Optional[str] = None,
        instruction_section: Optional[str] = None,
        specification_section: Optional[str] = None,
        performance_section: Optional[str] = None,
        example_section: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_budget: Optional[int] = None,
    ) -> GenerateCrispePromptOutput:
        """Synthesize a complete CRISPE framework prompt from user requirements."""
        # 1. Validate inputs
        try:
            inp = GenerateCrispePromptInput(
                idea=idea,
                context=context_section,
                role=role_section,
                instruction=instruction_section,
                specification=specification_section,
                performance=performance_section,
                example=example_section,
                model_id=model_id,
                temperature=temperature,
                thinking_budget=thinking_budget,
            )
        except Exception as exc:
            raise MCPInvalidInputError(
                f"Invalid prompt generation parameters: {sanitize_error_message(str(exc))}"
            ) from exc

        # 2. Validate model identifier against security whitelist
        if inp.model_id and inp.model_id not in ALLOWED_MODELS:
            raise MCPInvalidInputError(
                f"Model '{inp.model_id}' is not in approved model whitelist: {sorted(ALLOWED_MODELS)}"
            )

        # 3. Synthesize composite idea preserving user-provided section hints
        effective_idea = (inp.idea or "").strip()
        additional_hints = []
        if inp.context:
            additional_hints.append(f"Context: {inp.context.strip()}")
        if inp.role:
            additional_hints.append(f"Role: {inp.role.strip()}")
        if inp.instruction:
            additional_hints.append(f"Instruction: {inp.instruction.strip()}")
        if inp.specification:
            additional_hints.append(f"Specification: {inp.specification.strip()}")
        if inp.performance:
            additional_hints.append(f"Performance: {inp.performance.strip()}")
        if inp.example:
            additional_hints.append(f"Example: {inp.example.strip()}")

        if additional_hints:
            hints_block = "\n".join(additional_hints)
            effective_idea = f"{effective_idea}\n\nAdditional Requirements:\n{hints_block}" if effective_idea else hints_block

        # 4. Invoke the established application CrispeGenerationService
        try:
            gen_service = context.get_generation_service()
            req = CrispeGenerationRequest(
                idea=effective_idea,
                model_id=inp.model_id,
                temperature=inp.temperature,
                thinking_budget=inp.thinking_budget,
            )
            response = await gen_service.generate_crispe_prompt(req)

            return GenerateCrispePromptOutput(
                success=True,
                idea=response.idea,
                sections=response.sections,
                formatted_prompt=response.formatted_prompt,
                model_id=response.model_id,
                usage=response.usage,
                finish_reason=response.finish_reason,
                created_at=response.created_at.isoformat(),
            )
        except (EmptyPromptError, PromptValidationError) as exc:
            raise MCPInvalidInputError(str(exc)) from exc
        except ProviderError as exc:
            raise MCPError(
                message=sanitize_error_message(exc.message),
                code=exc.error_code or "PROVIDER_ERROR",
            ) from exc
        except Exception as exc:
            raise MCPInternalError(original_error=exc) from exc
