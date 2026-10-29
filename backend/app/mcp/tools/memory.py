"""Semantic prompt memory tools for the CRISPE-AI MCP Server (Step 30).

Exposes semantic prompt retrieval (vector similarity search with multi-signal ranking)
and durable prompt indexing (vector embedding generation and storage) scoped securely
to the authenticated user principal.
"""

from typing import Optional, TYPE_CHECKING

from app.core.exceptions import (
    EmptyPromptError,
    IncompatibleEmbeddingError,
    PromptValidationError,
    ProviderError,
    SemanticSearchError,
)
from app.mcp.errors import (
    MCPError,
    MCPInternalError,
    MCPInvalidInputError,
    MCPUnauthorizedError,
    sanitize_error_message,
)
from app.mcp.schemas import (
    PromptMemoryMatchItem,
    SavePromptMemoryInput,
    SavePromptMemoryOutput,
    SearchPromptMemoryInput,
    SearchPromptMemoryOutput,
)
from app.schemas.crispe import CrispeSections
from app.schemas.semantic_search import SemanticSearchQuery

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


def register_memory_tools(
    server: "MCPServer",
    config: "MCPServerConfig",
    context: "MCPAppContext",
) -> None:
    """Register semantic prompt memory capabilities on the MCP server."""
    app_context = context

    @server.tool(
        name="search_prompt_memory",
        description=(
            "Search the user's persistent CRISPE prompt memory using semantic similarity. "
            "Retrieves the closest previously synthesized or saved prompts based on high-dimensional "
            "vector embeddings and multi-signal ranking. Requires an authorized principal context."
        ),
    )
    async def search_prompt_memory(
        query: str,
        top_k: int = 5,
        task_type: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> SearchPromptMemoryOutput:
        """Retrieve semantically relevant saved prompts matching a natural language query."""
        # 1. Enforce principal authorization check
        principal = app_context.get_trusted_principal()
        if principal is None:
            raise MCPUnauthorizedError(
                "Authentication or active principal context is required to access prompt memory."
            )
        try:
            principal.ensure_active()
        except Exception as exc:
            raise MCPUnauthorizedError(f"Principal is inactive or suspended: {exc}") from exc

        # 2. Validate input parameters
        try:
            inp = SearchPromptMemoryInput(
                query=query,
                top_k=top_k,
                task_type=task_type,
                min_similarity=min_similarity,
            )
        except Exception as exc:
            raise MCPInvalidInputError(
                f"Invalid memory search parameters: {sanitize_error_message(str(exc))}"
            ) from exc

        # 3. Execute semantic search within active principal boundary
        try:
            session_factory = app_context.get_session_factory()
            search_service = app_context.get_semantic_search_service()

            async with session_factory() as session:
                query_obj = SemanticSearchQuery(
                    query=inp.query,
                    limit=inp.top_k,
                    task_type=inp.task_type,
                    min_similarity=inp.min_similarity,
                )
                search_response = await search_service.search(
                    session=session,
                    principal=principal,
                    query=query_obj,
                    enable_ranking=True,
                )

                results = [
                    PromptMemoryMatchItem(
                        prompt_id=str(item.prompt_id),
                        original_idea=item.original_idea,
                        formatted_prompt=item.formatted_prompt,
                        sections=item.sections,
                        similarity_score=float(item.similarity),
                        score_metric=search_response.search_metric,
                        task_type=item.task_type,
                        model_id=item.model_id,
                        created_at=item.created_at.isoformat(),
                    )
                    for item in search_response.items
                ]

                return SearchPromptMemoryOutput(
                    success=True,
                    query=search_response.query,
                    total_matches=len(results),
                    results=results,
                    owner_id=str(principal.id),
                )
        except (EmptyPromptError, PromptValidationError) as exc:
            raise MCPInvalidInputError(str(exc)) from exc
        except IncompatibleEmbeddingError as exc:
            raise MCPError(
                message=sanitize_error_message(str(exc)),
                code="INCOMPATIBLE_EMBEDDING",
            ) from exc
        except SemanticSearchError as exc:
            raise MCPError(
                message=sanitize_error_message(str(exc)),
                code="SEARCH_ERROR",
            ) from exc
        except MCPError:
            raise
        except Exception as exc:
            raise MCPInternalError(original_error=exc) from exc

    @server.tool(
        name="save_prompt_memory",
        description=(
            "Explicitly persist a high-quality CRISPE prompt to persistent semantic memory. "
            "Stores the prompt record and automatically generates and indexes its vector "
            "embedding for future semantic search. Requires an authorized principal context."
        ),
    )
    async def save_prompt_memory(
        idea: str,
        context: str,
        role: str,
        instruction: str,
        specification: str,
        performance: str,
        example: str,
        task_type: Optional[str] = None,
        model_id: Optional[str] = "gemma-4-26b-a4b-it",
    ) -> SavePromptMemoryOutput:
        """Durable indexing and embedding of a complete CRISPE prompt in persistent memory."""
        # 1. Enforce principal authorization check
        principal = app_context.get_trusted_principal()
        if principal is None:
            raise MCPUnauthorizedError(
                "Authentication or active principal context is required to save prompt to memory."
            )
        try:
            principal.ensure_active()
        except Exception as exc:
            raise MCPUnauthorizedError(f"Principal is inactive or suspended: {exc}") from exc

        # 2. Validate input parameters
        try:
            inp = SavePromptMemoryInput(
                idea=idea,
                context=context,
                role=role,
                instruction=instruction,
                specification=specification,
                performance=performance,
                example=example,
                task_type=task_type,
                model_id=model_id,
            )
        except Exception as exc:
            raise MCPInvalidInputError(
                f"Invalid prompt memory save parameters: {sanitize_error_message(str(exc))}"
            ) from exc

        # 3. Validate model identifier against security whitelist
        if inp.model_id and inp.model_id not in ALLOWED_MODELS:
            raise MCPInvalidInputError(
                f"Model '{inp.model_id}' is not in approved model whitelist: {sorted(ALLOWED_MODELS)}"
            )

        # 4. Construct validated CRISPE sections
        try:
            sections = CrispeSections(
                context=inp.context,
                role=inp.role,
                instruction=inp.instruction,
                specification=inp.specification,
                performance=inp.performance,
                example=inp.example,
            )
        except Exception as exc:
            raise MCPInvalidInputError(
                f"Invalid CRISPE section specification: {sanitize_error_message(str(exc))}"
            ) from exc

        # 5. Execute storage and embedding generation
        try:
            session_factory = app_context.get_session_factory()
            search_service = app_context.get_semantic_search_service()

            async with session_factory() as session:
                record, created = await search_service.save_prompt(
                    session=session,
                    principal=principal,
                    idea=inp.idea,
                    sections=sections,
                    task_type=inp.task_type,
                    model_id=inp.model_id,
                    auto_commit=True,
                )

                return SavePromptMemoryOutput(
                    success=True,
                    prompt_id=str(record.id),
                    status="created" if created else "already_exists",
                    owner_id=str(principal.id),
                    task_type=record.task_type,
                    created_at=record.created_at.isoformat(),
                )
        except (EmptyPromptError, PromptValidationError) as exc:
            raise MCPInvalidInputError(str(exc)) from exc
        except ProviderError as exc:
            raise MCPError(
                message=sanitize_error_message(exc.message),
                code=exc.error_code or "PROVIDER_ERROR",
            ) from exc
        except MCPError:
            raise
        except Exception as exc:
            raise MCPInternalError(original_error=exc) from exc
