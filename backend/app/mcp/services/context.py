"""Dependency injection container and service bridge for CRISPE-AI MCP Server (Step 29, 30).

Enforces lazy resolution of backend application services so that importing or initializing
the MCP server does not trigger database connections, network calls, or expensive setup.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from app.core.config import Settings
    from app.core.security import AuthorizedPrincipal
    from app.services.generation_service import CrispeGenerationService
    from app.services.optimization_job import OptimizationJobService
    from app.services.prompt_release import PromptReleaseService
    from app.services.semantic_search import SemanticSearchService


class MCPAppContext:
    """Dependency container bridging MCP tools to underlying application domain services.

    Allows tests and custom transports to supply mock or alternative service instances
    without modifying tool definitions or importing heavy backend resources prematurely.
    """

    def __init__(
        self,
        settings: Optional["Settings"] = None,
        trusted_principal: Optional["AuthorizedPrincipal"] = None,
        session_factory: Optional[Any] = None,
        generation_service_factory: Optional[Callable[[], "CrispeGenerationService"]] = None,
        semantic_search_service_factory: Optional[Callable[[], "SemanticSearchService"]] = None,
        optimization_job_service_factory: Optional[Callable[[], "OptimizationJobService"]] = None,
        prompt_release_service_factory: Optional[Callable[[], "PromptReleaseService"]] = None,
    ) -> None:
        self._settings = settings
        self._trusted_principal = trusted_principal
        self._session_factory = session_factory
        self._generation_service_factory = generation_service_factory
        self._semantic_search_service_factory = semantic_search_service_factory
        self._optimization_job_service_factory = optimization_job_service_factory
        self._prompt_release_service_factory = prompt_release_service_factory

        # Cached singleton instances once resolved lazily
        self._generation_service: Optional["CrispeGenerationService"] = None
        self._semantic_search_service: Optional["SemanticSearchService"] = None
        self._optimization_job_service: Optional["OptimizationJobService"] = None
        self._prompt_release_service: Optional["PromptReleaseService"] = None

    def get_settings(self) -> "Settings":
        """Retrieve application settings lazily."""
        if self._settings is None:
            from app.core.config import get_settings
            self._settings = get_settings()
        return self._settings

    def get_trusted_principal(self) -> Optional["AuthorizedPrincipal"]:
        """Retrieve the trusted execution principal governing user-scoped operations."""
        return self._trusted_principal

    def set_trusted_principal(self, principal: Optional["AuthorizedPrincipal"]) -> None:
        """Update or inject the active security principal context."""
        self._trusted_principal = principal

    def get_session_factory(self) -> Any:
        """Resolve database session factory lazily."""
        if self._session_factory is None:
            from app.db.session import get_session_factory
            self._session_factory = get_session_factory()
        return self._session_factory

    def get_generation_service(self) -> "CrispeGenerationService":
        """Resolve CRISPE prompt generation service lazily."""
        if self._generation_service is None:
            if self._generation_service_factory:
                self._generation_service = self._generation_service_factory()
            else:
                from app.services.generation_service import get_generation_service
                self._generation_service = get_generation_service()
        return self._generation_service

    def get_semantic_search_service(self) -> "SemanticSearchService":
        """Resolve semantic prompt memory search service lazily."""
        if self._semantic_search_service is None:
            if self._semantic_search_service_factory:
                self._semantic_search_service = self._semantic_search_service_factory()
            else:
                from app.services.semantic_search import get_semantic_search_service
                self._semantic_search_service = get_semantic_search_service()
        return self._semantic_search_service

    def get_optimization_job_service(self) -> "OptimizationJobService":
        """Resolve optimization job management service lazily."""
        if self._optimization_job_service is None:
            if self._optimization_job_service_factory:
                self._optimization_job_service = self._optimization_job_service_factory()
            else:
                from app.services.optimization_job import get_optimization_job_service
                self._optimization_job_service = get_optimization_job_service()
        return self._optimization_job_service

    def get_prompt_release_service(self) -> "PromptReleaseService":
        """Resolve prompt release management service lazily."""
        if self._prompt_release_service is None:
            if self._prompt_release_service_factory:
                self._prompt_release_service = self._prompt_release_service_factory()
            else:
                from app.services.prompt_release import get_prompt_release_service
                self._prompt_release_service = get_prompt_release_service()
        return self._prompt_release_service
