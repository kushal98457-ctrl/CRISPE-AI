"""Resource budget guards and provider-call accounting for GEPA experiments (Steps 24 & 25).

Maintains complete backward compatibility with Step 24's ResourceBudgetGuard while
delegating all enforcement, token accounting, and concurrency controls to the
centralized ResourceBudgetManager (Step 25).
"""

from typing import Any, Dict, List, Optional
import uuid

from app.services.resource_budget import (
    BudgetExceededError,
    ConcurrencyLimitError,
    ExperimentStopReason,
    OperationType,
    OptimizationTimeoutError,
    ProviderCallRecord,
    ReservationToken,
    ResourceBudgetConfig,
    ResourceBudgetManager,
    ResourceUsageSummary,
    estimate_call_cost,
    get_active_budget_manager,
)

__all__ = [
    "BudgetExceededError",
    "OptimizationTimeoutError",
    "ConcurrencyLimitError",
    "ExperimentStopReason",
    "OperationType",
    "ProviderCallRecord",
    "ReservationToken",
    "ResourceBudgetConfig",
    "ResourceBudgetManager",
    "ResourceBudgetGuard",
    "ResourceUsageSummary",
    "estimate_call_cost",
    "get_active_budget_manager",
]


class ResourceBudgetGuard(ResourceBudgetManager):
    """Backward-compatible budget guard wrapper around ResourceBudgetManager (Step 24 & 25)."""

    def __init__(
        self,
        max_gepa_metric_calls: int = 20,
        max_total_provider_calls: int = 40,
        max_elapsed_seconds: float = 300.0,
        concurrency_limit: int = 1,
        experiment_id: Optional[str] = None,
        max_retries_per_request: int = 3,
        max_total_tokens: Optional[int] = None,
        max_budget_usd: Optional[float] = None,
        pricing_version: str = "gemini-2025-standard-v1",
    ) -> None:
        config = ResourceBudgetConfig(
            experiment_id=experiment_id or f"guard_{uuid.uuid4().hex[:8]}",
            max_metric_calls=max_gepa_metric_calls,
            max_provider_requests=max_total_provider_calls,
            max_elapsed_seconds=max_elapsed_seconds,
            max_concurrent_requests=concurrency_limit,
            max_retries_per_request=max_retries_per_request,
            max_total_tokens=max_total_tokens,
            max_budget_usd=max_budget_usd,
            pricing_version=pricing_version,
        )
        super().__init__(config=config)

    @property
    def max_gepa_metric_calls(self) -> int:
        return self.config.max_metric_calls

    @property
    def max_total_provider_calls(self) -> int:
        return self.config.max_provider_requests

    @property
    def max_elapsed_seconds(self) -> float:
        return self.config.max_elapsed_seconds

    @property
    def concurrency_limit(self) -> int:
        return self.config.max_concurrent_requests

    @property
    def _gepa_metric_calls(self) -> int:
        return self._metric_calls_count

    @property
    def _total_provider_calls_count(self) -> int:
        return self._total_provider_calls
