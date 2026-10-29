"""Resource budget guards and provider-call accounting for GEPA experiments (Step 24).

Enforces strict, multi-tiered operational bounds:
1. GEPA metric call ceiling (e.g. <= 15-20 metric evaluations).
2. Total provider API call ceiling (generation + judge + reflection + retries).
3. Maximum wall-clock execution timeout.
4. Concurrency limits.
5. Auditable usage telemetry and failure containment.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, Optional

from app.core.logging import get_logger

logger = get_logger("gepa_budget_guard")


class BudgetExceededError(RuntimeError):
    """Raised when an optimization run exceeds a configured resource or API limit."""


class OptimizationTimeoutError(TimeoutError):
    """Raised when an optimization experiment exceeds its maximum wall-clock time."""


@dataclass
class ResourceUsageSummary:
    """Auditable resource consumption metrics for an optimization run."""

    gepa_metric_calls: int = 0
    generation_calls: int = 0
    reflection_calls: int = 0
    judge_calls: int = 0
    total_provider_calls: int = 0
    elapsed_seconds: float = 0.0
    is_budget_exceeded: bool = False
    budget_exceeded_reason: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finish_time: Optional[str] = None


class ResourceBudgetGuard:
    """Thread-safe accounting and limit-enforcement layer for prompt optimization runs."""

    def __init__(
        self,
        max_gepa_metric_calls: int = 20,
        max_total_provider_calls: int = 40,
        max_elapsed_seconds: float = 300.0,
        concurrency_limit: int = 1,
    ) -> None:
        """Initialize resource budget guard.

        Parameters:
            max_gepa_metric_calls: Maximum invocations of the GEPA feedback metric.
            max_total_provider_calls: Hard ceiling on cumulative external model requests.
            max_elapsed_seconds: Maximum permitted experiment runtime in seconds.
            concurrency_limit: Maximum concurrent model requests (default 1 for pilots).
        """
        if max_gepa_metric_calls <= 0:
            raise ValueError("max_gepa_metric_calls must be a positive integer.")
        if max_total_provider_calls <= 0:
            raise ValueError("max_total_provider_calls must be a positive integer.")
        if max_elapsed_seconds <= 0.0:
            raise ValueError("max_elapsed_seconds must be a positive float.")

        self.max_gepa_metric_calls = max_gepa_metric_calls
        self.max_total_provider_calls = max_total_provider_calls
        self.max_elapsed_seconds = max_elapsed_seconds
        self.concurrency_limit = concurrency_limit

        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._gepa_metric_calls = 0
        self._generation_calls = 0
        self._reflection_calls = 0
        self._judge_calls = 0
        self._total_provider_calls = 0
        self._is_exceeded = False
        self._exceeded_reason: Optional[str] = None
        self._semaphore = threading.Semaphore(concurrency_limit)

    def check_time_limit(self) -> None:
        """Verify experiment has not exceeded the wall-clock timeout limit."""
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.max_elapsed_seconds:
            with self._lock:
                self._is_exceeded = True
                self._exceeded_reason = (
                    f"Execution timeout: {elapsed:.1f}s exceeded max {self.max_elapsed_seconds:.1f}s"
                )
            logger.error("Budget guard: %s", self._exceeded_reason)
            raise OptimizationTimeoutError(self._exceeded_reason)

    def record_metric_call(self) -> None:
        """Register a single GEPA metric function invocation."""
        self.check_time_limit()
        with self._lock:
            if self._gepa_metric_calls >= self.max_gepa_metric_calls:
                self._is_exceeded = True
                self._exceeded_reason = (
                    f"Metric call budget exhausted: {self._gepa_metric_calls} >= {self.max_gepa_metric_calls}"
                )
                logger.warning("Budget guard: %s", self._exceeded_reason)
                raise BudgetExceededError(self._exceeded_reason)

            self._gepa_metric_calls += 1
            logger.debug(
                "Metric call recorded (%d/%d)",
                self._gepa_metric_calls,
                self.max_gepa_metric_calls,
            )

    def record_generation_call(self) -> None:
        """Register a generation model API invocation."""
        self._record_provider_call("generation")

    def record_reflection_call(self) -> None:
        """Register a reflection model API invocation."""
        self._record_provider_call("reflection")

    def record_judge_call(self) -> None:
        """Register a model-assisted evaluation judge API invocation."""
        self._record_provider_call("judge")

    def _record_provider_call(self, call_type: str) -> None:
        """Internal helper to count and enforce total provider request limits."""
        self.check_time_limit()
        with self._lock:
            if self._total_provider_calls >= self.max_total_provider_calls:
                self._is_exceeded = True
                self._exceeded_reason = (
                    f"Total provider request ceiling reached: {self._total_provider_calls} >= {self.max_total_provider_calls}"
                )
                logger.warning("Budget guard: %s", self._exceeded_reason)
                raise BudgetExceededError(self._exceeded_reason)

            self._total_provider_calls += 1
            if call_type == "generation":
                self._generation_calls += 1
            elif call_type == "reflection":
                self._reflection_calls += 1
            elif call_type == "judge":
                self._judge_calls += 1

            logger.debug(
                "Provider call recorded: type=%s, total=%d/%d",
                call_type,
                self._total_provider_calls,
                self.max_total_provider_calls,
            )

    def get_summary(self) -> ResourceUsageSummary:
        """Generate a complete telemetry snapshot of resource usage."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time
            return ResourceUsageSummary(
                gepa_metric_calls=self._gepa_metric_calls,
                generation_calls=self._generation_calls,
                reflection_calls=self._reflection_calls,
                judge_calls=self._judge_calls,
                total_provider_calls=self._total_provider_calls,
                elapsed_seconds=round(elapsed, 2),
                is_budget_exceeded=self._is_exceeded,
                budget_exceeded_reason=self._exceeded_reason,
                finish_time=datetime.now(timezone.utc).isoformat(),
            )
