"""Centralized AI Resource Budgeting, Provider-Call Accounting, and Execution Limits (Step 25).

Provides a comprehensive, auditable resource-control layer governing:
1. Metric-call budgets (GEPA feedback evaluations).
2. Provider API request ceilings and attempt-level retry accounting.
3. Wall-clock execution deadlines and timeouts.
4. Process-local request concurrency bounds.
5. Token accumulation and expenditure estimation.
6. Transparent stop reason categorization and immutable audit snapshots.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
import contextvars
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Union
import uuid

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.crispe import ModelUsageMetadata

logger = get_logger("resource_budget")


# =============================================================================
# Domain Exceptions
# =============================================================================

class BudgetExceededError(RuntimeError):
    """Raised when an operation would exceed or has exceeded a configured resource limit."""

    def __init__(self, message: str, stop_reason: str = "BUDGET_EXCEEDED") -> None:
        super().__init__(message)
        self.stop_reason = stop_reason


class OptimizationTimeoutError(TimeoutError):
    """Raised when an optimization run or model request exceeds its wall-clock deadline."""

    def __init__(self, message: str, stop_reason: str = "DEADLINE_EXCEEDED") -> None:
        super().__init__(message)
        self.stop_reason = stop_reason


class ConcurrencyLimitError(RuntimeError):
    """Raised when request concurrency slots are exhausted and cannot be acquired."""

    def __init__(self, message: str, stop_reason: str = "CONCURRENCY_BLOCKED") -> None:
        super().__init__(message)
        self.stop_reason = stop_reason


# =============================================================================
# Enumerations
# =============================================================================

class OperationType(str, Enum):
    """Categorization of logical AI model operations."""

    GENERATION = "generation"
    EVALUATION = "evaluation"
    REFLECTION = "reflection"
    RETRY = "retry"
    GENERAL = "general"


class ExperimentStopReason(str, Enum):
    """Formal termination reasons for an optimization experiment."""

    COMPLETED = "COMPLETED"
    METRIC_BUDGET_EXCEEDED = "METRIC_BUDGET_EXCEEDED"
    PROVIDER_REQUEST_BUDGET_EXCEEDED = "PROVIDER_REQUEST_BUDGET_EXCEEDED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    TOKEN_BUDGET_EXCEEDED = "TOKEN_BUDGET_EXCEEDED"
    EXPENDITURE_LIMIT_EXCEEDED = "EXPENDITURE_LIMIT_EXCEEDED"
    CONCURRENCY_BLOCKED = "CONCURRENCY_BLOCKED"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    EVALUATION_FAILURE = "EVALUATION_FAILURE"
    CANCELLED = "CANCELLED"


class CallStatus(str, Enum):
    """Lifecycle status of an individual provider request attempt."""

    RESERVED = "reserved"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Verified Pricing Configuration
# =============================================================================

@dataclass(frozen=True)
class ModelPricingRate:
    """Pricing rate in USD per 1,000,000 tokens for a specific model family."""

    input_cost_per_million: float
    output_cost_per_million: float
    cached_input_per_million: Optional[float] = None


# Official published standard rates (e.g. Gemini 2.5 Flash at <= 128k context)
PRICING_TABLE: Dict[str, Dict[str, ModelPricingRate]] = {
    "gemini-2025-standard-v1": {
        "gemini-2.5-flash": ModelPricingRate(
            input_cost_per_million=0.075,
            output_cost_per_million=0.30,
            cached_input_per_million=0.01875,
        ),
        "gemini-1.5-flash": ModelPricingRate(
            input_cost_per_million=0.075,
            output_cost_per_million=0.30,
        ),
        "gemma-4-26b-a4b-it": ModelPricingRate(
            input_cost_per_million=0.075,
            output_cost_per_million=0.30,
        ),
        "mock-student-gemini-flash": ModelPricingRate(
            input_cost_per_million=0.075,
            output_cost_per_million=0.30,
        ),
        "mock-reflection-gemini-flash": ModelPricingRate(
            input_cost_per_million=0.075,
            output_cost_per_million=0.30,
        ),
    }
}


def estimate_call_cost(
    model_id: str,
    prompt_tokens: Optional[int],
    candidates_tokens: Optional[int],
    pricing_version: str = "gemini-2025-standard-v1",
) -> Optional[float]:
    """Calculate estimated expenditure in USD based on reported token counts and rate table.

    Returns None if token counts are unavailable or the model pricing is unverified.
    """
    if prompt_tokens is None and candidates_tokens is None:
        return None

    rates_for_version = PRICING_TABLE.get(pricing_version)
    if not rates_for_version:
        return None

    # Strip prefixes like 'gemini/' if present
    normalized_model = model_id.split("/")[-1].lower()
    matched_rate = rates_for_version.get(normalized_model)

    if not matched_rate:
        # Partial match fallback (e.g. 'gemini-2.5-flash' in model string)
        for known_key, rate in rates_for_version.items():
            if known_key in normalized_model:
                matched_rate = rate
                break

    if not matched_rate:
        return None

    inp = prompt_tokens or 0
    out = candidates_tokens or 0

    cost = (inp * matched_rate.input_cost_per_million / 1_000_000.0) + (
        out * matched_rate.output_cost_per_million / 1_000_000.0
    )
    return round(cost, 6)


# =============================================================================
# Budget Configuration & Token Structures
# =============================================================================

@dataclass
class ResourceBudgetConfig:
    """Operational limits and thresholds for an individual experiment or run."""

    experiment_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    max_metric_calls: int = 20
    max_provider_requests: int = 50
    max_elapsed_seconds: float = 300.0
    max_concurrent_requests: int = 2
    max_retries_per_request: int = 3
    max_total_tokens: Optional[int] = None
    max_budget_usd: Optional[float] = None
    pricing_version: str = "gemini-2025-standard-v1"

    def __post_init__(self) -> None:
        """Validate all parameters and reject negative or nonsensical configurations."""
        if self.max_metric_calls <= 0:
            raise ValueError(f"max_metric_calls must be positive, got {self.max_metric_calls}")
        if self.max_provider_requests <= 0:
            raise ValueError(f"max_provider_requests must be positive, got {self.max_provider_requests}")
        if self.max_elapsed_seconds <= 0.0:
            raise ValueError(f"max_elapsed_seconds must be positive, got {self.max_elapsed_seconds}")
        if self.max_concurrent_requests <= 0:
            raise ValueError(f"max_concurrent_requests must be positive, got {self.max_concurrent_requests}")
        if self.max_retries_per_request < 0:
            raise ValueError(f"max_retries_per_request cannot be negative, got {self.max_retries_per_request}")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError(f"max_total_tokens must be positive, got {self.max_total_tokens}")
        if self.max_budget_usd is not None and self.max_budget_usd <= 0.0:
            raise ValueError(f"max_budget_usd must be positive, got {self.max_budget_usd}")


@dataclass
class ReservationToken:
    """Opaque reservation token issued before dispatching a provider call."""

    token_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    experiment_id: str = ""
    operation_type: OperationType = OperationType.GENERAL
    model_id: Optional[str] = None
    attempt_number: int = 1
    is_retry: bool = False
    reserved_monotonic: float = field(default_factory=time.monotonic)
    reserved_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    estimated_tokens: Optional[int] = None
    concurrency_acquired: bool = False
    is_released: bool = False


@dataclass
class ProviderCallRecord:
    """Auditable record of a single completed, failed, or retried provider API attempt."""

    call_id: str
    experiment_id: str
    operation_type: str
    provider_name: str
    model_id: str
    attempt_number: int
    is_retry: bool
    status: str
    start_time: str
    end_time: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    candidates_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ResourceUsageSummary:
    """Comprehensive resource consumption summary for an experiment run.

    Provides backward-compatible field names for Step 24 (`gepa_metric_calls`,
    `generation_calls`, `reflection_calls`, `judge_calls`, `total_provider_calls`,
    `elapsed_seconds`, `is_budget_exceeded`, `budget_exceeded_reason`).
    """

    experiment_id: str = ""
    status: str = ExperimentStopReason.COMPLETED.value
    is_budget_exceeded: bool = False
    budget_exceeded_reason: Optional[str] = None

    # Operation-level counters (Step 24 compat)
    gepa_metric_calls: int = 0
    metric_calls_failed: int = 0
    generation_calls: int = 0
    reflection_calls: int = 0
    judge_calls: int = 0
    total_provider_calls: int = 0
    retry_attempts_count: int = 0
    current_in_flight: int = 0

    # Timing
    elapsed_seconds: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finish_time: Optional[str] = None

    # Token & Cost Telemetry
    total_prompt_tokens: Optional[int] = None
    total_candidates_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    has_missing_usage: bool = False
    estimated_expenditure_usd: Optional[float] = None
    pricing_version: Optional[str] = None

    # Audit records
    call_records_count: int = 0
    call_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry summary to JSON-serializable dictionary."""
        return asdict(self)


# =============================================================================
# Context Variable for Ambient Budget Association
# =============================================================================

_active_budget_manager: contextvars.ContextVar[Optional[ResourceBudgetManager]] = (
    contextvars.ContextVar("active_budget_manager", default=None)
)


def get_active_budget_manager() -> Optional[ResourceBudgetManager]:
    """Retrieve the active ResourceBudgetManager in the current task/thread context."""
    return _active_budget_manager.get()


# =============================================================================
# ResourceBudgetManager Implementation
# =============================================================================

class ResourceBudgetManager:
    """Thread-safe and async-aware resource budget manager and provider-call accounting engine."""

    def __init__(
        self,
        config: Optional[ResourceBudgetConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize resource budget manager with validated configuration.

        If config is omitted, builds default configuration from Settings.
        """
        if config is None:
            cfg = settings or get_settings()
            config = ResourceBudgetConfig(
                max_metric_calls=cfg.BUDGET_MAX_METRIC_CALLS_DEFAULT,
                max_provider_requests=cfg.BUDGET_MAX_PROVIDER_REQUESTS_DEFAULT,
                max_elapsed_seconds=cfg.BUDGET_MAX_ELAPSED_SECONDS_DEFAULT,
                max_concurrent_requests=cfg.BUDGET_MAX_CONCURRENT_REQUESTS_DEFAULT,
                max_retries_per_request=cfg.BUDGET_MAX_RETRIES_DEFAULT,
                max_total_tokens=cfg.BUDGET_MAX_TOTAL_TOKENS_DEFAULT,
                max_budget_usd=cfg.MAX_BUDGET_USD_PER_RUN,
                pricing_version=cfg.BUDGET_PRICING_CONFIG_VERSION,
            )

        self.config = config
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(config.max_concurrent_requests)

        # Monotonic timeline
        self._start_monotonic = time.monotonic()
        self._start_iso = datetime.now(timezone.utc).isoformat()
        self._deadline_monotonic = self._start_monotonic + config.max_elapsed_seconds

        # Counters
        self._metric_calls_count = 0
        self._metric_calls_failed = 0
        self._generation_calls = 0
        self._reflection_calls = 0
        self._judge_calls = 0
        self._total_provider_calls = 0
        self._retry_attempts_count = 0
        self._in_flight_requests = 0

        # Token & cost accumulation
        self._accumulated_prompt_tokens: Optional[int] = None
        self._accumulated_candidates_tokens: Optional[int] = None
        self._accumulated_total_tokens: Optional[int] = None
        self._has_missing_usage = False
        self._accumulated_cost_usd: float = 0.0

        # Status & stop reasons
        self._status = ExperimentStopReason.COMPLETED
        self._is_exceeded = False
        self._exceeded_reason: Optional[str] = None
        self._finish_iso: Optional[str] = None

        # Detailed call records
        self._call_records: List[ProviderCallRecord] = []

    # -------------------------------------------------------------------------
    # Context Management & Scoping
    # -------------------------------------------------------------------------

    def activate(self) -> contextvars.Token:
        """Set this manager as the ambient active budget manager in the current context."""
        return _active_budget_manager.set(self)

    def deactivate(self, token: contextvars.Token) -> None:
        """Reset the ambient active budget manager."""
        _active_budget_manager.reset(token)

    # -------------------------------------------------------------------------
    # Execution Limits & Verification Checks
    # -------------------------------------------------------------------------

    def check_time_limit(self) -> None:
        """Verify experiment execution has not exceeded the wall-clock deadline."""
        now = time.monotonic()
        if now >= self._deadline_monotonic:
            elapsed = now - self._start_monotonic
            with self._lock:
                self._is_exceeded = True
                self._status = ExperimentStopReason.DEADLINE_EXCEEDED
                self._exceeded_reason = (
                    f"Execution timeout: deadline exceeded (elapsed {elapsed:.2f}s >= "
                    f"limit {self.config.max_elapsed_seconds:.2f}s)"
                )
            logger.error("Resource budget manager: %s", self._exceeded_reason)
            raise OptimizationTimeoutError(self._exceeded_reason)

    def check_budget_available(
        self,
        operation_type: OperationType,
        estimated_tokens: Optional[int] = None,
    ) -> None:
        """Pre-dispatch check to verify budget availability before making an external call."""
        self.check_time_limit()

        with self._lock:
            # 1. Hard provider request ceiling
            if self._total_provider_calls >= self.config.max_provider_requests:
                self._is_exceeded = True
                self._status = ExperimentStopReason.PROVIDER_REQUEST_BUDGET_EXCEEDED
                self._exceeded_reason = (
                    f"Total provider request ceiling reached: {self._total_provider_calls} >= "
                    f"{self.config.max_provider_requests}"
                )
                logger.warning("Resource budget manager: %s", self._exceeded_reason)
                raise BudgetExceededError(self._exceeded_reason, stop_reason=self._status.value)

            # 2. Token budget check (if configured and reported)
            if self.config.max_total_tokens is not None:
                current_tokens = self._accumulated_total_tokens or 0
                check_tokens = current_tokens + (estimated_tokens or 0)
                if check_tokens >= self.config.max_total_tokens:
                    self._is_exceeded = True
                    self._status = ExperimentStopReason.TOKEN_BUDGET_EXCEEDED
                    self._exceeded_reason = (
                        f"Token budget ceiling reached: estimated {check_tokens} >= "
                        f"{self.config.max_total_tokens}"
                    )
                    logger.warning("Resource budget manager: %s", self._exceeded_reason)
                    raise BudgetExceededError(self._exceeded_reason, stop_reason=self._status.value)

            # 3. Expenditure limit check (if configured and calculated)
            if self.config.max_budget_usd is not None:
                if self._accumulated_cost_usd >= self.config.max_budget_usd:
                    self._is_exceeded = True
                    self._status = ExperimentStopReason.EXPENDITURE_LIMIT_EXCEEDED
                    self._exceeded_reason = (
                        f"Expenditure ceiling reached: estimated ${self._accumulated_cost_usd:.4f} >= "
                        f"${self.config.max_budget_usd:.2f}"
                    )
                    logger.warning("Resource budget manager: %s", self._exceeded_reason)
                    raise BudgetExceededError(self._exceeded_reason, stop_reason=self._status.value)

    # -------------------------------------------------------------------------
    # Metric-Call Accounting
    # -------------------------------------------------------------------------

    def record_metric_call(self, success: bool = True, error: Optional[Exception] = None) -> None:
        """Register a GEPA metric function invocation, enforcing the metric ceiling."""
        self.check_time_limit()

        with self._lock:
            if self._metric_calls_count >= self.config.max_metric_calls:
                self._is_exceeded = True
                self._status = ExperimentStopReason.METRIC_BUDGET_EXCEEDED
                self._exceeded_reason = (
                    f"Metric call budget exhausted: {self._metric_calls_count} >= "
                    f"{self.config.max_metric_calls}"
                )
                logger.warning("Resource budget manager: %s", self._exceeded_reason)
                raise BudgetExceededError(self._exceeded_reason, stop_reason=self._status.value)

            self._metric_calls_count += 1
            if not success:
                self._metric_calls_failed += 1

            logger.debug(
                "Metric call recorded (%d/%d, success=%s)",
                self._metric_calls_count,
                self.config.max_metric_calls,
                success,
            )

    # -------------------------------------------------------------------------
    # Provider Request Reservation & Accounting
    # -------------------------------------------------------------------------

    def reserve_provider_call(
        self,
        operation_type: Union[OperationType, str] = OperationType.GENERAL,
        model_id: Optional[str] = None,
        is_retry: bool = False,
        attempt_number: int = 1,
        estimated_tokens: Optional[int] = None,
        acquire_concurrency: bool = True,
        concurrency_timeout: Optional[float] = 5.0,
    ) -> ReservationToken:
        """Reserve provider call capacity BEFORE dispatching a network request.

        1. Validates time limit and budget availability.
        2. Acquires process-local concurrency slot.
        3. Atomically increments in-flight and total attempt counters.
        4. Returns an auditable ReservationToken.
        """
        if isinstance(operation_type, str):
            try:
                op_type = OperationType(operation_type.lower())
            except ValueError:
                op_type = OperationType.GENERAL
        else:
            op_type = operation_type

        # Verify capacity before attempting concurrency acquisition
        self.check_budget_available(op_type, estimated_tokens=estimated_tokens)

        concurrency_acquired = False
        if acquire_concurrency:
            acquired = self._semaphore.acquire(timeout=concurrency_timeout)
            if not acquired:
                self._is_exceeded = True
                self._status = ExperimentStopReason.CONCURRENCY_BLOCKED
                msg = f"Failed to acquire concurrency slot within {concurrency_timeout}s"
                logger.error("Resource budget manager: %s", msg)
                raise ConcurrencyLimitError(msg)
            concurrency_acquired = True

        with self._lock:
            self._in_flight_requests += 1
            self._total_provider_calls += 1

            if is_retry:
                self._retry_attempts_count += 1

            if op_type == OperationType.GENERATION:
                self._generation_calls += 1
            elif op_type == OperationType.REFLECTION:
                self._reflection_calls += 1
            elif op_type == OperationType.EVALUATION:
                self._judge_calls += 1

        token = ReservationToken(
            experiment_id=self.config.experiment_id,
            operation_type=op_type,
            model_id=model_id,
            attempt_number=attempt_number,
            is_retry=is_retry,
            estimated_tokens=estimated_tokens,
            concurrency_acquired=concurrency_acquired,
        )

        logger.debug(
            "Reserved provider call: op=%s, attempt=%d, retry=%s, total=%d/%d, in_flight=%d",
            op_type.value,
            attempt_number,
            is_retry,
            self._total_provider_calls,
            self.config.max_provider_requests,
            self._in_flight_requests,
        )
        return token

    def record_call_completion(
        self,
        token: ReservationToken,
        usage: Optional[Union[ModelUsageMetadata, Dict[str, Any]]] = None,
        latency_ms: Optional[float] = None,
        provider_name: str = "google-genai",
        model_id: Optional[str] = None,
    ) -> ProviderCallRecord:
        """Register the successful completion of a reserved provider request."""
        self._release_concurrency(token)

        # Extract tokens
        p_tok, c_tok, t_tok = self._extract_tokens(usage)

        now_iso = datetime.now(timezone.utc).isoformat()
        resolved_model = model_id or token.model_id or "unknown"
        estimated_cost = estimate_call_cost(
            model_id=resolved_model,
            prompt_tokens=p_tok,
            candidates_tokens=c_tok,
            pricing_version=self.config.pricing_version,
        )

        with self._lock:
            # Accumulate token metrics (handling missing values cleanly)
            if p_tok is not None:
                self._accumulated_prompt_tokens = (self._accumulated_prompt_tokens or 0) + p_tok
            if c_tok is not None:
                self._accumulated_candidates_tokens = (self._accumulated_candidates_tokens or 0) + c_tok
            if t_tok is not None:
                self._accumulated_total_tokens = (self._accumulated_total_tokens or 0) + t_tok
            elif p_tok is not None and c_tok is not None:
                self._accumulated_total_tokens = (self._accumulated_total_tokens or 0) + (p_tok + c_tok)
            else:
                self._has_missing_usage = True

            if estimated_cost is not None:
                self._accumulated_cost_usd += estimated_cost

            record = ProviderCallRecord(
                call_id=token.token_id,
                experiment_id=token.experiment_id,
                operation_type=token.operation_type.value,
                provider_name=provider_name,
                model_id=resolved_model,
                attempt_number=token.attempt_number,
                is_retry=token.is_retry,
                status=CallStatus.COMPLETED.value,
                start_time=token.reserved_iso,
                end_time=now_iso,
                latency_ms=latency_ms,
                prompt_tokens=p_tok,
                candidates_tokens=c_tok,
                total_tokens=t_tok,
                estimated_cost_usd=estimated_cost,
            )
            self._call_records.append(record)

        logger.debug(
            "Completed provider call '%s' (tokens: %s, cost: %s)",
            token.token_id,
            t_tok if t_tok is not None else "N/A",
            f"${estimated_cost:.6f}" if estimated_cost is not None else "N/A",
        )
        return record

    def record_call_failure(
        self,
        token: ReservationToken,
        error: Exception,
        latency_ms: Optional[float] = None,
        provider_name: str = "google-genai",
        model_id: Optional[str] = None,
    ) -> ProviderCallRecord:
        """Register a failed provider request attempt, ensuring concurrency is freed."""
        self._release_concurrency(token)

        now_iso = datetime.now(timezone.utc).isoformat()
        resolved_model = model_id or token.model_id or "unknown"
        error_type = type(error).__name__
        error_msg = str(error)

        with self._lock:
            self._has_missing_usage = True
            record = ProviderCallRecord(
                call_id=token.token_id,
                experiment_id=token.experiment_id,
                operation_type=token.operation_type.value,
                provider_name=provider_name,
                model_id=resolved_model,
                attempt_number=token.attempt_number,
                is_retry=token.is_retry,
                status=CallStatus.FAILED.value,
                start_time=token.reserved_iso,
                end_time=now_iso,
                latency_ms=latency_ms,
                error_type=error_type,
                error_message=error_msg,
            )
            self._call_records.append(record)

        logger.warning(
            "Failed provider call attempt '%s' (type: %s, error: %s)",
            token.token_id,
            token.operation_type.value,
            error_msg,
        )
        return record

    def _release_concurrency(self, token: ReservationToken) -> None:
        """Safely release held concurrency slot and decrement in-flight counter."""
        if not token.is_released:
            token.is_released = True
            with self._lock:
                if self._in_flight_requests > 0:
                    self._in_flight_requests -= 1
            if token.concurrency_acquired:
                self._semaphore.release()

    def _extract_tokens(
        self, usage: Optional[Union[ModelUsageMetadata, Dict[str, Any]]]
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract prompt, candidates, and total tokens without coercing None to 0."""
        if usage is None:
            return None, None, None

        if isinstance(usage, ModelUsageMetadata):
            return usage.prompt_tokens, usage.candidates_tokens, usage.total_tokens

        if isinstance(usage, dict):
            p = usage.get("prompt_tokens") or usage.get("prompt_token_count")
            c = usage.get("candidates_tokens") or usage.get("completion_tokens") or usage.get("candidates_token_count")
            t = usage.get("total_tokens") or usage.get("total_token_count")
            return p, c, t

        return None, None, None

    # -------------------------------------------------------------------------
    # Context Managers
    # -------------------------------------------------------------------------

    @contextmanager
    def guarded_call(
        self,
        operation_type: Union[OperationType, str] = OperationType.GENERAL,
        model_id: Optional[str] = None,
        is_retry: bool = False,
        attempt_number: int = 1,
        estimated_tokens: Optional[int] = None,
    ) -> Generator[ReservationToken, None, None]:
        """Synchronous context manager enforcing pre-dispatch reservation and cleanup."""
        token = self.reserve_provider_call(
            operation_type=operation_type,
            model_id=model_id,
            is_retry=is_retry,
            attempt_number=attempt_number,
            estimated_tokens=estimated_tokens,
        )
        start = time.monotonic()
        try:
            yield token
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            self.record_call_failure(token, error=exc, latency_ms=latency_ms, model_id=model_id)
            raise
        else:
            latency_ms = (time.monotonic() - start) * 1000.0
            self.record_call_completion(token, latency_ms=latency_ms, model_id=model_id)

    @asynccontextmanager
    async def async_guarded_call(
        self,
        operation_type: Union[OperationType, str] = OperationType.GENERAL,
        model_id: Optional[str] = None,
        is_retry: bool = False,
        attempt_number: int = 1,
        estimated_tokens: Optional[int] = None,
    ):
        """Asynchronous context manager enforcing pre-dispatch reservation and cleanup."""
        token = self.reserve_provider_call(
            operation_type=operation_type,
            model_id=model_id,
            is_retry=is_retry,
            attempt_number=attempt_number,
            estimated_tokens=estimated_tokens,
        )
        start = time.monotonic()
        try:
            yield token
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            self.record_call_failure(token, error=exc, latency_ms=latency_ms, model_id=model_id)
            raise
        else:
            latency_ms = (time.monotonic() - start) * 1000.0
            self.record_call_completion(token, latency_ms=latency_ms, model_id=model_id)

    # -------------------------------------------------------------------------
    # Backward Compatibility Aliases for Step 24 (ResourceBudgetGuard)
    # -------------------------------------------------------------------------

    def record_generation_call(self) -> None:
        """Register generation call (Step 24 compat)."""
        tok = self.reserve_provider_call(OperationType.GENERATION, acquire_concurrency=False)
        self.record_call_completion(tok)

    def record_reflection_call(self) -> None:
        """Register reflection call (Step 24 compat)."""
        tok = self.reserve_provider_call(OperationType.REFLECTION, acquire_concurrency=False)
        self.record_call_completion(tok)

    def record_judge_call(self) -> None:
        """Register judge evaluation call (Step 24 compat)."""
        tok = self.reserve_provider_call(OperationType.EVALUATION, acquire_concurrency=False)
        self.record_call_completion(tok)

    # -------------------------------------------------------------------------
    # Telemetry Snapshot & Reporting
    # -------------------------------------------------------------------------

    def get_summary(self) -> ResourceUsageSummary:
        """Produce a complete, immutable telemetry snapshot of resource consumption."""
        with self._lock:
            elapsed = time.monotonic() - self._start_monotonic
            records_copy = [asdict(r) for r in self._call_records]

            return ResourceUsageSummary(
                experiment_id=self.config.experiment_id,
                status=self._status.value,
                is_budget_exceeded=self._is_exceeded,
                budget_exceeded_reason=self._exceeded_reason,
                gepa_metric_calls=self._metric_calls_count,
                metric_calls_failed=self._metric_calls_failed,
                generation_calls=self._generation_calls,
                reflection_calls=self._reflection_calls,
                judge_calls=self._judge_calls,
                total_provider_calls=self._total_provider_calls,
                retry_attempts_count=self._retry_attempts_count,
                current_in_flight=self._in_flight_requests,
                elapsed_seconds=round(elapsed, 2),
                start_time=self._start_iso,
                finish_time=self._finish_iso or datetime.now(timezone.utc).isoformat(),
                total_prompt_tokens=self._accumulated_prompt_tokens,
                total_candidates_tokens=self._accumulated_candidates_tokens,
                total_tokens=self._accumulated_total_tokens,
                has_missing_usage=self._has_missing_usage,
                estimated_expenditure_usd=round(self._accumulated_cost_usd, 6) if self._accumulated_cost_usd > 0 else None,
                pricing_version=self.config.pricing_version if self._accumulated_cost_usd > 0 else None,
                call_records_count=len(records_copy),
                call_records=records_copy,
            )
