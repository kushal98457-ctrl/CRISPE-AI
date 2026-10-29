"""Unit and integration tests for AI Resource Budgeting, Provider Accounting, and Limits (Step 25).

Covers:
A. Configuration tests (valid ranges, negative/zero rejection, defaults).
B. Request reservation tests (pre-dispatch blocking, attempts, retries, exception release).
C. Metric budget tests (metric separation, judge calls, reflection, retry containment).
D. Time and concurrency tests (deadlines, semaphore limits, exception cleanup, deadlock avoidance).
E. Token and expenditure accounting (accumulation, missing-data handling, cost estimates, ceiling stops).
F. Integration and regression tests (adapter integration, DSPy LM wrapper, production prompt safety).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, List, Optional
import pytest

from app.adapters.gemini_adapter import (
    BaseModelAdapter,
    ModelGenerationRequest,
    ModelGenerationResponse,
)
from app.core.config import Settings
from app.core.exceptions import ProviderAPIError, ProviderRateLimitError
from app.dspy_modules.budget_guard import (
    BudgetExceededError,
    ConcurrencyLimitError,
    ExperimentStopReason,
    OperationType,
    OptimizationTimeoutError,
    ProviderCallRecord,
    ReservationToken,
    ResourceBudgetConfig,
    ResourceBudgetGuard,
    ResourceBudgetManager,
    ResourceUsageSummary,
    estimate_call_cost,
    get_active_budget_manager,
)
from app.dspy_modules.model_client import BudgetGuardedLM
from app.schemas.crispe import ModelUsageMetadata
from app.services.generation_service import (
    CRISPE_SYSTEM_PROMPT,
    CrispeGenerationService,
)


# =============================================================================
# A. Configuration Tests
# =============================================================================

def test_resource_budget_config_valid() -> None:
    """Verify ResourceBudgetConfig initializes with valid explicit bounds."""
    cfg = ResourceBudgetConfig(
        experiment_id="test_exp_01",
        max_metric_calls=15,
        max_provider_requests=30,
        max_elapsed_seconds=120.0,
        max_concurrent_requests=3,
        max_retries_per_request=2,
        max_total_tokens=50000,
        max_budget_usd=3.50,
        pricing_version="gemini-2025-standard-v1",
    )
    assert cfg.experiment_id == "test_exp_01"
    assert cfg.max_metric_calls == 15
    assert cfg.max_provider_requests == 30
    assert cfg.max_elapsed_seconds == 120.0
    assert cfg.max_concurrent_requests == 3
    assert cfg.max_retries_per_request == 2
    assert cfg.max_total_tokens == 50000
    assert cfg.max_budget_usd == 3.50


@pytest.mark.parametrize(
    "kwargs,err_match",
    [
        ({"max_metric_calls": 0}, "max_metric_calls must be positive"),
        ({"max_metric_calls": -5}, "max_metric_calls must be positive"),
        ({"max_provider_requests": 0}, "max_provider_requests must be positive"),
        ({"max_provider_requests": -1}, "max_provider_requests must be positive"),
        ({"max_elapsed_seconds": 0.0}, "max_elapsed_seconds must be positive"),
        ({"max_elapsed_seconds": -10.0}, "max_elapsed_seconds must be positive"),
        ({"max_concurrent_requests": 0}, "max_concurrent_requests must be positive"),
        ({"max_retries_per_request": -1}, "max_retries_per_request cannot be negative"),
        ({"max_total_tokens": 0}, "max_total_tokens must be positive"),
        ({"max_budget_usd": 0.0}, "max_budget_usd must be positive"),
        ({"max_budget_usd": -2.0}, "max_budget_usd must be positive"),
    ],
)
def test_resource_budget_config_rejects_invalid_values(kwargs: Dict[str, Any], err_match: str) -> None:
    """Verify ResourceBudgetConfig rejects nonsensical, zero, or negative configuration limits."""
    with pytest.raises(ValueError, match=err_match):
        ResourceBudgetConfig(**kwargs)


def test_resource_budget_manager_default_from_settings() -> None:
    """Verify ResourceBudgetManager defaults load safely from application Settings."""
    settings = Settings()
    manager = ResourceBudgetManager(settings=settings)
    summary = manager.get_summary()

    assert summary.status == ExperimentStopReason.COMPLETED.value
    assert not summary.is_budget_exceeded
    assert manager.config.max_metric_calls == settings.BUDGET_MAX_METRIC_CALLS_DEFAULT
    assert manager.config.max_provider_requests == settings.BUDGET_MAX_PROVIDER_REQUESTS_DEFAULT
    assert manager.config.pricing_version == settings.BUDGET_PRICING_CONFIG_VERSION


# =============================================================================
# B. Request Reservation Tests
# =============================================================================

def test_request_blocked_before_dispatch_when_budget_exhausted() -> None:
    """Verify pre-dispatch reservation immediately blocks requests once provider ceiling is hit."""
    cfg = ResourceBudgetConfig(max_metric_calls=10, max_provider_requests=2)
    manager = ResourceBudgetManager(config=cfg)

    # First reservation succeeds
    tok1 = manager.reserve_provider_call(OperationType.GENERATION)
    manager.record_call_completion(tok1)

    # Second reservation succeeds
    tok2 = manager.reserve_provider_call(OperationType.GENERATION)
    manager.record_call_completion(tok2)

    # Third reservation must fail before any dispatch
    with pytest.raises(BudgetExceededError, match="Total provider request ceiling reached"):
        manager.reserve_provider_call(OperationType.GENERATION)

    summary = manager.get_summary()
    assert summary.total_provider_calls == 2
    assert summary.is_budget_exceeded
    assert summary.status == ExperimentStopReason.PROVIDER_REQUEST_BUDGET_EXCEEDED.value


def test_retries_consume_additional_provider_capacity() -> None:
    """Verify retries consume provider capacity and are tracked in retry metrics."""
    cfg = ResourceBudgetConfig(max_metric_calls=10, max_provider_requests=3)
    manager = ResourceBudgetManager(config=cfg)

    # Attempt 1 (initial request)
    tok1 = manager.reserve_provider_call(
        OperationType.GENERATION,
        is_retry=False,
        attempt_number=1,
    )
    manager.record_call_failure(tok1, error=RuntimeError("Transient 503"))

    # Attempt 2 (retry 1)
    tok2 = manager.reserve_provider_call(
        OperationType.GENERATION,
        is_retry=True,
        attempt_number=2,
    )
    manager.record_call_failure(tok2, error=RuntimeError("Transient 503"))

    # Attempt 3 (retry 2)
    tok3 = manager.reserve_provider_call(
        OperationType.GENERATION,
        is_retry=True,
        attempt_number=3,
    )
    manager.record_call_completion(tok3)

    summary = manager.get_summary()
    assert summary.total_provider_calls == 3
    assert summary.retry_attempts_count == 2
    assert summary.generation_calls == 3
    assert len(summary.call_records) == 3

    # Attempt 4 should be blocked (capacity exhausted)
    with pytest.raises(BudgetExceededError):
        manager.reserve_provider_call(OperationType.GENERATION, is_retry=True, attempt_number=4)


def test_concurrency_released_on_exceptions() -> None:
    """Verify concurrency semaphore is guaranteed to be released when exceptions occur."""
    cfg = ResourceBudgetConfig(max_metric_calls=10, max_provider_requests=10, max_concurrent_requests=1)
    manager = ResourceBudgetManager(config=cfg)

    with pytest.raises(ValueError):
        with manager.guarded_call(OperationType.GENERATION):
            raise ValueError("Unexpected processing explosion")

    summary = manager.get_summary()
    assert summary.current_in_flight == 0
    assert summary.total_provider_calls == 1

    # Able to acquire slot again without deadlock or ConcurrencyLimitError
    with manager.guarded_call(OperationType.GENERATION):
        pass

    summary2 = manager.get_summary()
    assert summary2.current_in_flight == 0
    assert summary2.total_provider_calls == 2


# =============================================================================
# C. Metric Budget Tests
# =============================================================================

def test_metric_budget_strictly_enforced_and_distinct_from_provider_calls() -> None:
    """Verify GEPA metric invocations are counted separately from provider requests."""
    cfg = ResourceBudgetConfig(max_metric_calls=3, max_provider_requests=20)
    manager = ResourceBudgetManager(config=cfg)

    manager.record_metric_call(success=True)
    manager.record_metric_call(success=True)
    manager.record_metric_call(success=False)

    # 4th metric call triggers budget exceeded
    with pytest.raises(BudgetExceededError, match="Metric call budget exhausted"):
        manager.record_metric_call()

    summary = manager.get_summary()
    assert summary.gepa_metric_calls == 3
    assert summary.metric_calls_failed == 1
    assert summary.total_provider_calls == 0  # Metric calls do not automatically increment provider calls
    assert summary.status == ExperimentStopReason.METRIC_BUDGET_EXCEEDED.value


def test_evaluator_judge_and_reflection_calls_consume_provider_budget() -> None:
    """Verify reflection and judge evaluations consume provider request reservations."""
    cfg = ResourceBudgetConfig(max_metric_calls=10, max_provider_requests=5)
    manager = ResourceBudgetManager(config=cfg)

    # 1. Student generation
    tok1 = manager.reserve_provider_call(OperationType.GENERATION)
    manager.record_call_completion(tok1)

    # 2. Judge evaluation
    tok2 = manager.reserve_provider_call(OperationType.EVALUATION)
    manager.record_call_completion(tok2)

    # 3. Reflection proposer
    tok3 = manager.reserve_provider_call(OperationType.REFLECTION)
    manager.record_call_completion(tok3)

    summary = manager.get_summary()
    assert summary.total_provider_calls == 3
    assert summary.generation_calls == 1
    assert summary.judge_calls == 1
    assert summary.reflection_calls == 1


# =============================================================================
# D. Time and Concurrency Tests
# =============================================================================

def test_deadline_expired_halts_new_provider_requests() -> None:
    """Verify requests cannot start after wall-clock deadline expires."""
    cfg = ResourceBudgetConfig(
        max_metric_calls=10,
        max_provider_requests=10,
        max_elapsed_seconds=0.02,  # 20ms
    )
    manager = ResourceBudgetManager(config=cfg)
    time.sleep(0.04)

    with pytest.raises(OptimizationTimeoutError, match="Execution timeout"):
        manager.reserve_provider_call(OperationType.GENERATION)

    summary = manager.get_summary()
    assert summary.is_budget_exceeded
    assert summary.status == ExperimentStopReason.DEADLINE_EXCEEDED.value


def test_concurrency_limits_enforced_across_threads() -> None:
    """Verify concurrency semaphore restricts simultaneously active requests."""
    cfg = ResourceBudgetConfig(
        max_metric_calls=20,
        max_provider_requests=20,
        max_concurrent_requests=2,
    )
    manager = ResourceBudgetManager(config=cfg)

    active_counts: List[int] = []
    lock = threading.Lock()

    def worker_job():
        with manager.guarded_call(OperationType.GENERATION):
            with lock:
                active_counts.append(manager._in_flight_requests)
            time.sleep(0.05)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_job) for _ in range(5)]
        for f in futures:
            f.result()

    summary = manager.get_summary()
    assert summary.total_provider_calls == 5
    assert summary.current_in_flight == 0
    # At no point should concurrent in-flight requests exceed limit of 2
    assert max(active_counts) <= 2


# =============================================================================
# E. Token and Expenditure Accounting Tests
# =============================================================================

def test_token_accumulation_and_missing_usage_handling() -> None:
    """Verify token accumulation accurately distinguishes reported counts from missing data."""
    cfg = ResourceBudgetConfig(max_metric_calls=10, max_provider_requests=10)
    manager = ResourceBudgetManager(config=cfg)

    # Call 1: Valid reported usage
    tok1 = manager.reserve_provider_call(OperationType.GENERATION)
    usage1 = ModelUsageMetadata(prompt_tokens=100, candidates_tokens=50, total_tokens=150)
    manager.record_call_completion(tok1, usage=usage1)

    # Call 2: Missing usage (None)
    tok2 = manager.reserve_provider_call(OperationType.GENERATION)
    manager.record_call_completion(tok2, usage=None)

    summary = manager.get_summary()
    assert summary.total_prompt_tokens == 100
    assert summary.total_candidates_tokens == 50
    assert summary.total_tokens == 150
    assert summary.has_missing_usage is True  # Flags that partial/missing telemetry occurred


def test_cost_estimation_with_verified_pricing() -> None:
    """Verify cost calculation accurately computes USD expenditure for known models."""
    cost = estimate_call_cost(
        model_id="gemini-2.5-flash",
        prompt_tokens=1_000_000,
        candidates_tokens=1_000_000,
        pricing_version="gemini-2025-standard-v1",
    )
    # 1M input ($0.075) + 1M output ($0.30) = $0.375
    assert cost == 0.375


def test_unknown_model_cost_returns_none() -> None:
    """Verify unverified model pricing returns None without fabricating estimates."""
    cost = estimate_call_cost(
        model_id="completely-unregistered-custom-model",
        prompt_tokens=500,
        candidates_tokens=500,
        pricing_version="gemini-2025-standard-v1",
    )
    assert cost is None


def test_token_budget_ceiling_halts_requests() -> None:
    """Verify reaching aggregate token budget halts future provider requests."""
    cfg = ResourceBudgetConfig(
        max_metric_calls=10,
        max_provider_requests=10,
        max_total_tokens=200,
    )
    manager = ResourceBudgetManager(config=cfg)

    # Call 1 consumes 150 tokens
    tok1 = manager.reserve_provider_call(OperationType.GENERATION)
    manager.record_call_completion(
        tok1,
        usage={"prompt_tokens": 100, "candidates_tokens": 50, "total_tokens": 150},
    )

    # Call 2 attempts reservation with estimated 100 tokens (150 + 100 > 200) -> blocked
    with pytest.raises(BudgetExceededError, match="Token budget ceiling reached"):
        manager.reserve_provider_call(OperationType.GENERATION, estimated_tokens=100)

    summary = manager.get_summary()
    assert summary.status == ExperimentStopReason.TOKEN_BUDGET_EXCEEDED.value


# =============================================================================
# F. Integration and Regression Tests
# =============================================================================

class MockFailingDiagnosticAdapter(BaseModelAdapter):
    """Test double simulating a transient error followed by success."""

    def __init__(self) -> None:
        self.attempts = 0

    async def generate_content(self, request: ModelGenerationRequest) -> ModelGenerationResponse:
        self.attempts += 1
        active_bm = request.budget_manager or get_active_budget_manager()
        is_retry = (self.attempts > 1)

        token = None
        if active_bm:
            token = active_bm.reserve_provider_call(
                operation_type=OperationType.GENERATION,
                model_id="mock-diag",
                is_retry=is_retry,
                attempt_number=self.attempts,
            )

        if self.attempts == 1:
            if active_bm and token:
                active_bm.record_call_failure(token, error=RuntimeError("Rate limit exceeded"))
            raise ProviderRateLimitError("Rate limit exceeded", retry_after=0.01)

        if active_bm and token:
            active_bm.record_call_completion(
                token=token,
                usage={"prompt_tokens": 50, "candidates_tokens": 50, "total_tokens": 100},
                latency_ms=10.0,
                model_id="mock-diag",
            )
        return ModelGenerationResponse(
            content='{"context": "C", "role": "R", "instruction": "I", "specification": "S", "performance": "P", "example": "E"}',
            model_id="mock-diag",
        )

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_adapter_retries_counted_in_budget_manager() -> None:
    """Verify adapter retries are accounted for as distinct requests in ResourceBudgetManager."""
    manager = ResourceBudgetManager(
        config=ResourceBudgetConfig(max_metric_calls=10, max_provider_requests=5)
    )
    adapter = MockFailingDiagnosticAdapter()

    # Attempt 1 fails transiently
    with pytest.raises(ProviderRateLimitError):
        await adapter.generate_content(
            ModelGenerationRequest(prompt="test", budget_manager=manager)
        )

    # Attempt 2 succeeds
    resp = await adapter.generate_content(
        ModelGenerationRequest(prompt="test", budget_manager=manager)
    )
    assert resp.content is not None

    summary = manager.get_summary()
    assert summary.total_provider_calls == 2
    assert summary.retry_attempts_count == 1
    assert summary.total_tokens == 100


def test_budget_guarded_lm_wrapper() -> None:
    """Verify BudgetGuardedLM wraps an arbitrary DSPy LM and tracks execution."""
    class DummyLM:
        def __init__(self):
            self.model = "dummy-model"
            self.history = [{"response": type("Resp", (), {"usage": type("U", (), {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100})})()}]

        def __call__(self, prompt=None, messages=None, **kwargs):
            return ["Sample prediction output"]

    manager = ResourceBudgetManager(
        config=ResourceBudgetConfig(max_metric_calls=10, max_provider_requests=5)
    )
    guarded_lm = BudgetGuardedLM(
        inner_lm=DummyLM(),
        budget_manager=manager,
        operation_type=OperationType.REFLECTION,
    )

    result = guarded_lm(prompt="Propose new instructions")
    assert result == ["Sample prediction output"]

    summary = manager.get_summary()
    assert summary.total_provider_calls == 1
    assert summary.reflection_calls == 1
    assert summary.total_tokens == 100


def test_production_instructions_remain_unmodified_step25() -> None:
    """Verify that CrispeGenerationService production instructions remain strictly unchanged."""
    service = CrispeGenerationService()
    assert service is not None

    assert "You are an elite AI prompt engineer" in CRISPE_SYSTEM_PROMPT
    assert "1. Context:" in CRISPE_SYSTEM_PROMPT
    assert "2. Role:" in CRISPE_SYSTEM_PROMPT
    assert "3. Instruction:" in CRISPE_SYSTEM_PROMPT
    assert "4. Specification:" in CRISPE_SYSTEM_PROMPT
    assert "5. Performance:" in CRISPE_SYSTEM_PROMPT
    assert "6. Example:" in CRISPE_SYSTEM_PROMPT
