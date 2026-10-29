"""Pydantic schemas and state-machine contracts for durable optimization jobs (Step 26).

Enforces:
1. Formal job lifecycle states and valid transitions.
2. Server-side validation of dataset versions, partition boundaries, models, and budgets.
3. Clean separation of request parameters from internal worker execution state.
4. Safe, non-leaking client responses with progress and audit telemetry.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.exceptions import InvalidJobStateTransitionError


class JobStatus(str, Enum):
    """Authoritative lifecycle states for durable optimization jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERRUPTED = "interrupted"


class JobStateMachine:
    """Validates permitted lifecycle transitions for durable optimization jobs."""

    TERMINAL_STATES = frozenset({
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
        JobStatus.BUDGET_EXHAUSTED,
    })

    PERMITTED_TRANSITIONS: Dict[JobStatus, frozenset[JobStatus]] = {
        JobStatus.QUEUED: frozenset({
            JobStatus.RUNNING,
            JobStatus.CANCEL_REQUESTED,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }),
        JobStatus.RUNNING: frozenset({
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCEL_REQUESTED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
            JobStatus.BUDGET_EXHAUSTED,
            JobStatus.RETRY_WAIT,
            JobStatus.INTERRUPTED,
        }),
        JobStatus.CANCEL_REQUESTED: frozenset({
            JobStatus.CANCELLED,
            JobStatus.FAILED,
            JobStatus.TIMED_OUT,
            JobStatus.BUDGET_EXHAUSTED,
        }),
        JobStatus.RETRY_WAIT: frozenset({
            JobStatus.RUNNING,
            JobStatus.CANCEL_REQUESTED,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }),
        JobStatus.INTERRUPTED: frozenset({
            JobStatus.RUNNING,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }),
        # Terminal states have empty target sets (immutable)
        JobStatus.SUCCEEDED: frozenset(),
        JobStatus.FAILED: frozenset(),
        JobStatus.CANCELLED: frozenset(),
        JobStatus.TIMED_OUT: frozenset(),
        JobStatus.BUDGET_EXHAUSTED: frozenset(),
    }

    @classmethod
    def is_terminal(cls, status: JobStatus) -> bool:
        """Return True if the job status represents a final, unmodifiable state."""
        return status in cls.TERMINAL_STATES

    @classmethod
    def validate_transition(cls, current: JobStatus, target: JobStatus) -> None:
        """Verify that transitioning from current to target status is strictly allowed.

        Raises InvalidJobStateTransitionError if the transition is illegal.
        """
        if current == target:
            return

        allowed = cls.PERMITTED_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise InvalidJobStateTransitionError(
                current_status=current.value,
                requested_status=target.value,
            )


# =============================================================================
# Request Schemas
# =============================================================================

class CreateOptimizationJobRequest(BaseModel):
    """Validated caller-provided specification for queueing an optimization job."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str = Field(
        default="v1",
        description="Dataset version identifier for evaluation subset selection.",
    )
    train_size: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of training examples selected from optimization partition.",
    )
    val_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of validation examples for candidate evaluation.",
    )
    generation_model_id: str = Field(
        default="gemma-4-26b-a4b-it",
        description="Primary CRISPE prompt generation model checkpoint.",
    )
    reflection_model_id: str = Field(
        default="gemini-2.5-flash",
        description="Reflection model for GEPA instruction mutation proposer.",
    )
    max_metric_calls: int = Field(
        default=15,
        ge=2,
        le=50,
        description="Strict ceiling on GEPA metric function invocations.",
    )
    max_total_provider_calls: int = Field(
        default=35,
        ge=2,
        le=100,
        description="Hard ceiling on cumulative external provider API requests.",
    )
    max_elapsed_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=1800.0,
        description="Maximum wall-clock execution time in seconds.",
    )
    concurrency_limit: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Maximum concurrent model requests during optimization.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility in optimizer candidate selection.",
    )
    enable_model_judge: bool = Field(
        default=False,
        description="Whether to execute LLM judge evaluation during metric scoring.",
    )
    is_mock: bool = Field(
        default=False,
        description="Whether to run using deterministic offline mock models.",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional client idempotency key to prevent duplicate job submissions.",
    )

    @field_validator("max_total_provider_calls")
    @classmethod
    def validate_provider_ceiling(cls, v: int, info) -> int:
        metric_calls = info.data.get("max_metric_calls", 15)
        if v < metric_calls:
            raise ValueError(
                f"max_total_provider_calls ({v}) must be greater than or equal to max_metric_calls ({metric_calls})"
            )
        return v


class CancelOptimizationJobRequest(BaseModel):
    """Request payload to initiate cooperative job cancellation."""

    model_config = ConfigDict(extra="ignore")

    reason: Optional[str] = Field(
        default="User requested cancellation",
        max_length=255,
        description="Explanation or context for cancelling the optimization job.",
    )


# =============================================================================
# Response Schemas
# =============================================================================

class OptimizationJobProgress(BaseModel):
    """Safe, measurable progress indicators for an active or completed job."""

    model_config = ConfigDict(extra="ignore")

    current_phase: str = Field(default="queued")
    progress_percent: Optional[float] = Field(default=None)
    progress_message: Optional[str] = Field(default=None)
    completed_metric_calls: int = Field(default=0)
    max_metric_calls: int = Field(default=15)
    completed_provider_calls: int = Field(default=0)
    max_provider_calls: int = Field(default=35)
    last_heartbeat_at: Optional[datetime] = Field(default=None)


class OptimizationJobResourceUsage(BaseModel):
    """Resource consumption telemetry persisted across job attempts."""

    model_config = ConfigDict(extra="ignore")

    used_metric_calls: int = Field(default=0)
    used_provider_calls: int = Field(default=0)
    used_retry_attempts: int = Field(default=0)
    used_prompt_tokens: int = Field(default=0)
    used_candidates_tokens: int = Field(default=0)
    used_total_tokens: int = Field(default=0)
    estimated_cost_usd: float = Field(default=0.0)
    pricing_version: str = Field(default="gemini-2025-standard-v1")


class OptimizationJobResponse(BaseModel):
    """Authoritative API response schema for a durable optimization job."""

    model_config = ConfigDict(extra="ignore", from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique job identifier.")
    owner_id: uuid.UUID = Field(..., description="Owning security principal identifier.")
    job_type: str = Field(..., description="Job classification type.")
    status: JobStatus = Field(..., description="Current lifecycle state.")
    dataset_version: str = Field(...)
    dataset_sha256: Optional[str] = Field(default=None)
    generation_model_id: str = Field(...)
    reflection_model_id: str = Field(...)
    is_mock: bool = Field(default=False)
    attempt_count: int = Field(default=0)
    max_attempts: int = Field(default=2)
    progress: OptimizationJobProgress = Field(...)
    resource_usage: OptimizationJobResourceUsage = Field(...)
    queued_at: datetime = Field(...)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    cancel_requested: bool = Field(default=False)
    failure_category: Optional[str] = Field(default=None)
    failure_summary: Optional[str] = Field(default=None)
    artifacts_dir: Optional[str] = Field(default=None)
    result_summary: Optional[Dict[str, Any]] = Field(default=None)


class OptimizationJobListResponse(BaseModel):
    """Owner-scoped paginated collection of optimization jobs."""

    model_config = ConfigDict(extra="ignore")

    items: List[OptimizationJobResponse] = Field(default_factory=list)
    total_count: int = Field(default=0)
