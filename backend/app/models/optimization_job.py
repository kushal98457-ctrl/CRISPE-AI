"""SQLAlchemy ORM model for durable optimization jobs (Step 26).

Stores job definitions, dataset partition bounds, validated resource limits,
worker lease fences, persistent usage telemetry, and execution outcomes.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UUID,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.schemas.optimization_job import (
    JobStatus,
    OptimizationJobProgress,
    OptimizationJobResourceUsage,
    OptimizationJobResponse,
)

if TYPE_CHECKING:
    from app.models.principal import Principal
    from app.models.prompt_candidate import PromptCandidate


class OptimizationJob(Base, TimestampMixin):
    """Authoritative persistent record of a durable GEPA optimization job."""

    __tablename__ = "optimization_jobs"
    __table_args__ = (
        Index("ix_optimization_jobs_status_lease", "status", "lease_expires_at"),
        Index("ix_optimization_jobs_owner_created", "owner_id", "created_at"),
        Index("ix_optimization_jobs_idempotency", "owner_id", "idempotency_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique primary key identifier for the optimization job.",
    )
    owner_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("principals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Owning security principal identifier providing tenant isolation.",
    )
    job_type: Mapped[str] = mapped_column(
        String(50),
        default="gepa_prompt_optimization",
        nullable=False,
        doc="Classification of the optimization workload.",
    )
    schema_version: Mapped[str] = mapped_column(
        String(20),
        default="v1",
        nullable=False,
        doc="Data schema revision version.",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default=JobStatus.QUEUED.value,
        nullable=False,
        index=True,
        doc="Current state machine status.",
    )

    # Dataset & Partition Boundaries
    dataset_version: Mapped[str] = mapped_column(
        String(50),
        default="v1",
        nullable=False,
    )
    dataset_sha256: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )
    partition_manifest: Mapped[str] = mapped_column(
        String(100),
        default="prompt_optimization_eval_v1_splits",
        nullable=False,
    )
    train_example_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Explicit list of training example IDs selected from optimization partition.",
    )
    val_example_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Explicit list of validation example IDs selected from validation partition.",
    )
    train_size: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
    )
    val_size: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    seed: Mapped[int] = mapped_column(
        Integer,
        default=42,
        nullable=False,
    )

    # Model Configuration
    generation_model_id: Mapped[str] = mapped_column(
        String(100),
        default="gemma-4-26b-a4b-it",
        nullable=False,
    )
    reflection_model_id: Mapped[str] = mapped_column(
        String(100),
        default="gemini-2.5-flash",
        nullable=False,
    )
    enable_model_judge: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    is_mock: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    # Budget Limits (Step 25 constraints)
    max_metric_calls: Mapped[int] = mapped_column(
        Integer,
        default=15,
        nullable=False,
    )
    max_total_provider_calls: Mapped[int] = mapped_column(
        Integer,
        default=35,
        nullable=False,
    )
    max_elapsed_seconds: Mapped[float] = mapped_column(
        Float,
        default=300.0,
        nullable=False,
    )
    concurrency_limit: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    max_retries_per_request: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    max_total_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    max_budget_usd: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    pricing_version: Mapped[str] = mapped_column(
        String(50),
        default="gemini-2025-standard-v1",
        nullable=False,
    )

    # Persistent Usage Telemetry (survives restarts)
    used_metric_calls: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    used_provider_calls: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    used_retry_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    used_prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    used_candidates_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    used_total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    estimated_cost_usd: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    # Worker Lease & Claiming Fences
    worker_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )
    claim_token: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="Fencing token preventing stale worker attempts from overwriting state.",
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        default=2,
        nullable=False,
    )
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    last_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Lifecycle Timestamps
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Progress & Cancellation Tracking
    current_phase: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
    )
    progress_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    progress_message: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    cancel_reason: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # Outcome & Artifact References
    stop_reason: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    failure_category: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    failure_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    artifacts_dir: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    result_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )

    # Relationships
    owner: Mapped["Principal"] = relationship(
        "Principal",
        back_populates="optimization_jobs",
    )
    candidates: Mapped[List["PromptCandidate"]] = relationship(
        "PromptCandidate",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="PromptCandidate.created_at",
    )

    def to_response_dto(self) -> OptimizationJobResponse:
        """Convert the persistent ORM model to a typed, safe API response DTO."""
        progress = OptimizationJobProgress(
            current_phase=self.current_phase,
            progress_percent=self.progress_percent,
            progress_message=self.progress_message,
            completed_metric_calls=self.used_metric_calls,
            max_metric_calls=self.max_metric_calls,
            completed_provider_calls=self.used_provider_calls,
            max_provider_calls=self.max_total_provider_calls,
            last_heartbeat_at=self.last_heartbeat_at,
        )
        usage = OptimizationJobResourceUsage(
            used_metric_calls=self.used_metric_calls,
            used_provider_calls=self.used_provider_calls,
            used_retry_attempts=self.used_retry_attempts,
            used_prompt_tokens=self.used_prompt_tokens,
            used_candidates_tokens=self.used_candidates_tokens,
            used_total_tokens=self.used_total_tokens,
            estimated_cost_usd=self.estimated_cost_usd,
            pricing_version=self.pricing_version,
        )
        return OptimizationJobResponse(
            id=self.id,
            owner_id=uuid.UUID(self.owner_id) if isinstance(self.owner_id, str) and "-" in self.owner_id else uuid.uuid5(uuid.NAMESPACE_DNS, self.owner_id),
            job_type=self.job_type,
            status=JobStatus(self.status),
            dataset_version=self.dataset_version,
            dataset_sha256=self.dataset_sha256,
            generation_model_id=self.generation_model_id,
            reflection_model_id=self.reflection_model_id,
            is_mock=self.is_mock,
            attempt_count=self.attempt_count,
            max_attempts=self.max_attempts,
            progress=progress,
            resource_usage=usage,
            queued_at=self.queued_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            cancel_requested=self.cancel_requested,
            failure_category=self.failure_category,
            failure_summary=self.failure_summary,
            artifacts_dir=self.artifacts_dir,
            result_summary=self.result_summary,
        )
