"""SQLAlchemy ORM models for prompt releases, production activation, and audit logs (Step 28).

Persists:
1. Versioned prompt releases representing validated, immutable candidate snapshots.
2. Production pointer state (is_active) with partial unique constraints ensuring at most one active release.
3. Structured validation reports and isolated holdout evaluation summaries.
4. Comprehensive, durable audit trail for release transitions, promotions, and rollbacks.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    UUID,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.schemas.prompt_release import (
    ReleaseDetailResponse,
    ReleaseStatus,
    ReleaseSummaryResponse,
)

if TYPE_CHECKING:
    from app.models.optimization_job import OptimizationJob
    from app.models.principal import Principal
    from app.models.prompt_candidate import PromptCandidate


class PromptRelease(Base, TimestampMixin):
    """Authoritative release record governing production prompt deployment and rollback."""

    __tablename__ = "prompt_releases"
    __table_args__ = (
        Index("ix_prompt_releases_candidate_status", "candidate_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique primary key identifier for the release.",
    )
    release_version: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        doc="Human-readable semantic release version label (e.g. 'v1.0.0').",
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prompt_candidates.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        doc="Target immutable prompt candidate represented by this release.",
    )
    candidate_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Cryptographic SHA-256 digest of the candidate instruction text.",
    )
    job_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("optimization_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Optimization job that originated the candidate.",
    )
    instruction_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Immutable snapshot of the CRISPE system instruction deployed in this release.",
    )
    runtime_compatibility: Mapped[str] = mapped_column(
        String(50),
        default="crispe-dspy-v1",
        nullable=False,
        doc="Runtime architecture version supported by this release.",
    )
    generation_model_id: Mapped[str] = mapped_column(
        String(100),
        default="gemma-4-26b-a4b-it",
        nullable=False,
        doc="Target model configured for this release (no credentials).",
    )
    provider_name: Mapped[str] = mapped_column(
        String(50),
        default="google-genai",
        nullable=False,
        doc="Model provider identifier.",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default=ReleaseStatus.DRAFT.value,
        nullable=False,
        index=True,
        doc="Lifecycle state machine status.",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="True if this release is the currently active production prompt pointer.",
    )
    previous_release_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prompt_releases.id", ondelete="SET NULL"),
        nullable=True,
        doc="Direct predecessor release identifier in the deployment lineage.",
    )
    validation_policy_version: Mapped[str] = mapped_column(
        String(50),
        default="release_policy_v1",
        nullable=False,
        doc="Version of validation policy applied to approve this release.",
    )
    validation_report: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Complete structured validation audit report and metric comparisons.",
    )
    holdout_evaluation_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Controlled holdout evaluation score summary, isolated from the optimization loop.",
    )
    created_by_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("principals.id"),
        nullable=False,
        doc="Security principal who initiated the release draft.",
    )
    approved_by_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        ForeignKey("principals.id"),
        nullable=True,
        doc="Authorized security principal who promoted or approved the release.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    validated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    promoted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    superseded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    rolled_back_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    rejection_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    rollback_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    audit_event_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    # Relationships
    candidate: Mapped["PromptCandidate"] = relationship("PromptCandidate")
    job: Mapped[Optional["OptimizationJob"]] = relationship("OptimizationJob")
    created_by: Mapped["Principal"] = relationship("Principal", foreign_keys=[created_by_id])
    approved_by: Mapped[Optional["Principal"]] = relationship("Principal", foreign_keys=[approved_by_id])
    previous_release: Mapped[Optional["PromptRelease"]] = relationship("PromptRelease", remote_side=[id])
    audit_events: Mapped[List["PromptReleaseAuditEvent"]] = relationship(
        "PromptReleaseAuditEvent",
        back_populates="release",
        cascade="all, delete-orphan",
        order_by="PromptReleaseAuditEvent.created_at",
    )

    def to_summary_dto(self) -> ReleaseSummaryResponse:
        """Convert to public release summary DTO."""
        return ReleaseSummaryResponse(
            id=self.id,
            release_version=self.release_version,
            candidate_id=self.candidate_id,
            candidate_hash=self.candidate_hash,
            status=self.status,
            is_active=self.is_active,
            generation_model_id=self.generation_model_id,
            runtime_compatibility=self.runtime_compatibility,
            created_by_id=self.created_by_id,
            approved_by_id=self.approved_by_id,
            created_at=self.created_at,
            promoted_at=self.promoted_at,
        )

    def to_detail_dto(self) -> ReleaseDetailResponse:
        """Convert to detailed release response DTO."""
        summary = self.to_summary_dto()
        return ReleaseDetailResponse(
            **summary.model_dump(),
            instruction_text=self.instruction_text,
            job_id=self.job_id,
            previous_release_id=self.previous_release_id,
            validation_policy_version=self.validation_policy_version,
            validation_report=self.validation_report,
            holdout_evaluation_summary=self.holdout_evaluation_summary,
            validated_at=self.validated_at,
            superseded_at=self.superseded_at,
            rolled_back_at=self.rolled_back_at,
            rejection_reason=self.rejection_reason,
            rollback_reason=self.rollback_reason,
        )


class PromptReleaseAuditEvent(Base, TimestampMixin):
    """Durable audit event recording release state transitions, approvals, and rollbacks."""

    __tablename__ = "prompt_release_audit_events"
    __table_args__ = (
        Index("ix_prompt_release_audit_release", "release_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    release_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prompt_releases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Event classification ('created', 'validating', 'validated', 'rejected', 'promoted', 'superseded', 'rolled_back').",
    )
    acting_principal_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("principals.id"),
        nullable=False,
        index=True,
        doc="Security principal who authorized or triggered this lifecycle action.",
    )
    from_status: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    to_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )
    reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    release: Mapped["PromptRelease"] = relationship(
        "PromptRelease",
        back_populates="audit_events",
    )
    acting_principal: Mapped["Principal"] = relationship("Principal")
