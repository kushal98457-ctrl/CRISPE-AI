"""SQLAlchemy ORM models for persistent prompt candidate versioning and optimization evaluations (Step 27).

Persists:
1. Immutable prompt candidates with exact instruction snapshots and cryptographic content digests.
2. Parent-child lineage and mutation provenance across optimization iterations.
3. Separation of candidate definitions from evaluation records across authorized splits.
4. Database-level constraints protecting the final holdout partition from optimization leakage.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
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
from app.schemas.prompt_candidate import (
    CandidateDetailResponse,
    CandidateSummaryResponse,
    EvaluationRecordResponse,
)

if TYPE_CHECKING:
    from app.models.optimization_job import OptimizationJob


class PromptCandidate(Base, TimestampMixin):
    """Authoritative persistent record of a prompt candidate produced or evaluated by an optimization job."""

    __tablename__ = "prompt_candidates"
    __table_args__ = (
        UniqueConstraint("job_id", "candidate_type", "content_hash", name="uq_prompt_candidates_job_type_content"),
        Index("ix_prompt_candidates_job_created", "job_id", "created_at"),
        Index("ix_prompt_candidates_type_status", "candidate_type", "lifecycle_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique primary key identifier for the candidate.",
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("optimization_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Parent optimization job that generated or evaluated this candidate.",
    )
    parent_candidate_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prompt_candidates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Direct parent candidate identifier in the optimization lineage.",
    )
    candidate_type: Mapped[str] = mapped_column(
        String(50),
        default="optimized",
        nullable=False,
        index=True,
        doc="Classification ('baseline', 'optimized', 'intermediate').",
    )
    instruction_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Exact immutable snapshot of the CRISPE instruction evaluated.",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Cryptographic SHA-256 digest of normalized instruction text for deduplication and integrity.",
    )
    lifecycle_status: Mapped[str] = mapped_column(
        String(50),
        default="proposed",
        nullable=False,
        index=True,
        doc="Current candidate lifecycle state ('proposed', 'evaluated', 'rejected').",
    )
    optimizer_version: Mapped[str] = mapped_column(
        String(50),
        default="gepa-0.1.0",
        nullable=False,
        doc="Optimizer library and module release version.",
    )
    program_version: Mapped[str] = mapped_column(
        String(50),
        default="crispe-dspy-v1",
        nullable=False,
        doc="CRISPE DSPy program architecture revision.",
    )
    model_id: Mapped[str] = mapped_column(
        String(100),
        default="gemma-4-26b-a4b-it",
        nullable=False,
        doc="Target generation model identifier evaluated with this candidate (no secrets).",
    )
    dataset_version: Mapped[str] = mapped_column(
        String(50),
        default="v1",
        nullable=False,
        doc="Evaluation benchmark dataset version.",
    )
    evaluation_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Configuration parameters, seeds, and rubric settings applied.",
    )
    provenance_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Lineage metadata, reflection notes, trial index, or mutation history.",
    )
    artifact_ref: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Uniform relative artifact reference for serialized program state or traces.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    job: Mapped["OptimizationJob"] = relationship(
        "OptimizationJob",
        back_populates="candidates",
    )
    parent: Mapped[Optional["PromptCandidate"]] = relationship(
        "PromptCandidate",
        remote_side=[id],
        back_populates="children",
    )
    children: Mapped[List["PromptCandidate"]] = relationship(
        "PromptCandidate",
        back_populates="parent",
    )
    evaluations: Mapped[List["CandidateEvaluation"]] = relationship(
        "CandidateEvaluation",
        back_populates="candidate",
        cascade="all, delete-orphan",
        order_by="CandidateEvaluation.created_at",
    )

    def to_summary_dto(self) -> CandidateSummaryResponse:
        """Convert to concise candidate summary DTO."""
        preview = self.instruction_text[:200] + "..." if len(self.instruction_text) > 200 else self.instruction_text
        return CandidateSummaryResponse(
            id=self.id,
            job_id=self.job_id,
            parent_candidate_id=self.parent_candidate_id,
            candidate_type=self.candidate_type,
            lifecycle_status=self.lifecycle_status,
            content_hash=self.content_hash,
            model_id=self.model_id,
            dataset_version=self.dataset_version,
            created_at=self.created_at,
            instruction_preview=preview,
        )

    def to_detail_dto(self) -> CandidateDetailResponse:
        """Convert to full candidate detail DTO with immutable instruction snapshot."""
        summary = self.to_summary_dto()
        return CandidateDetailResponse(
            **summary.model_dump(),
            instruction_text=self.instruction_text,
            optimizer_version=self.optimizer_version,
            program_version=self.program_version,
            evaluation_config=self.evaluation_config,
            provenance_metadata=self.provenance_metadata,
            artifact_ref=self.artifact_ref,
        )


class CandidateEvaluation(Base, TimestampMixin):
    """Authoritative evaluation result record connecting a candidate to evaluation metrics."""

    __tablename__ = "candidate_evaluations"
    __table_args__ = (
        CheckConstraint(
            "dataset_split NOT IN ('test', 'holdout', 'final_holdout')",
            name="ck_candidate_evaluations_no_holdout",
        ),
        UniqueConstraint(
            "candidate_id",
            "evaluation_type",
            "dataset_split",
            "example_id",
            name="uq_candidate_evaluations_identity",
        ),
        Index("ix_candidate_evaluations_type_split", "evaluation_type", "dataset_split"),
        Index("ix_candidate_evaluations_job_created", "job_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prompt_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("optimization_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    evaluation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Evaluation modality ('deterministic', 'model_assisted', 'combined').",
    )
    dataset_version: Mapped[str] = mapped_column(
        String(50),
        default="v1",
        nullable=False,
    )
    dataset_split: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Authorized dataset partition ('train', 'val', 'optimization', 'validation').",
    )
    example_id: Mapped[str] = mapped_column(
        String(100),
        default="__aggregate__",
        nullable=False,
        index=True,
        doc="Dataset sample identifier or '__aggregate__' for split-level aggregates.",
    )
    evaluator_version: Mapped[str] = mapped_column(
        String(50),
        default="v1",
        nullable=False,
    )
    rubric_version: Mapped[str] = mapped_column(
        String(50),
        default="v1",
        nullable=False,
    )
    metrics: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Key-value mapping of evaluated numeric scores (e.g. mean_score, scores).",
    )
    passed: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )
    is_critical_failure: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    sanitized_feedback: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    execution_status: Mapped[str] = mapped_column(
        String(50),
        default="completed",
        nullable=False,
    )
    artifact_ref: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    candidate: Mapped["PromptCandidate"] = relationship(
        "PromptCandidate",
        back_populates="evaluations",
    )

    def to_response_dto(self) -> EvaluationRecordResponse:
        """Convert to public API response DTO."""
        return EvaluationRecordResponse(
            id=self.id,
            candidate_id=self.candidate_id,
            job_id=self.job_id,
            evaluation_type=self.evaluation_type,
            dataset_version=self.dataset_version,
            dataset_split=self.dataset_split,
            example_id=self.example_id if self.example_id != "__aggregate__" else None,
            evaluator_version=self.evaluator_version,
            rubric_version=self.rubric_version,
            metrics=self.metrics,
            passed=self.passed,
            is_critical_failure=self.is_critical_failure,
            sanitized_feedback=self.sanitized_feedback,
            execution_status=self.execution_status,
            artifact_ref=self.artifact_ref,
            created_at=self.created_at,
        )
