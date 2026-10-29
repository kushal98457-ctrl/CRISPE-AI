"""Pydantic schemas and DTOs for prompt candidate versioning and optimization evaluations (Step 27).

Enforces:
1. Immutable snapshot representation for candidate prompts and parentage.
2. Separation of candidate definitions from evaluation records.
3. Strongly typed evaluation metrics, pass/fail indicators, and execution status.
4. Safe, non-leaking API models without internal file paths or credentials.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field


class CandidateType(str, Enum):
    """Classification of prompt candidate in the optimization lineage."""

    BASELINE = "baseline"
    OPTIMIZED = "optimized"
    INTERMEDIATE = "intermediate"


class CandidateLifecycleStatus(str, Enum):
    """Lifecycle progression states for prompt candidates.

    Note: Candidate promotion is strictly out-of-scope for Step 27 (reserved for Step 28).
    """

    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    REJECTED = "rejected"


class EvaluationType(str, Enum):
    """Classification of candidate evaluation execution."""

    DETERMINISTIC = "deterministic"
    MODEL_ASSISTED = "model_assisted"
    COMBINED = "combined"


# -----------------------------------------------------------------------------
# Prompt Candidate Response Models
# -----------------------------------------------------------------------------

class CandidateSummaryResponse(BaseModel):
    """Concise representation of a prompt candidate for list views."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique primary key identifier of the prompt candidate.")
    job_id: uuid.UUID = Field(..., description="Optimization job that produced or evaluated the candidate.")
    parent_candidate_id: Optional[uuid.UUID] = Field(
        None,
        description="Parent candidate identifier in the optimization lineage, if applicable.",
    )
    candidate_type: str = Field(..., description="Candidate classification ('baseline', 'optimized', 'intermediate').")
    lifecycle_status: str = Field(..., description="Candidate lifecycle state ('proposed', 'evaluated', 'rejected').")
    content_hash: str = Field(..., description="SHA-256 cryptographic digest of normalized instruction text.")
    model_id: str = Field(..., description="Target generation model identifier evaluated with this candidate.")
    dataset_version: str = Field(..., description="Evaluation benchmark dataset version used for testing.")
    created_at: datetime = Field(..., description="Timestamp when candidate snapshot was recorded.")
    instruction_preview: Optional[str] = Field(
        None,
        description="Truncated preview snippet of the candidate instruction text.",
    )


class CandidateDetailResponse(CandidateSummaryResponse):
    """Complete immutable snapshot and provenance details for a prompt candidate."""

    instruction_text: str = Field(
        ...,
        description="Exact CRISPE instruction or instruction-set snapshot evaluated.",
    )
    optimizer_version: str = Field(..., description="Optimizer library release version (e.g. gepa-0.1.0).")
    program_version: str = Field(..., description="CRISPE DSPy program architecture revision.")
    evaluation_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration parameters and rubric settings applied during evaluation.",
    )
    provenance_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Lineage, reflection history, trial index, or mutation provenance details.",
    )
    artifact_ref: Optional[str] = Field(
        None,
        description="Uniform relative artifact reference for serialized program state.",
    )


class PaginatedCandidatesResponse(BaseModel):
    """Paginated collection of prompt candidates produced by an optimization job."""

    items: List[CandidateSummaryResponse] = Field(..., description="List of prompt candidate summaries.")
    total_count: int = Field(..., ge=0, description="Total number of candidates matching the query.")
    page: int = Field(..., ge=1, description="Current page number.")
    page_size: int = Field(..., ge=1, description="Page size limit applied.")


# -----------------------------------------------------------------------------
# Candidate Evaluation Response Models
# -----------------------------------------------------------------------------

class EvaluationRecordResponse(BaseModel):
    """Authoritative evaluation result record connecting a candidate to evaluation metrics."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique evaluation record identifier.")
    candidate_id: uuid.UUID = Field(..., description="Evaluated prompt candidate identifier.")
    job_id: uuid.UUID = Field(..., description="Parent optimization job identifier.")
    evaluation_type: str = Field(
        ...,
        description="Evaluation modality: 'deterministic', 'model_assisted', or 'combined'.",
    )
    dataset_version: str = Field(..., description="Dataset benchmark version.")
    dataset_split: str = Field(
        ...,
        description="Evaluation dataset partition ('train', 'val', 'optimization', 'validation').",
    )
    example_id: Optional[str] = Field(
        None,
        description="Specific dataset example identifier, or '__aggregate__' for aggregate split metrics.",
    )
    evaluator_version: str = Field(..., description="Engine evaluator revision.")
    rubric_version: str = Field(..., description="Evaluation rubric revision.")
    metrics: Dict[str, Any] = Field(
        ...,
        description="Key-value mapping of evaluated numeric scores (e.g. mean_score, crispe_compliance).",
    )
    passed: Optional[bool] = Field(None, description="Binary threshold pass/fail indicator, if defined.")
    is_critical_failure: bool = Field(
        default=False,
        description="True if evaluation encountered a critical benchmark failure.",
    )
    sanitized_feedback: Optional[str] = Field(
        None,
        description="Sanitized feedback narrative or structured diagnostic message.",
    )
    execution_status: str = Field(
        default="completed",
        description="Evaluation execution status ('completed', 'failed').",
    )
    artifact_ref: Optional[str] = Field(
        None,
        description="Relative artifact reference to detailed traces or raw judge responses.",
    )
    created_at: datetime = Field(..., description="Timestamp when evaluation was executed and recorded.")


class CandidateEvaluationsResponse(BaseModel):
    """Aggregated evaluation overview and optional per-example breakdown for a candidate."""

    candidate_id: uuid.UUID = Field(..., description="Prompt candidate identifier.")
    aggregate_evaluations: List[EvaluationRecordResponse] = Field(
        ...,
        description="Aggregate metric evaluations for authorized dataset splits.",
    )
    per_example_evaluations: Optional[List[EvaluationRecordResponse]] = Field(
        None,
        description="Detailed per-example evaluation results, when requested.",
    )


# -----------------------------------------------------------------------------
# Optimization Results Summary Response Model
# -----------------------------------------------------------------------------

class JobOptimizationResultsResponse(BaseModel):
    """Comprehensive optimization results summary for a durable optimization job."""

    job_id: uuid.UUID = Field(..., description="Optimization job identifier.")
    status: str = Field(..., description="Job execution status.")
    stop_reason: Optional[str] = Field(None, description="Termination reason string.")
    baseline_candidate: Optional[CandidateSummaryResponse] = Field(
        None,
        description="Baseline prompt candidate evaluated at start of job.",
    )
    best_candidate: Optional[CandidateSummaryResponse] = Field(
        None,
        description="Best scoring candidate discovered during optimization.",
    )
    comparison_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Comparative metrics between baseline and optimized candidates on validation split.",
    )
    candidate_count: int = Field(default=0, ge=0, description="Total candidates recorded for this job.")
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of artifact names to uniform relative URNs.",
    )
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp, if finished.")
