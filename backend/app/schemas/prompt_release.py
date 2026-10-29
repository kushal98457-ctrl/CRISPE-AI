"""Pydantic schemas and state-machine contracts for safe candidate promotion and prompt release management (Step 28).

Enforces:
1. Versioned prompt release lifecycle states and valid transitions.
2. Configurable validation policy gates (coverage, thresholds, regressions, holdout).
3. Structured, auditable validation reports.
4. Strongly typed API request/response DTOs for release creation, validation, promotion, and rollback.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import InvalidReleaseStateTransitionError


class ReleaseStatus(str, Enum):
    """Authoritative lifecycle states for a prompt release."""

    DRAFT = "draft"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ROLLED_BACK = "rolled_back"


class ReleaseStateMachine:
    """Validates permitted lifecycle transitions for prompt releases."""

    TERMINAL_STATES = frozenset({
        ReleaseStatus.REJECTED,
        ReleaseStatus.ROLLED_BACK,
    })

    PERMITTED_TRANSITIONS: Dict[ReleaseStatus, frozenset[ReleaseStatus]] = {
        ReleaseStatus.DRAFT: frozenset({
            ReleaseStatus.VALIDATING,
            ReleaseStatus.REJECTED,
        }),
        ReleaseStatus.VALIDATING: frozenset({
            ReleaseStatus.VALIDATED,
            ReleaseStatus.REJECTED,
        }),
        ReleaseStatus.VALIDATED: frozenset({
            ReleaseStatus.ACTIVE,
            ReleaseStatus.REJECTED,
        }),
        ReleaseStatus.ACTIVE: frozenset({
            ReleaseStatus.SUPERSEDED,
            ReleaseStatus.ROLLED_BACK,
        }),
        ReleaseStatus.SUPERSEDED: frozenset({
            ReleaseStatus.ACTIVE,       # Re-activation via rollback restoration
            ReleaseStatus.ROLLED_BACK,
        }),
        ReleaseStatus.REJECTED: frozenset(),
        ReleaseStatus.ROLLED_BACK: frozenset(),
    }

    @classmethod
    def validate_transition(cls, current_status: ReleaseStatus, target_status: ReleaseStatus) -> None:
        """Enforce legal release state transitions, raising InvalidReleaseStateTransitionError if illegal."""
        if current_status == target_status:
            return

        permitted = cls.PERMITTED_TRANSITIONS.get(current_status, frozenset())
        if target_status not in permitted:
            raise InvalidReleaseStateTransitionError(
                current_status=current_status.value,
                requested_status=target_status.value,
            )


# -----------------------------------------------------------------------------
# Validation Policy and Report Models
# -----------------------------------------------------------------------------

class ValidationCheckResult(BaseModel):
    """Outcome of an individual validation policy rule check."""

    check_name: str = Field(..., description="Unique rule or gate identifier.")
    passed: bool = Field(..., description="Whether the rule check passed.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic values and metrics evaluated.")
    failure_reason: Optional[str] = Field(None, description="Actionable failure reason if check failed.")


class ReleaseValidationReport(BaseModel):
    """Structured, reproducible release validation audit report."""

    model_config = ConfigDict(from_attributes=True)

    release_id: uuid.UUID = Field(..., description="Evaluated release identifier.")
    candidate_id: uuid.UUID = Field(..., description="Evaluated candidate identifier.")
    candidate_hash: str = Field(..., description="Cryptographic SHA-256 digest of candidate instruction.")
    baseline_release_id: Optional[uuid.UUID] = Field(None, description="Baseline release compared against, if any.")
    baseline_hash: Optional[str] = Field(None, description="SHA-256 digest of baseline instruction.")
    policy_version: str = Field(..., description="Validation policy revision applied.")
    decision: str = Field(..., description="Final decision: 'passed', 'failed', or 'incomplete'.")
    checks: List[ValidationCheckResult] = Field(default_factory=list, description="Detailed outcomes for each gate.")
    metrics_comparison: Dict[str, Any] = Field(
        default_factory=dict,
        description="Comparative metrics against baseline on authorized partitions.",
    )
    evaluated_at: datetime = Field(..., description="Timestamp when validation report was generated.")
    actionable_summary: str = Field(..., description="Summary narrative explaining decision.")


class ReleaseValidationPolicy(BaseModel):
    """Configurable gates governing candidate promotion eligibility."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(default="release_policy_v1", description="Policy version string.")
    min_evaluation_coverage: int = Field(default=2, ge=1, description="Minimum validation examples required.")
    min_validation_mean_score: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable validation partition mean score.",
    )
    max_permitted_score_regression: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowed score drop compared to the active production baseline.",
    )
    max_critical_failures_allowed: int = Field(
        default=0,
        ge=0,
        description="Maximum permissible critical failures on benchmark splits.",
    )
    require_holdout_evaluation: bool = Field(
        default=True,
        description="Whether controlled final-holdout evaluation is mandatory before release.",
    )
    min_holdout_mean_score: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable score on the sealed holdout benchmark partition.",
    )
    allowed_generation_models: List[str] = Field(
        default_factory=lambda: [
            "gemma-4-26b-a4b-it",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "mock-student-gemini-flash",
        ],
        description="Whitelist of permitted runtime generation models.",
    )
    require_separation_of_duties: bool = Field(
        default=False,
        description="If True, the promoter must be different from the candidate/job creator.",
    )


# -----------------------------------------------------------------------------
# Request Models
# -----------------------------------------------------------------------------

class CreateReleaseRequest(BaseModel):
    """Request payload to initiate a new candidate release draft."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: uuid.UUID = Field(..., description="Target evaluated prompt candidate to release.")
    release_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique semantic release version label (e.g. 'v1.1.0' or 'release-20261029-01').",
    )
    notes: Optional[str] = Field(None, max_length=500, description="Optional release justification notes.")


class PromoteReleaseRequest(BaseModel):
    """Request payload to authorize production promotion of a validated release."""

    model_config = ConfigDict(extra="forbid")

    reason: Optional[str] = Field(None, max_length=500, description="Operator justification for promotion.")


class RollbackReleaseRequest(BaseModel):
    """Request payload to safely roll back production to an approved prior release."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(..., min_length=5, max_length=500, description="Mandatory operational rollback reason.")


# -----------------------------------------------------------------------------
# Response Models
# -----------------------------------------------------------------------------

class ReleaseSummaryResponse(BaseModel):
    """Concise representation of a prompt release for list views."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique release identifier.")
    release_version: str = Field(..., description="Release version tag.")
    candidate_id: uuid.UUID = Field(..., description="Underlying immutable prompt candidate identifier.")
    candidate_hash: str = Field(..., description="SHA-256 cryptographic digest of instruction text.")
    status: str = Field(..., description="Current release state machine status.")
    is_active: bool = Field(..., description="True if this release is the current active production pointer.")
    generation_model_id: str = Field(..., description="Target model configured for this prompt release.")
    runtime_compatibility: str = Field(..., description="Supported generation runtime format.")
    created_by_id: str = Field(..., description="Principal who created the release draft.")
    approved_by_id: Optional[str] = Field(None, description="Principal who approved or promoted the release.")
    created_at: datetime = Field(..., description="Release creation timestamp.")
    promoted_at: Optional[datetime] = Field(None, description="Timestamp of production promotion.")


class ReleaseDetailResponse(ReleaseSummaryResponse):
    """Full detail of a prompt release including instructions, validation, and audit links."""

    instruction_text: str = Field(..., description="Exact immutable CRISPE instruction snapshot.")
    job_id: Optional[uuid.UUID] = Field(None, description="Optimization job that originated the candidate.")
    previous_release_id: Optional[uuid.UUID] = Field(None, description="Direct predecessor release identifier.")
    validation_policy_version: str = Field(..., description="Validation policy version used.")
    validation_report: Optional[Dict[str, Any]] = Field(None, description="Full structured validation report.")
    holdout_evaluation_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Sanitized holdout evaluation scores and summary metrics.",
    )
    validated_at: Optional[datetime] = Field(None, description="Timestamp of validation completion.")
    superseded_at: Optional[datetime] = Field(None, description="Timestamp when replaced by newer release.")
    rolled_back_at: Optional[datetime] = Field(None, description="Timestamp when rolled back.")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason if status is rejected.")
    rollback_reason: Optional[str] = Field(None, description="Rollback explanation if status is rolled_back.")


class PaginatedReleasesResponse(BaseModel):
    """Paginated collection of prompt releases."""

    items: List[ReleaseSummaryResponse] = Field(..., description="List of release summaries.")
    total_count: int = Field(..., ge=0, description="Total count matching query.")
    limit: int = Field(..., ge=1, description="Page size limit applied.")
    offset: int = Field(..., ge=0, description="Page offset applied.")
