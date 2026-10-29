"""Deterministic check definitions, result schemas, and reporting models (Step 20).

Provides typed schemas for machine-checkable evaluation rules, individual check results,
per-example evaluation outcomes, and partition-level metric summaries for the
Self-Evolving Prompt Engine.
"""

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CheckType(str, Enum):
    """Supported deterministic check types for CRISPE prompt evaluation."""

    CRISPE_SCHEMA_VALID = "crispe_schema_valid"
    REQUIRED_SECTION_PRESENT = "required_section_present"
    MINIMUM_TEXT_LENGTH = "minimum_text_length"
    REQUIRED_TEXT = "required_text"
    REQUIRED_TERM_GROUP = "required_term_group"
    FORBIDDEN_TEXT = "forbidden_text"
    REGEX_CONSTRAINT = "regex_constraint"
    CONSTRAINT_PRESERVED = "constraint_preserved"
    OUTPUT_FORMAT_REQUIREMENT = "output_format_requirement"


class CheckOutcome(str, Enum):
    """Execution status and evaluation verdict of an individual check."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class CrispeSectionTarget(str, Enum):
    """Target CRISPE section to inspect during check execution."""

    ALL = "all"
    CONTEXT = "context"
    ROLE = "role"
    INSTRUCTION = "instruction"
    SPECIFICATION = "specification"
    PERFORMANCE = "performance"
    EXAMPLE = "example"


class EvaluationDimension(str, Enum):
    """Multi-dimensional scoring categorization for prompt quality evaluation."""

    STRUCTURE = "structure"
    REQUIREMENTS = "requirements"
    CONSTRAINTS = "constraints"
    PROHIBITIONS = "prohibitions"
    OUTPUT_FORMAT = "output_format"


class DeterministicCheckDefinition(BaseModel):
    """Specification of a machine-checkable prompt evaluation rule."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    check_id: str = Field(
        ...,
        min_length=3,
        max_length=96,
        description="Unique identifier for this check (slug format, e.g. 'req_unit_tests').",
    )
    check_type: CheckType = Field(
        ...,
        description="The deterministic check type to execute.",
    )
    dimension: EvaluationDimension = Field(
        default=EvaluationDimension.REQUIREMENTS,
        description="Scoring dimension this check contributes to.",
    )
    target_section: CrispeSectionTarget = Field(
        default=CrispeSectionTarget.ALL,
        description="Which section of the CRISPE prompt to inspect.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Check-specific parameters (e.g. search terms, length bounds, regex pattern).",
    )
    required: bool = Field(
        default=True,
        description="Whether passing this check is required for example completion.",
    )
    is_critical: bool = Field(
        default=False,
        description="If True, failure triggers the critical failure penalty cap on overall score.",
    )
    weight: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Relative importance weight within its evaluation dimension.",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Human-readable explanation of what this rule verifies.",
    )

    @field_validator("check_id")
    @classmethod
    def validate_check_id(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not re.match(r"^[a-z0-9_\-]+$", cleaned):
            raise ValueError(f"check_id '{v}' must contain only lowercase alphanumeric, hyphens, or underscores.")
        return cleaned

    @field_validator("parameters")
    @classmethod
    def validate_parameters_for_type(cls, params: Dict[str, Any], info: Any) -> Dict[str, Any]:
        """Validate parameters according to check type requirements."""
        # Note: info.data contains validated fields parsed so far
        check_type_val = info.data.get("check_type")
        if not check_type_val:
            return params

        # Cast to enum if needed
        check_type = CheckType(check_type_val) if isinstance(check_type_val, str) else check_type_val

        if check_type == CheckType.MINIMUM_TEXT_LENGTH:
            min_chars = params.get("min_chars")
            min_words = params.get("min_words")
            if min_chars is None and min_words is None:
                raise ValueError("MINIMUM_TEXT_LENGTH requires 'min_chars' or 'min_words' parameter.")
            if min_chars is not None and (not isinstance(min_chars, int) or min_chars <= 0):
                raise ValueError("'min_chars' must be a positive integer.")
            if min_words is not None and (not isinstance(min_words, int) or min_words <= 0):
                raise ValueError("'min_words' must be a positive integer.")

        elif check_type == CheckType.REQUIRED_TEXT:
            text = params.get("text")
            if not text or not isinstance(text, str) or not text.strip():
                raise ValueError("REQUIRED_TEXT requires non-empty string parameter 'text'.")

        elif check_type == CheckType.REQUIRED_TERM_GROUP:
            terms = params.get("terms")
            if not terms or not isinstance(terms, list) or len(terms) == 0:
                raise ValueError("REQUIRED_TERM_GROUP requires non-empty list parameter 'terms'.")
            for t in terms:
                if not isinstance(t, str) or not t.strip():
                    raise ValueError("All items in 'terms' must be non-empty strings.")

        elif check_type == CheckType.FORBIDDEN_TEXT:
            terms = params.get("forbidden_terms") or params.get("terms") or ([params.get("text")] if params.get("text") else None)
            if not terms or not isinstance(terms, list) or len(terms) == 0:
                raise ValueError("FORBIDDEN_TEXT requires non-empty list 'forbidden_terms' or 'text'.")

        elif check_type == CheckType.REGEX_CONSTRAINT:
            pattern = params.get("pattern")
            if not pattern or not isinstance(pattern, str) or not pattern.strip():
                raise ValueError("REGEX_CONSTRAINT requires non-empty string parameter 'pattern'.")
            if len(pattern) > 500:
                raise ValueError("Regex pattern exceeds maximum allowed length of 500 characters.")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression pattern: {exc}")

        return params


class CheckResult(BaseModel):
    """Execution output and diagnostic details for an individual check."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    check_id: str = Field(..., description="Unique check identifier matching the definition.")
    check_type: CheckType = Field(..., description="Check type executed.")
    dimension: EvaluationDimension = Field(..., description="Evaluation dimension of the check.")
    outcome: CheckOutcome = Field(..., description="Verdict: passed, failed, not_applicable, or error.")
    required: bool = Field(..., description="Whether passing was required.")
    is_critical: bool = Field(..., description="Whether failure triggers the critical penalty cap.")
    weight: float = Field(..., gt=0.0, description="Weight used in dimension scoring.")
    expected_condition: str = Field(..., description="Summary of expected condition.")
    actual_condition: str = Field(..., description="Observed state in the evaluated prompt.")
    diagnostic_message: str = Field(..., description="Human-readable diagnostic feedback.")
    execution_error: Optional[str] = Field(default=None, description="Error message if execution failed.")


class EvaluationExampleResult(BaseModel):
    """Complete evaluation report for a single CRISPE prompt example."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    evaluation_id: str = Field(..., description="Unique evaluation record identifier.")
    example_id: str = Field(..., description="Dataset example identifier.")
    dataset_version: str = Field(..., description="Evaluation dataset version.")
    scoring_policy_version: str = Field(..., description="Scoring policy version used.")
    status: str = Field(..., description="Overall status: 'SUCCESS', 'EVALUATION_FAILED', or 'GENERATION_FAILED'.")
    crispe_structure_valid: bool = Field(..., description="Whether prompt satisfies CrispeSections schema.")
    crispe_structural_issues: List[str] = Field(default_factory=list, description="Field-level schema errors if malformed.")
    composite_score: float = Field(..., ge=0.0, le=1.0, description="Normalized weighted overall score [0.0, 1.0].")
    dimension_scores: Dict[str, float] = Field(..., description="Score per dimension [0.0, 1.0].")
    dimension_check_counts: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Counts of passed, failed, error, and not_applicable checks per dimension.",
    )
    critical_failure: bool = Field(..., description="Whether a critical requirement check failed.")
    critical_failure_reasons: List[str] = Field(
        default_factory=list,
        description="Diagnostic explanations for triggered critical failures.",
    )
    checks: List[CheckResult] = Field(..., description="Full list of individual check results.")
    failed_required_checks: List[str] = Field(default_factory=list, description="IDs of all failed required checks.")
    execution_errors: List[str] = Field(default_factory=list, description="Errors encountered during check execution.")
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of evaluation.",
    )


class EvaluationSummaryReport(BaseModel):
    """Aggregate evaluation metrics across a dataset partition or baseline artifact."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    experiment_id: str = Field(..., description="Identifier of evaluated experiment or baseline run.")
    dataset_version: str = Field(..., description="Dataset version evaluated.")
    partition: str = Field(..., description="Partition evaluated (e.g. validation, optimization).")
    scoring_policy_version: str = Field(..., description="Scoring policy version applied.")
    total_examples: int = Field(..., ge=0, description="Total examples attempted/evaluated.")
    successful_generations: int = Field(..., ge=0, description="Generations completed successfully.")
    failed_generations: int = Field(..., ge=0, description="Generations that failed or timed out.")
    valid_crispe_outputs: int = Field(..., ge=0, description="Generations conforming to CrispeSections.")
    malformed_crispe_outputs: int = Field(..., ge=0, description="Outputs with invalid or missing sections.")
    total_checks_evaluated: int = Field(..., ge=0, description="Total individual checks executed.")
    passed_checks: int = Field(..., ge=0, description="Total checks passed.")
    failed_checks: int = Field(..., ge=0, description="Total checks failed.")
    errored_checks: int = Field(..., ge=0, description="Checks that encountered runtime execution errors.")
    required_check_pass_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="passed_required_checks / applicable_required_checks.",
    )
    critical_failure_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="examples_with_critical_failure / total_examples.",
    )
    mean_composite_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average composite score across evaluated examples.",
    )
    dimension_mean_scores: Dict[str, float] = Field(
        ...,
        description="Mean score for each evaluation dimension.",
    )
    score_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Binned histogram of composite scores (e.g. '0.0-0.2', '0.2-0.4', etc.).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC completion timestamp of summary report.",
    )
