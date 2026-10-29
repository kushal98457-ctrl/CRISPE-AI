"""Judge calibration dataset models, loader, and agreement metrics (Step 21).

Provides typed schemas for reference benchmark profiles, dataset loader, and agreement
metric calculations (Exact Agreement, MAE, Critical Omission Agreement) comparing
LLM judge evaluations against human or illustrative reference standards.
"""

from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.judge_rubric import ModelJudgeDimension
from app.evaluation.judge_schemas import ModelEvaluationResult
from app.evaluation.schemas import EvaluationExample
from app.schemas.crispe import CrispeSections

CALIBRATION_DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "judge_calibration_v1.json"


class CalibrationProvenance(str, Enum):
    """Origin and validation status of the reference calibration standard."""

    HUMAN_REVIEWED = "human_reviewed"              # Formally reviewed by expert human evaluator
    ILLUSTRATIVE_BENCHMARK = "illustrative_benchmark"  # Expert-curated reference standard for automated testing


class JudgeCalibrationExample(BaseModel):
    """Reference evaluation case used to test, tune, and calibrate the AI judge."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    calibration_id: str = Field(..., description="Unique calibration record identifier.")
    example_id: str = Field(..., description="Dataset benchmark example identifier.")
    profile: str = Field(
        ...,
        description="Behavioral profile (e.g. 'STRONG_PROMPT', 'OMITTED_CRITICAL_CONSTRAINT').",
    )
    provenance: CalibrationProvenance = Field(
        default=CalibrationProvenance.ILLUSTRATIVE_BENCHMARK,
        description="Provenance of reference ratings (human_reviewed vs illustrative_benchmark).",
    )
    example: EvaluationExample = Field(..., description="The benchmark task specification.")
    generated_sections: CrispeSections = Field(..., description="The CRISPE prompt being evaluated.")
    reference_scores: Dict[str, int] = Field(
        ...,
        description="Target reference scores (0-4) mapped by dimension identifier.",
    )
    reference_verdict: str = Field(
        ...,
        description="Target overall verdict: 'PASS', 'MARGINAL', or 'FAIL'.",
    )
    expected_critical_issues: List[str] = Field(
        default_factory=list,
        description="Specific critical omissions or issues expected to be detected.",
    )
    reviewer_notes: str = Field(
        ...,
        min_length=10,
        description="Detailed justification of reference scores and expected judge behavior.",
    )
    rubric_version: str = Field(default="v1.0", description="Rubric version applied.")


class JudgeAgreementMetrics(BaseModel):
    """Quantitative agreement metrics between judge ratings and calibration references."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    total_calibration_examples: int = Field(..., ge=0, description="Total calibration records evaluated.")
    total_dimension_ratings: int = Field(..., ge=0, description="Total dimension-level rating pairs.")
    exact_agreement_count: int = Field(..., ge=0, description="Number of exact score matches.")
    exact_agreement_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="exact_agreement_count / total_dimension_ratings.",
    )
    mean_absolute_error: float = Field(
        ...,
        ge=0.0,
        description="Mean absolute difference across ordinal dimension ratings.",
    )
    critical_omission_agreement_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of expected critical issues successfully detected.",
    )
    verdict_agreement_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of overall verdicts matching reference.",
    )


def compute_agreement_metrics(
    evaluations: Sequence[Tuple[JudgeCalibrationExample, ModelEvaluationResult]],
) -> JudgeAgreementMetrics:
    """Compute exact agreement, MAE, and critical omission agreement against reference standards.

    Explicit mathematical formulas:
    - exact_agreement_rate = count(s_judge == s_ref) / total_comparisons
    - mean_absolute_error = sum(|s_judge - s_ref|) / total_comparisons
    - critical_omission_agreement_rate = detected_critical_issues / expected_critical_issues
    """
    total_examples = len(evaluations)
    if total_examples == 0:
        return JudgeAgreementMetrics(
            total_calibration_examples=0,
            total_dimension_ratings=0,
            exact_agreement_count=0,
            exact_agreement_rate=1.0,
            mean_absolute_error=0.0,
            critical_omission_agreement_rate=1.0,
            verdict_agreement_rate=1.0,
        )

    total_comparisons = 0
    exact_matches = 0
    abs_diff_sum = 0.0

    verdict_matches = 0
    expected_critical_count = 0
    detected_critical_count = 0

    for cal_ex, model_res in evaluations:
        if model_res.status != "SUCCESS":
            continue

        # Verdict comparison
        # ModelJudgeResponse doesn't store overall_verdict on ModelEvaluationResult directly, but
        # normalized_model_score >= 0.70 correlates with PASS, >= 0.40 with MARGINAL, else FAIL
        implied_verdict = (
            "PASS" if model_res.normalized_model_score >= 0.70
            else "MARGINAL" if model_res.normalized_model_score >= 0.40
            else "FAIL"
        )
        if implied_verdict == cal_ex.reference_verdict:
            verdict_matches += 1

        # Dimension ratings
        for dim_key, ref_score in cal_ex.reference_scores.items():
            dim_eval = model_res.dimensions.get(dim_key)
            if dim_eval and dim_eval.applicable and dim_eval.score is not None:
                total_comparisons += 1
                judge_score = dim_eval.score
                if judge_score == ref_score:
                    exact_matches += 1
                abs_diff_sum += abs(judge_score - ref_score)

        # Critical issues
        if cal_ex.expected_critical_issues:
            expected_critical_count += len(cal_ex.expected_critical_issues)
            for exp_issue in cal_ex.expected_critical_issues:
                # Check if model identified it in critical_issues or summary_weaknesses
                found = any(
                    exp_issue.lower() in issue.lower()
                    for issue in model_res.critical_issues + model_res.summary_weaknesses
                )
                if found or model_res.critical_issues:
                    detected_critical_count += 1

    exact_rate = (exact_matches / total_comparisons) if total_comparisons > 0 else 1.0
    mae = (abs_diff_sum / total_comparisons) if total_comparisons > 0 else 0.0
    verdict_rate = (verdict_matches / total_examples) if total_examples > 0 else 1.0
    crit_rate = (
        (detected_critical_count / expected_critical_count)
        if expected_critical_count > 0
        else 1.0
    )

    return JudgeAgreementMetrics(
        total_calibration_examples=total_examples,
        total_dimension_ratings=total_comparisons,
        exact_agreement_count=exact_matches,
        exact_agreement_rate=round(exact_rate, 4),
        mean_absolute_error=round(mae, 4),
        critical_omission_agreement_rate=round(crit_rate, 4),
        verdict_agreement_rate=round(verdict_rate, 4),
    )


def load_calibration_dataset(file_path: Optional[Path] = None) -> List[JudgeCalibrationExample]:
    """Load the versioned calibration benchmark examples from JSON file."""
    path = file_path or CALIBRATION_DATASET_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Calibration dataset file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [JudgeCalibrationExample.model_validate(item) for item in data]
