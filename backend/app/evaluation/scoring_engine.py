"""Scoring policy, evaluation execution, and metric aggregation engine (Step 20).

Implements multi-dimensional evaluation scoring, explicit critical failure capping,
and deterministic metric aggregation with well-defined mathematical denominators.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid
from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.check_schemas import (
    CheckOutcome,
    CheckResult,
    CheckType,
    DeterministicCheckDefinition,
    EvaluationDimension,
    EvaluationExampleResult,
    EvaluationSummaryReport,
)
from app.evaluation.rule_builder import build_checks_for_example
from app.evaluation.rule_engine import execute_check, validate_crispe_structure
from app.evaluation.schemas import EvaluationExample
from app.schemas.crispe import CrispeSections


class ScoringPolicy(BaseModel):
    """Configuration governing dimensional weights and critical penalty caps."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(default="v1", description="Semantic version of scoring policy.")
    dimension_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            EvaluationDimension.STRUCTURE.value: 0.25,
            EvaluationDimension.REQUIREMENTS.value: 0.35,
            EvaluationDimension.CONSTRAINTS.value: 0.20,
            EvaluationDimension.OUTPUT_FORMAT.value: 0.10,
            EvaluationDimension.PROHIBITIONS.value: 0.10,
        },
        description="Relative weights for each evaluation dimension (must sum to ~1.0).",
    )
    critical_failure_score_cap: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Upper score ceiling applied when any critical requirement check fails.",
    )
    schema_failure_score_cap: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Upper score ceiling applied when prompt fails canonical CRISPE schema validation.",
    )


class DeterministicScorer:
    """Coordinates check execution, multi-dimensional scoring, and metric aggregation."""

    def __init__(self, policy: Optional[ScoringPolicy] = None) -> None:
        self.policy = policy or ScoringPolicy()

    def evaluate_example(
        self,
        example: EvaluationExample,
        sections: Optional[CrispeSections],
        raw_text: Optional[str] = None,
        generation_status: str = "SUCCESS",
        generation_error: Optional[str] = None,
        custom_checks: Optional[List[DeterministicCheckDefinition]] = None,
    ) -> EvaluationExampleResult:
        """Evaluate a single example against deterministic checks and compute scores.

        Parameters:
            example: Benchmark record from evaluation dataset.
            sections: Generated CrispeSections (or None if generation failed).
            raw_text: Rendered Markdown prompt or raw text output.
            generation_status: 'SUCCESS' or 'FAILED'.
            generation_error: Error message if generation failed.
            custom_checks: Optional custom check definitions (defaults to rule builder).

        Returns:
            EvaluationExampleResult with dimensional and composite scores.
        """
        evaluation_id = f"eval_{uuid.uuid4().hex[:12]}"
        dataset_version = example.example_version

        # Handle upstream generation failure
        if generation_status == "FAILED" or sections is None and not raw_text:
            err_msg = generation_error or "Generation failed to produce output."
            return EvaluationExampleResult(
                evaluation_id=evaluation_id,
                example_id=example.example_id,
                dataset_version=dataset_version,
                scoring_policy_version=self.policy.policy_version,
                status="GENERATION_FAILED",
                crispe_structure_valid=False,
                crispe_structural_issues=[f"Upstream generation failed: {err_msg}"],
                composite_score=0.0,
                dimension_scores={dim.value: 0.0 for dim in EvaluationDimension},
                dimension_check_counts={
                    dim.value: {"passed": 0, "failed": 0, "error": 0, "not_applicable": 0}
                    for dim in EvaluationDimension
                },
                critical_failure=True,
                critical_failure_reasons=[f"Generation failure: {err_msg}"],
                checks=[],
                failed_required_checks=["generation_failed"],
                execution_errors=[],
                evaluated_at=datetime.now(timezone.utc),
            )

        # 1. CRISPE Structural Validation
        is_struct_valid, struct_issues = validate_crispe_structure(sections)

        # 2. Collect Checks
        checks_to_run = custom_checks or build_checks_for_example(example)

        # 3. Execute Checks
        check_results: List[CheckResult] = []
        for check_def in checks_to_run:
            res = execute_check(check_def, sections, raw_text=raw_text)
            check_results.append(res)

        # 4. Group results by dimension
        dim_passed_weight: Dict[str, float] = defaultdict(float)
        dim_total_weight: Dict[str, float] = defaultdict(float)
        dim_counts: Dict[str, Dict[str, int]] = {
            dim.value: {"passed": 0, "failed": 0, "error": 0, "not_applicable": 0}
            for dim in EvaluationDimension
        }

        critical_failures: List[str] = []
        failed_required: List[str] = []
        exec_errors: List[str] = []

        for cr in check_results:
            dim_key = cr.dimension.value if isinstance(cr.dimension, EvaluationDimension) else str(cr.dimension)
            outcome_val = cr.outcome.value if isinstance(cr.outcome, CheckOutcome) else str(cr.outcome)

            if dim_key in dim_counts and outcome_val in dim_counts[dim_key]:
                dim_counts[dim_key][outcome_val] += 1

            if cr.outcome == CheckOutcome.ERROR:
                exec_errors.append(f"Check '{cr.check_id}': {cr.execution_error or 'Execution error'}")

            # Calculate dimensional weightings for applicable checks
            if cr.outcome in (CheckOutcome.PASSED, CheckOutcome.FAILED, CheckOutcome.ERROR):
                dim_total_weight[dim_key] += cr.weight
                if cr.outcome == CheckOutcome.PASSED:
                    dim_passed_weight[dim_key] += cr.weight

            # Track required failures
            if cr.required and cr.outcome in (CheckOutcome.FAILED, CheckOutcome.ERROR):
                failed_required.append(cr.check_id)

            # Track critical failures
            if cr.is_critical and cr.outcome in (CheckOutcome.FAILED, CheckOutcome.ERROR):
                critical_failures.append(f"Critical check '{cr.check_id}' failed: {cr.diagnostic_message}")

        # If schema is invalid, that is also a critical structural failure
        if not is_struct_valid:
            critical_failures.append(f"Canonical CRISPE schema invalid: {'; '.join(struct_issues)}")

        # 5. Compute dimensional scores
        dimension_scores: Dict[str, float] = {}
        for dim in EvaluationDimension:
            key = dim.value
            tot = dim_total_weight[key]
            if tot > 0.0:
                score_d = min(1.0, max(0.0, dim_passed_weight[key] / tot))
            else:
                # If no applicable checks configured for this dimension, default to 1.0 (neutral)
                score_d = 1.0
            dimension_scores[key] = round(score_d, 4)

        # 6. Compute uncapped composite score
        active_weights_sum = 0.0
        weighted_sum = 0.0
        for dim_key, d_weight in self.policy.dimension_weights.items():
            if dim_total_weight.get(dim_key, 0.0) > 0.0:
                active_weights_sum += d_weight
                weighted_sum += d_weight * dimension_scores.get(dim_key, 1.0)

        if active_weights_sum > 0.0:
            uncapped_score = weighted_sum / active_weights_sum
        else:
            uncapped_score = 1.0 if is_struct_valid else 0.0

        composite_score = uncapped_score

        # 7. Apply Critical Penalty Caps
        has_critical_failure = len(critical_failures) > 0

        if has_critical_failure:
            composite_score = min(composite_score, self.policy.critical_failure_score_cap)

        if not is_struct_valid:
            composite_score = min(composite_score, self.policy.schema_failure_score_cap)

        composite_score = round(max(0.0, min(1.0, composite_score)), 4)

        overall_status = "SUCCESS"
        if len(exec_errors) > 0 and len(failed_required) == len(check_results):
            overall_status = "EVALUATION_FAILED"

        return EvaluationExampleResult(
            evaluation_id=evaluation_id,
            example_id=example.example_id,
            dataset_version=dataset_version,
            scoring_policy_version=self.policy.policy_version,
            status=overall_status,
            crispe_structure_valid=is_struct_valid,
            crispe_structural_issues=struct_issues,
            composite_score=composite_score,
            dimension_scores=dimension_scores,
            dimension_check_counts=dim_counts,
            critical_failure=has_critical_failure,
            critical_failure_reasons=critical_failures,
            checks=check_results,
            failed_required_checks=failed_required,
            execution_errors=exec_errors,
            evaluated_at=datetime.now(timezone.utc),
        )

    def aggregate_results(
        self,
        results: Sequence[EvaluationExampleResult],
        experiment_id: str,
        dataset_version: str,
        partition: str,
    ) -> EvaluationSummaryReport:
        """Aggregate per-example evaluation results into a comprehensive summary report.

        Explicit Denominators:
        - total_examples: len(results)
        - required_check_pass_rate: total_passed_required / total_applicable_required
        - critical_failure_rate: count(critical_failure == True) / total_examples
        - mean_composite_score: sum(composite_score) / total_examples
        """
        total_examples = len(results)
        if total_examples == 0:
            return EvaluationSummaryReport(
                experiment_id=experiment_id,
                dataset_version=dataset_version,
                partition=partition,
                scoring_policy_version=self.policy.policy_version,
                total_examples=0,
                successful_generations=0,
                failed_generations=0,
                valid_crispe_outputs=0,
                malformed_crispe_outputs=0,
                total_checks_evaluated=0,
                passed_checks=0,
                failed_checks=0,
                errored_checks=0,
                required_check_pass_rate=1.0,
                critical_failure_rate=0.0,
                mean_composite_score=0.0,
                dimension_mean_scores={dim.value: 0.0 for dim in EvaluationDimension},
                score_distribution={
                    "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0
                },
                created_at=datetime.now(timezone.utc),
            )

        successful_generations = 0
        failed_generations = 0
        valid_crispe_outputs = 0
        malformed_crispe_outputs = 0

        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        errored_checks = 0

        total_required_applicable = 0
        total_required_passed = 0

        critical_failure_count = 0
        composite_score_sum = 0.0

        dim_scores_sum: Dict[str, float] = defaultdict(float)
        score_bins: Dict[str, int] = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for res in results:
            if res.status == "GENERATION_FAILED":
                failed_generations += 1
            else:
                successful_generations += 1

            if res.crispe_structure_valid:
                valid_crispe_outputs += 1
            elif res.status != "GENERATION_FAILED":
                malformed_crispe_outputs += 1

            if res.critical_failure:
                critical_failure_count += 1

            composite_score_sum += res.composite_score

            # Binning
            score = res.composite_score
            if score < 0.2:
                score_bins["0.0-0.2"] += 1
            elif score < 0.4:
                score_bins["0.2-0.4"] += 1
            elif score < 0.6:
                score_bins["0.4-0.6"] += 1
            elif score < 0.8:
                score_bins["0.6-0.8"] += 1
            else:
                score_bins["0.8-1.0"] += 1

            for dim_key, dim_score in res.dimension_scores.items():
                dim_scores_sum[dim_key] += dim_score

            for cr in res.checks:
                total_checks += 1
                if cr.outcome == CheckOutcome.PASSED:
                    passed_checks += 1
                elif cr.outcome == CheckOutcome.FAILED:
                    failed_checks += 1
                elif cr.outcome == CheckOutcome.ERROR:
                    errored_checks += 1

                if cr.required:
                    if cr.outcome in (CheckOutcome.PASSED, CheckOutcome.FAILED, CheckOutcome.ERROR):
                        total_required_applicable += 1
                        if cr.outcome == CheckOutcome.PASSED:
                            total_required_passed += 1

        required_pass_rate = (
            (total_required_passed / total_required_applicable)
            if total_required_applicable > 0
            else 1.0
        )
        critical_fail_rate = critical_failure_count / total_examples
        mean_composite = composite_score_sum / total_examples

        dimension_mean_scores = {
            dim.value: round(dim_scores_sum[dim.value] / total_examples, 4)
            for dim in EvaluationDimension
        }

        return EvaluationSummaryReport(
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            partition=partition,
            scoring_policy_version=self.policy.policy_version,
            total_examples=total_examples,
            successful_generations=successful_generations,
            failed_generations=failed_generations,
            valid_crispe_outputs=valid_crispe_outputs,
            malformed_crispe_outputs=malformed_crispe_outputs,
            total_checks_evaluated=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            errored_checks=errored_checks,
            required_check_pass_rate=round(required_pass_rate, 4),
            critical_failure_rate=round(critical_fail_rate, 4),
            mean_composite_score=round(mean_composite, 4),
            dimension_mean_scores=dimension_mean_scores,
            score_distribution=score_bins,
            created_at=datetime.now(timezone.utc),
        )
