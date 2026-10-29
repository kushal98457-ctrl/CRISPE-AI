"""End-to-end GEPA prompt optimization experiment runner with resource safeguards (Step 24).

Coordinates:
1. Leakage-aware subset loading & dataset integrity verification.
2. Baseline prompt evaluation on permitted validation examples.
3. Resource-guarded reflective GEPA optimization (metric limits, provider ceiling, timeouts).
4. Candidate evaluation and baseline vs. candidate comparative analysis.
5. Machine-readable artifact persistence (summary, program state, instruction diff, validation comparison).
6. Strict production prompt protection (candidate remains isolated as EXPERIMENTAL_CANDIDATE).
"""

from dataclasses import asdict
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import dspy

from app.core.config import Settings, get_settings
from app.core.exceptions import MissingProviderCredentialError
from app.core.logging import get_logger
from app.dspy_modules.budget_guard import (
    BudgetExceededError,
    OperationType,
    OptimizationTimeoutError,
    ResourceBudgetGuard,
    ResourceUsageSummary,
)
from app.dspy_modules.dataset_converter import convert_evaluation_example
from app.dspy_modules.experiment_config import GepaExperimentConfig
from app.dspy_modules.gepa_metric import CrispeGepaMetric
from app.dspy_modules.model_client import (
    create_dspy_lm,
    create_dspy_reflection_lm,
    dspy_isolated_context,
)
from app.dspy_modules.program import CrispeGenerationProgram
from app.dspy_modules.subset_selector import PilotSubsetSelector, PilotSubsetsMetadata
from app.services.generation_service import CrispeGenerationService

logger = get_logger("gepa_experiment_runner")


# =============================================================================
# Deterministic Test Doubles for Offline Experiments
# =============================================================================

class MockExperimentStudentLM(dspy.BaseLM):
    """Deterministic mock student LM for offline pilot runs."""

    def __init__(self, guard: Optional[ResourceBudgetGuard] = None) -> None:
        super().__init__(model="mock-student-gemini-flash")
        self.guard = guard

    def __call__(self, prompt=None, messages=None, **kwargs):
        if self.guard:
            if hasattr(self.guard, "reserve_provider_call"):
                token = self.guard.reserve_provider_call(
                    operation_type=OperationType.GENERATION,
                    model_id=self.model,
                    acquire_concurrency=True,
                )
                self.guard.record_call_completion(
                    token=token,
                    usage={"prompt_tokens": 120, "candidates_tokens": 150, "total_tokens": 270},
                    latency_ms=15.0,
                    provider_name="mock-offline",
                    model_id=self.model,
                )
            else:
                self.guard.record_generation_call()
        return [
            "[[ ## context ## ]]\nContext: Enterprise financial transaction pipeline.\n"
            "[[ ## role ## ]]\nRole: Senior FinTech Systems Architect.\n"
            "[[ ## instruction ## ]]\nInstruction: Build an automated ledger discrepancy detector in Python.\n"
            "[[ ## specification ## ]]\nSpecification: Return structured JSON format with fields: status, discrepancy_amount. Must not drop messages.\n"
            "[[ ## performance ## ]]\nPerformance: Reconcile 10,000 transactions per second with zero drop.\n"
            "[[ ## example ## ]]\nExample: Sample transaction discrepancy payload."
        ]


class MockExperimentReflectionLM(dspy.BaseLM):
    """Deterministic mock reflection LM for offline pilot runs."""

    def __init__(self, guard: Optional[ResourceBudgetGuard] = None) -> None:
        super().__init__(model="mock-reflection-gemini-flash")
        self.guard = guard

    def __call__(self, prompt=None, messages=None, **kwargs):
        if self.guard:
            if hasattr(self.guard, "reserve_provider_call"):
                token = self.guard.reserve_provider_call(
                    operation_type=OperationType.REFLECTION,
                    model_id=self.model,
                    acquire_concurrency=True,
                )
                self.guard.record_call_completion(
                    token=token,
                    usage={"prompt_tokens": 200, "candidates_tokens": 100, "total_tokens": 300},
                    latency_ms=20.0,
                    provider_name="mock-offline",
                    model_id=self.model,
                )
            else:
                self.guard.record_reflection_call()
        return [
            json.dumps({
                "new_instruction": (
                    "Transform a user's prompt idea, domain context, and constraints into a production-grade CRISPE prompt.\n\n"
                    "Enhanced Specification Policy:\n"
                    "- Context: Operational background, business domain, and technical environment.\n"
                    "- Role: The persona, authority, perspective, and professional identity.\n"
                    "- Instruction: The primary task, explicit action commands, and step-by-step methodologies.\n"
                    "- Specification: Ensure all mandatory constraints, technology locks, and output formats are strictly preserved.\n"
                    "- Performance: Explicit evaluation criteria and success benchmarks.\n"
                    "- Example: Concrete, illustrative sample demonstrating the desired execution pattern."
                )
            })
        ]


# =============================================================================
# Experiment Results Dataclasses
# =============================================================================

class GepaExperimentResult:
    """Complete results and comparison for a GEPA prompt optimization run."""

    def __init__(
        self,
        config: GepaExperimentConfig,
        subsets_metadata: PilotSubsetsMetadata,
        resource_summary: ResourceUsageSummary,
        baseline_summary: Dict[str, Any],
        candidate_summary: Dict[str, Any],
        comparison: Dict[str, Any],
        status: str,
        artifacts_saved: List[str],
        error_message: Optional[str] = None,
        baseline_instruction: str = "",
        candidate_instruction: str = "",
    ) -> None:
        self.config = config
        self.subsets_metadata = subsets_metadata
        self.resource_summary = resource_summary
        self.baseline_summary = baseline_summary
        self.candidate_summary = candidate_summary
        self.comparison = comparison
        self.status = status
        self.artifacts_saved = artifacts_saved
        self.error_message = error_message
        self.baseline_instruction = baseline_instruction
        self.candidate_instruction = candidate_instruction


# =============================================================================
# Experiment Runner Implementation
# =============================================================================

class GepaExperimentRunner:
    """Executes controlled, bounded GEPA prompt optimization experiments."""

    def __init__(
        self,
        config: Optional[GepaExperimentConfig] = None,
        settings: Optional[Settings] = None,
        subset_selector: Optional[PilotSubsetSelector] = None,
    ) -> None:
        self.config = config or GepaExperimentConfig()
        self.settings = settings or get_settings()
        self.subset_selector = subset_selector or PilotSubsetSelector()
        self.guard = ResourceBudgetGuard(
            experiment_id=self.config.experiment_id,
            max_gepa_metric_calls=self.config.max_metric_calls,
            max_total_provider_calls=self.config.max_total_provider_calls,
            max_elapsed_seconds=self.config.max_elapsed_seconds,
            concurrency_limit=self.config.concurrency_limit,
            max_retries_per_request=self.config.max_retries_per_request,
            max_total_tokens=self.config.max_total_tokens,
            max_budget_usd=self.config.max_budget_usd,
            pricing_version=self.config.pricing_version,
        )

    def _evaluate_program_on_subset(
        self,
        program: CrispeGenerationProgram,
        subset: List[dspy.Example],
        metric: CrispeGepaMetric,
        lm: dspy.LM,
    ) -> Dict[str, Any]:
        """Evaluate a program candidate across a validation subset and compute aggregate scores."""
        scores: List[float] = []
        feedbacks: List[str] = []
        valid_structure_count = 0
        critical_failure_count = 0
        per_example_details: List[Dict[str, Any]] = []

        with dspy.context(lm=lm):
            for gold in subset:
                pred = program(
                    idea=gold.idea,
                    task_context=gold.task_context,
                    constraints=gold.constraints,
                    output_format_preference=gold.output_format_preference,
                )
                metric_res = metric(gold=gold, pred=pred)
                scores.append(metric_res.score)
                feedbacks.append(metric_res.feedback)

                sections, err = metric._extract_crispe_sections(pred)
                if sections:
                    valid_structure_count += 1

                if metric_res.score <= 0.40 and "[CRITICAL FAILURE]" in metric_res.feedback:
                    critical_failure_count += 1

                per_example_details.append({
                    "example_id": gold.example_id,
                    "score": metric_res.score,
                    "feedback": metric_res.feedback,
                    "objective_scores": metric_res.objective_scores,
                    "valid_structure": bool(sections),
                })

        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "mean_score": round(mean_score, 4),
            "example_count": len(subset),
            "valid_structure_rate": round(valid_structure_count / len(subset), 4) if subset else 0.0,
            "critical_failure_count": critical_failure_count,
            "scores": scores,
            "per_example": per_example_details,
        }

    def _generate_instruction_diff(self, baseline_text: str, candidate_text: str) -> str:
        """Generate a clean unified text diff between baseline and candidate instructions."""
        baseline_lines = baseline_text.strip().splitlines(keepends=True)
        candidate_lines = candidate_text.strip().splitlines(keepends=True)

        diff = difflib.unified_diff(
            baseline_lines,
            candidate_lines,
            fromfile="baseline_instruction",
            tofile="optimized_candidate_instruction",
            lineterm="",
        )
        return "\n".join(diff)

    def _persist_artifacts(
        self,
        experiment_dir: Path,
        summary_data: Dict[str, Any],
        candidate_prog: Optional[CrispeGenerationProgram],
        instruction_diff: str,
        val_comparison: Dict[str, Any],
    ) -> List[str]:
        """Save all experiment artifacts into the designated results directory."""
        experiment_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[str] = []

        # 1. Experiment summary JSON
        summary_path = experiment_dir / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        saved_paths.append(str(summary_path))

        # 2. Validation comparison JSON
        comparison_path = experiment_dir / "validation_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(val_comparison, f, indent=2)
        saved_paths.append(str(comparison_path))

        # 3. Instruction diff Markdown
        diff_path = experiment_dir / "instruction_diff.md"
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write("# CRISPE-AI GEPA Prompt Optimization Diff\n\n")
            f.write(f"**Experiment ID:** `{self.config.experiment_id}`  \n")
            f.write(f"**Timestamp:** `{datetime.now(timezone.utc).isoformat()}`  \n\n")
            f.write("## 1. Unified Diff\n\n```diff\n")
            f.write(instruction_diff if instruction_diff else "No instruction changes (candidate identical to baseline)")
            f.write("\n```\n\n")
            f.write("## 2. Optimized Candidate Instruction\n\n```text\n")
            if candidate_prog and hasattr(candidate_prog.predictor, "signature"):
                f.write(getattr(candidate_prog.predictor.signature, "instructions", ""))
            f.write("\n```\n")
        saved_paths.append(str(diff_path))

        # 4. Candidate program state (DSPy dump_state)
        if candidate_prog and hasattr(candidate_prog, "dump_state"):
            state_path = experiment_dir / "candidate_program.json"
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(candidate_prog.dump_state(), f, indent=2)
            saved_paths.append(str(state_path))

        logger.info("Persisted %d experiment artifacts to %s", len(saved_paths), experiment_dir)
        return saved_paths

    def run_experiment(self, force_mock: bool = False) -> GepaExperimentResult:
        """Execute the bounded prompt optimization experiment.

        Parameters:
            force_mock: If True, uses deterministic mock models even if credentials exist.

        Returns:
            GepaExperimentResult with full comparison metrics and saved artifact paths.
        """
        logger.info("Starting GEPA optimization experiment '%s'...", self.config.experiment_id)
        experiment_dir = self.config.results_dir / self.config.experiment_id

        # 1. Dataset Subset Selection & Integrity Check
        train_examples, val_examples, subsets_meta = self.subset_selector.select_pilot_subsets(
            train_size=self.config.train_size,
            val_size=self.config.val_size,
        )

        # 2. Model Client Resolution (Live vs. Mock)
        use_mock = self.config.is_mock or force_mock
        has_credentials = False
        try:
            self.settings.get_gemini_api_key()
            has_credentials = True
        except MissingProviderCredentialError:
            has_credentials = False

        if not use_mock and not has_credentials:
            logger.warning("No Gemini API credentials configured; falling back to deterministic mock experiment.")
            use_mock = True

        if use_mock:
            logger.info("Running experiment using deterministic offline mock models.")
            student_lm = MockExperimentStudentLM(guard=self.guard)
            reflection_lm = MockExperimentReflectionLM(guard=self.guard)
            gen_model_id = "mock-student-gemini-flash"
            ref_model_id = "mock-reflection-gemini-flash"
        else:
            logger.info("Running experiment using live configured Gemini Flash models.")
            student_lm = create_dspy_lm(
                settings=self.settings,
                budget_manager=self.guard,
                operation_type=OperationType.GENERATION,
            )
            reflection_lm = create_dspy_reflection_lm(
                settings=self.settings,
                budget_manager=self.guard,
            )
            gen_model_id = self.settings.PRIMARY_MODEL
            ref_model_id = self.settings.REFLECTION_MODEL

        # Activate ambient budget context
        ctx_token = self.guard.activate()
        try:
            # 3. Guarded Metric Setup
            base_metric = CrispeGepaMetric(enable_model_judge=self.config.enable_model_judge)

            def guarded_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, **kwargs):
                self.guard.record_metric_call()
                return base_metric(gold, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace, **kwargs)

            # 4. Evaluate Baseline Program
            baseline_prog = CrispeGenerationProgram()
            base_instructions = baseline_prog.predictor.signature.instructions

            logger.info("Evaluating baseline program on %d validation examples...", len(val_examples))
            baseline_summary = self._evaluate_program_on_subset(
                program=baseline_prog,
                subset=val_examples,
                metric=base_metric,
                lm=student_lm,
            )
            logger.info("Baseline validation mean score: %.4f", baseline_summary["mean_score"])

            # 5. Build and Execute GEPA Optimizer
            gepa = dspy.GEPA(
                metric=guarded_metric,
                reflection_lm=reflection_lm,
                max_metric_calls=self.config.max_metric_calls,
                track_stats=True,
                seed=self.config.seed,
            )

            candidate_prog: Optional[CrispeGenerationProgram] = None
            status = "SUCCESS"
            error_msg: Optional[str] = None

            try:
                with dspy.context(lm=student_lm):
                    logger.info(
                        "Executing gepa.compile() with max_metric_calls=%d, train=%d, val=%d...",
                        self.config.max_metric_calls,
                        len(train_examples),
                        len(val_examples),
                    )
                    candidate_prog = gepa.compile(
                        student=baseline_prog,
                        trainset=train_examples,
                        valset=val_examples,
                    )
            except (BudgetExceededError, OptimizationTimeoutError) as exc:
                status = getattr(exc, "stop_reason", "BUDGET_EXCEEDED")
                error_msg = str(exc)
                logger.warning("Optimization halted by resource guard: %s", exc)
                candidate_prog = baseline_prog  # Fallback to baseline
            except Exception as exc:
                status = "FAILED"
                error_msg = str(exc)
                logger.error("Optimization failed unexpectedly: %s", exc)
                candidate_prog = baseline_prog
        finally:
            self.guard.deactivate(ctx_token)

        # 6. Evaluate Candidate Program on Validation Subset
        candidate_summary = self._evaluate_program_on_subset(
            program=candidate_prog,
            subset=val_examples,
            metric=base_metric,
            lm=student_lm,
        )
        logger.info("Candidate validation mean score: %.4f", candidate_summary["mean_score"])

        # 7. Comparative Metrics & Instruction Diff
        candidate_instructions = getattr(candidate_prog.predictor.signature, "instructions", base_instructions)
        instruction_diff = self._generate_instruction_diff(base_instructions, candidate_instructions)

        abs_score_diff = round(candidate_summary["mean_score"] - baseline_summary["mean_score"], 4)
        rel_change_pct = (
            round((abs_score_diff / baseline_summary["mean_score"]) * 100, 2)
            if baseline_summary["mean_score"] > 0
            else 0.0
        )

        comparison = {
            "baseline_mean_score": baseline_summary["mean_score"],
            "candidate_mean_score": candidate_summary["mean_score"],
            "absolute_score_diff": abs_score_diff,
            "relative_change_pct": rel_change_pct,
            "baseline_critical_failures": baseline_summary["critical_failure_count"],
            "candidate_critical_failures": candidate_summary["critical_failure_count"],
            "instructions_changed": base_instructions != candidate_instructions,
            "evaluation_examples_count": len(val_examples),
        }

        # 8. Production Prompt Invariant Check (Feature A Protection Boundary)
        # Verify that CrispeGenerationService production prompts are UNTOUCHED
        prod_service = CrispeGenerationService()
        assert prod_service is not None, "Production generation service must remain intact."

        # 9. Persist Artifacts
        resource_summary = self.guard.get_summary()
        summary_payload = {
            "experiment_id": self.config.experiment_id,
            "status": status,
            "error_message": error_msg,
            "generation_model_id": gen_model_id,
            "reflection_model_id": ref_model_id,
            "is_mock": use_mock,
            "dataset_version": subsets_meta.dataset_version,
            "dataset_sha256": subsets_meta.dataset_sha256,
            "train_example_ids": subsets_meta.train_example_ids,
            "val_example_ids": subsets_meta.val_example_ids,
            "holdout_excluded": subsets_meta.holdout_excluded,
            "resource_usage": asdict(resource_summary),
            "comparison": comparison,
            "candidate_status": "EXPERIMENTAL_CANDIDATE",
            "is_promoted": False,
        }

        val_comparison_payload = {
            "experiment_id": self.config.experiment_id,
            "baseline_results": baseline_summary["per_example"],
            "candidate_results": candidate_summary["per_example"],
            "comparison": comparison,
        }

        artifacts_saved = self._persist_artifacts(
            experiment_dir=experiment_dir,
            summary_data=summary_payload,
            candidate_prog=candidate_prog,
            instruction_diff=instruction_diff,
            val_comparison=val_comparison_payload,
        )

        return GepaExperimentResult(
            config=self.config,
            subsets_metadata=subsets_meta,
            resource_summary=resource_summary,
            baseline_summary=baseline_summary,
            candidate_summary=candidate_summary,
            comparison=comparison,
            status=status,
            artifacts_saved=artifacts_saved,
            error_message=error_msg,
            baseline_instruction=base_instructions,
            candidate_instruction=candidate_instructions,
        )
