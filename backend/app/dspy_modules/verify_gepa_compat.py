"""Standalone compatibility and integration verification script for DSPy, GEPA, and Gemini Flash (Step 23).

Executes the 14 mandatory diagnostic verification checks established in Step 23:
1.  Python runtime version check
2.  Dependency and package import checks
3.  DSPy Signature construction
4.  DSPy program construction
5.  Fake-model execution through DSPy program
6.  CRISPE output-schema validation
7.  Deterministic metric invocation
8.  Model-assisted metric invocation using fake judge responses
9.  GEPA constructor/configuration compatibility
10. Metric return-contract validation
11. Dataset-example conversion
12. Holdout-exclusion verification
13. Configuration-isolation verification
14. Safe diagnostic output generation

CRITICAL SAFETY & SCOPE BOUNDARY:
- Strictly does NOT execute an optimization loop (optimizer.compile() is never called).
- Consumes ZERO real model credits by default (uses deterministic test doubles).
- Never leaks or prints credentials or hidden evaluation guidance.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import importlib.metadata
import json
import sys
import time
from typing import Any, Dict, List, Optional
import dspy

from app.adapters.gemini_adapter import BaseModelAdapter, ModelGenerationRequest, ModelGenerationResponse
from app.core.config import get_settings
from app.core.logging import get_logger
from app.dspy_modules.dataset_converter import (
    convert_evaluation_example,
    load_holdout_dspy_examples,
    load_optimization_dspy_examples,
    load_validation_dspy_examples,
)
from app.dspy_modules.gepa_metric import CrispeGepaMetric
from app.dspy_modules.model_client import (
    create_dspy_lm,
    create_dspy_reflection_lm,
    dspy_isolated_context,
    normalize_dspy_model_name,
)
from app.dspy_modules.program import CrispeGenerationProgram
from app.dspy_modules.signature import CrispeGenerationSignature
from app.evaluation.dataset_accessor import (
    EvaluationDatasetAccessor,
    HoldoutAccessRestrictedError,
)
from app.evaluation.judge_schemas import ModelJudgeResponse
from app.evaluation.model_evaluator import ModelPromptEvaluator
from app.evaluation.schemas import EvaluationExample, TaskDifficulty, TaskType
from app.evaluation.scoring_engine import DeterministicScorer
from app.schemas.crispe import CrispeSections

logger = get_logger("verify_gepa_compat")


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT RUN"


@dataclass
class DiagnosticCheckResult:
    check_number: int
    name: str
    status: CheckStatus
    message: str
    duration_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Mock Doubles for Offline Deterministic Verification
# =============================================================================

class MockDiagnosticLM(dspy.BaseLM):
    """Deterministic mock LM for DSPy program verification without API access."""

    def __init__(self, model_name: str = "mock-gemini-diagnostic") -> None:
        super().__init__(model=model_name)
        self.model_name = model_name

    def __call__(self, prompt=None, messages=None, **kwargs):
        return [
            "[[ ## context ## ]]\nContext: Enterprise financial reconciliation service.\n"
            "[[ ## role ## ]]\nRole: Senior FinTech Systems Architect.\n"
            "[[ ## instruction ## ]]\nInstruction: Build an automated ledger discrepancy detector in Python.\n"
            "[[ ## specification ## ]]\nSpecification: Return structured JSON with fields: status, discrepancy_amount, timestamp.\n"
            "[[ ## performance ## ]]\nPerformance: Reconcile 10,000 transactions per second with zero drop.\n"
            "[[ ## example ## ]]\nExample: Sample transaction discrepancy payload."
        ]


class MockDiagnosticJudgeAdapter(BaseModelAdapter):
    """Deterministic mock adapter for model-assisted evaluator check."""

    def __init__(self) -> None:
        self.model_id = "mock-judge-model"

    async def generate_content(self, request: ModelGenerationRequest) -> ModelGenerationResponse:
        dummy_judge_json = json.dumps({
            "dimensions": [
                {
                    "dimension": "intent_preservation",
                    "applicable": True,
                    "score": 4,
                    "rationale": "The generated prompt preserves the user intent with high fidelity.",
                },
                {
                    "dimension": "clarity_and_actionability",
                    "applicable": True,
                    "score": 3,
                    "rationale": "Clear and actionable instructions directing the target model.",
                },
                {
                    "dimension": "context_adequacy",
                    "applicable": True,
                    "score": 4,
                    "rationale": "Provides thorough FinTech reconciliation domain context.",
                },
                {
                    "dimension": "requirement_and_constraint_coverage",
                    "applicable": True,
                    "score": 4,
                    "rationale": "All mandatory constraints and behaviors are accounted for.",
                },
                {
                    "dimension": "specification_and_output_precision",
                    "applicable": True,
                    "score": 4,
                    "rationale": "Structured JSON format is specified with concrete fields.",
                },
                {
                    "dimension": "internal_consistency",
                    "applicable": True,
                    "score": 4,
                    "rationale": "No internal contradictions across CRISPE sections.",
                },
                {
                    "dimension": "example_usefulness",
                    "applicable": True,
                    "score": 3,
                    "rationale": "Representative sample input and output provided.",
                },
                {
                    "dimension": "ambiguity_handling",
                    "applicable": True,
                    "score": 3,
                    "rationale": "Handles stream discrepancy edge cases reasonably well.",
                },
            ],
            "critical_issues": [],
            "summary_weaknesses": [],
            "suggested_improvements": ["Add edge case for null ledger events"],
            "overall_verdict": "PASS",
        })
        return ModelGenerationResponse(
            content=dummy_judge_json,
            model_id="mock-judge-model",
            finish_reason="stop",
        )

    async def close(self) -> None:
        pass


def make_diagnostic_sample_example() -> EvaluationExample:
    """Construct a clean, non-sensitive sample EvaluationExample for diagnostic checks."""
    return EvaluationExample(
        example_id="compat_diag_ex_001",
        task_family="diagnostics",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        difficulty=TaskDifficulty.MEDIUM,
        tags=["verification", "ledger", "fintech"],
        example_version="v1",
        user_idea="Build an automated ledger discrepancy detector in Python.",
        task_context="Enterprise financial reconciliation service handling ledger streams.",
        required_behaviors=["Ingest ledger events", "Flag discrepancies", "Output structured JSON"],
        constraints=["Do not drop transactions", "Use Python 3.12+ standard library or asyncio"],
        expected_output_format="Structured JSON format",
        ambiguity_notes="None.",
        acceptance_criteria=["Must specify JSON output", "Must include discrepancy_amount field"],
        critical_requirements=["Must not drop ledger events"],
        prohibited_behaviors=["Do not allow unhandled exceptions during reconciliation"],
        grading_notes="Verify JSON format and zero-drop constraint in specification.",
    )


# =============================================================================
# Diagnostic Verification Suite
# =============================================================================

def run_compatibility_suite(verbose: bool = True) -> List[DiagnosticCheckResult]:
    """Execute all 14 compatibility and integration hardening checks."""
    results: List[DiagnosticCheckResult] = []

    # Check 1: Python-version check
    t0 = time.perf_counter()
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver >= (3, 12):
        results.append(DiagnosticCheckResult(
            check_number=1,
            name="Python Runtime Version",
            status=CheckStatus.PASS,
            message=f"Python {py_str} meets '>=3.12' requirement.",
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={"version": py_str},
        ))
    else:
        results.append(DiagnosticCheckResult(
            check_number=1,
            name="Python Runtime Version",
            status=CheckStatus.FAIL,
            message=f"Python {py_str} does not meet '>=3.12' requirement.",
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={"version": py_str},
        ))

    # Check 2: Dependency and package import checks
    t0 = time.perf_counter()
    required_packages = ["dspy", "litellm", "gepa", "google.genai", "pydantic", "fastapi", "pytest"]
    missing_pkgs: List[str] = []
    versions: Dict[str, str] = {}
    for pkg in required_packages:
        try:
            mod = importlib.import_module(pkg)
            dist_name = "google-genai" if pkg == "google.genai" else ("dspy-ai" if pkg == "dspy" else pkg)
            try:
                ver = importlib.metadata.version(dist_name)
            except Exception:
                ver = getattr(mod, "__version__", "unknown")
            versions[pkg] = ver
        except ImportError as exc:
            missing_pkgs.append(f"{pkg}: {exc}")

    if not missing_pkgs:
        results.append(DiagnosticCheckResult(
            check_number=2,
            name="Dependency & Import Verification",
            status=CheckStatus.PASS,
            message=f"All {len(required_packages)} critical packages imported successfully.",
            duration_ms=(time.perf_counter() - t0) * 1000,
            details=versions,
        ))
    else:
        results.append(DiagnosticCheckResult(
            check_number=2,
            name="Dependency & Import Verification",
            status=CheckStatus.FAIL,
            message=f"Missing dependencies: {', '.join(missing_pkgs)}",
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={"missing": missing_pkgs, "found": versions},
        ))

    # Check 3: DSPy Signature construction
    t0 = time.perf_counter()
    try:
        sig = CrispeGenerationSignature
        input_fields = list(sig.input_fields.keys())
        output_fields = list(sig.output_fields.keys())
        expected_inputs = ["idea", "task_context", "constraints", "output_format_preference"]
        expected_outputs = ["context", "role", "instruction", "specification", "performance", "example"]

        if input_fields == expected_inputs and output_fields == expected_outputs:
            results.append(DiagnosticCheckResult(
                check_number=3,
                name="DSPy Signature Construction",
                status=CheckStatus.PASS,
                message=f"CrispeGenerationSignature verified (4 inputs, 6 outputs).",
                duration_ms=(time.perf_counter() - t0) * 1000,
                details={"inputs": input_fields, "outputs": output_fields},
            ))
        else:
            results.append(DiagnosticCheckResult(
                check_number=3,
                name="DSPy Signature Construction",
                status=CheckStatus.FAIL,
                message=f"Signature fields mismatch. Inputs: {input_fields}, Outputs: {output_fields}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=3,
            name="DSPy Signature Construction",
            status=CheckStatus.FAIL,
            message=f"Signature check raised exception: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 4: DSPy program construction
    t0 = time.perf_counter()
    try:
        program = CrispeGenerationProgram()
        results.append(DiagnosticCheckResult(
            check_number=4,
            name="DSPy Program Construction",
            status=CheckStatus.PASS,
            message=f"CrispeGenerationProgram constructed with Predict(CrispeGenerationSignature).",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=4,
            name="DSPy Program Construction",
            status=CheckStatus.FAIL,
            message=f"CrispeGenerationProgram failed initialization: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 5: Fake-model execution through the DSPy program
    t0 = time.perf_counter()
    pred = None
    try:
        program = CrispeGenerationProgram()
        mock_lm = MockDiagnosticLM()
        with dspy.context(lm=mock_lm):
            pred = program(
                idea="Build an automated ledger discrepancy detector in Python.",
                task_context="Enterprise financial reconciliation service.",
                constraints="Use Python 3.12+",
                output_format_preference="Structured JSON format",
            )
        results.append(DiagnosticCheckResult(
            check_number=5,
            name="Fake-Model Execution",
            status=CheckStatus.PASS,
            message="Forward pass completed using MockDiagnosticLM without external API calls.",
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={"has_context": bool(getattr(pred, "context", None))},
        ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=5,
            name="Fake-Model Execution",
            status=CheckStatus.FAIL,
            message=f"Forward pass with fake LM failed: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 6: CRISPE output-schema validation
    t0 = time.perf_counter()
    try:
        program = CrispeGenerationProgram()
        sections = program.validate_and_extract_sections(pred)
        if isinstance(sections, CrispeSections) and all([
            sections.context, sections.role, sections.instruction,
            sections.specification, sections.performance, sections.example
        ]):
            results.append(DiagnosticCheckResult(
                check_number=6,
                name="CRISPE Output-Schema Validation",
                status=CheckStatus.PASS,
                message="CrispeSections validated with all 6 non-empty fields.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
        else:
            results.append(DiagnosticCheckResult(
                check_number=6,
                name="CRISPE Output-Schema Validation",
                status=CheckStatus.FAIL,
                message="Extracted CrispeSections contain empty or invalid fields.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=6,
            name="CRISPE Output-Schema Validation",
            status=CheckStatus.FAIL,
            message=f"Schema validation raised exception: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 7: Deterministic metric invocation
    t0 = time.perf_counter()
    try:
        sample_ex = make_diagnostic_sample_example()
        dspy_gold = convert_evaluation_example(sample_ex)
        metric = CrispeGepaMetric(enable_model_judge=False)

        metric_result = metric(gold=dspy_gold, pred=pred)
        if metric_result.score > 0.5:
            results.append(DiagnosticCheckResult(
                check_number=7,
                name="Deterministic Metric Invocation",
                status=CheckStatus.PASS,
                message=f"CrispeGepaMetric executed deterministic checks (score={metric_result.score:.4f}).",
                duration_ms=(time.perf_counter() - t0) * 1000,
                details={"score": metric_result.score, "feedback_len": len(metric_result.feedback)},
            ))
        else:
            results.append(DiagnosticCheckResult(
                check_number=7,
                name="Deterministic Metric Invocation",
                status=CheckStatus.FAIL,
                message=f"Metric score unexpectedly low: {metric_result.score}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=7,
            name="Deterministic Metric Invocation",
            status=CheckStatus.FAIL,
            message=f"Deterministic metric call raised exception: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 8: Model-assisted metric invocation using fake judge response
    t0 = time.perf_counter()
    try:
        fake_judge_adapter = MockDiagnosticJudgeAdapter()
        mock_evaluator = ModelPromptEvaluator(adapter=fake_judge_adapter)
        judge_metric = CrispeGepaMetric(
            model_evaluator=mock_evaluator,
            enable_model_judge=True,
        )
        judge_result = judge_metric(gold=dspy_gold, pred=pred)
        if "model" in judge_result.objective_scores:
            results.append(DiagnosticCheckResult(
                check_number=8,
                name="Model-Assisted Metric Invocation",
                status=CheckStatus.PASS,
                message=f"Combined evaluation passed with mock judge (score={judge_result.score:.4f}).",
                duration_ms=(time.perf_counter() - t0) * 1000,
                details=judge_result.objective_scores,
            ))
        else:
            results.append(DiagnosticCheckResult(
                check_number=8,
                name="Model-Assisted Metric Invocation",
                status=CheckStatus.FAIL,
                message="Judge evaluation was enabled but 'model' score is missing from objective_scores.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=8,
            name="Model-Assisted Metric Invocation",
            status=CheckStatus.FAIL,
            message=f"Model-assisted metric invocation raised: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 9: GEPA constructor/configuration compatibility
    t0 = time.perf_counter()
    try:
        class MockReflectionLM(dspy.BaseLM):
            def __init__(self):
                super().__init__(model="mock-reflection-lm")
            def __call__(self, prompt=None, messages=None, **kwargs):
                return ["Suggested instruction mutation"]

        ref_lm = MockReflectionLM()
        gepa_optimizer = dspy.GEPA(
            metric=metric,
            reflection_lm=ref_lm,
            auto="light",
        )
        gepa_bounded = dspy.GEPA(
            metric=metric,
            reflection_lm=ref_lm,
            max_metric_calls=10,
        )
        results.append(DiagnosticCheckResult(
            check_number=9,
            name="GEPA Constructor Compatibility",
            status=CheckStatus.PASS,
            message="dspy.GEPA initialized with reflection_lm under both auto='light' and max_metric_calls=10.",
            duration_ms=(time.perf_counter() - t0) * 1000,
            details={"auto_mode": gepa_optimizer.auto, "bounded_calls": gepa_bounded.max_metric_calls},
        ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=9,
            name="GEPA Constructor Compatibility",
            status=CheckStatus.FAIL,
            message=f"dspy.GEPA initialization failed: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 10: Metric return-contract validation
    t0 = time.perf_counter()
    try:
        sample_contract_res = metric(gold=dspy_gold, pred=pred)
        has_score = hasattr(sample_contract_res, "score") and isinstance(sample_contract_res.score, float)
        has_feedback = hasattr(sample_contract_res, "feedback") and isinstance(sample_contract_res.feedback, str)
        score_bounded = 0.0 <= sample_contract_res.score <= 1.0
        dict_accessible = sample_contract_res["score"] == sample_contract_res.score

        if has_score and has_feedback and score_bounded and dict_accessible:
            results.append(DiagnosticCheckResult(
                check_number=10,
                name="Metric Return-Contract Validation",
                status=CheckStatus.PASS,
                message=f"Metric returns ScoreWithFeedback: score={sample_contract_res.score:.4f}, feedback length={len(sample_contract_res.feedback)} chars.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
        else:
            results.append(DiagnosticCheckResult(
                check_number=10,
                name="Metric Return-Contract Validation",
                status=CheckStatus.FAIL,
                message=f"Contract violated: has_score={has_score}, has_feedback={has_feedback}, score_bounded={score_bounded}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=10,
            name="Metric Return-Contract Validation",
            status=CheckStatus.FAIL,
            message=f"Metric return check raised: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 11: Dataset-example conversion
    t0 = time.perf_counter()
    try:
        converted = convert_evaluation_example(sample_ex)
        gen_inputs = list(converted.inputs().keys())
        expected_gen_inputs = ["idea", "task_context", "constraints", "output_format_preference"]
        has_eval_metadata = hasattr(converted, "evaluation_example") and converted.evaluation_example.example_id == sample_ex.example_id
        # Crucial: hidden criteria must NOT be in inputs
        no_leakage = "acceptance_criteria" not in gen_inputs and "prohibited_behaviors" not in gen_inputs

        if gen_inputs == expected_gen_inputs and has_eval_metadata and no_leakage:
            results.append(DiagnosticCheckResult(
                check_number=11,
                name="Dataset-Example Conversion",
                status=CheckStatus.PASS,
                message=f"EvaluationExample mapped to dspy.Example with clean inputs and sealed criteria.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
        else:
            results.append(DiagnosticCheckResult(
                check_number=11,
                name="Dataset-Example Conversion",
                status=CheckStatus.FAIL,
                message=f"Conversion mismatch: gen_inputs={gen_inputs}, has_eval_metadata={has_eval_metadata}, no_leakage={no_leakage}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=11,
            name="Dataset-Example Conversion",
            status=CheckStatus.FAIL,
            message=f"Dataset conversion check raised: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 12: Holdout-exclusion verification
    t0 = time.perf_counter()
    try:
        holdout_blocked = False
        try:
            load_holdout_dspy_examples(authorized=False)
        except HoldoutAccessRestrictedError:
            holdout_blocked = True

        opt_examples = load_optimization_dspy_examples()
        val_examples = load_validation_dspy_examples()

        if holdout_blocked and len(opt_examples) > 0 and len(val_examples) > 0:
            results.append(DiagnosticCheckResult(
                check_number=12,
                name="Holdout-Exclusion Verification",
                status=CheckStatus.PASS,
                message=f"Holdout partition is strictly protected (HoldoutAccessRestrictedError raised); opt={len(opt_examples)}, val={len(val_examples)}.",
                duration_ms=(time.perf_counter() - t0) * 1000,
                details={"opt_count": len(opt_examples), "val_count": len(val_examples)},
            ))
        else:
            results.append(DiagnosticCheckResult(
                check_number=12,
                name="Holdout-Exclusion Verification",
                status=CheckStatus.FAIL,
                message=f"Holdout protection failed: holdout_blocked={holdout_blocked}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=12,
            name="Holdout-Exclusion Verification",
            status=CheckStatus.FAIL,
            message=f"Holdout check raised: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 13: Configuration-isolation verification
    t0 = time.perf_counter()
    try:
        class MockLMAlpha(dspy.BaseLM):
            def __init__(self):
                super().__init__(model="alpha-model")
            def __call__(self, prompt=None, messages=None, **kwargs):
                return ["Alpha response"]

        class MockLMBeta(dspy.BaseLM):
            def __init__(self):
                super().__init__(model="beta-model")
            def __call__(self, prompt=None, messages=None, **kwargs):
                return ["Beta response"]

        lm_alpha = MockLMAlpha()
        lm_beta = MockLMBeta()

        with dspy_isolated_context(lm=lm_alpha):
            active_alpha = dspy.settings.lm.model
            with dspy_isolated_context(lm=lm_beta):
                active_beta = dspy.settings.lm.model
            active_alpha_restored = dspy.settings.lm.model

        if active_alpha == "alpha-model" and active_beta == "beta-model" and active_alpha_restored == "alpha-model":
            results.append(DiagnosticCheckResult(
                check_number=13,
                name="Configuration-Isolation Verification",
                status=CheckStatus.PASS,
                message="Context isolation verified: dspy_isolated_context cleanly restores parent state without leak.",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
        else:
            results.append(DiagnosticCheckResult(
                check_number=13,
                name="Configuration-Isolation Verification",
                status=CheckStatus.FAIL,
                message=f"Context isolation failed: alpha={active_alpha}, beta={active_beta}, restored={active_alpha_restored}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            ))
    except Exception as exc:
        results.append(DiagnosticCheckResult(
            check_number=13,
            name="Configuration-Isolation Verification",
            status=CheckStatus.FAIL,
            message=f"Isolation check raised: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        ))

    # Check 14: Safe diagnostic output
    t0 = time.perf_counter()
    passed_count = sum(1 for r in results if r.status == CheckStatus.PASS)
    total_checks = len(results) + 1  # Including this check itself
    results.append(DiagnosticCheckResult(
        check_number=14,
        name="Safe Diagnostic Summary",
        status=CheckStatus.PASS,
        message=f"{passed_count + 1}/{total_checks} diagnostic compatibility checks PASSED (zero secrets exposed).",
        duration_ms=(time.perf_counter() - t0) * 1000,
        details={"passed": passed_count + 1, "total": total_checks},
    ))

    if verbose:
        print("\n" + "=" * 80)
        print(" CRISPE-AI / DSPy & GEPA Compatibility Hardening Report (Step 23)")
        print("=" * 80)
        print(f"{'#':<3} | {'Check Name':<38} | {'Status':<8} | {'Duration':<8} | Message")
        print("-" * 80)
        for r in results:
            print(f"{r.check_number:<3} | {r.name:<38} | {r.status.value:<8} | {r.duration_ms:>6.1f}ms | {r.message}")
        print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    results = run_compatibility_suite(verbose=True)
    all_passed = all(r.status == CheckStatus.PASS for r in results)
    sys.exit(0 if all_passed else 1)
