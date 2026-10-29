"""GEPA Metric Adapter for CRISPE Prompt Evaluation (Step 23).

Bridges the deterministic (Step 20) and model-assisted (Step 21) evaluation engines
with DSPy's GEPA reflective prompt optimization loop.

Implements the GEPAFeedbackMetric protocol:
(gold, pred, trace, pred_name, pred_trace) -> ScoreWithFeedback
"""

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Union
import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

from app.core.exceptions import PromptValidationError
from app.core.logging import get_logger
from app.dspy_modules.program import CrispeGenerationProgram
from app.evaluation.check_schemas import CheckOutcome, EvaluationExampleResult
from app.evaluation.judge_schemas import CombinedEvaluationResult, ModelEvaluationResult
from app.evaluation.model_evaluator import ModelPromptEvaluator
from app.evaluation.schemas import EvaluationExample
from app.evaluation.scoring_engine import DeterministicScorer
from app.schemas.crispe import CrispeSections

logger = get_logger("gepa_metric")


def _run_async_in_worker_thread(coro) -> Any:
    """Execute an asynchronous coroutine safely in a worker thread.

    Prevents 'RuntimeError: This event loop is already running' when invoked
    from within an existing async event loop during GEPA optimization.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop in current thread
        return asyncio.run(coro)

    # Event loop is already running in active thread -> run in dedicated background thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro))
        return future.result()


class CrispeGepaMetric:
    """GEPA-compatible evaluation metric providing composite scores and actionable reflection feedback.

    Satisfies the dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric interface.
    Accepts 5 positional arguments: (gold, pred, trace, pred_name, pred_trace).
    Returns ScoreWithFeedback(score: float, feedback: str, objective_scores: dict).
    """

    def __init__(
        self,
        deterministic_scorer: Optional[DeterministicScorer] = None,
        model_evaluator: Optional[ModelPromptEvaluator] = None,
        enable_model_judge: bool = False,
        program: Optional[CrispeGenerationProgram] = None,
    ) -> None:
        """Initialize the GEPA CRISPE evaluation metric.

        Parameters:
            deterministic_scorer: Scorer instance for machine-checkable rules.
            model_evaluator: Optional LLM-as-a-judge evaluator for semantic dimensions.
            enable_model_judge: Whether to execute model evaluation (consumes model calls).
            program: Optional program for canonical CrispeSections validation.
        """
        self.deterministic_scorer = deterministic_scorer or DeterministicScorer()
        self.model_evaluator = model_evaluator
        self.enable_model_judge = enable_model_judge
        self.program = program or CrispeGenerationProgram()

    def _resolve_evaluation_example(self, gold: Any) -> Optional[EvaluationExample]:
        """Extract the canonical EvaluationExample from the gold example or container."""
        if isinstance(gold, EvaluationExample):
            return gold

        # Check for attached evaluation record
        for attr in ("_evaluation_example", "evaluation_example"):
            val = getattr(gold, attr, None)
            if isinstance(val, EvaluationExample):
                return val

        # If gold is a dict or Example with fields, try to construct or resolve
        if isinstance(gold, dspy.Example):
            data = gold.toDict() if hasattr(gold, "toDict") else {}
            if "_evaluation_example" in data and isinstance(data["_evaluation_example"], EvaluationExample):
                return data["_evaluation_example"]
            if "evaluation_example" in data and isinstance(data["evaluation_example"], EvaluationExample):
                return data["evaluation_example"]

        return None

    def _extract_crispe_sections(
        self, pred: Any
    ) -> tuple[Optional[CrispeSections], Optional[str]]:
        """Extract and validate CrispeSections from the DSPy prediction.

        Returns:
            Tuple of (CrispeSections or None, error_message or None).
        """
        if isinstance(pred, CrispeSections):
            return pred, None

        if isinstance(pred, dspy.Prediction):
            try:
                sections = self.program.validate_and_extract_sections(pred)
                return sections, None
            except PromptValidationError as exc:
                return None, f"Schema validation error: {str(exc)}"
            except Exception as exc:
                return None, f"Unexpected extraction error: {str(exc)}"

        if isinstance(pred, dict):
            try:
                sections = CrispeSections.model_validate(pred)
                return sections, None
            except Exception as exc:
                return None, f"Dict validation error: {str(exc)}"

        return None, f"Unsupported prediction type: {type(pred).__name__}"

    def _build_actionable_feedback(
        self,
        det_result: EvaluationExampleResult,
        model_result: Optional[ModelEvaluationResult] = None,
        extraction_error: Optional[str] = None,
    ) -> str:
        """Formulate prioritized, diagnostic reflection feedback for GEPA's instruction proposer."""
        feedback_items: List[str] = []

        if extraction_error:
            feedback_items.append(f"[SCHEMA FAILURE] {extraction_error}. Ensure all six CRISPE sections are outputted.")

        # 1. Critical Failures (Highest Priority)
        if det_result.critical_failure:
            feedback_items.append("[CRITICAL FAILURE] The generated prompt violated essential safety or task constraints.")

        # 2. Failed Deterministic Checks
        for check in det_result.checks:
            if check.outcome == CheckOutcome.FAILED:
                dim_str = getattr(check.dimension, "value", str(check.dimension))
                category_label = dim_str.upper()
                msg = f"[{category_label}] Check '{check.check_id}' failed: {check.diagnostic_message or 'Verification rule not met'}"
                feedback_items.append(msg)

        # 3. Model-assisted Judge Feedback (if available)
        if model_result and model_result.dimensions:
            for dim_name, dim in model_result.dimensions.items():
                if dim.score is not None and dim.score <= 2:  # Below acceptable threshold (out of 4)
                    feedback_items.append(
                        f"[JUDGE WEAKNESS - {dim_name} ({dim.score}/4)]: {dim.rationale}"
                    )

        if not feedback_items:
            if det_result.composite_score >= 0.95:
                return "Excellent CRISPE prompt. All deterministic constraints and quality specifications are fully satisfied."
            return f"Prompt generation acceptable with score {det_result.composite_score:.2f}."

        return "\n".join(feedback_items)

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Optional[Any] = None,
        pred_name: Optional[str] = None,
        pred_trace: Optional[Any] = None,
        program_trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ScoreWithFeedback:
        """Compute evaluation score and reflection feedback for GEPA optimizer.

        Parameters:
            gold: dspy.Example containing generator inputs and evaluation metadata.
            pred: dspy.Prediction returned by CrispeGenerationProgram.
            trace: Optional DSPy execution trace.
            pred_name: Target predictor identifier currently optimized by GEPA.
            pred_trace: Sub-trace for the active predictor.
            program_trace: Optional full program trace.

        Returns:
            ScoreWithFeedback with normalized composite score, diagnostic feedback,
            and component objective scores.
        """
        example = self._resolve_evaluation_example(gold)
        if not example:
            logger.error("CrispeGepaMetric: Cannot resolve EvaluationExample from gold example.")
            return ScoreWithFeedback(
                score=0.0,
                feedback="Evaluation failed: gold example does not contain a valid EvaluationExample reference.",
                objective_scores={"deterministic": 0.0},
            )

        # 1. Extract CRISPE sections from prediction
        sections, extraction_error = self._extract_crispe_sections(pred)
        generation_status = "SUCCESS" if sections else "FAILED"

        # 2. Deterministic Evaluation (Step 20)
        try:
            det_res = self.deterministic_scorer.evaluate_example(
                example=example,
                sections=sections,
                raw_text=sections.to_markdown() if sections else None,
                generation_status=generation_status,
                generation_error=extraction_error,
            )
        except Exception as exc:
            logger.error("DeterministicScorer failed during GEPA metric: %s", exc)
            return ScoreWithFeedback(
                score=0.0,
                feedback=f"Deterministic evaluation error: {str(exc)}",
                objective_scores={"deterministic": 0.0},
            )

        composite_score = det_res.composite_score
        objective_scores: Dict[str, float] = {"deterministic": composite_score}
        model_result: Optional[ModelEvaluationResult] = None

        # 3. Optional Model-Assisted Evaluation (Step 21)
        if self.enable_model_judge and self.model_evaluator and sections:
            try:
                model_result = _run_async_in_worker_thread(
                    self.model_evaluator.evaluate_prompt(example, sections)
                )
                if model_result and model_result.normalized_model_score is not None:
                    combined_res: CombinedEvaluationResult = self.model_evaluator.combine_results(
                        deterministic_result=det_res,
                        model_result=model_result,
                        example_id=example.example_id,
                        dataset_version=example.example_version,
                    )
                    composite_score = combined_res.composite_score
                    objective_scores["model"] = model_result.normalized_model_score
                    objective_scores["combined"] = combined_res.composite_score
            except Exception as exc:
                logger.warning("Model judge evaluation failed during GEPA metric (preserving det score): %s", exc)
                objective_scores["model_error"] = 1.0

        # Enforce critical failure cap
        if det_res.critical_failure:
            composite_score = min(composite_score, 0.40)

        # 4. Formulate actionable diagnostic feedback
        feedback = self._build_actionable_feedback(
            det_result=det_res,
            model_result=model_result,
            extraction_error=extraction_error,
        )

        return ScoreWithFeedback(
            score=round(composite_score, 4),
            feedback=feedback,
            objective_scores=objective_scores,
        )


def crispe_gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None,
    **kwargs: Any,
) -> ScoreWithFeedback:
    """Convenience functional metric conforming to the GEPAFeedbackMetric interface."""
    metric = CrispeGepaMetric()
    return metric(gold, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace, **kwargs)
