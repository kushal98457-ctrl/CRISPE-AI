"""Evaluation interface adapter connecting DSPy programs to deterministic and model evaluators (Step 22).

Extracts clean generator inputs from EvaluationExample records while strictly guarding
hidden evaluation criteria, and feeds DSPy predictions to Steps 20 & 21 evaluators.
"""

from typing import Dict, Optional
import dspy

from app.core.logging import get_logger
from app.dspy_modules.program import CrispeGenerationProgram
from app.evaluation.check_schemas import EvaluationExampleResult
from app.evaluation.judge_schemas import CombinedEvaluationResult
from app.evaluation.model_evaluator import ModelPromptEvaluator
from app.evaluation.schemas import EvaluationExample
from app.evaluation.scoring_engine import DeterministicScorer
from app.schemas.crispe import CrispeSections

logger = get_logger("dspy_evaluation_adapter")


class DspyEvaluationAdapter:
    """Adapter bridging DSPy predictions and evaluation benchmark datasets."""

    @staticmethod
    def prepare_generator_inputs(example: EvaluationExample) -> Dict[str, str]:
        """Extract only permissible generator input fields from an evaluation benchmark record.

        CRITICAL SECURITY & EVALUATION BOUNDARY:
        Never reveals acceptance_criteria, critical_requirements, prohibited_behaviors,
        or grading_notes to the generator.
        """
        constraints_str = "; ".join(example.constraints) if example.constraints else ""
        return {
            "idea": example.user_idea,
            "task_context": example.task_context,
            "constraints": constraints_str,
            "output_format_preference": example.expected_output_format,
        }

    @staticmethod
    async def evaluate_prediction(
        example: EvaluationExample,
        prediction: dspy.Prediction,
        program: Optional[CrispeGenerationProgram] = None,
        deterministic_scorer: Optional[DeterministicScorer] = None,
        model_evaluator: Optional[ModelPromptEvaluator] = None,
        enable_model_judge: bool = False,
    ) -> CombinedEvaluationResult:
        """Evaluate a generated DSPy prediction using existing deterministic and model evaluators.

        Parameters:
            example: Benchmark record from evaluation dataset.
            prediction: Raw DSPy prediction returned by CrispeGenerationProgram.
            program: Optional program instance to validate prediction.
            deterministic_scorer: Scorer instance from Step 20.
            model_evaluator: Evaluator instance from Step 21.
            enable_model_judge: Whether to execute model-assisted judge evaluation.

        Returns:
            CombinedEvaluationResult containing separate deterministic and model metrics.
        """
        prog = program or CrispeGenerationProgram()
        scorer = deterministic_scorer or DeterministicScorer()

        # 1. Attempt canonical CRISPE extraction
        sections: Optional[CrispeSections] = None
        generation_status = "SUCCESS"
        generation_error: Optional[str] = None

        try:
            sections = prog.validate_and_extract_sections(prediction)
        except Exception as exc:
            generation_status = "FAILED"
            generation_error = str(exc)
            logger.warning("DSPy prediction failed extraction for '%s': %s", example.example_id, exc)

        # 2. Execute Deterministic Evaluation
        det_res = scorer.evaluate_example(
            example=example,
            sections=sections,
            raw_text=sections.to_markdown() if sections else None,
            generation_status=generation_status,
            generation_error=generation_error,
        )

        # 3. If model evaluator is enabled and sections are valid, run combined evaluation
        judge = model_evaluator or ModelPromptEvaluator()
        return await judge.evaluate_combined(
            example=example,
            sections=sections,
            raw_text=sections.to_markdown() if sections else None,
            deterministic_result=det_res,
            enable_model_judge=enable_model_judge,
        )
