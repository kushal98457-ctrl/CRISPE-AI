"""Dataset conversion adapter bridging CRISPE evaluation partitions to DSPy Examples (Step 23).

Transforms EvaluationExample records into typed dspy.Example instances for DSPy and GEPA
workflows while enforcing strict partition boundaries and preventing leakage of evaluator guidance.
"""

from typing import List, Optional, Sequence
import dspy

from app.core.logging import get_logger
from app.evaluation.dataset_accessor import (
    EvaluationDatasetAccessor,
    HoldoutAccessRestrictedError,
)
from app.evaluation.schemas import EvaluationExample

logger = get_logger("dspy_dataset_converter")


# Generator input fields exposed to the DSPy signature
DSPY_GENERATOR_INPUT_FIELDS = (
    "idea",
    "task_context",
    "constraints",
    "output_format_preference",
)


def convert_evaluation_example(example: EvaluationExample) -> dspy.Example:
    """Convert a single EvaluationExample into a typed dspy.Example.

    Security & Evaluation Leakage Guarantees:
    1. Generator inputs are designated strictly using .with_inputs(...).
    2. Hidden evaluator guidance (acceptance_criteria, critical_requirements,
       prohibited_behaviors, grading_notes) are stored as evaluation metadata for
       the metric, and NEVER appear in the predictor's input dictionary.
    3. The complete EvaluationExample is preserved under '_evaluation_example' and
       'evaluation_example' for evaluator access.

    Parameters:
        example: Canonical benchmark EvaluationExample record.

    Returns:
        Configured dspy.Example with designated inputs.
    """
    constraints_str = "; ".join(example.constraints) if example.constraints else ""

    dspy_ex = dspy.Example(
        # Generation inputs
        idea=example.user_idea,
        task_context=example.task_context,
        constraints=constraints_str,
        output_format_preference=example.expected_output_format,
        # Evaluation metadata (labels / ground truth context for the metric)
        example_id=example.example_id,
        task_family=example.task_family or "",
        task_type=str(example.task_type),
        difficulty=str(example.difficulty),
        evaluation_example=example,
        _evaluation_example=example,
    ).with_inputs(*DSPY_GENERATOR_INPUT_FIELDS)

    return dspy_ex


def convert_evaluation_examples(
    examples: Sequence[EvaluationExample],
) -> List[dspy.Example]:
    """Convert a sequence of EvaluationExample records into dspy.Example objects.

    Parameters:
        examples: Sequence of evaluation records.

    Returns:
        List of converted dspy.Example instances.
    """
    return [convert_evaluation_example(ex) for ex in examples]


def load_optimization_dspy_examples(
    accessor: Optional[EvaluationDatasetAccessor] = None,
) -> List[dspy.Example]:
    """Load optimization partition examples (~60%) as dspy.Example records.

    Safe for DSPy / GEPA prompt tuning and candidate proposal.
    """
    ds_accessor = accessor or EvaluationDatasetAccessor()
    raw_examples = ds_accessor.get_optimization_examples()
    logger.info(
        "Converted %d optimization examples for DSPy/GEPA optimization",
        len(raw_examples),
    )
    return convert_evaluation_examples(raw_examples)


def load_validation_dspy_examples(
    accessor: Optional[EvaluationDatasetAccessor] = None,
) -> List[dspy.Example]:
    """Load validation partition examples (~20%) as dspy.Example records.

    Safe for candidate comparison, hyperparameter selection, and evaluation.
    """
    ds_accessor = accessor or EvaluationDatasetAccessor()
    raw_examples = ds_accessor.get_validation_examples()
    logger.info(
        "Converted %d validation examples for DSPy/GEPA candidate evaluation",
        len(raw_examples),
    )
    return convert_evaluation_examples(raw_examples)


def load_holdout_dspy_examples(
    accessor: Optional[EvaluationDatasetAccessor] = None,
    authorized: bool = False,
    reason: Optional[str] = None,
) -> List[dspy.Example]:
    """Load final holdout partition examples (~20%) as dspy.Example records.

    SEALED HOLDOUT ACCESS GUARD:
    Holdout examples must never enter routine optimization or tuning loops.
    Raises HoldoutAccessRestrictedError if authorized is not True.

    Parameters:
        accessor: Dataset accessor instance.
        authorized: Must be explicitly set to True.
        reason: Optional audit justification string.

    Returns:
        List of holdout dspy.Example instances.

    Raises:
        HoldoutAccessRestrictedError: If authorized is False.
    """
    if not authorized:
        logger.warning("Blocked unauthorized attempt to convert holdout examples for DSPy.")
        raise HoldoutAccessRestrictedError()

    ds_accessor = accessor or EvaluationDatasetAccessor()
    raw_examples = ds_accessor.get_holdout_examples(authorized=authorized, reason=reason)
    logger.info("Converted %d holdout examples under authorized audit", len(raw_examples))
    return convert_evaluation_examples(raw_examples)
