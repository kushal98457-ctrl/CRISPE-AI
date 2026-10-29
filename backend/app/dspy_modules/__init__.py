"""DSPy and GEPA integration package for programmable CRISPE prompt generation (Steps 22, 23 & 24)."""

from app.dspy_modules.budget_guard import (
    BudgetExceededError,
    OptimizationTimeoutError,
    ResourceBudgetGuard,
    ResourceUsageSummary,
)
from app.dspy_modules.dataset_converter import (
    convert_evaluation_example,
    convert_evaluation_examples,
    load_holdout_dspy_examples,
    load_optimization_dspy_examples,
    load_validation_dspy_examples,
)
from app.dspy_modules.evaluation_adapter import (
    DspyEvaluationAdapter,
)
from app.dspy_modules.experiment_config import (
    GepaExperimentConfig,
)
from app.dspy_modules.experiment_runner import (
    GepaExperimentResult,
    GepaExperimentRunner,
)
from app.dspy_modules.gepa_metric import (
    CrispeGepaMetric,
    crispe_gepa_metric,
)
from app.dspy_modules.model_client import (
    create_dspy_lm,
    create_dspy_reflection_lm,
    dspy_isolated_context,
    normalize_dspy_model_name,
)
from app.dspy_modules.program import (
    CrispeGenerationProgram,
    create_crispe_program,
)
from app.dspy_modules.service import (
    DspyCrispeGenerationService,
)
from app.dspy_modules.signature import (
    CrispeGenerationSignature,
)
from app.dspy_modules.subset_selector import (
    PilotSubsetSelector,
    PilotSubsetsMetadata,
)

__all__ = [
    "CrispeGenerationSignature",
    "CrispeGenerationProgram",
    "create_crispe_program",
    "normalize_dspy_model_name",
    "create_dspy_lm",
    "create_dspy_reflection_lm",
    "dspy_isolated_context",
    "DspyCrispeGenerationService",
    "DspyEvaluationAdapter",
    "convert_evaluation_example",
    "convert_evaluation_examples",
    "load_optimization_dspy_examples",
    "load_validation_dspy_examples",
    "load_holdout_dspy_examples",
    "CrispeGepaMetric",
    "crispe_gepa_metric",
    "ResourceBudgetGuard",
    "ResourceUsageSummary",
    "BudgetExceededError",
    "OptimizationTimeoutError",
    "PilotSubsetSelector",
    "PilotSubsetsMetadata",
    "GepaExperimentConfig",
    "GepaExperimentRunner",
    "GepaExperimentResult",
]
