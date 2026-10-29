"""Prompt optimization evaluation, deterministic quality assessment, and model-assisted judging (Step 21)."""

from app.evaluation.baseline_evaluator import (
    BaselineEvaluator,
)
from app.evaluation.baseline_runner import (
    BaselineExampleResult,
    BaselineExperimentSummary,
    BaselineRunner,
)
from app.evaluation.calibration import (
    CalibrationProvenance,
    JudgeAgreementMetrics,
    JudgeCalibrationExample,
    compute_agreement_metrics,
    load_calibration_dataset,
)
from app.evaluation.check_schemas import (
    CheckOutcome,
    CheckResult,
    CheckType,
    CrispeSectionTarget,
    DeterministicCheckDefinition,
    EvaluationDimension,
    EvaluationExampleResult,
    EvaluationSummaryReport,
)
from app.evaluation.dataset_accessor import (
    EvaluationDatasetAccessor,
    HoldoutAccessRestrictedError,
    get_dataset_accessor,
)
from app.evaluation.judge_prompt import (
    JUDGE_INSTRUCTION_VERSION,
    JUDGE_SYSTEM_INSTRUCTION,
    build_judge_user_prompt,
)
from app.evaluation.judge_rubric import (
    DEFAULT_RUBRIC_V1,
    DimensionCriterion,
    EvaluationRubric,
    ModelJudgeDimension,
    RubricScoreLevel,
    get_default_rubric,
)
from app.evaluation.judge_schemas import (
    CombinedEvaluationResult,
    DimensionEvaluation,
    ModelEvaluationResult,
    ModelJudgeResponse,
)
from app.evaluation.model_evaluator import (
    ModelPromptEvaluator,
)
from app.evaluation.rule_builder import (
    build_checks_for_example,
)
from app.evaluation.rule_engine import (
    execute_check,
    validate_crispe_structure,
    validate_safe_regex,
)
from app.evaluation.schemas import (
    EvaluationExample,
    TaskDifficulty,
    TaskType,
)
from app.evaluation.scoring_engine import (
    DeterministicScorer,
    ScoringPolicy,
)
from app.evaluation.split_manifest import (
    DEFAULT_MANIFEST_PATH,
    PartitionName,
    SplitManifest,
    SplitPartitionSummary,
    compute_file_sha256,
    create_stratified_split,
    load_manifest,
    save_manifest,
    validate_split_manifest,
)
from app.evaluation.validator import (
    DEFAULT_DATASET_PATH,
    DatasetValidationReport,
    load_dataset,
    print_report,
    validate_dataset_file,
)

__all__ = [
    # Schemas & Types
    "EvaluationExample",
    "TaskType",
    "TaskDifficulty",
    "load_dataset",
    "validate_dataset_file",
    "DatasetValidationReport",
    "print_report",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MANIFEST_PATH",
    # Partitions & Manifest
    "PartitionName",
    "SplitPartitionSummary",
    "SplitManifest",
    "compute_file_sha256",
    "create_stratified_split",
    "validate_split_manifest",
    "load_manifest",
    "save_manifest",
    "HoldoutAccessRestrictedError",
    "EvaluationDatasetAccessor",
    "get_dataset_accessor",
    # Baseline Runner
    "BaselineExampleResult",
    "BaselineExperimentSummary",
    "BaselineRunner",
    # Deterministic Checks & Scoring (Step 20)
    "CheckType",
    "CheckOutcome",
    "CrispeSectionTarget",
    "EvaluationDimension",
    "DeterministicCheckDefinition",
    "CheckResult",
    "EvaluationExampleResult",
    "EvaluationSummaryReport",
    "validate_crispe_structure",
    "validate_safe_regex",
    "execute_check",
    "build_checks_for_example",
    "ScoringPolicy",
    "DeterministicScorer",
    "BaselineEvaluator",
    # Model-Assisted Evaluation (Step 21)
    "ModelJudgeDimension",
    "RubricScoreLevel",
    "DimensionCriterion",
    "EvaluationRubric",
    "DEFAULT_RUBRIC_V1",
    "get_default_rubric",
    "DimensionEvaluation",
    "ModelJudgeResponse",
    "ModelEvaluationResult",
    "CombinedEvaluationResult",
    "ModelPromptEvaluator",
    "JUDGE_SYSTEM_INSTRUCTION",
    "JUDGE_INSTRUCTION_VERSION",
    "build_judge_user_prompt",
    "CalibrationProvenance",
    "JudgeCalibrationExample",
    "JudgeAgreementMetrics",
    "compute_agreement_metrics",
    "load_calibration_dataset",
]
