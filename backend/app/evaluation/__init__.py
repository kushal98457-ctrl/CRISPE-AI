"""Prompt optimization evaluation, deterministic quality assessment, and baselining (Step 20)."""

from app.evaluation.baseline_evaluator import (
    BaselineEvaluator,
)
from app.evaluation.baseline_runner import (
    BaselineExampleResult,
    BaselineExperimentSummary,
    BaselineRunner,
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
]
