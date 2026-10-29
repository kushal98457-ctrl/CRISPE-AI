"""Prompt optimization evaluation dataset, partitioning, and baseline experiment package (Step 19)."""

from app.evaluation.baseline_runner import (
    BaselineExampleResult,
    BaselineExperimentSummary,
    BaselineRunner,
)
from app.evaluation.dataset_accessor import (
    EvaluationDatasetAccessor,
    HoldoutAccessRestrictedError,
    get_dataset_accessor,
)
from app.evaluation.schemas import (
    EvaluationExample,
    TaskDifficulty,
    TaskType,
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
    "EvaluationExample",
    "TaskType",
    "TaskDifficulty",
    "load_dataset",
    "validate_dataset_file",
    "DatasetValidationReport",
    "print_report",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MANIFEST_PATH",
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
    "BaselineExampleResult",
    "BaselineExperimentSummary",
    "BaselineRunner",
]
