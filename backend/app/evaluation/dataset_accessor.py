"""Dataset Accessor providing partition isolation and sealed holdout protection (Step 19).

Enforces strict partition boundaries:
- Optimization examples: available for DSPy/GEPA prompt tuning.
- Validation examples: available for candidate selection and hyperparameter evaluation.
- Final Holdout examples: sealed; blocked by default and accessible only with explicit authorization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from app.core.logging import get_logger
from app.evaluation.schemas import EvaluationExample
from app.evaluation.split_manifest import (
    DEFAULT_MANIFEST_PATH,
    PartitionName,
    SplitManifest,
    load_manifest,
)
from app.evaluation.validator import DEFAULT_DATASET_PATH, load_dataset

logger = get_logger("dataset_accessor")


class HoldoutAccessRestrictedError(PermissionError):
    """Raised when an unauthorized attempt is made to access the final holdout partition."""

    def __init__(
        self,
        message: str = (
            "Access to the final holdout partition is strictly restricted. "
            "Holdout examples are sealed and cannot be accessed during routine "
            "optimization or baseline runs without explicit authorization (authorized=True)."
        ),
    ) -> None:
        super().__init__(message)


class EvaluationDatasetAccessor:
    """Provides access to evaluation dataset examples with partition isolation and leakage protection."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.dataset_path = dataset_path or DEFAULT_DATASET_PATH
        self.manifest_path = manifest_path or DEFAULT_MANIFEST_PATH

        # Load dataset examples and manifest
        self._examples: List[EvaluationExample] = load_dataset(self.dataset_path)
        self._manifest: SplitManifest = load_manifest(self.manifest_path)

        # Index examples by ID
        self._id_to_example: Dict[str, EvaluationExample] = {
            ex.example_id: ex for ex in self._examples
        }

        # Cache partition ID sets
        self._opt_ids = set(self._manifest.partitions[PartitionName.OPTIMIZATION.value].example_ids)
        self._val_ids = set(self._manifest.partitions[PartitionName.VALIDATION.value].example_ids)
        self._hld_ids = set(self._manifest.partitions[PartitionName.FINAL_HOLDOUT.value].example_ids)

    @property
    def manifest(self) -> SplitManifest:
        """Return the active split manifest."""
        return self._manifest

    def get_optimization_examples(self) -> List[EvaluationExample]:
        """Retrieve examples assigned to the optimization partition (~60%).

        Safe for candidate generation, tuning, and prompt reflection.
        """
        return [self._id_to_example[eid] for eid in self._manifest.partitions[PartitionName.OPTIMIZATION.value].example_ids]

    def get_validation_examples(self) -> List[EvaluationExample]:
        """Retrieve examples assigned to the validation partition (~20%).

        Safe for candidate comparison, hyperparameter tuning, and operational baselining.
        """
        return [self._id_to_example[eid] for eid in self._manifest.partitions[PartitionName.VALIDATION.value].example_ids]

    def get_holdout_examples(
        self,
        authorized: bool = False,
        reason: Optional[str] = None,
    ) -> List[EvaluationExample]:
        """Retrieve examples assigned to the final holdout partition (~20%).

        Parameters:
            authorized: Must be explicitly passed as True. If False, raises HoldoutAccessRestrictedError.
            reason: Optional justification string for logging the access.

        Raises:
            HoldoutAccessRestrictedError: If authorized is False.
        """
        if not authorized:
            logger.warning("Unauthorized attempt to access final holdout partition blocked.")
            raise HoldoutAccessRestrictedError()

        logger.info("Authorized access to final holdout partition granted. Reason: %s", reason or "Explicit user request")
        return [self._id_to_example[eid] for eid in self._manifest.partitions[PartitionName.FINAL_HOLDOUT.value].example_ids]

    def get_partition_examples(
        self,
        partition: Union[PartitionName, str],
        allow_holdout: bool = False,
    ) -> List[EvaluationExample]:
        """Retrieve examples for a specific partition by name.

        Parameters:
            partition: Target partition name (optimization, validation, final_holdout).
            allow_holdout: Required to be True if partition is 'final_holdout'.
        """
        p_name = partition.value if isinstance(partition, PartitionName) else str(partition).lower()

        if p_name == PartitionName.OPTIMIZATION.value:
            return self.get_optimization_examples()
        elif p_name == PartitionName.VALIDATION.value:
            return self.get_validation_examples()
        elif p_name == PartitionName.FINAL_HOLDOUT.value:
            return self.get_holdout_examples(authorized=allow_holdout, reason="get_partition_examples")
        else:
            raise ValueError(
                f"Unknown partition '{partition}'. Must be one of: "
                f"{[p.value for p in PartitionName]}"
            )

    def get_example_by_id(self, example_id: str) -> Optional[EvaluationExample]:
        """Retrieve an example by its unique identifier."""
        return self._id_to_example.get(example_id)


def get_dataset_accessor(
    dataset_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> EvaluationDatasetAccessor:
    """Factory creating an EvaluationDatasetAccessor instance."""
    return EvaluationDatasetAccessor(dataset_path=dataset_path, manifest_path=manifest_path)
