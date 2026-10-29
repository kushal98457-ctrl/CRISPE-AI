"""Leakage-aware representative subset selection for bounded GEPA experiments (Step 24).

Selects balanced, diverse subsets from the authorized optimization and validation partitions
while cryptographically verifying dataset integrity and strictly preventing holdout leakage.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import dspy

from app.core.logging import get_logger
from app.dspy_modules.dataset_converter import convert_evaluation_example
from app.evaluation.dataset_accessor import EvaluationDatasetAccessor, HoldoutAccessRestrictedError
from app.evaluation.schemas import EvaluationExample
from app.evaluation.split_manifest import DEFAULT_MANIFEST_PATH, SplitManifest, load_manifest
from app.evaluation.validator import DEFAULT_DATASET_PATH

logger = get_logger("gepa_subset_selector")


@dataclass
class PilotSubsetsMetadata:
    """Audit metadata recording dataset provenance, checksums, and selected partition IDs."""

    dataset_version: str
    dataset_sha256: str
    manifest_version: str
    manifest_path: str
    train_example_ids: List[str]
    val_example_ids: List[str]
    train_domains: Dict[str, int]
    val_domains: Dict[str, int]
    holdout_excluded: bool = True


def compute_file_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hex digest of a dataset file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class PilotSubsetSelector:
    """Selects stratified subsets from optimization and validation partitions with zero-leakage guarantees."""

    def __init__(
        self,
        accessor: Optional[EvaluationDatasetAccessor] = None,
        manifest_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
    ) -> None:
        self.dataset_path = dataset_path or DEFAULT_DATASET_PATH
        self.manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
        self.accessor = accessor or EvaluationDatasetAccessor(
            dataset_path=self.dataset_path,
            manifest_path=self.manifest_path,
        )
        self.manifest = self.accessor.manifest

    def verify_dataset_integrity(self) -> str:
        """Verify the SHA-256 checksum of the dataset against the manifest record."""
        computed_sha = compute_file_sha256(self.dataset_path)
        expected_sha = self.manifest.dataset_sha256

        if computed_sha != expected_sha:
            msg = (
                f"Dataset integrity check failed! Computed {computed_sha} "
                f"does not match manifest record {expected_sha}"
            )
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Dataset SHA-256 integrity verified: %s", computed_sha[:16])
        return computed_sha

    def select_pilot_subsets(
        self,
        train_size: int = 5,
        val_size: int = 3,
    ) -> Tuple[List[dspy.Example], List[dspy.Example], PilotSubsetsMetadata]:
        """Select representative training and validation subsets for the GEPA pilot.

        Guarantees:
        - Diverse domain representation (software engineering, AI/ML, product design, edge cases, research).
        - Deterministic ordering by example_id for exact experiment reproducibility.
        - Zero cross-partition leakage.
        - Final holdout partition is strictly protected and untouched.

        Parameters:
            train_size: Number of training examples to select (from optimization partition).
            val_size: Number of validation examples to select (from validation partition).

        Returns:
            Tuple of (train_dspy_examples, val_dspy_examples, metadata).
        """
        sha = self.verify_dataset_integrity()

        # 1. Load optimization and validation examples via authorized accessor
        opt_examples = self.accessor.get_optimization_examples()
        val_examples = self.accessor.get_validation_examples()

        # Verify holdout cannot be accessed without explicit authorization
        try:
            self.accessor.get_holdout_examples(authorized=False)
            raise RuntimeError("Holdout isolation breach: accessor allowed unauthorized holdout access!")
        except HoldoutAccessRestrictedError:
            logger.debug("Holdout partition isolation confirmed.")

        # 2. Select diverse domain training subset
        # Order by task_type then example_id for deterministic domain diversity
        sorted_opt = sorted(opt_examples, key=lambda x: (str(x.task_type), x.example_id))
        selected_train: List[EvaluationExample] = []
        seen_domains: set = set()

        # First pass: one from each distinct domain up to train_size
        for ex in sorted_opt:
            domain = str(ex.task_type)
            if domain not in seen_domains and len(selected_train) < train_size:
                selected_train.append(ex)
                seen_domains.add(domain)

        # Fill remaining slots if train_size > distinct domains
        for ex in sorted_opt:
            if len(selected_train) >= train_size:
                break
            if ex not in selected_train:
                selected_train.append(ex)

        # 3. Select diverse validation subset
        sorted_val = sorted(val_examples, key=lambda x: (str(x.task_type), x.example_id))
        selected_val: List[EvaluationExample] = []
        val_seen_domains: set = set()

        for ex in sorted_val:
            domain = str(ex.task_type)
            if domain not in val_seen_domains and len(selected_val) < val_size:
                selected_val.append(ex)
                val_seen_domains.add(domain)

        for ex in sorted_val:
            if len(selected_val) >= val_size:
                break
            if ex not in selected_val:
                selected_val.append(ex)

        # 4. Strict Leakage Invariants Verification
        train_ids = [ex.example_id for ex in selected_train]
        val_ids = [ex.example_id for ex in selected_val]
        holdout_ids = set(self.manifest.partitions["final_holdout"].example_ids)

        assert len(set(train_ids) & set(val_ids)) == 0, "Train and validation subsets must have zero overlapping IDs!"
        assert len(set(train_ids) & holdout_ids) == 0, "Holdout leakage detected in training subset!"
        assert len(set(val_ids) & holdout_ids) == 0, "Holdout leakage detected in validation subset!"

        train_domains: Dict[str, int] = {}
        for ex in selected_train:
            train_domains[str(ex.task_type)] = train_domains.get(str(ex.task_type), 0) + 1

        val_domains: Dict[str, int] = {}
        for ex in selected_val:
            val_domains[str(ex.task_type)] = val_domains.get(str(ex.task_type), 0) + 1

        metadata = PilotSubsetsMetadata(
            dataset_version=self.manifest.dataset_version,
            dataset_sha256=sha,
            manifest_version=self.manifest.manifest_version,
            manifest_path=str(self.manifest_path),
            train_example_ids=train_ids,
            val_example_ids=val_ids,
            train_domains=train_domains,
            val_domains=val_domains,
            holdout_excluded=True,
        )

        logger.info(
            "Selected pilot subsets: train=%d (domains=%s), val=%d (domains=%s)",
            len(selected_train),
            list(train_domains.keys()),
            len(selected_val),
            list(val_domains.keys()),
        )

        # Convert to dspy.Example instances with designated inputs
        train_dspy = [convert_evaluation_example(ex) for ex in selected_train]
        val_dspy = [convert_evaluation_example(ex) for ex in selected_val]

        return train_dspy, val_dspy, metadata
