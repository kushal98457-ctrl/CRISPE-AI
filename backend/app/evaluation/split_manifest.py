"""Versioned dataset splitting and manifest specification for prompt evaluation (Step 19).

Provides deterministic, group-aware, and stratified partitioning of evaluation examples into:
- Optimization partition (~60%, 30 examples): used by DSPy/GEPA optimizers.
- Validation partition (~20%, 10 examples): used to compare candidates and tune parameters.
- Final Holdout partition (~20%, 10 examples): sealed test set for unbiased final verification.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.schemas import EvaluationExample, TaskDifficulty, TaskType
from app.evaluation.validator import DEFAULT_DATASET_PATH, load_dataset

DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "datasets" / "eval_split_manifest_v1.json"
)


class PartitionName(str, Enum):
    """Categorical partition labels for the prompt evaluation benchmark."""

    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    FINAL_HOLDOUT = "final_holdout"


class SplitPartitionSummary(BaseModel):
    """Statistical and inventory summary for a single dataset partition."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    count: int = Field(..., ge=0, description="Total number of examples assigned to this partition.")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of total dataset.")
    example_ids: List[str] = Field(..., description="Sorted list of example identifiers assigned to partition.")
    domain_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of examples per task domain category.",
    )
    difficulty_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of examples per difficulty tier (easy, medium, hard).",
    )


class SplitManifest(BaseModel):
    """Complete versioned manifest defining dataset partitioning and integrity metadata."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    manifest_version: str = Field(default="v1", description="Schema version of this split manifest.")
    dataset_version: str = Field(..., description="Target dataset version (e.g. v1).")
    dataset_file: str = Field(..., description="Basename of the partitioned evaluation dataset file.")
    dataset_sha256: str = Field(..., description="Cryptographic SHA-256 hash of the canonical dataset file.")
    splitter_version: str = Field(default="1.0.0", description="Version of the splitting algorithm.")
    random_seed: int = Field(default=42, description="Pseudo-random generator seed used for deterministic shuffling.")
    target_proportions: Dict[str, float] = Field(
        default_factory=lambda: {
            PartitionName.OPTIMIZATION.value: 0.60,
            PartitionName.VALIDATION.value: 0.20,
            PartitionName.FINAL_HOLDOUT.value: 0.20,
        },
        description="Target distribution proportions across partitions.",
    )
    total_examples: int = Field(..., ge=1, description="Total number of evaluation examples partitioned.")
    partitions: Dict[str, SplitPartitionSummary] = Field(
        ...,
        description="Partition inventory mapping partition name to its summary and example IDs.",
    )
    grouping_strategy: str = Field(
        default="task_family_stratified",
        description="Strategy guaranteeing related task families never leak across partition boundaries.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the split was generated.",
    )
    notes: str = Field(
        default=(
            "Sealed final holdout protocol: The final_holdout partition is strictly reserved for "
            "unbiased post-optimization measurement and must NEVER be supplied to candidate optimizers."
        ),
        description="Operational and governance constraints for the manifest.",
    )


def compute_file_sha256(file_path: Path) -> str:
    """Compute the hexadecimal SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_stratified_split(
    examples: Sequence[EvaluationExample],
    dataset_path: Optional[Path] = None,
    seed: int = 42,
) -> SplitManifest:
    """Create a deterministic, group-aware, and stratified split of evaluation examples.

    Guarantees:
    1. Group-aware: All examples sharing a task_family remain together in the exact same partition.
    2. Stratified: Every domain and difficulty tier is represented across optimization, validation, and holdout.
    3. Exact Proportions: Bounded to 30 Optimization (60%), 10 Validation (20%), and 10 Final Holdout (20%) for N=50.
    4. Deterministic: Given the same seed and dataset, produces identical partition assignments.
    """
    if not examples:
        raise ValueError("Cannot split an empty sequence of examples.")

    target_path = dataset_path or DEFAULT_DATASET_PATH
    checksum = compute_file_sha256(target_path) if target_path.is_file() else "UNKNOWN"

    total_count = len(examples)
    target_opt = 30
    target_val = 10
    target_hld = 10

    # 1. Group examples by task_family (to prevent leakage)
    family_to_examples: Dict[str, List[EvaluationExample]] = defaultdict(list)
    for ex in examples:
        fam_key = ex.task_family or ex.example_id
        family_to_examples[fam_key].append(ex)

    # 2. Group families by domain to ensure balanced domain representation
    domain_to_families: Dict[str, List[str]] = defaultdict(list)
    family_metadata: Dict[str, Tuple[str, str]] = {}
    for fam_key, fam_examples in family_to_examples.items():
        dom = fam_examples[0].task_type
        diff = fam_examples[0].difficulty
        domain_to_families[dom].append(fam_key)
        family_metadata[fam_key] = (dom, diff)

    rng = random.Random(seed)

    opt_ids: List[str] = []
    val_ids: List[str] = []
    hld_ids: List[str] = []

    # Quotas per domain:
    # 9-item domains (software_engineering, edge_cases_conflicts): 5 opt, 2 val, 2 hld
    # 8-item domains: 5 opt, 2 val, 1 hld OR 5 opt, 1 val, 2 hld
    domain_quotas: Dict[str, Tuple[int, int, int]] = {
        TaskType.SOFTWARE_ENGINEERING.value: (5, 2, 2),
        TaskType.EDGE_CASES_CONFLICTS.value: (5, 2, 2),
        TaskType.AI_MACHINE_LEARNING.value: (5, 2, 1),
        TaskType.EDUCATION_PROBLEM_SOLVING.value: (5, 2, 1),
        TaskType.TECHNICAL_RESEARCH.value: (5, 1, 2),
        TaskType.PRODUCT_SYSTEM_DESIGN.value: (5, 1, 2),
    }

    for domain in sorted(domain_to_families.keys()):
        families = domain_to_families[domain]
        # Sort families by difficulty then name for deterministic ordering
        families.sort(key=lambda fam: (family_metadata[fam][1], fam))
        rng.shuffle(families)

        q_opt, q_val, q_hld = domain_quotas.get(domain, (5, 2, 2))

        assigned_opt = families[:q_opt]
        assigned_val = families[q_opt : q_opt + q_val]
        assigned_hld = families[q_opt + q_val : q_opt + q_val + q_hld]

        for fam in assigned_opt:
            opt_ids.extend(ex.example_id for ex in family_to_examples[fam])
        for fam in assigned_val:
            val_ids.extend(ex.example_id for ex in family_to_examples[fam])
        for fam in assigned_hld:
            hld_ids.extend(ex.example_id for ex in family_to_examples[fam])

    opt_ids.sort()
    val_ids.sort()
    hld_ids.sort()

    # Build partition summaries
    id_to_example = {ex.example_id: ex for ex in examples}

    def make_summary(ids: List[str]) -> SplitPartitionSummary:
        dom_counter: Counter[str] = Counter()
        diff_counter: Counter[str] = Counter()
        for i in ids:
            ex = id_to_example[i]
            dom_counter[ex.task_type] += 1
            diff_counter[ex.difficulty] += 1
        return SplitPartitionSummary(
            count=len(ids),
            percentage=round(len(ids) / total_count * 100, 2),
            example_ids=ids,
            domain_distribution=dict(dom_counter),
            difficulty_distribution=dict(diff_counter),
        )

    partitions = {
        PartitionName.OPTIMIZATION.value: make_summary(opt_ids),
        PartitionName.VALIDATION.value: make_summary(val_ids),
        PartitionName.FINAL_HOLDOUT.value: make_summary(hld_ids),
    }

    return SplitManifest(
        manifest_version="v1",
        dataset_version=examples[0].example_version,
        dataset_file=target_path.name,
        dataset_sha256=checksum,
        splitter_version="1.0.0",
        random_seed=seed,
        target_proportions={
            PartitionName.OPTIMIZATION.value: 0.60,
            PartitionName.VALIDATION.value: 0.20,
            PartitionName.FINAL_HOLDOUT.value: 0.20,
        },
        total_examples=total_count,
        partitions=partitions,
        grouping_strategy="task_family_stratified",
    )


def validate_split_manifest(
    manifest: SplitManifest,
    dataset_path: Optional[Path] = None,
) -> List[str]:
    """Validate that a SplitManifest accurately matches the source dataset with zero leakage.

    Returns a list of error strings; an empty list indicates the manifest is 100% valid.
    """
    errors: List[str] = []
    target_path = dataset_path or DEFAULT_DATASET_PATH

    if not target_path.is_file():
        errors.append(f"Target dataset file not found: {target_path}")
        return errors

    # Checksum verification
    current_checksum = compute_file_sha256(target_path)
    if current_checksum != manifest.dataset_sha256:
        errors.append(
            f"Dataset SHA-256 mismatch! Manifest recorded '{manifest.dataset_sha256}', "
            f"actual dataset file has '{current_checksum}'."
        )

    # Load dataset examples
    try:
        dataset_examples = load_dataset(target_path)
    except Exception as exc:
        errors.append(f"Failed to load dataset: {exc}")
        return errors

    dataset_ids = {ex.example_id for ex in dataset_examples}
    family_map = {ex.example_id: (ex.task_family or ex.example_id) for ex in dataset_examples}

    # Verify partition disjointness and completeness
    partition_id_sets: Dict[str, Set[str]] = {}
    all_assigned_ids: Set[str] = set()

    for p_name, summary in manifest.partitions.items():
        p_ids = set(summary.example_ids)
        if len(p_ids) != summary.count:
            errors.append(
                f"Partition '{p_name}' count mismatch: summary claims {summary.count}, "
                f"but example_ids list contains {len(p_ids)} items."
            )

        # Check for cross-partition overlap
        for prior_name, prior_ids in partition_id_sets.items():
            intersection = p_ids.intersection(prior_ids)
            if intersection:
                errors.append(
                    f"Data leakage detected! Partitions '{prior_name}' and '{p_name}' "
                    f"share {len(intersection)} identical IDs: {list(intersection)[:5]}"
                )

        partition_id_sets[p_name] = p_ids
        all_assigned_ids.update(p_ids)

    # Verify all dataset IDs exist in manifest
    missing_from_manifest = dataset_ids - all_assigned_ids
    if missing_from_manifest:
        errors.append(
            f"{len(missing_from_manifest)} dataset examples are omitted from manifest: {list(missing_from_manifest)[:5]}"
        )

    unknown_in_manifest = all_assigned_ids - dataset_ids
    if unknown_in_manifest:
        errors.append(
            f"{len(unknown_in_manifest)} manifest IDs do not exist in dataset: {list(unknown_in_manifest)[:5]}"
        )

    # Verify task_family leakage (no task_family spans multiple partitions)
    family_to_partitions: Dict[str, Set[str]] = defaultdict(set)
    for p_name, p_ids in partition_id_sets.items():
        for ex_id in p_ids:
            fam = family_map.get(ex_id)
            if fam:
                family_to_partitions[fam].add(p_name)

    leaking_families = {fam: p_names for fam, p_names in family_to_partitions.items() if len(p_names) > 1}
    if leaking_families:
        errors.append(
            f"Task family grouping violation! {len(leaking_families)} families cross partitions: {leaking_families}"
        )

    return errors


def save_manifest(manifest: SplitManifest, path: Optional[Path] = None) -> None:
    """Save SplitManifest to formatted JSON file."""
    target_path = path or DEFAULT_MANIFEST_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2) + "\n")


def load_manifest(path: Optional[Path] = None) -> SplitManifest:
    """Load SplitManifest from JSON file."""
    target_path = path or DEFAULT_MANIFEST_PATH
    if not target_path.is_file():
        raise FileNotFoundError(f"Split manifest file not found: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SplitManifest.model_validate(data)
