"""Unit and integrity tests for dataset splitting, holdout protection, and baseline generation (Step 19).

Tests:
1. Split manifest integrity and SHA-256 verification.
2. Leakage-free, mutually exclusive partition assignments.
3. Group-aware task family isolation across partitions.
4. Sealed holdout access restrictions and permission enforcement.
5. Baseline runner execution with mock generation service.
6. Checkpoint resumption and idempotence.
7. Graceful error classification and recovery.
8. Accurate operational metric reporting.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

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
from app.evaluation.schemas import EvaluationExample, TaskDifficulty, TaskType
from app.evaluation.split_manifest import (
    DEFAULT_MANIFEST_PATH,
    PartitionName,
    SplitManifest,
    compute_file_sha256,
    create_stratified_split,
    load_manifest,
    validate_split_manifest,
)
from app.evaluation.validator import DEFAULT_DATASET_PATH, load_dataset
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeGenerationResponse,
    CrispeSections,
    ModelUsageMetadata,
)
from app.services.generation_service import CrispeGenerationService


# -----------------------------------------------------------------------------
# Test Fixtures & Mock Service
# -----------------------------------------------------------------------------

class MockCrispeGenerationService(CrispeGenerationService):
    """Deterministic mock generation service for testing the baseline runner without live APIs."""

    def __init__(self, failure_ideas: Optional[set] = None) -> None:
        self.call_count = 0
        self.processed_ideas: list[str] = []
        self.failure_ideas = failure_ideas or set()

    async def generate_crispe_prompt(
        self, request: CrispeGenerationRequest
    ) -> CrispeGenerationResponse:
        self.call_count += 1
        self.processed_ideas.append(request.idea)

        # Simulate targeted failure if requested
        for fail_text in self.failure_ideas:
            if fail_text in request.idea:
                raise RuntimeError(f"Simulated generation provider failure for {fail_text[:30]}")

        sections = CrispeSections(
            context="Baseline mock operational context.",
            role="Principal prompt architect.",
            instruction="Execute baseline benchmark evaluation task.",
            specification="Adhere to strict formatting constraints.",
            performance="Measure schema validity and operational latency.",
            example="Mock concrete input/output pair.",
        )
        return CrispeGenerationResponse(
            idea=request.idea,
            sections=sections,
            formatted_prompt=sections.to_markdown(),
            model_id="mock-gemini-2.5-flash",
            usage=ModelUsageMetadata(prompt_tokens=50, candidates_tokens=150, total_tokens=200),
            finish_reason="STOP",
        )


@pytest.fixture
def official_manifest() -> SplitManifest:
    """Load the official repository split manifest."""
    return load_manifest(DEFAULT_MANIFEST_PATH)


@pytest.fixture
def accessor() -> EvaluationDatasetAccessor:
    """Instantiate dataset accessor with official dataset and manifest."""
    return get_dataset_accessor(DEFAULT_DATASET_PATH, DEFAULT_MANIFEST_PATH)


# -----------------------------------------------------------------------------
# Split Manifest & Partitioning Tests
# -----------------------------------------------------------------------------

def test_manifest_loads_and_passes_validation(official_manifest: SplitManifest) -> None:
    """The official split manifest matches the source dataset with zero integrity violations."""
    errors = validate_split_manifest(official_manifest, DEFAULT_DATASET_PATH)
    assert not errors, f"Split manifest validation failed: {errors}"
    assert official_manifest.total_examples == 50


def test_manifest_checksum_matches_dataset_sha256(official_manifest: SplitManifest) -> None:
    """Manifest dataset_sha256 exactly matches the physical SHA-256 of the dataset file."""
    actual_hash = compute_file_sha256(DEFAULT_DATASET_PATH)
    assert official_manifest.dataset_sha256 == actual_hash


def test_partition_target_proportions_and_counts(official_manifest: SplitManifest) -> None:
    """Partitions match the 60% / 20% / 20% distribution (30 opt, 10 val, 10 holdout)."""
    p = official_manifest.partitions

    assert p[PartitionName.OPTIMIZATION.value].count == 30
    assert p[PartitionName.OPTIMIZATION.value].percentage == 60.0

    assert p[PartitionName.VALIDATION.value].count == 10
    assert p[PartitionName.VALIDATION.value].percentage == 20.0

    assert p[PartitionName.FINAL_HOLDOUT.value].count == 10
    assert p[PartitionName.FINAL_HOLDOUT.value].percentage == 20.0


def test_zero_partition_leakage(official_manifest: SplitManifest) -> None:
    """No example ID appears in more than one partition (mutually exclusive sets)."""
    opt_set = set(official_manifest.partitions[PartitionName.OPTIMIZATION.value].example_ids)
    val_set = set(official_manifest.partitions[PartitionName.VALIDATION.value].example_ids)
    hld_set = set(official_manifest.partitions[PartitionName.FINAL_HOLDOUT.value].example_ids)

    assert len(opt_set.intersection(val_set)) == 0, "Leakage between optimization and validation!"
    assert len(opt_set.intersection(hld_set)) == 0, "Leakage between optimization and holdout!"
    assert len(val_set.intersection(hld_set)) == 0, "Leakage between validation and holdout!"

    # Union covers all 50 examples
    all_assigned = opt_set | val_set | hld_set
    assert len(all_assigned) == 50


def test_all_domains_represented_in_every_partition(official_manifest: SplitManifest) -> None:
    """Every task domain is present across optimization, validation, and holdout sets."""
    all_domains = {t.value for t in TaskType}

    for p_name, p_sum in official_manifest.partitions.items():
        present_domains = set(p_sum.domain_distribution.keys())
        missing = all_domains - present_domains
        assert not missing, f"Partition '{p_name}' lacks domain representation for: {missing}"


def test_all_difficulty_tiers_represented_in_every_partition(official_manifest: SplitManifest) -> None:
    """Every difficulty tier (easy, medium, hard) is present in each partition."""
    all_diffs = {d.value for d in TaskDifficulty}

    for p_name, p_sum in official_manifest.partitions.items():
        present_diffs = set(p_sum.difficulty_distribution.keys())
        missing = all_diffs - present_diffs
        assert not missing, f"Partition '{p_name}' lacks difficulty representation for: {missing}"


def test_task_family_grouping_preserved(accessor: EvaluationDatasetAccessor, official_manifest: SplitManifest) -> None:
    """No task family spans multiple partition boundaries."""
    family_to_partition = {}
    for p_name, p_sum in official_manifest.partitions.items():
        for eid in p_sum.example_ids:
            ex = accessor.get_example_by_id(eid)
            assert ex is not None
            fam = ex.task_family or ex.example_id
            if fam in family_to_partition:
                assert family_to_partition[fam] == p_name, f"Family '{fam}' leaked into '{p_name}' and '{family_to_partition[fam]}'"
            family_to_partition[fam] = p_name


def test_deterministic_split_reproducibility() -> None:
    """Running create_stratified_split with the same seed generates identical partitions."""
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    split_1 = create_stratified_split(dataset, seed=42)
    split_2 = create_stratified_split(dataset, seed=42)

    assert split_1.partitions[PartitionName.OPTIMIZATION.value].example_ids == split_2.partitions[PartitionName.OPTIMIZATION.value].example_ids
    assert split_1.partitions[PartitionName.VALIDATION.value].example_ids == split_2.partitions[PartitionName.VALIDATION.value].example_ids
    assert split_1.partitions[PartitionName.FINAL_HOLDOUT.value].example_ids == split_2.partitions[PartitionName.FINAL_HOLDOUT.value].example_ids


# -----------------------------------------------------------------------------
# Holdout Access Isolation Tests
# -----------------------------------------------------------------------------

def test_holdout_access_blocked_by_default(accessor: EvaluationDatasetAccessor) -> None:
    """Attempting to access final holdout examples without authorization raises HoldoutAccessRestrictedError."""
    with pytest.raises(HoldoutAccessRestrictedError):
        accessor.get_holdout_examples(authorized=False)

    with pytest.raises(HoldoutAccessRestrictedError):
        accessor.get_partition_examples(PartitionName.FINAL_HOLDOUT, allow_holdout=False)


def test_holdout_access_allowed_with_explicit_authorization(accessor: EvaluationDatasetAccessor) -> None:
    """Explicitly authorized calls successfully retrieve the sealed holdout examples."""
    holdout = accessor.get_holdout_examples(authorized=True, reason="Final verification drill")
    assert len(holdout) == 10
    assert all(isinstance(ex, EvaluationExample) for ex in holdout)


def test_optimization_and_validation_access(accessor: EvaluationDatasetAccessor) -> None:
    """Optimization and validation partition accessors return correct record sets without restriction."""
    opt_examples = accessor.get_optimization_examples()
    assert len(opt_examples) == 30

    val_examples = accessor.get_validation_examples()
    assert len(val_examples) == 10


def test_invalid_partition_name_rejected(accessor: EvaluationDatasetAccessor) -> None:
    """Requesting an unknown partition name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown partition"):
        accessor.get_partition_examples("non_existent_partition")


# -----------------------------------------------------------------------------
# Split Validator Negative Tests
# -----------------------------------------------------------------------------

def test_split_validator_catches_checksum_mismatch(official_manifest: SplitManifest, tmp_path: Path) -> None:
    """Validator detects when dataset file has been modified or checksum diverges."""
    bad_manifest = official_manifest.model_copy(update={"dataset_sha256": "0000000000000000000000000000000000000000000000000000000000000000"})
    errors = validate_split_manifest(bad_manifest, DEFAULT_DATASET_PATH)
    assert any("SHA-256 mismatch" in err for err in errors)


def test_split_validator_catches_duplicate_leakage(official_manifest: SplitManifest) -> None:
    """Validator catches when an ID is accidentally placed in both validation and optimization."""
    leaked_id = official_manifest.partitions[PartitionName.VALIDATION.value].example_ids[0]

    # Tamper manifest to duplicate ID
    opt_ids = list(official_manifest.partitions[PartitionName.OPTIMIZATION.value].example_ids) + [leaked_id]
    tampered_parts = dict(official_manifest.partitions)
    tampered_parts[PartitionName.OPTIMIZATION.value] = official_manifest.partitions[PartitionName.OPTIMIZATION.value].model_copy(
        update={"count": len(opt_ids), "example_ids": opt_ids}
    )
    tampered_manifest = official_manifest.model_copy(update={"partitions": tampered_parts})

    errors = validate_split_manifest(tampered_manifest, DEFAULT_DATASET_PATH)
    assert any("Data leakage detected" in err for err in errors)


# -----------------------------------------------------------------------------
# Baseline Runner Execution Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_baseline_runner_validation_partition(tmp_path: Path, accessor: EvaluationDatasetAccessor) -> None:
    """Baseline runner executes successfully across the 10 validation examples using mock generator."""
    mock_service = MockCrispeGenerationService()
    runner = BaselineRunner(
        generation_service=mock_service,
        dataset_accessor=accessor,
        results_dir=tmp_path,
    )

    exp_id = "test_baseline_val_run_001"
    summary, results = await runner.run_baseline(
        experiment_id=exp_id,
        partition=PartitionName.VALIDATION,
    )

    assert summary.total_attempted == 10
    assert summary.successful_responses == 10
    assert summary.failed_requests == 0
    assert summary.valid_crispe_outputs == 10
    assert summary.generation_success_rate == 1.0
    assert summary.schema_validity_rate == 1.0
    assert len(results) == 10
    assert mock_service.call_count == 10

    # Verify results JSONL file was written
    result_file = tmp_path / f"{exp_id}.jsonl"
    assert result_file.is_file()
    with open(result_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 10

    # Verify summary JSON was written
    summary_file = tmp_path / f"{exp_id}_summary.json"
    assert summary_file.is_file()


@pytest.mark.asyncio
async def test_baseline_runner_checkpoint_resumption(tmp_path: Path, accessor: EvaluationDatasetAccessor) -> None:
    """Resuming an interrupted baseline run skips already completed examples without duplicating calls."""
    mock_service = MockCrispeGenerationService()
    runner = BaselineRunner(
        generation_service=mock_service,
        dataset_accessor=accessor,
        results_dir=tmp_path,
    )

    exp_id = "test_resume_run_002"

    # 1. Run only first 4 examples
    summary_1, results_1 = await runner.run_baseline(
        experiment_id=exp_id,
        partition=PartitionName.VALIDATION,
        max_examples=4,
    )
    assert summary_1.total_attempted == 4
    assert mock_service.call_count == 4

    # 2. Resume run for full 10 examples
    summary_2, results_2 = await runner.run_baseline(
        experiment_id=exp_id,
        partition=PartitionName.VALIDATION,
        max_examples=10,
    )
    # Total attempted is now 10, but only 6 NEW calls were made to mock service
    assert summary_2.total_attempted == 10
    assert mock_service.call_count == 10  # 4 from run 1 + 6 from run 2
    assert len(results_2) == 10


@pytest.mark.asyncio
async def test_baseline_runner_error_handling(tmp_path: Path, accessor: EvaluationDatasetAccessor) -> None:
    """Runner records individual example failures with error classification without crashing."""
    val_examples = accessor.get_validation_examples()
    fail_ex = val_examples[0]

    mock_service = MockCrispeGenerationService(failure_ideas={fail_ex.user_idea})
    runner = BaselineRunner(
        generation_service=mock_service,
        dataset_accessor=accessor,
        results_dir=tmp_path,
    )

    exp_id = "test_error_handling_run_003"
    summary, results = await runner.run_baseline(
        experiment_id=exp_id,
        partition=PartitionName.VALIDATION,
    )

    assert summary.total_attempted == 10
    assert summary.successful_responses == 9
    assert summary.failed_requests == 1
    assert summary.generation_success_rate == 0.90
    assert summary.valid_crispe_outputs == 9

    failed_rec = next(r for r in results if r.example_id == fail_ex.example_id)
    assert failed_rec.status == "FAILED"
    assert failed_rec.error_category == "RuntimeError"
    assert "Simulated generation provider failure" in (failed_rec.error_message or "")


@pytest.mark.asyncio
async def test_baseline_runner_blocks_unauthorized_holdout(tmp_path: Path, accessor: EvaluationDatasetAccessor) -> None:
    """Baseline runner prevents executing the final holdout partition unless allow_holdout=True."""
    mock_service = MockCrispeGenerationService()
    runner = BaselineRunner(
        generation_service=mock_service,
        dataset_accessor=accessor,
        results_dir=tmp_path,
    )

    with pytest.raises(HoldoutAccessRestrictedError):
        await runner.run_baseline(
            experiment_id="test_holdout_blocked",
            partition=PartitionName.FINAL_HOLDOUT,
            allow_holdout=False,
        )


def test_baseline_summary_handles_zero_attempts(tmp_path: Path, accessor: EvaluationDatasetAccessor) -> None:
    """Summary calculation safely handles zero-attempt runs without division by zero."""
    runner = BaselineRunner(dataset_accessor=accessor, results_dir=tmp_path)
    summary = runner.calculate_summary(
        experiment_id="empty_run",
        partition="validation",
        results=[],
        results_file=str(tmp_path / "empty.jsonl"),
    )
    assert summary.total_attempted == 0
    assert summary.generation_success_rate == 0.0
    assert summary.schema_validity_rate == 0.0
    assert summary.overall_validity_rate == 0.0
    assert summary.mean_latency_ms is None
