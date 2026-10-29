"""Automated unit and integration tests for the GEPA prompt optimization experiment (Step 24).

Verifies:
- Leakage-aware subset selection and cryptographic dataset integrity
- Strict multi-tiered budget safeguards (metric calls, provider requests, timeouts)
- End-to-end experiment orchestration with deterministic test doubles
- Artifact persistence (summary, candidate program state, instruction diff, validation comparisons)
- Production prompt protection invariants (experimental candidates never replace active production prompts)
- Zero real API credits consumed during routine test runs
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import pytest

from app.core.config import Settings
from app.dspy_modules.budget_guard import (
    BudgetExceededError,
    OptimizationTimeoutError,
    ResourceBudgetGuard,
)
from app.dspy_modules.experiment_config import GepaExperimentConfig
from app.dspy_modules.experiment_runner import GepaExperimentRunner
from app.dspy_modules.subset_selector import PilotSubsetSelector, compute_file_sha256
from app.evaluation.dataset_accessor import EvaluationDatasetAccessor, HoldoutAccessRestrictedError
from app.services.generation_service import CrispeGenerationService


# =============================================================================
# 1. Dataset Subset Selection & Leakage Prevention Tests
# =============================================================================

def test_dataset_sha256_integrity() -> None:
    """Verify that dataset file SHA-256 matches the cryptographic digest in the manifest."""
    selector = PilotSubsetSelector()
    computed_sha = selector.verify_dataset_integrity()
    assert len(computed_sha) == 64
    assert computed_sha == selector.manifest.dataset_sha256


def test_pilot_subset_selector_isolation_and_diversity() -> None:
    """Verify train/validation subset selection maintains strict partition boundaries and domain diversity."""
    selector = PilotSubsetSelector()
    train_dspy, val_dspy, meta = selector.select_pilot_subsets(train_size=5, val_size=3)

    assert len(train_dspy) == 5
    assert len(val_dspy) == 3

    # Check partition integrity
    manifest = selector.manifest
    opt_ids = set(manifest.partitions["optimization"].example_ids)
    val_ids = set(manifest.partitions["validation"].example_ids)
    holdout_ids = set(manifest.partitions["final_holdout"].example_ids)

    # 1. Training examples must come exclusively from optimization partition
    for ex in train_dspy:
        assert ex.example_id in opt_ids
        assert ex.example_id not in val_ids
        assert ex.example_id not in holdout_ids

    # 2. Validation examples must come exclusively from validation partition
    for ex in val_dspy:
        assert ex.example_id in val_ids
        assert ex.example_id not in opt_ids
        assert ex.example_id not in holdout_ids

    # 3. No overlap between train and val
    train_id_set = {ex.example_id for ex in train_dspy}
    val_id_set = {ex.example_id for ex in val_dspy}
    assert len(train_id_set & val_id_set) == 0

    # 4. Domain diversity
    assert len(meta.train_domains) >= 3, f"Expected diverse domains, got: {meta.train_domains}"
    assert meta.holdout_excluded is True


def test_holdout_access_strictly_prevented_in_selector() -> None:
    """Verify that selector accessor rejects unauthorized holdout access."""
    selector = PilotSubsetSelector()
    with pytest.raises(HoldoutAccessRestrictedError):
        selector.accessor.get_holdout_examples(authorized=False)


# =============================================================================
# 2. Resource Budget Guard Tests
# =============================================================================

def test_budget_guard_metric_limit_enforced() -> None:
    """Verify ResourceBudgetGuard halts execution when max_gepa_metric_calls is reached."""
    guard = ResourceBudgetGuard(max_gepa_metric_calls=3, max_total_provider_calls=10)

    guard.record_metric_call()
    guard.record_metric_call()
    guard.record_metric_call()

    with pytest.raises(BudgetExceededError, match="Metric call budget exhausted"):
        guard.record_metric_call()

    summary = guard.get_summary()
    assert summary.is_budget_exceeded is True
    assert summary.gepa_metric_calls == 3


def test_budget_guard_provider_limit_enforced() -> None:
    """Verify ResourceBudgetGuard halts execution when max_total_provider_calls is reached."""
    guard = ResourceBudgetGuard(max_gepa_metric_calls=10, max_total_provider_calls=2)

    guard.record_generation_call()
    guard.record_reflection_call()

    with pytest.raises(BudgetExceededError, match="Total provider request ceiling reached"):
        guard.record_generation_call()

    summary = guard.get_summary()
    assert summary.is_budget_exceeded is True
    assert summary.total_provider_calls == 2


def test_budget_guard_timeout_enforced() -> None:
    """Verify ResourceBudgetGuard halts execution when wall-clock execution time exceeds limit."""
    guard = ResourceBudgetGuard(
        max_gepa_metric_calls=10,
        max_total_provider_calls=10,
        max_elapsed_seconds=0.01,  # 10ms limit
    )
    time.sleep(0.02)

    with pytest.raises(OptimizationTimeoutError, match="Execution timeout"):
        guard.record_metric_call()


def test_budget_guard_summary_telemetry() -> None:
    """Verify get_summary() produces auditable counts across call types."""
    guard = ResourceBudgetGuard(max_gepa_metric_calls=10, max_total_provider_calls=20)
    guard.record_generation_call()
    guard.record_generation_call()
    guard.record_reflection_call()
    guard.record_judge_call()

    summary = guard.get_summary()
    assert summary.generation_calls == 2
    assert summary.reflection_calls == 1
    assert summary.judge_calls == 1
    assert summary.total_provider_calls == 4
    assert summary.is_budget_exceeded is False


# =============================================================================
# 3. Experiment Config & Runner Tests
# =============================================================================

def test_experiment_config_validation() -> None:
    """Verify GepaExperimentConfig validates parameter bounds and rejects provider ceiling conflicts."""
    config = GepaExperimentConfig(max_metric_calls=10, max_total_provider_calls=25)
    assert config.max_metric_calls == 10
    assert config.max_total_provider_calls == 25

    with pytest.raises(ValueError, match="must be greater than or equal to max_metric_calls"):
        GepaExperimentConfig(max_metric_calls=20, max_total_provider_calls=10)


def test_end_to_end_gepa_experiment_runner_mock() -> None:
    """Verify full end-to-end execution of GepaExperimentRunner using deterministic mock models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = GepaExperimentConfig(
            experiment_id="test_pilot_e2e_01",
            train_size=3,
            val_size=2,
            max_metric_calls=8,
            max_total_provider_calls=20,
            results_dir=Path(temp_dir),
            is_mock=True,
            seed=42,
        )

        runner = GepaExperimentRunner(config=config)
        result = runner.run_experiment(force_mock=True)

        assert result.status in ("SUCCESS", "BUDGET_EXCEEDED")
        assert result.config.experiment_id == "test_pilot_e2e_01"
        assert len(result.subsets_metadata.train_example_ids) == 3
        assert len(result.subsets_metadata.val_example_ids) == 2
        assert result.subsets_metadata.holdout_excluded is True

        # Validation comparison
        comp = result.comparison
        assert "baseline_mean_score" in comp
        assert "candidate_mean_score" in comp
        assert "absolute_score_diff" in comp
        assert 0.0 <= comp["baseline_mean_score"] <= 1.0
        assert 0.0 <= comp["candidate_mean_score"] <= 1.0

        # Verify all 4 artifacts were persisted to the temp directory
        exp_dir = Path(temp_dir) / "test_pilot_e2e_01"
        assert (exp_dir / "experiment_summary.json").exists()
        assert (exp_dir / "validation_comparison.json").exists()
        assert (exp_dir / "instruction_diff.md").exists()
        assert (exp_dir / "candidate_program.json").exists()

        # Check content of summary JSON
        with open(exp_dir / "experiment_summary.json", "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            assert summary_data["experiment_id"] == "test_pilot_e2e_01"
            assert summary_data["candidate_status"] == "EXPERIMENTAL_CANDIDATE"
            assert summary_data["is_promoted"] is False
            assert summary_data["holdout_excluded"] is True


def test_instruction_diff_generation() -> None:
    """Verify that _generate_instruction_diff produces clean unified diff output."""
    runner = GepaExperimentRunner()
    base_text = "Step 1: Do X.\nStep 2: Do Y."
    candidate_text = "Step 1: Do X precisely.\nStep 2: Do Y.\nStep 3: Output JSON."

    diff = runner._generate_instruction_diff(base_text, candidate_text)
    assert "--- baseline_instruction" in diff
    assert "+++ optimized_candidate_instruction" in diff
    assert "+Step 1: Do X precisely." in diff
    assert "+Step 3: Output JSON." in diff


# =============================================================================
# 4. Production Protection Invariant Tests (Step Boundary)
# =============================================================================

def test_production_instructions_remain_unmodified() -> None:
    """Verify that GEPA experiments do NOT alter the production CrispeGenerationService prompt."""
    from app.services.generation_service import CRISPE_SYSTEM_PROMPT, CRISPE_USER_PROMPT_TEMPLATE

    # Record initial production system prompt
    initial_sys_prompt = CRISPE_SYSTEM_PROMPT
    initial_user_template = CRISPE_USER_PROMPT_TEMPLATE
    assert "CRISPE" in initial_sys_prompt
    assert "1. Context:" in initial_sys_prompt

    # Run a mock experiment to completion
    with tempfile.TemporaryDirectory() as temp_dir:
        config = GepaExperimentConfig(
            experiment_id="test_safety_guard_01",
            train_size=2,
            val_size=2,
            max_metric_calls=6,
            results_dir=Path(temp_dir),
            is_mock=True,
        )
        runner = GepaExperimentRunner(config=config)
        result = runner.run_experiment(force_mock=True)
        assert result.status in ("SUCCESS", "BUDGET_EXCEEDED")

    # Re-verify production prompts post-optimization
    from app.services import generation_service
    assert generation_service.CRISPE_SYSTEM_PROMPT == initial_sys_prompt
    assert generation_service.CRISPE_USER_PROMPT_TEMPLATE == initial_user_template
