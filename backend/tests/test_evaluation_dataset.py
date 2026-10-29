"""Unit and integrity tests for the Prompt Optimization Evaluation Dataset (Step 18).

Verifies:
1. Schema correctness and field invariants.
2. Full dataset loading and record validity.
3. Comprehensive domain coverage across all 6 taxonomical categories.
4. Balanced difficulty distribution (easy, medium, hard).
5. Uniqueness of example identifiers.
6. Absence of credentials or sensitive personal information.
7. Proper rejection of malformed or invalid records by validator.
"""

from collections import Counter
import json
from pathlib import Path
import re
import pytest
from pydantic import ValidationError

from app.evaluation.schemas import EvaluationExample, TaskDifficulty, TaskType
from app.evaluation.validator import (
    DEFAULT_DATASET_PATH,
    load_dataset,
    validate_dataset_file,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset_examples():
    """Load the full evaluation dataset once for the test module."""
    return load_dataset(DEFAULT_DATASET_PATH)


# -----------------------------------------------------------------------------
# Schema & Record Invariant Tests
# -----------------------------------------------------------------------------

def test_dataset_size_within_expected_bounds(dataset_examples) -> None:
    """Dataset contains between 40 and 60 carefully curated records."""
    count = len(dataset_examples)
    assert 40 <= count <= 60, f"Expected 40-60 examples, got {count}"
    assert count == 50, f"Expected exactly 50 examples in v1, got {count}"


def test_example_ids_are_unique(dataset_examples) -> None:
    """Every record in the dataset has a unique example_id."""
    seen = set()
    duplicates = []
    for ex in dataset_examples:
        if ex.example_id in seen:
            duplicates.append(ex.example_id)
        seen.add(ex.example_id)

    assert not duplicates, f"Found duplicate example_ids: {duplicates}"


def test_example_id_format(dataset_examples) -> None:
    """All example_ids follow slug naming convention (lowercase, digits, underscores, hyphens)."""
    pattern = re.compile(r"^[a-z0-9_\-]+$")
    for ex in dataset_examples:
        assert pattern.match(ex.example_id), f"Invalid example_id format: '{ex.example_id}'"


def test_required_collections_are_non_empty(dataset_examples) -> None:
    """Every record must have non-empty behaviors, criteria, constraints, and prohibitions."""
    for ex in dataset_examples:
        assert len(ex.required_behaviors) >= 1, f"{ex.example_id}: required_behaviors empty"
        assert len(ex.constraints) >= 1, f"{ex.example_id}: constraints empty"
        assert len(ex.acceptance_criteria) >= 1, f"{ex.example_id}: acceptance_criteria empty"
        assert len(ex.critical_requirements) >= 1, f"{ex.example_id}: critical_requirements empty"
        assert len(ex.prohibited_behaviors) >= 1, f"{ex.example_id}: prohibited_behaviors empty"

        # Check item content is substantive
        for item in ex.acceptance_criteria:
            assert len(item.strip()) >= 5, f"{ex.example_id}: acceptance_criterion too short: '{item}'"


def test_text_fields_substantive_depth(dataset_examples) -> None:
    """String fields meet length requirements and contain meaningful guidance."""
    for ex in dataset_examples:
        assert len(ex.user_idea.strip()) >= 10, f"{ex.example_id}: user_idea too short"
        assert len(ex.task_context.strip()) >= 10, f"{ex.example_id}: task_context too short"
        assert len(ex.expected_output_format.strip()) >= 5, f"{ex.example_id}: expected_output_format too short"
        assert len(ex.ambiguity_notes.strip()) >= 4, f"{ex.example_id}: ambiguity_notes too short"
        assert len(ex.grading_notes.strip()) >= 10, f"{ex.example_id}: grading_notes too short"


# -----------------------------------------------------------------------------
# Taxonomical Coverage & Diversity Tests
# -----------------------------------------------------------------------------

def test_all_domains_represented(dataset_examples) -> None:
    """All 6 core task domains must be represented in the dataset."""
    present_domains = {ex.task_type for ex in dataset_examples}
    expected_domains = {t.value for t in TaskType}
    missing = expected_domains - present_domains
    assert not missing, f"Missing domain categories: {missing}"


def test_domain_distribution_balance(dataset_examples) -> None:
    """No single domain dominates the dataset (>30% of total), ensuring balance."""
    counts = Counter(ex.task_type for ex in dataset_examples)
    total = len(dataset_examples)

    for domain, count in counts.items():
        pct = count / total
        assert pct <= 0.30, f"Domain '{domain}' exceeds 30% of total dataset ({pct * 100:.1f}%)"
        assert count >= 6, f"Domain '{domain}' has insufficient examples ({count} < 6)"


def test_difficulty_distribution_coverage(dataset_examples) -> None:
    """Dataset includes a balanced spread of easy, medium, and hard tasks."""
    counts = Counter(ex.difficulty for ex in dataset_examples)

    assert counts[TaskDifficulty.EASY.value] >= 8, "Expected at least 8 easy examples"
    assert counts[TaskDifficulty.MEDIUM.value] >= 15, "Expected at least 15 medium examples"
    assert counts[TaskDifficulty.HARD.value] >= 8, "Expected at least 8 hard examples"


def test_edge_cases_and_conflicts_included(dataset_examples) -> None:
    """Edge cases domain specifically includes ambiguity, missing information, and conflicts."""
    edge_cases = [ex for ex in dataset_examples if ex.task_type == TaskType.EDGE_CASES_CONFLICTS.value]
    assert len(edge_cases) >= 8

    # Verify presence of conflicting constraints and vague inputs
    tags_in_edge = {tag for ex in edge_cases for tag in ex.tags}
    assert "conflicts" in tags_in_edge
    assert "ambiguity" in tags_in_edge or "missing-info" in tags_in_edge


# -----------------------------------------------------------------------------
# Safety, Privacy, and Credential Tests
# -----------------------------------------------------------------------------

def test_no_credentials_or_sensitive_data_in_dataset() -> None:
    """Dataset JSONL contains no live API keys, secrets, or real private emails."""
    with open(DEFAULT_DATASET_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Generic credential patterns
    assert "AIzaSy" not in content, "Found potential Google API key"
    assert "sk-" not in content or "scikit-" in content, "Found potential secret key prefix"
    assert "BEGIN RSA PRIVATE KEY" not in content
    assert "Bearer " not in content

    # Check for real user personal emails (excluding synthetic placeholders like user@internal)
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@(?!internal\b|example\b)[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    matches = email_pattern.findall(content)
    assert not matches, f"Found potential personal email addresses: {matches}"


# -----------------------------------------------------------------------------
# Validator Utility & Edge-Case Rejection Tests
# -----------------------------------------------------------------------------

def test_validator_on_official_dataset() -> None:
    """validate_dataset_file reports is_valid=True with zero errors on the official dataset."""
    report = validate_dataset_file(DEFAULT_DATASET_PATH)
    assert report.is_valid is True
    assert report.total_records == 50
    assert report.valid_records == 50
    assert report.invalid_records == 0
    assert len(report.validation_errors) == 0


def test_validator_detects_non_existent_file(tmp_path) -> None:
    """Validator handles missing files gracefully with an error report."""
    fake_path = tmp_path / "does_not_exist.jsonl"
    report = validate_dataset_file(fake_path)
    assert report.is_valid is False
    assert any("File not found" in err for err in report.validation_errors)


def test_validator_detects_malformed_json(tmp_path) -> None:
    """Validator flags syntax errors in corrupted JSON lines."""
    corrupted_file = tmp_path / "corrupted.jsonl"
    corrupted_file.write_text("{this is not valid json}\n", encoding="utf-8")

    report = validate_dataset_file(corrupted_file)
    assert report.is_valid is False
    assert report.invalid_records == 1
    assert any("Malformed JSON" in err for err in report.validation_errors)


def test_validator_detects_duplicate_example_id(tmp_path, dataset_examples) -> None:
    """Validator rejects datasets containing duplicate example_ids."""
    dup_file = tmp_path / "duplicate.jsonl"
    first_ex = dataset_examples[0]
    # Write same example twice
    dup_file.write_text(
        first_ex.model_dump_json() + "\n" + first_ex.model_dump_json() + "\n",
        encoding="utf-8",
    )

    report = validate_dataset_file(dup_file)
    assert report.is_valid is False
    assert report.invalid_records == 1
    assert any("Duplicate example_id" in err for err in report.validation_errors)


def test_validator_detects_schema_violation(tmp_path, dataset_examples) -> None:
    """Validator catches records missing mandatory fields or using invalid enum values."""
    bad_file = tmp_path / "bad_schema.jsonl"
    bad_data = dataset_examples[0].model_dump()
    bad_data["task_type"] = "invalid_non_existent_domain"

    bad_file.write_text(json.dumps(bad_data) + "\n", encoding="utf-8")

    report = validate_dataset_file(bad_file)
    assert report.is_valid is False
    assert report.invalid_records == 1
    assert any("Schema error" in err for err in report.validation_errors)


def test_load_dataset_raises_on_invalid_file(tmp_path) -> None:
    """load_dataset raises ValueError on schema violations."""
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("{\"example_id\": \"invalid\"}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Schema validation failure"):
        load_dataset(bad_file)
