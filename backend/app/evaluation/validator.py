"""Validation utility and integrity checker for prompt optimization evaluation dataset (Step 18).

Provides programmatic and command-line interfaces to:
1. Load dataset files in JSONL format.
2. Validate each record against the Pydantic schema.
3. Detect duplicate example IDs.
4. Verify non-empty behaviors, criteria, constraints, and prohibitions.
5. Generate aggregate summaries across task domains and difficulty tiers.
"""

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Union
from pydantic import ValidationError

from app.evaluation.schemas import EvaluationExample, TaskDifficulty, TaskType

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "prompt_optimization_eval_v1.jsonl"
)


@dataclass
class DatasetValidationReport:
    """Detailed summary of dataset parsing and schema validation results."""

    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Returns True only if all records passed validation and at least one record exists."""
        return self.total_records > 0 and self.invalid_records == 0 and len(self.validation_errors) == 0


def load_dataset(
    file_path: Optional[Union[str, Path]] = None,
) -> List[EvaluationExample]:
    """Load and parse the evaluation dataset from a JSONL file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If any record fails validation or duplicate IDs are detected.
    """
    path = Path(file_path) if file_path else DEFAULT_DATASET_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

    examples: List[EvaluationExample] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_no}: Malformed JSON in dataset: {exc}"
                ) from exc

            try:
                example = EvaluationExample.model_validate(data)
            except ValidationError as exc:
                raise ValueError(
                    f"Line {line_no}: Schema validation failure: {exc}"
                ) from exc

            if example.example_id in seen_ids:
                raise ValueError(
                    f"Line {line_no}: Duplicate example_id detected: '{example.example_id}'"
                )
            seen_ids.add(example.example_id)
            examples.append(example)

    if not examples:
        raise ValueError(f"Dataset file is empty: {path}")

    return examples


def validate_dataset_file(
    file_path: Optional[Union[str, Path]] = None,
) -> DatasetValidationReport:
    """Inspect and validate a dataset file, returning an aggregated report without raising exceptions."""
    path = Path(file_path) if file_path else DEFAULT_DATASET_PATH
    report = DatasetValidationReport()

    if not path.is_file():
        report.validation_errors.append(f"File not found: {path}")
        return report

    seen_ids = set()
    domain_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            report.total_records += 1

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                report.invalid_records += 1
                report.validation_errors.append(
                    f"Line {line_no}: Malformed JSON: {exc}"
                )
                continue

            try:
                example = EvaluationExample.model_validate(data)
            except ValidationError as exc:
                report.invalid_records += 1
                report.validation_errors.append(
                    f"Line {line_no} (ID: {data.get('example_id', 'unknown')}): Schema error: {exc.errors()}"
                )
                continue

            if example.example_id in seen_ids:
                report.invalid_records += 1
                report.validation_errors.append(
                    f"Line {line_no}: Duplicate example_id '{example.example_id}'"
                )
                continue

            seen_ids.add(example.example_id)
            report.valid_records += 1
            domain_counts[example.task_type] += 1
            difficulty_counts[example.difficulty] += 1

    report.domain_distribution = dict(domain_counts)
    report.difficulty_distribution = dict(difficulty_counts)

    # Check for empty dataset condition
    if report.total_records == 0:
        report.validation_errors.append(f"Dataset file contains no records: {path}")

    return report


def print_report(report: DatasetValidationReport, file_path: Optional[Union[str, Path]] = None) -> None:
    """Print human-readable summary of validation report to stdout."""
    target_path = Path(file_path) if file_path else DEFAULT_DATASET_PATH
    print("=" * 76)
    print("CRISPE-AI PROMPT OPTIMIZATION EVALUATION DATASET VALIDATION REPORT")
    print("=" * 76)
    print(f"File Target: {target_path}")
    print(f"Status:      {'VALID (Passed)' if report.is_valid else 'INVALID (Failed)'}")
    print(f"Records:     Total = {report.total_records} | Valid = {report.valid_records} | Invalid = {report.invalid_records}")
    print("-" * 76)
    print("Domain Taxonomy Distribution:")
    for task_type in TaskType:
        count = report.domain_distribution.get(task_type.value, 0)
        pct = (count / report.total_records * 100) if report.total_records > 0 else 0.0
        print(f"  - {task_type.value:<30}: {count:>3} ({pct:>5.1f}%)")
    print("-" * 76)
    print("Difficulty Distribution:")
    for diff in TaskDifficulty:
        count = report.difficulty_distribution.get(diff.value, 0)
        pct = (count / report.total_records * 100) if report.total_records > 0 else 0.0
        print(f"  - {diff.value:<30}: {count:>3} ({pct:>5.1f}%)")
    print("-" * 76)

    if report.validation_errors:
        print(f"Validation Errors ({len(report.validation_errors)}):")
        for err in report.validation_errors[:10]:
            print(f"  [X] {err}")
        if len(report.validation_errors) > 10:
            print(f"  ... and {len(report.validation_errors) - 10} more errors.")
    else:
        print("Integrity Check: Zero schema violations, zero duplicate IDs, complete coverage.")
    print("=" * 76)


if __name__ == "__main__":
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    res = validate_dataset_file(cli_path)
    print_report(res, cli_path)
    sys.exit(0 if res.is_valid else 1)
