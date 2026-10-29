"""Automated unit and integration tests for Deterministic Evaluation Engine (Step 20).

Verifies:
- CRISPE structural validation (completeness, empty fields, invalid types)
- All 9 deterministic check types (passing, failing, missing sections, unicode, case, regex safety, prohibitions)
- Multi-dimensional scoring formula with hand-calculated proofs
- Critical failure penalty capping (0.40) and schema failure capping (0.20)
- Division-by-zero protection on empty inputs / zero applicable checks
- Baseline artifact evaluation adapter with mock baseline artifacts
- Sealed holdout partition protection and leakage safeguards
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
from typing import Dict, List

from app.evaluation.baseline_evaluator import BaselineEvaluator
from app.evaluation.baseline_runner import BaselineExampleResult
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
)
from app.evaluation.rule_builder import build_checks_for_example
from app.evaluation.rule_engine import (
    execute_check,
    is_negated_mention,
    normalize_text,
    validate_crispe_structure,
    validate_safe_regex,
)
from app.evaluation.schemas import EvaluationExample, TaskDifficulty, TaskType
from app.evaluation.scoring_engine import DeterministicScorer, ScoringPolicy
from app.evaluation.split_manifest import PartitionName
from app.schemas.crispe import CrispeSections


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_valid_sections() -> CrispeSections:
    """Fixture providing a fully valid 6-section CRISPE prompt."""
    return CrispeSections(
        context="Developing a high-throughput payment processing pipeline in Python 3.12 with asyncio.",
        role="Act as a Principal Fintech Security Architect and Systems Engineer.",
        instruction="Design an idempotent webhook handler that validates cryptographic signatures and persists idempotency keys.",
        specification="Must use Redis SETNX with 24-hour TTL. Return RFC-7807 problem details JSON on verification failure.",
        performance="Ensure maximum latency is below 50ms at 10,000 requests per second. Zero race conditions under concurrent retries.",
        example="Sample webhook payload:\n{\n  'event_id': 'evt_99182',\n  'amount': 5000\n}",
    )


@pytest.fixture
def sample_example() -> EvaluationExample:
    """Fixture providing a valid EvaluationExample benchmark record."""
    return EvaluationExample(
        example_id="fintech_idempotency_01",
        task_family="fintech_transactions",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        difficulty=TaskDifficulty.MEDIUM,
        tags=["payment", "idempotency", "redis", "security"],
        example_version="v1",
        user_idea="Create a prompt for designing a webhook handler with idempotency and signature validation.",
        task_context="High-volume financial transaction gateway processing millions of daily webhooks.",
        required_behaviors=[
            "Instruct the AI to validate incoming HMAC SHA-256 signatures",
            "Require idempotent storage using Redis keys with explicit TTL",
        ],
        constraints=[
            "Do not allow raw SQL queries without parameterized bounds",
            "Must return RFC-7807 compliant error format",
        ],
        expected_output_format="Structured Markdown with architecture breakdown and Python asyncio snippet",
        ambiguity_notes="None. All requirements specified.",
        acceptance_criteria=[
            "Prompt includes explicit instruction for HMAC signature verification",
            "Prompt requires Redis idempotency key verification",
        ],
        critical_requirements=[
            "Must enforce HMAC SHA-256 signature verification before processing body",
        ],
        prohibited_behaviors=[
            "Do not permit in-memory idempotency storage that resets upon restart",
        ],
        grading_notes="Verify prompt enforces Redis TTL and cryptographic signature checking.",
    )


# =============================================================================
# A. CRISPE Structural Validation Tests
# =============================================================================

class TestCrispeStructuralValidation:
    """Tests for validating canonical 6-section CRISPE schema."""

    def test_complete_valid_crispe_sections(self, sample_valid_sections: CrispeSections) -> None:
        """Valid CrispeSections passes structural validation with zero issues."""
        is_valid, issues = validate_crispe_structure(sample_valid_sections)
        assert is_valid is True
        assert issues == []

    def test_valid_dict_structure(self) -> None:
        """Valid dictionary with all 6 keys passes structural validation."""
        dict_data = {
            "context": "Context text here.",
            "role": "Role persona here.",
            "instruction": "Instruction command here.",
            "specification": "Specification details.",
            "performance": "Performance metrics.",
            "example": "Demonstration example.",
        }
        is_valid, issues = validate_crispe_structure(dict_data)
        assert is_valid is True
        assert issues == []

    def test_missing_required_section(self) -> None:
        """Dictionary missing a section fails with detailed issue message."""
        incomplete = {
            "context": "Context text.",
            "role": "Role persona.",
            # instruction is missing
            "specification": "Specification.",
            "performance": "Performance.",
            "example": "Example.",
        }
        is_valid, issues = validate_crispe_structure(incomplete)
        assert is_valid is False
        assert any("instruction" in issue for issue in issues)

    def test_empty_or_whitespace_section(self) -> None:
        """Whitespace-only section is detected and rejected."""
        empty_sec = {
            "context": "Context text.",
            "role": "   \n\t   ",  # empty whitespace
            "instruction": "Instruction command.",
            "specification": "Specification.",
            "performance": "Performance.",
            "example": "Example.",
        }
        is_valid, issues = validate_crispe_structure(empty_sec)
        assert is_valid is False
        assert any("role" in issue and "empty" in issue for issue in issues)

    def test_none_sections_input(self) -> None:
        """None input fails gracefully without raising an unhandled exception."""
        is_valid, issues = validate_crispe_structure(None)
        assert is_valid is False
        assert len(issues) > 0

    def test_invalid_type_input(self) -> None:
        """Passing an integer or list fails validation gracefully."""
        is_valid, issues = validate_crispe_structure("random string")
        assert is_valid is False
        assert "CrispeSections" in issues[0]


# =============================================================================
# B. Check Engine Unit Tests (All 9 Check Types)
# =============================================================================

class TestDeterministicCheckEngine:
    """Tests executing each of the supported check types."""

    def test_crispe_schema_valid_check(self, sample_valid_sections: CrispeSections) -> None:
        """CRISPE_SCHEMA_VALID returns PASSED for valid prompt and FAILED for malformed."""
        check_def = DeterministicCheckDefinition(
            check_id="test_schema_valid",
            check_type=CheckType.CRISPE_SCHEMA_VALID,
            dimension=EvaluationDimension.STRUCTURE,
            description="Test schema validity",
        )
        res = execute_check(check_def, sample_valid_sections)
        assert res.outcome == CheckOutcome.PASSED
        assert res.is_critical is False

        # Now test against malformed dict
        bad_sections = {"context": "only context"}
        res_bad = execute_check(check_def, bad_sections)
        assert res_bad.outcome == CheckOutcome.FAILED
        assert "Malformed CRISPE" in res_bad.actual_condition

    def test_required_section_present_check(self, sample_valid_sections: CrispeSections) -> None:
        """REQUIRED_SECTION_PRESENT verifies section has minimum character length."""
        check_def = DeterministicCheckDefinition(
            check_id="test_role_present",
            check_type=CheckType.REQUIRED_SECTION_PRESENT,
            target_section=CrispeSectionTarget.ROLE,
            parameters={"min_chars": 10},
            description="Role must be present",
        )
        res = execute_check(check_def, sample_valid_sections)
        assert res.outcome == CheckOutcome.PASSED

        # Empty/short role check
        check_fail = DeterministicCheckDefinition(
            check_id="test_role_too_long",
            check_type=CheckType.REQUIRED_SECTION_PRESENT,
            target_section=CrispeSectionTarget.ROLE,
            parameters={"min_chars": 500},
            description="Role impossibly long",
        )
        res_fail = execute_check(check_fail, sample_valid_sections)
        assert res_fail.outcome == CheckOutcome.FAILED

    def test_minimum_text_length_check(self, sample_valid_sections: CrispeSections) -> None:
        """MINIMUM_TEXT_LENGTH evaluates char and word count constraints."""
        check_def = DeterministicCheckDefinition(
            check_id="test_min_length",
            check_type=CheckType.MINIMUM_TEXT_LENGTH,
            target_section=CrispeSectionTarget.INSTRUCTION,
            parameters={"min_chars": 20, "min_words": 5},
            description="Instruction minimum length",
        )
        res = execute_check(check_def, sample_valid_sections)
        assert res.outcome == CheckOutcome.PASSED

        check_fail = DeterministicCheckDefinition(
            check_id="test_min_len_fail",
            check_type=CheckType.MINIMUM_TEXT_LENGTH,
            target_section=CrispeSectionTarget.INSTRUCTION,
            parameters={"min_chars": 1000},
            description="Too long",
        )
        res_fail = execute_check(check_fail, sample_valid_sections)
        assert res_fail.outcome == CheckOutcome.FAILED

    def test_required_text_check(self, sample_valid_sections: CrispeSections) -> None:
        """REQUIRED_TEXT tests exact phrase containment and case sensitivity."""
        # Case insensitive pass
        check_def = DeterministicCheckDefinition(
            check_id="test_req_text",
            check_type=CheckType.REQUIRED_TEXT,
            target_section=CrispeSectionTarget.SPECIFICATION,
            parameters={"text": "redis setnx", "case_sensitive": False},
            description="Must require Redis SETNX",
        )
        res = execute_check(check_def, sample_valid_sections)
        assert res.outcome == CheckOutcome.PASSED

        # Case sensitive fail
        check_case = DeterministicCheckDefinition(
            check_id="test_req_text_case",
            check_type=CheckType.REQUIRED_TEXT,
            target_section=CrispeSectionTarget.SPECIFICATION,
            parameters={"text": "redis setnx", "case_sensitive": True},
            description="Case sensitive match",
        )
        res_case = execute_check(check_case, sample_valid_sections)
        assert res_case.outcome == CheckOutcome.FAILED  # Actual prompt has uppercase "Redis SETNX"

    def test_required_term_group_check(self, sample_valid_sections: CrispeSections) -> None:
        """REQUIRED_TERM_GROUP passes if at least min_matches terms are found."""
        check_def = DeterministicCheckDefinition(
            check_id="test_term_group",
            check_type=CheckType.REQUIRED_TERM_GROUP,
            target_section=CrispeSectionTarget.ALL,
            parameters={"terms": ["webhook", "idempotent", "unrelated_term_xyz"], "min_matches": 2},
            description="Find webhook and idempotent",
        )
        res = execute_check(check_def, sample_valid_sections)
        assert res.outcome == CheckOutcome.PASSED
        assert "webhook" in res.actual_condition and "idempotent" in res.actual_condition

        # Failing min_matches
        check_fail = DeterministicCheckDefinition(
            check_id="test_term_group_fail",
            check_type=CheckType.REQUIRED_TERM_GROUP,
            target_section=CrispeSectionTarget.ALL,
            parameters={"terms": ["webhook", "unrelated_1", "unrelated_2"], "min_matches": 2},
            description="Requires 2 matches",
        )
        res_fail = execute_check(check_fail, sample_valid_sections)
        assert res_fail.outcome == CheckOutcome.FAILED

    def test_forbidden_text_with_negation_awareness(self) -> None:
        """FORBIDDEN_TEXT correctly differentiates negative boundary instructions from violations."""
        # Scenario A: Prompt says "You should use MD5 hashing" -> VIOLATION
        bad_sections = CrispeSections(
            context="Security context.",
            role="Security engineer.",
            instruction="You must use MD5 hashing for the token generation.",
            specification="Standard rules.",
            performance="Fast performance.",
            example="md5 example.",
        )
        check_def = DeterministicCheckDefinition(
            check_id="test_no_md5",
            check_type=CheckType.FORBIDDEN_TEXT,
            target_section=CrispeSectionTarget.ALL,
            parameters={"forbidden_terms": ["md5"], "allow_contextual_negation": True},
            description="Prohibit MD5 hashing",
        )
        res_bad = execute_check(check_def, bad_sections)
        assert res_bad.outcome == CheckOutcome.FAILED
        assert "md5" in res_bad.actual_condition

        # Scenario B: Prompt says "Do not use MD5 hashing. Avoid md5." -> PASS (negation context)
        good_sections = CrispeSections(
            context="Security context.",
            role="Security engineer.",
            instruction="Generate secure SHA-256 tokens. Do not use MD5 under any circumstances.",
            specification="Must avoid md5.",
            performance="High security.",
            example="sha256 example.",
        )
        res_good = execute_check(check_def, good_sections)
        assert res_good.outcome == CheckOutcome.PASSED
        assert res_good.actual_condition == "No prohibited patterns detected."

    def test_regex_constraint_check(self, sample_valid_sections: CrispeSections) -> None:
        """REGEX_CONSTRAINT matches prevalidated safe regular expressions."""
        check_def = DeterministicCheckDefinition(
            check_id="test_regex_status",
            check_type=CheckType.REGEX_CONSTRAINT,
            target_section=CrispeSectionTarget.PERFORMANCE,
            parameters={"pattern": r"\b\d{1,3}(,\d{3})*\s*requests per second\b"},
            description="Regex for requests per second",
        )
        res = execute_check(check_def, sample_valid_sections)
        assert res.outcome == CheckOutcome.PASSED

    def test_regex_safety_and_redos_prevention(self) -> None:
        """Catastrophic backtracking patterns and over-length patterns are rejected."""
        # Nested quantifier
        with pytest.raises(ValueError, match="Potential catastrophic backtracking"):
            validate_safe_regex(r"(a+)+")

        with pytest.raises(ValueError, match="Potential catastrophic backtracking"):
            validate_safe_regex(r"(a|b+)+")

        # Exceeds maximum length
        with pytest.raises(ValueError, match="maximum allowed length"):
            validate_safe_regex("a" * 501)

    def test_constraint_preserved_check(self, sample_valid_sections: CrispeSections) -> None:
        """CONSTRAINT_PRESERVED validates terms or patterns."""
        check_def = DeterministicCheckDefinition(
            check_id="test_con_rfc",
            check_type=CheckType.CONSTRAINT_PRESERVED,
            target_section=CrispeSectionTarget.SPECIFICATION,
            parameters={"terms": ["rfc-7807", "problem details"]},
            description="RFC-7807 error format",
        )
        res = execute_check(check_def, sample_valid_sections)
        assert res.outcome == CheckOutcome.PASSED

    def test_output_format_requirement_check(self, sample_valid_sections: CrispeSections) -> None:
        """OUTPUT_FORMAT_REQUIREMENT checks for requested format indicators."""
        check_def = DeterministicCheckDefinition(
            check_id="test_fmt_json",
            check_type=CheckType.OUTPUT_FORMAT_REQUIREMENT,
            target_section=CrispeSectionTarget.ALL,
            parameters={"format_indicators": ["json", "problem details"]},
            description="JSON format requirement",
        )
        res = execute_check(check_def, sample_valid_sections)
        assert res.outcome == CheckOutcome.PASSED


# =============================================================================
# C. Multi-Dimensional Scoring Tests
# =============================================================================

class TestDeterministicScoringEngine:
    """Mathematical verification of scoring formula, critical caps, and denominators."""

    def test_hand_calculated_scoring_formula(self, sample_example: EvaluationExample) -> None:
        """Verify dimensional and uncapped composite score against exact hand-calculated math.

        Setup 4 custom checks across 2 dimensions:
        Dimension STRUCTURE (weight=0.25):
          - check1: weight=1.0, outcome=PASSED
          - check2: weight=1.0, outcome=FAILED
          => dim_structure = 1.0 / 2.0 = 0.5000

        Dimension REQUIREMENTS (weight=0.35):
          - check3: weight=2.0, outcome=PASSED
          - check4: weight=1.0, outcome=PASSED
          => dim_requirements = 3.0 / 3.0 = 1.0000

        Active weights sum = 0.25 + 0.35 = 0.60
        Weighted sum = (0.25 * 0.5) + (0.35 * 1.0) = 0.125 + 0.35 = 0.475
        Uncapped composite score = 0.475 / 0.60 = 0.791666... -> rounded to 0.7917
        """
        policy = ScoringPolicy(
            policy_version="test_v1",
            dimension_weights={
                EvaluationDimension.STRUCTURE.value: 0.25,
                EvaluationDimension.REQUIREMENTS.value: 0.35,
                EvaluationDimension.CONSTRAINTS.value: 0.20,
                EvaluationDimension.OUTPUT_FORMAT.value: 0.10,
                EvaluationDimension.PROHIBITIONS.value: 0.10,
            },
            critical_failure_score_cap=0.40,
            schema_failure_score_cap=0.20,
        )
        scorer = DeterministicScorer(policy=policy)

        custom_checks = [
            DeterministicCheckDefinition(
                check_id="chk_struct_1",
                check_type=CheckType.REQUIRED_TEXT,
                dimension=EvaluationDimension.STRUCTURE,
                parameters={"text": "context"},
                weight=1.0,
                description="Check struct 1",
            ),
            DeterministicCheckDefinition(
                check_id="chk_struct_2",
                check_type=CheckType.REQUIRED_TEXT,
                dimension=EvaluationDimension.STRUCTURE,
                parameters={"text": "non_existent_text_xyz"},
                weight=1.0,
                description="Check struct 2",
            ),
            DeterministicCheckDefinition(
                check_id="chk_req_1",
                check_type=CheckType.REQUIRED_TEXT,
                dimension=EvaluationDimension.REQUIREMENTS,
                parameters={"text": "instruction"},
                weight=2.0,
                description="Check req 1",
            ),
            DeterministicCheckDefinition(
                check_id="chk_req_2",
                check_type=CheckType.REQUIRED_TEXT,
                dimension=EvaluationDimension.REQUIREMENTS,
                parameters={"text": "specification"},
                weight=1.0,
                description="Check req 2",
            ),
        ]

        dummy_sections = CrispeSections(
            context="context text",
            role="role text",
            instruction="instruction text",
            specification="specification text",
            performance="performance text",
            example="example text",
        )

        result = scorer.evaluate_example(
            example=sample_example,
            sections=dummy_sections,
            custom_checks=custom_checks,
        )

        assert result.dimension_scores[EvaluationDimension.STRUCTURE.value] == 0.5000
        assert result.dimension_scores[EvaluationDimension.REQUIREMENTS.value] == 1.0000
        assert pytest.approx(result.composite_score, abs=1e-4) == 0.7917
        assert result.critical_failure is False

    def test_critical_failure_penalty_capping(self, sample_example: EvaluationExample) -> None:
        """When a critical requirement check fails, composite score is strictly capped at 0.40."""
        scorer = DeterministicScorer()
        critical_check = DeterministicCheckDefinition(
            check_id="crit_signature_fail",
            check_type=CheckType.REQUIRED_TEXT,
            dimension=EvaluationDimension.REQUIREMENTS,
            target_section=CrispeSectionTarget.ALL,
            parameters={"text": "mandatory_critical_token_not_found"},
            required=True,
            is_critical=True,
            weight=1.0,
            description="Critical signature verification requirement",
        )

        sections = CrispeSections(
            context="Payment gateway context with excellent details.",
            role="Expert architect role description.",
            instruction="Handle the webhooks properly.",
            specification="Detailed specifications.",
            performance="Strict performance benchmarks.",
            example="Sample code snippet.",
        )

        res = scorer.evaluate_example(
            example=sample_example,
            sections=sections,
            custom_checks=[critical_check],
        )

        assert res.critical_failure is True
        assert len(res.critical_failure_reasons) > 0
        assert res.composite_score <= 0.40

    def test_schema_failure_penalty_capping(self, sample_example: EvaluationExample) -> None:
        """When canonical CRISPE schema fails, composite score is capped at 0.20."""
        scorer = DeterministicScorer()
        # Missing sections
        malformed_sections = {"context": "only context"}

        res = scorer.evaluate_example(
            example=sample_example,
            sections=malformed_sections,  # type: ignore
        )

        assert res.crispe_structure_valid is False
        assert res.critical_failure is True
        assert res.composite_score <= 0.20

    def test_zero_applicable_checks_and_empty_dataset(self) -> None:
        """Zero applicable checks or empty datasets do not cause division by zero."""
        scorer = DeterministicScorer()
        report = scorer.aggregate_results(
            results=[],
            experiment_id="empty_exp",
            dataset_version="v1",
            partition="validation",
        )

        assert report.total_examples == 0
        assert report.required_check_pass_rate == 1.0
        assert report.critical_failure_rate == 0.0
        assert report.mean_composite_score == 0.0

    def test_deterministic_idempotency(
        self,
        sample_example: EvaluationExample,
        sample_valid_sections: CrispeSections,
    ) -> None:
        """Evaluating identical inputs twice produces identical scores and check counts."""
        scorer = DeterministicScorer()
        res1 = scorer.evaluate_example(sample_example, sample_valid_sections)
        res2 = scorer.evaluate_example(sample_example, sample_valid_sections)

        assert res1.composite_score == res2.composite_score
        assert res1.dimension_scores == res2.dimension_scores
        assert res1.critical_failure == res2.critical_failure
        assert len(res1.checks) == len(res2.checks)


# =============================================================================
# D. Baseline Artifact Adapter Integration Tests
# =============================================================================

class TestBaselineEvaluatorAdapter:
    """Tests evaluating saved baseline artifacts without calling live LLMs."""

    def test_evaluate_mock_baseline_results(
        self,
        tmp_path: Path,
        sample_example: EvaluationExample,
        sample_valid_sections: CrispeSections,
    ) -> None:
        """Evaluates synthetic baseline results file and generates valid summary report."""
        exp_id = "test_baseline_mock_01"
        res_file = tmp_path / f"{exp_id}.jsonl"

        # Construct 2 mock baseline results
        b1 = BaselineExampleResult(
            experiment_id=exp_id,
            example_id="fintech_idempotency_01",
            partition="validation",
            task_type="software_engineering",
            difficulty="medium",
            model_id="gemini-2.5-flash",
            status="SUCCESS",
            sections=sample_valid_sections,
            formatted_prompt=sample_valid_sections.to_markdown(),
            schema_valid=True,
        )
        b2 = BaselineExampleResult(
            experiment_id=exp_id,
            example_id="fintech_idempotency_01",
            partition="validation",
            task_type="software_engineering",
            difficulty="medium",
            model_id="gemini-2.5-flash",
            status="FAILED",
            error_message="ResourceExhausted quota exceeded",
            sections=None,
            formatted_prompt=None,
            schema_valid=False,
        )

        with open(res_file, "w", encoding="utf-8") as f:
            f.write(b1.model_dump_json() + "\n")
            f.write(b2.model_dump_json() + "\n")

        # Mock dataset accessor with single example
        class MockDatasetAccessor(EvaluationDatasetAccessor):
            def __init__(self) -> None:
                self._id_to_example = {sample_example.example_id: sample_example}
                self._hld_ids = set()
                self._opt_ids = set()
                self._val_ids = {sample_example.example_id}
                self._manifest = type("Manifest", (), {"dataset_version": "v1"})()

        mock_accessor = MockDatasetAccessor()
        evaluator = BaselineEvaluator(dataset_accessor=mock_accessor, results_dir=tmp_path)

        summary, evals = evaluator.evaluate_file(res_file, experiment_id=exp_id, save_report=True)

        assert summary.total_examples == 2
        assert summary.successful_generations == 1
        assert summary.failed_generations == 1
        assert summary.valid_crispe_outputs == 1
        assert len(evals) == 2

        # Verify evaluation details and summary file artifacts were generated on disk
        summary_path = tmp_path / f"{exp_id}_eval_summary.json"
        details_path = tmp_path / f"{exp_id}_eval_details.jsonl"
        assert summary_path.is_file()
        assert details_path.is_file()

        # Denominator check: failed generation is distinct from evaluation failure
        assert evals[1].status == "GENERATION_FAILED"
        assert evals[1].composite_score == 0.0


# =============================================================================
# E. Partition Isolation and Holdout Protection Tests
# =============================================================================

class TestPartitionProtectionAndHoldout:
    """Verifies sealed holdout access prevention during evaluation."""

    def test_evaluating_holdout_without_auth_raises_error(self, tmp_path: Path) -> None:
        """Evaluating baseline results on final_holdout raises HoldoutAccessRestrictedError without auth."""
        exp_id = "test_holdout_leak_check"
        res_file = tmp_path / f"{exp_id}.jsonl"

        holdout_result = BaselineExampleResult(
            experiment_id=exp_id,
            example_id="holdout_secret_ex_01",
            partition="final_holdout",
            task_type="software_engineering",
            difficulty="hard",
            model_id="gemini-2.5-flash",
            status="SUCCESS",
            sections=None,
            schema_valid=False,
        )

        with open(res_file, "w", encoding="utf-8") as f:
            f.write(holdout_result.model_dump_json() + "\n")

        evaluator = BaselineEvaluator(results_dir=tmp_path)

        with pytest.raises(HoldoutAccessRestrictedError, match="Cannot evaluate final holdout results"):
            evaluator.evaluate_file(res_file, allow_holdout=False)

    def test_accessor_get_example_by_id_blocks_holdout(self) -> None:
        """DatasetAccessor.get_example_by_id blocks holdout examples if allow_holdout=False."""
        accessor = EvaluationDatasetAccessor()
        holdout_ids = accessor.manifest.partitions[PartitionName.FINAL_HOLDOUT.value].example_ids
        assert len(holdout_ids) > 0
        secret_id = holdout_ids[0]

        # Blocked without auth
        with pytest.raises(HoldoutAccessRestrictedError):
            accessor.get_example_by_id(secret_id, allow_holdout=False)

        # Allowed with explicit auth
        ex = accessor.get_example_by_id(secret_id, allow_holdout=True)
        assert ex is not None
        assert ex.example_id == secret_id
