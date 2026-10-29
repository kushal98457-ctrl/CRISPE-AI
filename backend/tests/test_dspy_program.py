"""Automated unit and integration tests for the DSPy CRISPE Generation Program (Step 22).

Verifies:
- CrispeGenerationSignature input/output contracts, descriptions, and exclusion of hidden criteria
- CrispeGenerationProgram forward execution, canonical CrispeSections mapping, and validation failures
- Concurrency isolation under dspy.context(lm=...) without cross-request state leakage
- DspyCrispeGenerationService request/response parity and error handling
- DspyEvaluationAdapter integration with Step 20 & 21 evaluators
- Zero live API credits consumed during routine tests
"""

import asyncio
from typing import Dict, List, Optional
import dspy
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    EmptyPromptError,
    PromptValidationError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
)
from app.dspy_modules.evaluation_adapter import DspyEvaluationAdapter
from app.dspy_modules.model_client import (
    create_dspy_lm,
    dspy_isolated_context,
    normalize_dspy_model_name,
)
from app.dspy_modules.program import (
    CrispeGenerationProgram,
    create_crispe_program,
)
from app.dspy_modules.service import DspyCrispeGenerationService
from app.dspy_modules.signature import CrispeGenerationSignature
from app.evaluation.check_schemas import EvaluationExampleResult
from app.evaluation.judge_schemas import CombinedEvaluationResult
from app.evaluation.schemas import EvaluationExample, TaskDifficulty, TaskType
from app.evaluation.scoring_engine import DeterministicScorer
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeGenerationResponse,
    CrispeSections,
)


# =============================================================================
# Mock DSPy Language Models
# =============================================================================

class MockDummyLM(dspy.BaseLM):
    """Deterministic mock LM returning canonical CRISPE markers for DSPy adapters."""

    def __init__(
        self,
        model_name: str = "mock-gemini",
        override_sections: Optional[Dict[str, str]] = None,
        should_fail: Optional[Exception] = None,
    ) -> None:
        super().__init__(model=model_name)
        self.model_name = model_name
        self.override_sections = override_sections or {
            "context": "Context: Distributed transaction processing pipeline.",
            "role": "Role: Senior Systems Reliability Engineer.",
            "instruction": "Instruction: Build an automated failover controller.",
            "specification": "Specification: Must maintain zero message loss and log structured JSON.",
            "performance": "Performance: Failover completed in under 5 seconds.",
            "example": "Example: Sample heartbeat failure trigger.",
        }
        self.should_fail = should_fail
        self.call_history: List[Dict[str, Any]] = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.call_history.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
        if self.should_fail:
            raise self.should_fail

        s = self.override_sections
        response_text = (
            f"[[ ## context ## ]]\n{s.get('context', '')}\n"
            f"[[ ## role ## ]]\n{s.get('role', '')}\n"
            f"[[ ## instruction ## ]]\n{s.get('instruction', '')}\n"
            f"[[ ## specification ## ]]\n{s.get('specification', '')}\n"
            f"[[ ## performance ## ]]\n{s.get('performance', '')}\n"
            f"[[ ## example ## ]]\n{s.get('example', '')}"
        )
        return [response_text]


@pytest.fixture
def sample_eval_example() -> EvaluationExample:
    return EvaluationExample(
        example_id="dspy_test_ex_01",
        task_family="distributed_systems",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        difficulty=TaskDifficulty.MEDIUM,
        tags=["failover", "consensus"],
        example_version="v1",
        user_idea="Design a leader election watchdog in Python asyncio.",
        task_context="Raft-based key-value store cluster.",
        required_behaviors=["Implement randomized election timeouts", "Heartbeat broadcasting"],
        constraints=["Do not use third-party distributed lock managers"],
        expected_output_format="Python asyncio implementation snippet",
        ambiguity_notes="Assume 5-node cluster.",
        acceptance_criteria=["Randomized timer logic", "Heartbeat handling"],
        critical_requirements=["Must enforce Raft randomized election timeouts"],
        prohibited_behaviors=["Do not permit split-brain without quorum"],
        grading_notes="Check for quorum check and randomized timer.",
    )


# =============================================================================
# A. Signature Tests
# =============================================================================

class TestCrispeGenerationSignature:
    """Verifies CrispeGenerationSignature inputs, outputs, and descriptions."""

    def test_signature_input_fields(self) -> None:
        """Signature defines expected generator input fields."""
        input_names = set(CrispeGenerationSignature.input_fields.keys())
        expected_inputs = {"idea", "task_context", "constraints", "output_format_preference"}
        assert input_names == expected_inputs

    def test_signature_output_fields(self) -> None:
        """Signature defines all 6 canonical CRISPE output fields."""
        output_names = set(CrispeGenerationSignature.output_fields.keys())
        expected_outputs = {"context", "role", "instruction", "specification", "performance", "example"}
        assert output_names == expected_outputs

    def test_hidden_criteria_excluded_from_inputs(self) -> None:
        """Hidden evaluator criteria and answer keys are strictly absent from inputs."""
        input_names = set(CrispeGenerationSignature.input_fields.keys())
        forbidden_inputs = {
            "acceptance_criteria",
            "critical_requirements",
            "prohibited_behaviors",
            "grading_notes",
            "answers",
            "rubric",
        }
        assert input_names.isdisjoint(forbidden_inputs)

    def test_field_descriptions_are_meaningful(self) -> None:
        """Every field carries descriptive guidance explaining its semantic purpose."""
        for field_name, field_info in CrispeGenerationSignature.input_fields.items():
            desc = field_info.json_schema_extra.get("desc", "")
            assert len(desc) >= 15, f"Input field '{field_name}' description is too short."

        for field_name, field_info in CrispeGenerationSignature.output_fields.items():
            desc = field_info.json_schema_extra.get("desc", "")
            assert len(desc) >= 20, f"Output field '{field_name}' description is too short."


# =============================================================================
# B. Program Execution Tests
# =============================================================================

class TestCrispeGenerationProgram:
    """Verifies program forward pass, schema extraction, and validation failures."""

    def test_program_execution_with_mock_lm(self) -> None:
        """Program executes successfully and returns validated CrispeSections."""
        mock_lm = MockDummyLM()
        program = create_crispe_program()

        with dspy.context(lm=mock_lm):
            sections, prediction = program.execute(
                idea="Create an automated failover controller",
                task_context="Distributed cluster",
                constraints="Zero message loss",
            )

        assert isinstance(sections, CrispeSections)
        assert sections.context == "Context: Distributed transaction processing pipeline."
        assert sections.role == "Role: Senior Systems Reliability Engineer."
        assert sections.instruction == "Instruction: Build an automated failover controller."
        assert sections.specification == "Specification: Must maintain zero message loss and log structured JSON."
        assert sections.performance == "Performance: Failover completed in under 5 seconds."
        assert sections.example == "Example: Sample heartbeat failure trigger."

    def test_empty_idea_raises_empty_prompt_error(self) -> None:
        """Empty or whitespace-only idea raises EmptyPromptError."""
        program = create_crispe_program()
        with pytest.raises(EmptyPromptError, match="cannot be empty"):
            program.forward(idea="   \n\t  ")

    def test_missing_section_raises_validation_error(self) -> None:
        """Prediction missing a required CRISPE section raises PromptValidationError."""
        # Create a mock prediction with missing 'instruction'
        fake_pred = dspy.Prediction(
            context="Valid context.",
            role="Valid role.",
            # instruction is missing
            specification="Valid spec.",
            performance="Valid perf.",
            example="Valid example.",
        )

        program = create_crispe_program()
        with pytest.raises(PromptValidationError, match="Missing required CRISPE section: 'instruction'"):
            program.validate_and_extract_sections(fake_pred)

    def test_whitespace_section_raises_validation_error(self) -> None:
        """Prediction with whitespace-only section raises PromptValidationError."""
        fake_pred = dspy.Prediction(
            context="Valid context.",
            role="   \n\t   ",  # whitespace only
            instruction="Valid instruction.",
            specification="Valid spec.",
            performance="Valid perf.",
            example="Valid example.",
        )

        program = create_crispe_program()
        with pytest.raises(PromptValidationError, match="Section 'role' is empty or whitespace only"):
            program.validate_and_extract_sections(fake_pred)


# =============================================================================
# C. Concurrency and Context Isolation Tests
# =============================================================================

class TestConfigurationAndIsolation:
    """Verifies that model configuration is isolated per request without cross-thread pollution."""

    def test_model_name_normalization(self) -> None:
        """Model IDs are normalized to include provider prefix."""
        assert normalize_dspy_model_name("gemini-2.5-flash") == "gemini/gemini-2.5-flash"
        assert normalize_dspy_model_name("gemma-4-26b-a4b-it") == "gemini/gemma-4-26b-a4b-it"
        assert normalize_dspy_model_name("gemini/custom-model") == "gemini/custom-model"
        assert normalize_dspy_model_name("openai/gpt-4o") == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_concurrent_request_isolation(self) -> None:
        """Concurrent async executions use their respective isolated LMs without interference."""
        program = create_crispe_program()

        async def worker(lm_model: str, task_name: str) -> CrispeSections:
            mock_lm = MockDummyLM(
                model_name=lm_model,
                override_sections={
                    "context": f"Context for {task_name}",
                    "role": f"Role for {task_name}",
                    "instruction": f"Instruction for {task_name}",
                    "specification": f"Specification for {task_name}",
                    "performance": f"Performance for {task_name}",
                    "example": f"Example for {task_name}",
                },
            )
            # Use isolated context
            with dspy_isolated_context(lm=mock_lm):
                # Small async pause simulating network delay
                await asyncio.sleep(0.01)
                sections, _ = await program.execute_async(idea=f"Task {task_name}")
                return sections

        # Run 5 concurrent workers with different target configurations simultaneously
        results = await asyncio.gather(
            worker("model-1", "A"),
            worker("model-2", "B"),
            worker("model-3", "C"),
            worker("model-4", "D"),
            worker("model-5", "E"),
        )

        assert results[0].context == "Context for A"
        assert results[1].context == "Context for B"
        assert results[2].context == "Context for C"
        assert results[3].context == "Context for D"
        assert results[4].context == "Context for E"


# =============================================================================
# D. DspyCrispeGenerationService Parity Tests
# =============================================================================

class TestDspyGenerationService:
    """Verifies parity of DspyCrispeGenerationService with canonical API schema."""

    @pytest.mark.asyncio
    async def test_service_generates_canonical_response(self) -> None:
        """Service transforms CrispeGenerationRequest into valid CrispeGenerationResponse."""
        mock_lm = MockDummyLM(model_name="gemini/gemini-2.5-flash")
        service = DspyCrispeGenerationService(lm=mock_lm)

        request = CrispeGenerationRequest(
            idea="Design an event-driven notification microservice in Python",
        )

        response: CrispeGenerationResponse = await service.generate_crispe_prompt(request)

        assert isinstance(response, CrispeGenerationResponse)
        assert response.idea == request.idea
        assert isinstance(response.sections, CrispeSections)
        assert "# C - Context" in response.formatted_prompt
        assert response.model_id == "gemini/gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_service_translates_provider_auth_error(self) -> None:
        """Provider authentication error is translated into ProviderAuthenticationError."""
        mock_lm = MockDummyLM(should_fail=Exception("Authentication failed (invalid api_key)"))
        service = DspyCrispeGenerationService(lm=mock_lm)

        request = CrispeGenerationRequest(idea="Test idea")
        with pytest.raises(ProviderAuthenticationError, match="authentication failed"):
            await service.generate_crispe_prompt(request)

    @pytest.mark.asyncio
    async def test_service_translates_rate_limit_error(self) -> None:
        """Provider rate limit error is translated into ProviderRateLimitError."""
        mock_lm = MockDummyLM(should_fail=Exception("Rate limit 429 quota exceeded"))
        service = DspyCrispeGenerationService(lm=mock_lm)

        request = CrispeGenerationRequest(idea="Test idea")
        with pytest.raises(ProviderRateLimitError, match="rate limit exceeded"):
            await service.generate_crispe_prompt(request)


# =============================================================================
# E. Evaluation Adapter Integration Tests
# =============================================================================

class TestDspyEvaluationAdapter:
    """Verifies integration between DSPy predictions and Steps 20 & 21 evaluators."""

    def test_prepare_generator_inputs_boundary(self, sample_eval_example: EvaluationExample) -> None:
        """prepare_generator_inputs strictly extracts only allowed generation fields."""
        inputs = DspyEvaluationAdapter.prepare_generator_inputs(sample_eval_example)

        assert inputs["idea"] == sample_eval_example.user_idea
        assert inputs["task_context"] == sample_eval_example.task_context
        assert "zero message loss" not in inputs["constraints"].lower()
        assert sample_eval_example.expected_output_format in inputs["output_format_preference"]

        # Ensure no hidden evaluator checkpoint fields are present
        assert "acceptance_criteria" not in inputs
        assert "critical_requirements" not in inputs
        assert "prohibited_behaviors" not in inputs
        assert "grading_notes" not in inputs

    @pytest.mark.asyncio
    async def test_evaluate_prediction_deterministic_and_model(
        self,
        sample_eval_example: EvaluationExample,
    ) -> None:
        """DspyEvaluationAdapter evaluates prediction through deterministic and model evaluators."""
        prediction = dspy.Prediction(
            context="Operating on a 5-node cluster running Raft consensus.",
            role="Distributed Consensus Protocol Specialist.",
            instruction="Implement randomized election timeouts between 150ms and 300ms. Handle heartbeat broadcasting.",
            specification="Do not use third-party distributed lock managers. Avoid split-brain.",
            performance="Election completes within one term without network partitioning.",
            example="Sample raft timer loop.",
        )

        comb_res: CombinedEvaluationResult = await DspyEvaluationAdapter.evaluate_prediction(
            example=sample_eval_example,
            prediction=prediction,
            enable_model_judge=False,  # Deterministic only in unit test
        )

        assert comb_res.combined_status == "PARTIAL_SUCCESS"
        assert comb_res.deterministic_result.crispe_structure_valid is True
        assert comb_res.deterministic_score > 0.0
        assert comb_res.composite_score == comb_res.deterministic_score
