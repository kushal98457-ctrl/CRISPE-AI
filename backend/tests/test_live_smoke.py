"""Optional opt-in live smoke test for Google GenAI integration.

This test is skipped by default in all automated test runs and CI pipelines.
It executes ONLY when:
1. RUN_LIVE_TESTS=1 is set in the environment.
2. A valid GEMINI_API_KEY (or GOOGLE_API_KEY) is available.

Never logs or prints real API credentials.
"""

import math
import os
import pytest
from app.adapters.embedding_adapter import GoogleGenAIEmbeddingAdapter
from app.adapters.gemini_adapter import GeminiModelAdapter
from app.core.config import get_settings
from app.schemas.crispe import CrispeGenerationRequest
from app.schemas.embedding import EmbeddingTaskType
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import CrispeGenerationService

RUN_LIVE = os.getenv("RUN_LIVE_TESTS", "0") == "1"
HAS_KEY = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


@pytest.mark.skipif(
    not (RUN_LIVE and HAS_KEY),
    reason="Live smoke test requires RUN_LIVE_TESTS=1 and a configured GEMINI_API_KEY.",
)
@pytest.mark.asyncio
async def test_live_crispe_generation_smoke() -> None:
    """Opt-in live smoke test verifying real end-to-end communication with Google AI."""
    settings = get_settings()
    adapter = GeminiModelAdapter(settings=settings)
    service = CrispeGenerationService(adapter=adapter)

    request = CrispeGenerationRequest(
        idea="A python script to calculate Fibonacci numbers efficiently",
        temperature=0.2,
    )

    try:
        response = await service.generate_crispe_prompt(request)
        assert response.sections.context
        assert response.sections.role
        assert response.sections.instruction
        assert response.sections.specification
        assert response.sections.performance
        assert response.sections.example
        assert "# C - Context" in response.formatted_prompt
    finally:
        await adapter.close()


@pytest.mark.skipif(
    not (RUN_LIVE and HAS_KEY),
    reason="Live smoke test requires RUN_LIVE_TESTS=1 and a configured GEMINI_API_KEY.",
)
@pytest.mark.asyncio
async def test_live_embedding_smoke() -> None:
    """Opt-in live smoke test verifying real embedding vector generation via Google GenAI."""
    settings = get_settings()
    adapter = GoogleGenAIEmbeddingAdapter(settings=settings)
    service = EmbeddingService(settings=settings, adapter=adapter)

    test_content = "CRISPE framework context optimization for semantic search"

    try:
        result = await service.embed_text(
            test_content,
            task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        )
        assert result.dimension == 768
        assert len(result.vector) == 768
        assert all(isinstance(v, float) and math.isfinite(v) for v in result.vector)
        assert result.model_id == "text-embedding-004"
        assert result.config_version == "v1"
    finally:
        await adapter.close()


@pytest.mark.skipif(
    not (RUN_LIVE and HAS_KEY),
    reason="Live smoke test requires RUN_LIVE_TESTS=1 and a configured GEMINI_API_KEY.",
)
@pytest.mark.asyncio
async def test_live_judge_evaluation_smoke() -> None:
    """Opt-in live smoke test verifying real LLM-as-a-judge evaluation via Google GenAI (Step 21)."""
    from app.evaluation.judge_schemas import ModelEvaluationResult
    from app.evaluation.model_evaluator import ModelPromptEvaluator
    from app.evaluation.schemas import EvaluationExample, TaskDifficulty, TaskType
    from app.schemas.crispe import CrispeSections

    settings = get_settings()
    adapter = GeminiModelAdapter(settings=settings)
    evaluator = ModelPromptEvaluator(adapter=adapter, settings=settings)

    test_example = EvaluationExample(
        example_id="live_smoke_test_01",
        task_family="smoke_testing",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        difficulty=TaskDifficulty.EASY,
        tags=["smoke", "testing"],
        example_version="v1",
        user_idea="Create a prompt for a Python CLI that prints system uptime.",
        task_context="Standard Linux sysadmin workstation.",
        required_behaviors=["Use Python 3.12 standard library", "Read from /proc/uptime"],
        constraints=["Do not require root privileges"],
        expected_output_format="Python script snippet",
        ambiguity_notes="None.",
        acceptance_criteria=["Uses /proc/uptime", "Standard library only"],
        critical_requirements=["Must not require root privileges"],
        prohibited_behaviors=["Do not use external pip dependencies"],
        grading_notes="Check for /proc/uptime and standard library only.",
    )

    test_sections = CrispeSections(
        context="Operating on a non-root Linux workstation.",
        role="Senior Linux Systems Engineer.",
        instruction="Write a Python 3.12 CLI script that reads /proc/uptime and prints human-readable uptime.",
        specification="Standard library only (e.g. pathlib, datetime). Do not use external packages. Do not require root.",
        performance="Script must execute in < 100ms.",
        example="Sample output: 'System uptime: 3 days, 4 hours, 12 minutes'.",
    )

    try:
        result: ModelEvaluationResult = await evaluator.evaluate_prompt(test_example, test_sections)
        assert result.status == "SUCCESS"
        assert result.mean_dimension_score >= 2.5
        assert result.normalized_model_score >= 0.60
        assert len(result.dimensions) >= 5
        assert result.evaluator_model_id
    finally:
        await adapter.close()
