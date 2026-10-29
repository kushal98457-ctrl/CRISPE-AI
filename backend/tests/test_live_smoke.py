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
