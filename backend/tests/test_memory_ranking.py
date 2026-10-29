"""Unit tests for Semantic Memory multi-signal ranking engine (Step 17).

Tests:
- RankingPolicy configuration and weight invariant validation.
- Individual signal computation (semantic, category, lexical, structural).
- Handling of absent or partial metadata.
- Deterministic candidate tie-breaking order.
- Top-K limit enforcement.
- Explainable score breakdown generation.
"""

from datetime import datetime, timezone
import uuid
import pytest

from app.core.exceptions import PromptValidationError
from app.schemas.crispe import CrispeSections
from app.schemas.semantic_search import SemanticSearchItem
from app.services.memory_ranking import (
    MemoryRanker,
    RankingPolicy,
    get_memory_ranker,
)


def make_test_item(
    *,
    prompt_id: uuid.UUID | None = None,
    original_idea: str = "Test prompt idea",
    similarity: float = 0.80,
    task_type: str | None = "coding",
    context: str = "A" * 30,
    role: str = "B" * 30,
    instruction: str = "C" * 30,
    specification: str = "D" * 30,
    performance: str = "E" * 30,
    example: str = "F" * 30,
    formatted_prompt: str = "Canonical formatted prompt text.",
    created_at: datetime | None = None,
) -> SemanticSearchItem:
    """Helper creating a valid SemanticSearchItem with configurable attributes."""
    pid = prompt_id or uuid.uuid4()
    sections = CrispeSections(
        context=context,
        role=role,
        instruction=instruction,
        specification=specification,
        performance=performance,
        example=example,
    )
    dist = round(1.0 - similarity, 6)
    return SemanticSearchItem(
        prompt_id=pid,
        original_idea=original_idea,
        sections=sections,
        formatted_prompt=formatted_prompt,
        task_type=task_type,
        model_id="text-embedding-004",
        distance=dist,
        similarity=similarity,
        config_version="v1",
        created_at=created_at or datetime.now(timezone.utc),
    )


# -----------------------------------------------------------------------------
# RankingPolicy Validation Tests
# -----------------------------------------------------------------------------

def test_ranking_policy_defaults() -> None:
    """Default policy has valid non-negative weights that sum to 1.0."""
    policy = RankingPolicy()
    assert policy.semantic_weight == 0.65
    assert policy.category_weight == 0.15
    assert policy.lexical_weight == 0.10
    assert policy.structural_weight == 0.10
    total = (
        policy.semantic_weight
        + policy.category_weight
        + policy.lexical_weight
        + policy.structural_weight
    )
    assert total == pytest.approx(1.0)


def test_ranking_policy_invalid_sum_rejected() -> None:
    """Weights that do not sum to 1.0 raise PromptValidationError."""
    with pytest.raises(PromptValidationError, match="must sum to 1.0"):
        RankingPolicy(semantic_weight=0.5, category_weight=0.2, lexical_weight=0.1, structural_weight=0.1)


def test_ranking_policy_negative_weight_rejected() -> None:
    """Negative weights raise PromptValidationError."""
    with pytest.raises(PromptValidationError, match="must be non-negative"):
        RankingPolicy(semantic_weight=1.1, category_weight=-0.1, lexical_weight=0.0, structural_weight=0.0)


# -----------------------------------------------------------------------------
# Signal Computation Tests
# -----------------------------------------------------------------------------

def test_semantic_score_clamping() -> None:
    """Semantic score extracts candidate similarity and clamps to [0.0, 1.0]."""
    ranker = get_memory_ranker()

    item_normal = make_test_item(similarity=0.85)
    assert ranker.compute_semantic_score(item_normal) == pytest.approx(0.85)

    item_negative = make_test_item(similarity=-0.2)
    assert ranker.compute_semantic_score(item_negative) == 0.0


def test_category_score_computation() -> None:
    """Category scoring returns 1.0 on match, 0.0 on mismatch, and 0.5 when metadata is missing."""
    ranker = get_memory_ranker()

    # Exact match (case insensitive)
    assert ranker.compute_category_score("coding", "Coding") == 1.0

    # Domain mismatch
    assert ranker.compute_category_score("creative_writing", "coding") == 0.0

    # Missing metadata handled with neutral 0.5 score
    assert ranker.compute_category_score(None, "coding") == 0.5
    assert ranker.compute_category_score("coding", None) == 0.5
    assert ranker.compute_category_score(None, None) == 0.5


def test_lexical_score_token_containment() -> None:
    """Lexical score accurately measures containment of query tokens in candidate text."""
    ranker = get_memory_ranker()

    item = make_test_item(
        original_idea="Debugging React JWT expired tokens in AuthContext",
        formatted_prompt="Interceptor refresh logic for Axios and React state.",
    )

    # Query with 3 content tokens: ["jwt", "expired", "tokens"]
    query = "JWT expired tokens"
    score = ranker.compute_lexical_score(query, item)
    assert score == pytest.approx(1.0)

    # Query with partial match: 3 of 4 tokens present
    query_partial = "handle JWT expired tokens"
    score_partial = ranker.compute_lexical_score(query_partial, item)
    assert score_partial == pytest.approx(0.75)

    # Query with non-matching keywords
    query_unrelated = "PostgreSQL foreign keys migration"
    score_unrelated = ranker.compute_lexical_score(query_unrelated, item)
    assert score_unrelated == pytest.approx(0.0)

    # Query with only stopwords
    assert ranker.compute_lexical_score("and or the in on", item) == 0.0


def test_structural_score_completeness() -> None:
    """Structural score measures proportion of substantial CRISPE sections."""
    ranker = get_memory_ranker()

    # Full 6 sections >= 20 chars
    item_full = make_test_item(
        context="A" * 25,
        role="B" * 25,
        instruction="C" * 25,
        specification="D" * 25,
        performance="E" * 25,
        example="F" * 25,
    )
    assert ranker.compute_structural_score(item_full) == pytest.approx(1.0)

    # Half populated: 3 substantial, 3 short (< 20 chars)
    item_half = make_test_item(
        context="A" * 25,
        role="Short",
        instruction="C" * 25,
        specification="Short",
        performance="E" * 25,
        example="",
    )
    assert ranker.compute_structural_score(item_half) == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# Multi-Signal Ranking & Tie-Breaking Tests
# -----------------------------------------------------------------------------

def test_rank_candidates_multi_signal_reordering() -> None:
    """Multi-signal ranking boosts candidate with domain and lexical match over pure similarity."""
    ranker = get_memory_ranker()

    # Candidate A: High vector similarity (0.90), but wrong domain and zero lexical match
    item_a = make_test_item(
        original_idea="General OAuth token overview",
        similarity=0.90,
        task_type="creative_writing",
        context="A" * 25,
        role="B" * 25,
        instruction="C" * 25,
        specification="D" * 25,
        performance="E" * 25,
        example="F" * 25,
        formatted_prompt="Overview text.",
    )

    # Candidate B: Slightly lower vector similarity (0.86), but exact domain match, exact lexical match, complete structure
    item_b = make_test_item(
        original_idea="Fix React Axios JWT expired refresh cycle",
        similarity=0.86,
        task_type="coding",
        context="A" * 25,
        role="B" * 25,
        instruction="C" * 25,
        specification="D" * 25,
        performance="E" * 25,
        example="F" * 25,
        formatted_prompt="React Axios JWT refresh cycle implementation.",
    )

    query = "React Axios JWT expired refresh"
    results = ranker.rank_candidates(
        query_text=query,
        candidates=[item_a, item_b],
        query_task_type="coding",
        top_k=2,
    )

    # Candidate B should outrank Candidate A due to category (1.0 vs 0.0) and lexical overlap (1.0 vs 0.0)
    assert len(results) == 2
    assert results[0].prompt_id == item_b.prompt_id
    assert results[1].prompt_id == item_a.prompt_id
    assert results[0].ranking_score is not None
    assert results[1].ranking_score is not None
    assert results[0].ranking_score > results[1].ranking_score


def test_rank_candidates_deterministic_tie_breaking() -> None:
    """Identical ranking scores break ties deterministically by similarity, created_at, then UUID."""
    ranker = get_memory_ranker()

    t_older = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    t_newer = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)

    id_1 = uuid.UUID("00000000-0000-0000-0000-000000000001")
    id_2 = uuid.UUID("00000000-0000-0000-0000-000000000002")

    # Two items with identical similarity and content, but different created_at
    item_older = make_test_item(prompt_id=id_1, similarity=0.80, created_at=t_older)
    item_newer = make_test_item(prompt_id=id_2, similarity=0.80, created_at=t_newer)

    ranked = ranker.rank_candidates(
        query_text="irrelevant",
        candidates=[item_older, item_newer],
        top_k=2,
    )

    # Newer item should be placed first
    assert ranked[0].prompt_id == id_2
    assert ranked[1].prompt_id == id_1


def test_rank_candidates_top_k_limit() -> None:
    """Ranker strictly enforces top_k limit."""
    ranker = get_memory_ranker()
    items = [make_test_item(similarity=0.70 + (i * 0.02)) for i in range(10)]

    ranked_3 = ranker.rank_candidates("query", items, top_k=3)
    assert len(ranked_3) == 3

    ranked_empty = ranker.rank_candidates("query", items, top_k=0)
    assert len(ranked_empty) == 0


def test_rank_candidates_score_breakdown_included() -> None:
    """Returned items include transparent score breakdown of all signals."""
    ranker = get_memory_ranker()
    item = make_test_item(similarity=0.80)

    ranked = ranker.rank_candidates("test query", [item], top_k=1)
    assert len(ranked) == 1
    assert ranked[0].score_breakdown is not None
    assert "semantic" in ranked[0].score_breakdown
    assert "category" in ranked[0].score_breakdown
    assert "lexical" in ranked[0].score_breakdown
    assert "structural" in ranked[0].score_breakdown
