"""Unit tests verifying mathematical correctness and boundary handling of IR metrics (Step 17).

Tests:
- Precision@K
- Recall@K
- Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)
- Discounted Cumulative Gain (DCG@K)
- Normalized Discounted Cumulative Gain (NDCG@K)
"""

import math
import pytest

from app.services.retrieval_metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# -----------------------------------------------------------------------------
# Precision@K Tests
# -----------------------------------------------------------------------------

def test_precision_at_k_normal_case() -> None:
    """Precision@K correctly calculates the ratio of relevant items in top-K."""
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    relevant = {"doc_b", "doc_c", "doc_x"}

    # Top-1: ["doc_a"] -> 0 / 1 = 0.0
    assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)

    # Top-2: ["doc_a", "doc_b"] -> 1 / 2 = 0.5
    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)

    # Top-3: ["doc_a", "doc_b", "doc_c"] -> 2 / 3 = 0.66667
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2.0 / 3.0)

    # Top-4: ["doc_a", "doc_b", "doc_c", "doc_d"] -> 2 / 4 = 0.5
    assert precision_at_k(retrieved, relevant, k=4) == pytest.approx(0.5)


def test_precision_at_k_under_retrieval_penalization() -> None:
    """Precision@K strictly uses K as denominator even when retrieved list is shorter than K."""
    retrieved = ["doc_a", "doc_b"]
    relevant = {"doc_a", "doc_b"}

    # System requested 5 items, returned only 2 (both relevant) -> 2 / 5 = 0.4
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.4)


def test_precision_at_k_boundary_conditions() -> None:
    """Precision@K gracefully returns 0.0 for empty or invalid boundaries."""
    assert precision_at_k([], {"doc_a"}, k=5) == 0.0
    assert precision_at_k(["doc_a"], set(), k=5) == 0.0
    assert precision_at_k(["doc_a"], {"doc_a"}, k=0) == 0.0
    assert precision_at_k(["doc_a"], {"doc_a"}, k=-1) == 0.0


# -----------------------------------------------------------------------------
# Recall@K Tests
# -----------------------------------------------------------------------------

def test_recall_at_k_normal_case() -> None:
    """Recall@K calculates ratio of all relevant items successfully retrieved in top-K."""
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    relevant = {"doc_b", "doc_c", "doc_x"}  # 3 total relevant items

    # Top-1: ["doc_a"] -> 0 / 3 = 0.0
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)

    # Top-2: ["doc_a", "doc_b"] -> 1 / 3 = 0.33333
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1.0 / 3.0)

    # Top-3: ["doc_a", "doc_b", "doc_c"] -> 2 / 3 = 0.66667
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2.0 / 3.0)


def test_recall_at_k_boundary_conditions() -> None:
    """Recall@K gracefully returns 0.0 for empty or invalid boundaries."""
    assert recall_at_k([], {"doc_a"}, k=5) == 0.0
    assert recall_at_k(["doc_a"], set(), k=5) == 0.0
    assert recall_at_k(["doc_a"], {"doc_a"}, k=0) == 0.0
    assert recall_at_k(["doc_a"], {"doc_a"}, k=-2) == 0.0


# -----------------------------------------------------------------------------
# Reciprocal Rank & MRR Tests
# -----------------------------------------------------------------------------

def test_reciprocal_rank() -> None:
    """Reciprocal rank returns 1/rank of first relevant item."""
    relevant = {"doc_target"}

    # Rank 1 -> 1 / 1 = 1.0
    assert reciprocal_rank(["doc_target", "doc_x"], relevant) == pytest.approx(1.0)

    # Rank 2 -> 1 / 2 = 0.5
    assert reciprocal_rank(["doc_other", "doc_target"], relevant) == pytest.approx(0.5)

    # Rank 4 -> 1 / 4 = 0.25
    assert reciprocal_rank(["d1", "d2", "d3", "doc_target"], relevant) == pytest.approx(0.25)

    # Not found -> 0.0
    assert reciprocal_rank(["d1", "d2"], relevant) == 0.0
    assert reciprocal_rank([], relevant) == 0.0
    assert reciprocal_rank(["d1"], set()) == 0.0


def test_mean_reciprocal_rank() -> None:
    """MRR computes the arithmetic mean of reciprocal ranks across queries."""
    query_rankings = [
        (["d1", "d2"], {"d1"}),          # RR = 1.0
        (["d1", "d2", "d3"], {"d2"}),    # RR = 0.5
        (["d1", "d2", "d3"], {"d3"}),    # RR = 0.333333
        (["d1", "d2"], {"d99"}),         # RR = 0.0
    ]
    # Sum = 1.0 + 0.5 + 0.333333 + 0.0 = 1.833333
    # Mean = 1.833333 / 4 = 0.458333
    expected_mrr = (1.0 + 0.5 + (1.0 / 3.0) + 0.0) / 4.0
    assert mean_reciprocal_rank(query_rankings) == pytest.approx(expected_mrr)

    # Empty queries -> 0.0
    assert mean_reciprocal_rank([]) == 0.0


# -----------------------------------------------------------------------------
# DCG@K Tests
# -----------------------------------------------------------------------------

def test_dcg_at_k_hand_calculated() -> None:
    """DCG@K adheres to standard exponential gain formula: (2^rel - 1) / log2(rank + 1).

    Hand Proof:
    Scores = [3.0, 2.0, 1.0], K=3
    Rank 1: (2^3 - 1) / log2(2) = 7.0 / 1.0 = 7.0
    Rank 2: (2^2 - 1) / log2(3) = 3.0 / 1.58496250 = 1.89278926
    Rank 3: (2^1 - 1) / log2(4) = 1.0 / 2.0 = 0.5
    Total DCG@3 = 7.0 + 1.89278926 + 0.5 = 9.39278926
    """
    scores = [3.0, 2.0, 1.0]
    expected_dcg3 = 7.0 + (3.0 / math.log2(3)) + 0.5
    assert dcg_at_k(scores, k=3) == pytest.approx(expected_dcg3, rel=1e-5)

    # K=1 only sums first item
    assert dcg_at_k(scores, k=1) == pytest.approx(7.0)

    # K=2 sums first two items
    expected_dcg2 = 7.0 + (3.0 / math.log2(3))
    assert dcg_at_k(scores, k=2) == pytest.approx(expected_dcg2, rel=1e-5)


def test_dcg_at_k_boundary_conditions() -> None:
    """DCG@K handles negative, empty, or non-positive K gracefully."""
    assert dcg_at_k([], k=5) == 0.0
    assert dcg_at_k([3.0, 2.0], k=0) == 0.0
    assert dcg_at_k([3.0, 2.0], k=-3) == 0.0

    # Negative relevance scores clamped to 0.0
    assert dcg_at_k([-1.0, -2.0], k=2) == 0.0


# -----------------------------------------------------------------------------
# NDCG@K Tests
# -----------------------------------------------------------------------------

def test_ndcg_at_k_perfect_ranking() -> None:
    """Perfect ranking where items are ordered by descending relevance yields NDCG=1.0."""
    retrieved = ["doc_3", "doc_2", "doc_1"]
    ground_truth = {"doc_3": 3.0, "doc_2": 2.0, "doc_1": 1.0}

    assert ndcg_at_k(retrieved, ground_truth, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(retrieved, ground_truth, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(retrieved, ground_truth, k=1) == pytest.approx(1.0)


def test_ndcg_at_k_suboptimal_ranking() -> None:
    """Suboptimal ordering yields mathematically expected NDCG < 1.0.

    Proof:
    Ground truth: {A: 3.0, B: 2.0, C: 1.0} -> IDCG@3 = 9.39278926
    Retrieved: [C, B, A] -> Scores: [1.0, 2.0, 3.0]
    Rank 1: (2^1 - 1) / log2(2) = 1.0
    Rank 2: (2^2 - 1) / log2(3) = 3.0 / 1.5849625 = 1.89278926
    Rank 3: (2^3 - 1) / log2(4) = 7.0 / 2.0 = 3.5
    Actual DCG@3 = 1.0 + 1.89278926 + 3.5 = 6.39278926
    NDCG@3 = 6.39278926 / 9.39278926 = 0.680606
    """
    retrieved = ["doc_c", "doc_b", "doc_a"]
    ground_truth = {"doc_a": 3.0, "doc_b": 2.0, "doc_c": 1.0}

    expected_idcg = 7.0 + (3.0 / math.log2(3)) + 0.5
    expected_actual = 1.0 + (3.0 / math.log2(3)) + 3.5
    expected_ndcg = expected_actual / expected_idcg

    assert ndcg_at_k(retrieved, ground_truth, k=3) == pytest.approx(expected_ndcg, rel=1e-5)


def test_ndcg_at_k_boundary_conditions() -> None:
    """NDCG@K handles boundary cases without division by zero."""
    ground_truth = {"doc_a": 3.0}

    assert ndcg_at_k([], ground_truth, k=5) == 0.0
    assert ndcg_at_k(["doc_a"], {}, k=5) == 0.0
    assert ndcg_at_k(["doc_a"], ground_truth, k=0) == 0.0
    assert ndcg_at_k(["doc_a"], ground_truth, k=-1) == 0.0

    # Ground truth with only 0.0 relevance has IDCG = 0.0 -> returns 0.0
    assert ndcg_at_k(["doc_a"], {"doc_a": 0.0}, k=1) == 0.0
