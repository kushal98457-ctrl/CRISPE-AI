"""Standard Information Retrieval (IR) evaluation metrics for Semantic Memory retrieval.

Provides mathematically verified implementations of:
- Precision@K
- Recall@K
- Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)
- Discounted Cumulative Gain (DCG@K)
- Normalized Discounted Cumulative Gain (NDCG@K)

Formulations follow standard IR literature (Manning et al., Introduction to Information Retrieval;
Järvelin & Kekäläinen, ACM TOIS 2002).
"""

import math
from typing import Any, Dict, Sequence, Set, Tuple


def precision_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Set[Any],
    k: int,
) -> float:
    """Calculate Precision@K: proportion of top-K retrieved items that are relevant.

    Formula:
        Precision@K = |{retrieved[0..k] ∩ relevant}| / K

    Boundary handling:
        - If k <= 0, returns 0.0.
        - If retrieved_ids is empty or relevant_ids is empty, returns 0.0.
        - Denominator is strictly K (not min(k, len(retrieved_ids))), penalizing
          systems that under-retrieve for the requested budget K.

    Parameters:
        retrieved_ids: Ordered sequence of retrieved document/prompt identifiers.
        relevant_ids: Set of known relevant document/prompt identifiers.
        k: Positive integer cutoff rank.

    Returns:
        Float precision score in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for item_id in top_k if item_id in relevant_ids)
    return relevant_retrieved / float(k)


def recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Set[Any],
    k: int,
) -> float:
    """Calculate Recall@K: proportion of all known relevant items retrieved in top-K.

    Formula:
        Recall@K = |{retrieved[0..k] ∩ relevant}| / |relevant|

    Boundary handling:
        - If k <= 0, returns 0.0.
        - If relevant_ids is empty, returns 0.0 (no relevant items exist to recall).
        - If retrieved_ids is empty, returns 0.0.

    Parameters:
        retrieved_ids: Ordered sequence of retrieved document/prompt identifiers.
        relevant_ids: Set of known relevant document/prompt identifiers.
        k: Positive integer cutoff rank.

    Returns:
        Float recall score in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for item_id in top_k if item_id in relevant_ids)
    return relevant_retrieved / float(len(relevant_ids))


def reciprocal_rank(
    retrieved_ids: Sequence[Any],
    relevant_ids: Set[Any],
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    Formula:
        RR = 1 / rank_first_relevant (1-indexed)
        RR = 0.0 if no relevant items appear in retrieved_ids.

    Boundary handling:
        - If retrieved_ids is empty or relevant_ids is empty, returns 0.0.

    Parameters:
        retrieved_ids: Ordered sequence of retrieved document/prompt identifiers.
        relevant_ids: Set of known relevant document/prompt identifiers.

    Returns:
        Float reciprocal rank in [0.0, 1.0].
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / float(rank)
    return 0.0


def mean_reciprocal_rank(
    query_rankings: Sequence[Tuple[Sequence[Any], Set[Any]]],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across multiple queries.

    Formula:
        MRR = (1 / |Q|) * Σ_{q ∈ Q} RR(q)

    Boundary handling:
        - If query_rankings is empty, returns 0.0.

    Parameters:
        query_rankings: Sequence of tuples, each containing (retrieved_ids, relevant_ids).

    Returns:
        Float MRR score in [0.0, 1.0].
    """
    if not query_rankings:
        return 0.0

    total_rr = sum(
        reciprocal_rank(retrieved_ids, relevant_ids)
        for retrieved_ids, relevant_ids in query_rankings
    )
    return total_rr / float(len(query_rankings))


def dcg_at_k(
    relevance_scores: Sequence[float],
    k: int,
) -> float:
    """Calculate Discounted Cumulative Gain at rank K using standard exponential gain.

    Formula:
        DCG@K = Σ_{i=1}^{min(K, N)} (2^{rel_i} - 1) / log2(i + 1)

    Logarithmic discount uses base 2:
        - Rank 1: (2^rel - 1) / log2(2) = (2^rel - 1) / 1.0
        - Rank 2: (2^rel - 1) / log2(3) = (2^rel - 1) / 1.58496
        - Rank 3: (2^rel - 1) / log2(4) = (2^rel - 1) / 2.0

    Boundary handling:
        - If k <= 0 or relevance_scores is empty, returns 0.0.
        - Negative relevance scores are treated as 0.0.

    Parameters:
        relevance_scores: Sequence of graded relevance scores in retrieval order.
        k: Positive integer cutoff rank.

    Returns:
        Float DCG score >= 0.0.
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    dcg = 0.0
    for rank, rel in enumerate(relevance_scores[:k], start=1):
        rel_clamped = max(0.0, float(rel))
        gain = (2.0 ** rel_clamped) - 1.0
        discount = math.log2(rank + 1)
        dcg += gain / discount

    return dcg


def ndcg_at_k(
    retrieved_ids: Sequence[Any],
    ground_truth: Dict[Any, float],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) for a query.

    Formula:
        NDCG@K = DCG@K / IDCG@K
        where IDCG@K is the Ideal DCG achievable by sorting all ground truth
        items descending by relevance.

    Boundary handling:
        - If k <= 0, returns 0.0.
        - If ground_truth contains no positive relevance items, IDCG is 0.0 and returns 0.0.
        - If retrieved_ids is empty, returns 0.0.
        - Items in retrieved_ids not found in ground_truth receive relevance 0.0.

    Parameters:
        retrieved_ids: Ordered sequence of retrieved document/prompt identifiers.
        ground_truth: Mapping from document/prompt identifier to graded relevance score (>= 0.0).
        k: Positive integer cutoff rank.

    Returns:
        Float NDCG score in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_ids or not ground_truth:
        return 0.0

    # 1. Compute actual DCG@K for retrieved ordering
    retrieved_scores = [
        max(0.0, float(ground_truth.get(item_id, 0.0)))
        for item_id in retrieved_ids[:k]
    ]
    actual_dcg = dcg_at_k(retrieved_scores, k)

    # 2. Compute Ideal DCG@K from ground truth items
    ideal_scores = sorted(
        [max(0.0, float(score)) for score in ground_truth.values() if score > 0.0],
        reverse=True,
    )
    idcg = dcg_at_k(ideal_scores, k)

    if idcg <= 0.0:
        return 0.0

    ndcg = actual_dcg / idcg
    return min(max(ndcg, 0.0), 1.0)
