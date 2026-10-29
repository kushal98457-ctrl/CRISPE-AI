"""Semantic Memory multi-signal ranking engine for CRISPE-AI.

Provides explainable, deterministic candidate reordering combining:
1. Semantic Similarity (cosine similarity from pgvector)
2. Task Category Compatibility (domain alignment)
3. Lexical Overlap (keyword containment in original idea and CRISPE prompt)
4. Structural Completeness (presence of substantial CRISPE sections)

Follows two-stage retrieval architecture:
Stage 1: Bounded candidate retrieval via pgvector ANN/exact cosine distance.
Stage 2: In-memory multi-signal ranking, deterministic tie-breaking, and Top-K cutoff.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Sequence, Set
import uuid

from app.core.exceptions import PromptValidationError
from app.core.logging import get_logger
from app.schemas.semantic_search import SemanticSearchItem

logger = get_logger("memory_ranking")

# Standard English stop words to filter out during lexical tokenization
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
    "does", "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your",
    "yours", "yourself", "yourselves",
}


@dataclass(frozen=True)
class RankingPolicy:
    """Configurable scoring weights and parameters for multi-signal ranking.

    Default Weights:
        - semantic_weight: 0.65 (primary semantic similarity signal)
        - category_weight: 0.15 (task domain alignment)
        - lexical_weight:  0.10 (exact token containment)
        - structural_weight: 0.10 (CRISPE section completeness)
    """

    semantic_weight: float = 0.65
    category_weight: float = 0.15
    lexical_weight: float = 0.10
    structural_weight: float = 0.10

    def __post_init__(self) -> None:
        """Validate that all weights are non-negative and sum approximately to 1.0."""
        weights = [
            self.semantic_weight,
            self.category_weight,
            self.lexical_weight,
            self.structural_weight,
        ]
        for w in weights:
            if w < 0.0:
                raise PromptValidationError(f"Ranking weights must be non-negative, got {w}")
        total = sum(weights)
        if abs(total - 1.0) > 1e-4:
            raise PromptValidationError(
                f"Ranking weights must sum to 1.0, got sum = {total:.4f}"
            )


@dataclass
class RankedCandidate:
    """Individual candidate prompt scored by the ranking engine."""

    prompt_id: uuid.UUID
    original_idea: str
    semantic_similarity: float
    category_score: float
    lexical_score: float
    structural_score: float
    ranking_score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    item: Optional[SemanticSearchItem] = None


class MemoryRanker:
    """In-memory multi-signal ranking engine for Semantic Memory candidates."""

    def __init__(self, policy: Optional[RankingPolicy] = None) -> None:
        self.policy = policy or RankingPolicy()

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Extract alphanumeric lowercase tokens of length >= 2, excluding stopwords."""
        if not text:
            return set()
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        return {tok for tok in raw_tokens if tok not in STOP_WORDS}

    def compute_semantic_score(self, candidate: SemanticSearchItem) -> float:
        """Extract and clamp semantic similarity score into [0.0, 1.0]."""
        # Cosine similarity is in [-1.0, 1.0]; clamp to [0.0, 1.0] for ranking
        return round(max(0.0, min(1.0, candidate.similarity)), 6)

    def compute_category_score(
        self,
        candidate_task_type: Optional[str],
        query_task_type: Optional[str],
    ) -> float:
        """Compute task category compatibility score.

        Scoring Rules:
        - If query_task_type is not specified or candidate_task_type is not specified:
          Return neutral 0.5 (unpenalized missing metadata).
        - If both are specified and match case-insensitively:
          Return 1.0 (positive match).
        - If both are specified and mismatch:
          Return 0.0 (incompatible domain).
        """
        if not query_task_type or not candidate_task_type:
            return 0.5

        if query_task_type.strip().lower() == candidate_task_type.strip().lower():
            return 1.0
        return 0.0

    def compute_lexical_score(
        self,
        query_text: str,
        candidate: SemanticSearchItem,
    ) -> float:
        """Compute lexical keyword containment of query tokens within candidate text.

        Checks candidate's original idea and CRISPE prompt body.
        Score = |query_tokens ∩ candidate_tokens| / |query_tokens|
        """
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return 0.0

        candidate_text = f"{candidate.original_idea} {candidate.formatted_prompt}"
        candidate_tokens = self._tokenize(candidate_text)

        overlap = query_tokens.intersection(candidate_tokens)
        return round(len(overlap) / float(len(query_tokens)), 6)

    def compute_structural_score(self, candidate: SemanticSearchItem) -> float:
        """Compute CRISPE prompt structural completeness score in [0.0, 1.0].

        Evaluates the 6 CRISPE sections (Context, Role, Instruction, Specification,
        Performance, Example). Each section with >= 20 characters contributes 1/6.
        """
        sections = candidate.sections
        if not sections:
            # Fallback based on formatted prompt depth
            length = len(candidate.formatted_prompt.strip())
            return round(min(1.0, max(0.0, length / 300.0)), 6)

        checked_sections = [
            sections.context,
            sections.role,
            sections.instruction,
            sections.specification,
            sections.performance,
            sections.example,
        ]
        substantial_count = sum(
            1 for s in checked_sections if s and len(s.strip()) >= 20
        )
        return round(substantial_count / 6.0, 6)

    def score_candidate(
        self,
        query_text: str,
        candidate: SemanticSearchItem,
        query_task_type: Optional[str] = None,
    ) -> RankedCandidate:
        """Calculate composite multi-signal score and individual breakdown for one candidate."""
        s_semantic = self.compute_semantic_score(candidate)
        s_category = self.compute_category_score(candidate.task_type, query_task_type)
        s_lexical = self.compute_lexical_score(query_text, candidate)
        s_structural = self.compute_structural_score(candidate)

        composite_score = (
            self.policy.semantic_weight * s_semantic
            + self.policy.category_weight * s_category
            + self.policy.lexical_weight * s_lexical
            + self.policy.structural_weight * s_structural
        )
        composite_score = round(min(1.0, max(0.0, composite_score)), 6)

        breakdown = {
            "semantic": s_semantic,
            "category": s_category,
            "lexical": s_lexical,
            "structural": s_structural,
        }

        return RankedCandidate(
            prompt_id=candidate.prompt_id,
            original_idea=candidate.original_idea,
            semantic_similarity=s_semantic,
            category_score=s_category,
            lexical_score=s_lexical,
            structural_score=s_structural,
            ranking_score=composite_score,
            score_breakdown=breakdown,
            item=candidate,
        )

    def rank_candidates(
        self,
        query_text: str,
        candidates: Sequence[SemanticSearchItem],
        query_task_type: Optional[str] = None,
        top_k: int = 10,
    ) -> List[SemanticSearchItem]:
        """Rank a candidate set of SemanticSearchItem objects and return top-K items.

        Deterministic Tie-Breaking Policy:
        1. Composite ranking_score (descending)
        2. Vector semantic similarity (descending)
        3. Creation timestamp (descending, newer preferred among identical scores)
        4. Prompt UUID string (ascending, deterministic stable order)

        Parameters:
            query_text: Raw user search query string.
            candidates: Sequence of retrieved candidate SemanticSearchItem objects.
            query_task_type: Optional user task domain filter/tag.
            top_k: Maximum number of reordered items to return.

        Returns:
            List of top-K SemanticSearchItem objects with ranking_score and score_breakdown.
        """
        if not candidates or top_k <= 0:
            return []

        scored_candidates: List[RankedCandidate] = [
            self.score_candidate(query_text, c, query_task_type=query_task_type)
            for c in candidates
        ]

        def sort_key(rc: RankedCandidate) -> tuple:
            created_ts = 0.0
            if rc.item and rc.item.created_at:
                if rc.item.created_at.tzinfo is None:
                    created_ts = rc.item.created_at.replace(tzinfo=timezone.utc).timestamp()
                else:
                    created_ts = rc.item.created_at.timestamp()
            return (
                -rc.ranking_score,
                -rc.semantic_similarity,
                -created_ts,
                str(rc.prompt_id),
            )

        scored_candidates.sort(key=sort_key)
        top_scored = scored_candidates[:top_k]

        result_items: List[SemanticSearchItem] = []
        for rc in top_scored:
            if rc.item is not None:
                # Update candidate item with ranking metadata
                updated_item = rc.item.model_copy(
                    update={
                        "ranking_score": rc.ranking_score,
                        "score_breakdown": rc.score_breakdown,
                    }
                )
                result_items.append(updated_item)

        return result_items


def get_memory_ranker(policy: Optional[RankingPolicy] = None) -> MemoryRanker:
    """Factory creating a MemoryRanker instance."""
    return MemoryRanker(policy=policy)
