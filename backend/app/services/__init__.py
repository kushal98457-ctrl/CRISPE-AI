"""Backend domain and application services for CRISPE-AI."""

from app.services.artifact_storage import (
    ArtifactStorageService,
    get_artifact_storage_service,
)
from app.services.candidate_persistence import (
    CandidatePersistenceService,
    get_candidate_persistence_service,
)
from app.services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    get_embedding_service_dependency,
)
from app.services.generation_service import (
    CrispeGenerationService,
    get_generation_service,
    get_generation_service_dependency,
)
from app.services.memory_ranking import (
    MemoryRanker,
    RankedCandidate,
    RankingPolicy,
    get_memory_ranker,
)
from app.services.retrieval_metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.services.semantic_search import (
    SemanticSearchService,
    get_semantic_search_service,
    get_semantic_search_service_dependency,
)

__all__ = [
    "ArtifactStorageService",
    "get_artifact_storage_service",
    "CandidatePersistenceService",
    "get_candidate_persistence_service",
    "CrispeGenerationService",
    "get_generation_service",
    "get_generation_service_dependency",
    "EmbeddingService",
    "get_embedding_service",
    "get_embedding_service_dependency",
    "SemanticSearchService",
    "get_semantic_search_service",
    "get_semantic_search_service_dependency",
    "MemoryRanker",
    "RankingPolicy",
    "RankedCandidate",
    "get_memory_ranker",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
]
