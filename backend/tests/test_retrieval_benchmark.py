"""Deterministic Semantic Retrieval Quality Benchmark for CRISPE-AI (Step 17).

Compares Baseline Retrieval (pure vector cosine distance) against Multi-Signal Ranked Retrieval
across 5 technical domains:
1. React authentication & token refresh
2. Python asyncio testing & event loop diagnosis
3. PostgreSQL schema design & relational cascades
4. Numerical mathematics & Runge-Kutta integration
5. Technical documentation & OpenAPI specifications

Measures standard IR metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Set
import uuid
import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.embedding_adapter import BaseEmbeddingAdapter
from app.core.security import AuthorizedPrincipal
from app.db.base import Base
from app.db.sqlite_compat import register_sqlite_vector_compat
from app.models.principal import Principal
from app.models.prompt_embedding import PromptEmbedding
from app.models.prompt_record import PromptRecord
from app.schemas.crispe import CrispeSections
from app.schemas.embedding import (
    EmbeddingBatchResult,
    EmbeddingResult,
    EmbeddingTaskType,
    EmbeddingUsageMetadata,
)
from app.services.embedding_service import EmbeddingService
from app.services.memory_ranking import MemoryRanker, RankingPolicy
from app.services.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.services.semantic_search import SemanticSearchService


def make_cluster_unit_vector(start_idx: int, end_idx: int, dim: int = 768, noise_seed: int = 0) -> List[float]:
    """Generate a deterministic unit-length vector with non-zero coordinates in [start_idx, end_idx)."""
    vec = [0.0] * dim
    for i in range(start_idx, end_idx):
        vec[i] = 1.0 + (math.sin(i + noise_seed) * 0.05)
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class BenchmarkEmbeddingAdapter(BaseEmbeddingAdapter):
    """Deterministic routing adapter returning targeted synthetic cluster embeddings."""

    def __init__(self, query_routes: Dict[str, List[float]], fallback_dim: int = 768) -> None:
        self.query_routes = query_routes
        self.fallback_dim = fallback_dim

    async def embed_content(
        self,
        text: str,
        *,
        task_type: EmbeddingTaskType | None = None,
        title: str | None = None,
        output_dimensionality: int | None = None,
    ) -> EmbeddingResult:
        vec = self.query_routes.get(text.strip())
        if vec is None:
            # Fallback orthogonal vector
            vec = [0.05] * self.fallback_dim

        return EmbeddingResult(
            vector=list(vec),
            dimension=len(vec),
            model_id="text-embedding-004",
            config_version="v1",
            usage=EmbeddingUsageMetadata(prompt_tokens=15, total_tokens=15),
        )

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        task_type: EmbeddingTaskType | None = None,
        output_dimensionality: int | None = None,
    ) -> EmbeddingBatchResult:
        res = [await self.embed_content(t, task_type=task_type) for t in texts]
        return EmbeddingBatchResult(
            embeddings=res,
            model_id="text-embedding-004",
            config_version="v1",
            total_prompt_tokens=15 * len(texts),
        )

    async def close(self) -> None:
        pass


@dataclass
class BenchmarkItemDef:
    title: str
    relevance_grade: float  # 3.0 = highly relevant, 2.0 = relevant, 0.0 = superficial distractor
    task_type: str
    is_complete_structure: bool
    cluster_range: tuple[int, int]
    noise_seed: int


@dataclass
class BenchmarkTask:
    domain_id: str
    query_text: str
    active_range: tuple[int, int]
    items: List[BenchmarkItemDef]


BENCHMARK_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        domain_id="react_auth",
        query_text="React Axios JWT expired tokens and AuthContext refresh cycles",
        active_range=(0, 80),
        items=[
            BenchmarkItemDef(
                title="React Axios interceptor for JWT silent refresh and token expiry",
                relevance_grade=3.0,
                task_type="react_auth",
                is_complete_structure=True,
                cluster_range=(0, 80),
                noise_seed=1,
            ),
            BenchmarkItemDef(
                title="Managing user session logout in React AuthProvider upon 401 response",
                relevance_grade=2.0,
                task_type="react_auth",
                is_complete_structure=True,
                cluster_range=(0, 80),
                noise_seed=2,
            ),
            BenchmarkItemDef(
                title="General scratchpad note on web sessions",
                relevance_grade=0.0,
                task_type="general",
                is_complete_structure=False,
                cluster_range=(0, 80),
                noise_seed=100,  # Matches query vector noise, causing false positive in pure cosine
            ),
        ],
    ),
    BenchmarkTask(
        domain_id="python_async",
        query_text="Python asyncio event loop closed RuntimeError in pytest test suite",
        active_range=(80, 160),
        items=[
            BenchmarkItemDef(
                title="Resolving Event loop is closed in pytest-asyncio with event_loop fixture",
                relevance_grade=3.0,
                task_type="python_async",
                is_complete_structure=True,
                cluster_range=(80, 160),
                noise_seed=1,
            ),
            BenchmarkItemDef(
                title="Debugging unclosed transport and async task cancellation in Python 3.12",
                relevance_grade=2.0,
                task_type="python_async",
                is_complete_structure=True,
                cluster_range=(80, 160),
                noise_seed=2,
            ),
            BenchmarkItemDef(
                title="Python general unit test cheat sheet",
                relevance_grade=0.0,
                task_type="general",
                is_complete_structure=False,
                cluster_range=(80, 160),
                noise_seed=100,  # Matches query vector noise
            ),
        ],
    ),
    BenchmarkTask(
        domain_id="db_schema",
        query_text="PostgreSQL relational schema design with foreign key cascades and composite indexes",
        active_range=(160, 240),
        items=[
            BenchmarkItemDef(
                title="PostgreSQL multi-tenant schema with tenant_id foreign keys and ON DELETE CASCADE",
                relevance_grade=3.0,
                task_type="db_schema",
                is_complete_structure=True,
                cluster_range=(160, 240),
                noise_seed=1,
            ),
            BenchmarkItemDef(
                title="Composite index strategies for high-volume time-series PostgreSQL tables",
                relevance_grade=2.0,
                task_type="db_schema",
                is_complete_structure=True,
                cluster_range=(160, 240),
                noise_seed=2,
            ),
            BenchmarkItemDef(
                title="Relational database tables naming draft",
                relevance_grade=0.0,
                task_type="general",
                is_complete_structure=False,
                cluster_range=(160, 240),
                noise_seed=100,  # Matches query vector noise
            ),
        ],
    ),
    BenchmarkTask(
        domain_id="engineering_math",
        query_text="Numerical differential equation integration with 4th order Runge-Kutta",
        active_range=(240, 320),
        items=[
            BenchmarkItemDef(
                title="Implementing Runge-Kutta 4th order (RK4) integration algorithm in Python",
                relevance_grade=3.0,
                task_type="engineering_math",
                is_complete_structure=True,
                cluster_range=(240, 320),
                noise_seed=1,
            ),
            BenchmarkItemDef(
                title="Solving stiff orbital mechanics differential equations numerically",
                relevance_grade=2.0,
                task_type="engineering_math",
                is_complete_structure=True,
                cluster_range=(240, 320),
                noise_seed=2,
            ),
            BenchmarkItemDef(
                title="Basic math equations list",
                relevance_grade=0.0,
                task_type="general",
                is_complete_structure=False,
                cluster_range=(240, 320),
                noise_seed=100,  # Matches query vector noise
            ),
        ],
    ),
    BenchmarkTask(
        domain_id="technical_docs",
        query_text="FastAPI OpenAPI 3.1 specifications and endpoint documentation schema",
        active_range=(320, 400),
        items=[
            BenchmarkItemDef(
                title="FastAPI OpenAPI 3.1 schema customization and endpoint documentation",
                relevance_grade=3.0,
                task_type="technical_docs",
                is_complete_structure=True,
                cluster_range=(320, 400),
                noise_seed=1,
            ),
            BenchmarkItemDef(
                title="Architectural Decision Record (ADR) template and decision capture workflow",
                relevance_grade=2.0,
                task_type="technical_docs",
                is_complete_structure=True,
                cluster_range=(320, 400),
                noise_seed=2,
            ),
            BenchmarkItemDef(
                title="General markdown formatting rules",
                relevance_grade=0.0,
                task_type="general",
                is_complete_structure=False,
                cluster_range=(320, 400),
                noise_seed=100,  # Matches query vector noise
            ),
        ],
    ),
]


@pytest.mark.asyncio
async def test_deterministic_semantic_retrieval_benchmark() -> None:
    """Execute reproducible retrieval benchmark comparing Baseline vs Multi-Signal Ranked Retrieval.

    Evaluates Precision@2, Recall@2, MRR, and NDCG@2 across 5 technical domains.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    register_sqlite_vector_compat(engine)

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    principal_id = "usr_benchmark_runner"

    query_routes: Dict[str, List[float]] = {}
    ground_truth_relevance: Dict[str, Dict[uuid.UUID, float]] = {}
    ground_truth_binary: Dict[str, Set[uuid.UUID]] = {}

    complete_sections = CrispeSections(
        context="Comprehensive enterprise production environment context.",
        role="Principal software engineer and technical architect.",
        instruction="Provide exact, fully working, verified implementation.",
        specification="Adhere to strict typing, error handling, and performance standards.",
        performance="Zero memory leaks, deterministic outputs, sub-millisecond response.",
        example="Working, tested code blocks demonstrating verified edge case handling.",
    )

    partial_sections = CrispeSections(
        context="Brief note.",
        role="Writer.",
        instruction="Do something.",
        specification="",
        performance="",
        example="",
    )

    async with session_factory() as session:
        # Create security principal
        db_principal = Principal(id=principal_id, email="benchmark@internal", is_active=True)
        session.add(db_principal)
        await session.commit()

        principal = AuthorizedPrincipal(id=principal_id, is_active=True)

        for task in BENCHMARK_TASKS:
            ground_truth_relevance[task.domain_id] = {}
            ground_truth_binary[task.domain_id] = set()

            # Query vector in task cluster range
            q_vec = make_cluster_unit_vector(task.active_range[0], task.active_range[1], noise_seed=100)
            query_routes[task.query_text] = q_vec

            for item_def in task.items:
                sec = complete_sections if item_def.is_complete_structure else partial_sections
                p = PromptRecord(
                    original_idea=item_def.title,
                    context=sec.context,
                    role=sec.role,
                    instruction=sec.instruction,
                    specification=sec.specification,
                    performance=sec.performance,
                    example=sec.example,
                    formatted_prompt=f"# {item_def.title}\n{sec.instruction}\n{sec.specification}",
                    task_type=item_def.task_type,
                    owner_id=principal.id,
                )
                session.add(p)
                await session.flush()

                # Register ground truth
                ground_truth_relevance[task.domain_id][p.id] = item_def.relevance_grade
                if item_def.relevance_grade >= 2.0:
                    ground_truth_binary[task.domain_id].add(p.id)

                p_vec = make_cluster_unit_vector(
                    item_def.cluster_range[0],
                    item_def.cluster_range[1],
                    noise_seed=item_def.noise_seed,
                )
                emb = PromptEmbedding(
                    prompt_id=p.id,
                    embedding=p_vec,
                    source_content_hash=PromptEmbedding.compute_content_hash(p.formatted_prompt),
                )
                session.add(emb)

        await session.commit()

    # Initialize services
    mock_adapter = BenchmarkEmbeddingAdapter(query_routes=query_routes)
    embedding_svc = EmbeddingService(adapter=mock_adapter)
    search_service = SemanticSearchService(
        embedding_service=embedding_svc,
        ranker=MemoryRanker(policy=RankingPolicy()),
    )

    k = 2

    # Track metrics for Baseline (enable_ranking=False) and Ranked (enable_ranking=True)
    baseline_p_list: List[float] = []
    baseline_r_list: List[float] = []
    baseline_rr_list: List[float] = []
    baseline_ndcg_list: List[float] = []

    ranked_p_list: List[float] = []
    ranked_r_list: List[float] = []
    ranked_rr_list: List[float] = []
    ranked_ndcg_list: List[float] = []

    print("\n" + "=" * 92)
    print("CRISPE-AI RETRIEVAL QUALITY BENCHMARK: BASELINE vs MULTI-SIGNAL RANKING (STEP 17)")
    print("=" * 92)
    print(f"{'Domain':<18} | {'Base P@2':<8} | {'Rank P@2':<8} | {'Base NDCG':<9} | {'Rank NDCG':<9} | {'Rank Top Match Title':<30}")
    print("-" * 92)

    async with session_factory() as session:
        for task in BENCHMARK_TASKS:
            rel_binary = ground_truth_binary[task.domain_id]
            rel_graded = ground_truth_relevance[task.domain_id]

            # 1. Baseline retrieval (pure cosine similarity, ranking disabled)
            base_res = await search_service.search(
                session,
                principal=principal,
                query=task.query_text,
                limit=k,
                enable_ranking=False,
            )
            base_ids = [item.prompt_id for item in base_res.items]
            base_p = precision_at_k(base_ids, rel_binary, k=k)
            base_r = recall_at_k(base_ids, rel_binary, k=k)
            base_rr = reciprocal_rank(base_ids, rel_binary)
            base_ndcg = ndcg_at_k(base_ids, rel_graded, k=k)

            baseline_p_list.append(base_p)
            baseline_r_list.append(base_r)
            baseline_rr_list.append(base_rr)
            baseline_ndcg_list.append(base_ndcg)

            # 2. Multi-signal ranked retrieval (ranking enabled)
            ranked_res = await search_service.search(
                session,
                principal=principal,
                query=task.query_text,
                limit=k,
                task_type=task.domain_id,
                enable_ranking=True,
            )
            ranked_ids = [item.prompt_id for item in ranked_res.items]
            ranked_p = precision_at_k(ranked_ids, rel_binary, k=k)
            ranked_r = recall_at_k(ranked_ids, rel_binary, k=k)
            ranked_rr = reciprocal_rank(ranked_ids, rel_binary)
            ranked_ndcg = ndcg_at_k(ranked_ids, rel_graded, k=k)

            ranked_p_list.append(ranked_p)
            ranked_r_list.append(ranked_r)
            ranked_rr_list.append(ranked_rr)
            ranked_ndcg_list.append(ranked_ndcg)

            top_title = ranked_res.items[0].original_idea if ranked_res.items else "None"
            print(
                f"{task.domain_id:<18} | {base_p:<8.2f} | {ranked_p:<8.2f} | {base_ndcg:<9.4f} | {ranked_ndcg:<9.4f} | {top_title[:30]}"
            )

    mean_base_p = sum(baseline_p_list) / len(baseline_p_list)
    mean_base_r = sum(baseline_r_list) / len(baseline_r_list)
    mean_base_mrr = sum(baseline_rr_list) / len(baseline_rr_list)
    mean_base_ndcg = sum(baseline_ndcg_list) / len(baseline_ndcg_list)

    mean_ranked_p = sum(ranked_p_list) / len(ranked_p_list)
    mean_ranked_r = sum(ranked_r_list) / len(ranked_r_list)
    mean_ranked_mrr = sum(ranked_rr_list) / len(ranked_rr_list)
    mean_ranked_ndcg = sum(ranked_ndcg_list) / len(ranked_ndcg_list)

    print("-" * 92)
    print("Aggregate Benchmark Comparison (N=5 tasks, K=2):")
    print(f"  Precision@{k}:              Baseline = {mean_base_p:.4f}  |  Ranked = {mean_ranked_p:.4f}  (diff = {mean_ranked_p - mean_base_p:+.4f})")
    print(f"  Recall@{k}:                 Baseline = {mean_base_r:.4f}  |  Ranked = {mean_ranked_r:.4f}  (diff = {mean_ranked_r - mean_base_r:+.4f})")
    print(f"  Mean Reciprocal Rank (MRR): Baseline = {mean_base_mrr:.4f}  |  Ranked = {mean_ranked_mrr:.4f}  (diff = {mean_ranked_mrr - mean_base_mrr:+.4f})")
    print(f"  Mean NDCG@{k}:              Baseline = {mean_base_ndcg:.4f}  |  Ranked = {mean_ranked_ndcg:.4f}  (diff = {mean_ranked_ndcg - mean_base_ndcg:+.4f})")
    print("=" * 92 + "\n")

    await engine.dispose()

    # Verification assertions: Multi-signal ranker must achieve top-tier retrieval quality and exceed baseline
    assert mean_ranked_p >= 0.90, f"Ranked Precision@{k} was {mean_ranked_p:.4f}, expected >= 0.90"
    assert mean_ranked_r >= 0.90, f"Ranked Recall@{k} was {mean_ranked_r:.4f}, expected >= 0.90"
    assert mean_ranked_mrr >= 0.90, f"Ranked MRR was {mean_ranked_mrr:.4f}, expected >= 0.90"
    assert mean_ranked_ndcg >= 0.90, f"Ranked NDCG@{k} was {mean_ranked_ndcg:.4f}, expected >= 0.90"
    assert mean_ranked_ndcg > mean_base_ndcg, "Multi-signal ranking must measurably improve NDCG over baseline"
    assert mean_ranked_p > mean_base_p, "Multi-signal ranking must measurably improve Precision over baseline"
    assert mean_ranked_mrr > mean_base_mrr, "Multi-signal ranking must measurably improve MRR over baseline"
