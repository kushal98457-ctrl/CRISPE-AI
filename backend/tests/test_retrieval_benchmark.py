"""Deterministic Semantic Retrieval Quality Benchmark for CRISPE-AI (Step 12).

Measures Precision@K, Recall@K, and Mean Reciprocal Rank (MRR) across 5 representative
technical domains:
1. React authentication debugging
2. Python error diagnosis
3. Database schema design
4. Engineering mathematics explanations
5. Technical research and documentation
"""

from dataclasses import dataclass
import math
from typing import AsyncGenerator, Dict, List, Sequence, Set
import uuid
import pytest
import pytest_asyncio
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
class BenchmarkTask:
    domain_id: str
    query_text: str
    active_range: tuple[int, int]
    prompt_titles: List[str]


BENCHMARK_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        domain_id="react_auth",
        query_text="Debugging React expired JWT tokens and AuthContext refresh cycles",
        active_range=(0, 80),
        prompt_titles=[
            "React Axios interceptor for JWT silent refresh and token expiry",
            "Managing user session logout in React AuthProvider upon 401 response",
        ],
    ),
    BenchmarkTask(
        domain_id="python_diagnosis",
        query_text="Python asyncio event loop closed RuntimeError in pytest test suite",
        active_range=(80, 160),
        prompt_titles=[
            "Resolving Event loop is closed in pytest-asyncio with event_loop fixture",
            "Debugging unclosed transport and async task cancellation in Python 3.12",
        ],
    ),
    BenchmarkTask(
        domain_id="db_schema",
        query_text="PostgreSQL relational schema design with foreign key cascades and composite indexes",
        active_range=(160, 240),
        prompt_titles=[
            "PostgreSQL multi-tenant schema with tenant_id foreign keys and ON DELETE CASCADE",
            "Composite index strategies for high-volume time-series PostgreSQL tables",
        ],
    ),
    BenchmarkTask(
        domain_id="engineering_math",
        query_text="Numerical differential equation integration with 4th order Runge-Kutta",
        active_range=(240, 320),
        prompt_titles=[
            "Implementing Runge-Kutta 4th order (RK4) integration algorithm in Python",
            "Solving stiff orbital mechanics differential equations numerically",
        ],
    ),
    BenchmarkTask(
        domain_id="technical_docs",
        query_text="Writing OpenAPI 3.1 specifications and Architectural Decision Records for microservices",
        active_range=(320, 400),
        prompt_titles=[
            "FastAPI OpenAPI 3.1 schema customization and endpoint documentation",
            "Architectural Decision Record (ADR) template and decision capture workflow",
        ],
    ),
]


@pytest.mark.asyncio
async def test_deterministic_semantic_retrieval_benchmark() -> None:
    """Execute deterministic semantic search benchmark measuring Precision@K, Recall@K, and MRR.

    Evaluates retrieval quality across 5 technical domains, totaling 10 distinct prompts.
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

    # 1. Populate principals, prompts, and embeddings
    query_routes: Dict[str, List[float]] = {}
    ground_truth_relevant: Dict[str, Set[uuid.UUID]] = {}

    sections = CrispeSections(
        context="Benchmark Context",
        role="Benchmark Evaluator",
        instruction="Evaluate retrieval accuracy",
        specification="Measure Precision, Recall, and MRR",
        performance="Target 100% Precision@2 and MRR=1.0",
        example="Valid CRISPE sections",
    )

    async with session_factory() as session:
        # Create database principal
        db_principal = Principal(id=principal_id, email="benchmark@internal", is_active=True)
        session.add(db_principal)
        await session.commit()

        principal = AuthorizedPrincipal(id=principal_id, is_active=True)

        for task in BENCHMARK_TASKS:
            ground_truth_relevant[task.domain_id] = set()

            # Assign query embedding
            q_vec = make_cluster_unit_vector(task.active_range[0], task.active_range[1], noise_seed=100)
            query_routes[task.query_text] = q_vec

            # Create 2 relevant prompts for this task
            for idx, title in enumerate(task.prompt_titles, start=1):
                p = PromptRecord(
                    original_idea=title,
                    context=sections.context,
                    role=sections.role,
                    instruction=sections.instruction,
                    specification=sections.specification,
                    performance=sections.performance,
                    example=sections.example,
                    formatted_prompt=f"# {title}\nCanonical text.",
                    task_type=task.domain_id,
                    owner_id=principal.id,
                )
                session.add(p)
                await session.flush()
                ground_truth_relevant[task.domain_id].add(p.id)

                p_vec = make_cluster_unit_vector(task.active_range[0], task.active_range[1], noise_seed=idx)
                emb = PromptEmbedding(
                    prompt_id=p.id,
                    embedding=p_vec,
                    source_content_hash=PromptEmbedding.compute_content_hash(p.formatted_prompt),
                )
                session.add(emb)

        await session.commit()

    # 2. Initialize search service with routing adapter
    mock_adapter = BenchmarkEmbeddingAdapter(query_routes=query_routes)
    service = SemanticSearchService(embedding_service=EmbeddingService(adapter=mock_adapter))

    # 3. Evaluate each domain query at K=2
    k = 2
    precisions: List[float] = []
    recalls: List[float] = []
    reciprocal_ranks: List[float] = []

    print("\n" + "=" * 80)
    print("CRISPE-AI SEMANTIC RETRIEVAL QUALITY BENCHMARK (STEP 15)")
    print("=" * 80)
    print(f"{'Domain':<22} | {'P@2':<8} | {'R@2':<8} | {'RR':<8} | {'Top Match Title':<30}")
    print("-" * 80)

    async with session_factory() as session:
        for task in BENCHMARK_TASKS:
            relevant_ids = ground_truth_relevant[task.domain_id]
            response = await service.search(
                session,
                principal=principal,
                query=task.query_text,
                limit=k,
            )

            retrieved_ids = [item.prompt_id for item in response.items]

            # Compute Precision@K
            relevant_retrieved = sum(1 for pid in retrieved_ids if pid in relevant_ids)
            p_at_k = relevant_retrieved / k if k > 0 else 0.0
            precisions.append(p_at_k)

            # Compute Recall@K
            r_at_k = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
            recalls.append(r_at_k)

            # Compute Reciprocal Rank
            rr = 0.0
            for rank, pid in enumerate(retrieved_ids, start=1):
                if pid in relevant_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            top_title = response.items[0].original_idea if response.items else "None"
            print(f"{task.domain_id:<22} | {p_at_k:<8.2f} | {r_at_k:<8.2f} | {rr:<8.2f} | {top_title[:30]}")

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("-" * 80)
    print(f"Overall Benchmark Summary (N={len(BENCHMARK_TASKS)} queries, K={k}):")
    print(f"  Mean Precision@{k}: {mean_precision:.4f} ({mean_precision * 100:.1f}%)")
    print(f"  Mean Recall@{k}:    {mean_recall:.4f} ({mean_recall * 100:.1f}%)")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f} ({mrr * 100:.1f}%)")
    print("=" * 80 + "\n")

    await engine.dispose()

    # 4. Strict benchmark quality assertions
    assert mean_precision >= 0.9, f"Mean Precision@{k} was {mean_precision:.4f}, expected >= 0.9"
    assert mean_recall >= 0.9, f"Mean Recall@{k} was {mean_recall:.4f}, expected >= 0.9"
    assert mrr >= 0.9, f"MRR was {mrr:.4f}, expected >= 0.9"
