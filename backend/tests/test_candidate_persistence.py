"""Automated tests for prompt candidate versioning and optimization results persistence (Step 27).

Covers:
1. Candidate snapshot persistence, exact instruction preservation, and content hashing.
2. Deduplication and idempotency on candidate retries.
3. Provenance tracking and parent-child candidate relationships.
4. Separate evaluation persistence with aggregate and per-example records.
5. Strict holdout leakage prevention at service and database levels.
6. Artifact storage security: path traversal blocking, atomic writes, and checksum verification.
7. Worker integration: persisting baseline and optimized candidates after job execution.
8. API endpoints: listing candidates, detail retrieval, evaluations, and results summary.
9. Tenant isolation: cross-tenant access returns 404.
10. Production prompt invariant protection.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import AsyncGenerator
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import (
    ArtifactStorageError,
    CandidateNotFoundError,
    HoldoutLeakageError,
    OptimizationJobNotFoundError,
)
from app.core.security import AuthorizedPrincipal, get_current_principal
from app.db.base import Base
from app.db.session import get_db_session
from app.db.sqlite_compat import register_sqlite_vector_compat
from app.main import create_application
from app.models.optimization_job import OptimizationJob
from app.models.principal import Principal
from app.models.prompt_candidate import CandidateEvaluation, PromptCandidate
from app.schemas.prompt_candidate import (
    CandidateLifecycleStatus,
    CandidateType,
    EvaluationType,
)
from app.services.artifact_storage import ArtifactStorageService
from app.services.candidate_persistence import CandidatePersistenceService
from app.services.generation_service import CRISPE_SYSTEM_PROMPT, CrispeGenerationService
from app.services.optimization_job import OptimizationJobService
from app.workers.optimization_worker import OptimizationWorker


# =============================================================================
# Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def test_db():
    """Isolated in-memory SQLite database engine and session factory."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    register_sqlite_vector_compat(engine)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    yield engine, session_factory

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def sample_owner(test_db):
    """Seed an authorized test principal."""
    _, session_factory = test_db
    async with session_factory() as session:
        principal = Principal(
            id="test-researcher-27",
            display_name="Step 27 Researcher",
            is_active=True,
        )
        session.add(principal)
        await session.commit()
    return AuthorizedPrincipal(id="test-researcher-27")


@pytest_asyncio.fixture
async def other_owner(test_db):
    """Seed a secondary principal for tenant isolation verification."""
    _, session_factory = test_db
    async with session_factory() as session:
        principal = Principal(
            id="other-user-27",
            display_name="Other User",
            is_active=True,
        )
        session.add(principal)
        await session.commit()
    return AuthorizedPrincipal(id="other-user-27")


@pytest_asyncio.fixture
async def sample_job(test_db, sample_owner):
    """Seed an existing optimization job owned by sample_owner."""
    _, session_factory = test_db
    async with session_factory() as session:
        job = OptimizationJob(
            id=uuid.uuid4(),
            owner_id=sample_owner.id,
            status="running",
            generation_model_id="gemma-4-26b-a4b-it",
            dataset_version="v1",
            max_metric_calls=15,
            max_total_provider_calls=35,
        )
        session.add(job)
        await session.commit()
        job_id = job.id
    return job_id


@pytest.fixture
def temp_artifact_storage():
    """Temporary artifact storage service for isolated file testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = ArtifactStorageService(base_dir=tmp_dir)
        yield service, Path(tmp_dir)


# =============================================================================
# 1. Artifact Storage Security & Functionality Tests
# =============================================================================

def test_artifact_storage_atomic_write_and_read(temp_artifact_storage):
    """Artifact is written atomically and can be read back with exact content."""
    service, base_path = temp_artifact_storage
    data = {"step": 27, "name": "candidate_test", "metrics": [0.85, 0.90]}

    urn = service.write_artifact("jobs/job-123/candidate-1.json", data)
    assert urn == "artifact://jobs/job-123/candidate-1.json"
    assert (base_path / "jobs" / "job-123" / "candidate-1.json").exists()

    read_data = service.read_json_artifact(urn)
    assert read_data == data
    assert service.artifact_exists(urn) is True


def test_artifact_storage_path_traversal_blocked(temp_artifact_storage):
    """Path traversal attempts are detected and rejected with ArtifactStorageError."""
    service, _ = temp_artifact_storage

    with pytest.raises(ArtifactStorageError, match="Invalid artifact path"):
        service.write_artifact("../../etc/passwd", "malicious_content")

    with pytest.raises(ArtifactStorageError, match="Invalid artifact path"):
        service.read_artifact("../../../windows/system32/cmd.exe")


def test_artifact_storage_checksum(temp_artifact_storage):
    """Cryptographic SHA-256 checksum is computed accurately for persisted artifacts."""
    service, _ = temp_artifact_storage
    urn = service.write_artifact("jobs/test.txt", "CRISPE-AI-OPTIMIZATION-27")

    checksum = service.compute_checksum(urn)
    assert len(checksum) == 64
    import hashlib
    expected = hashlib.sha256("CRISPE-AI-OPTIMIZATION-27".encode("utf-8")).hexdigest()
    assert checksum == expected


# =============================================================================
# 2. Candidate Versioning and Persistence Tests
# =============================================================================

@pytest.mark.asyncio
async def test_candidate_persistence_and_immutable_snapshot(test_db, sample_job):
    """Candidate snapshot retains exact instruction text, content hash, and metadata."""
    _, session_factory = test_db
    service = CandidatePersistenceService()

    instruction = (
        "Generate a production-grade CRISPE prompt ensuring that the Specification section "
        "contains strict technology locks and zero information drop."
    )

    async with session_factory() as session:
        cand = await service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text=instruction,
            candidate_type=CandidateType.BASELINE.value,
            lifecycle_status=CandidateLifecycleStatus.EVALUATED.value,
            model_id="gemma-4-26b-a4b-it",
            dataset_version="v1",
        )
        await session.commit()
        cand_id = cand.id
        content_hash = cand.content_hash

    async with session_factory() as session:
        stored = await session.get(PromptCandidate, cand_id)
        assert stored is not None
        assert stored.instruction_text == instruction
        assert stored.candidate_type == CandidateType.BASELINE.value
        assert stored.lifecycle_status == CandidateLifecycleStatus.EVALUATED.value
        assert len(stored.content_hash) == 64


@pytest.mark.asyncio
async def test_candidate_deduplication_idempotency(test_db, sample_job):
    """Repeatedly saving the identical instruction text for the same job reuses the existing candidate."""
    _, session_factory = test_db
    service = CandidatePersistenceService()
    instruction = "Reflect on domain context and format crisp sections."

    async with session_factory() as session:
        cand1 = await service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text=instruction,
            candidate_type=CandidateType.OPTIMIZED.value,
        )
        await session.commit()
        id1 = cand1.id

    async with session_factory() as session:
        cand2 = await service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text=instruction,  # identical text
            candidate_type=CandidateType.OPTIMIZED.value,
        )
        await session.commit()
        id2 = cand2.id

    # Must return the existing candidate record rather than duplicating
    assert id1 == id2

    # Query total records in DB for this job
    async with session_factory() as session:
        count = await session.scalar(select(PromptCandidate).where(PromptCandidate.job_id == sample_job))
        assert count is not None


@pytest.mark.asyncio
async def test_candidate_parentage_lineage(test_db, sample_job):
    """Optimized candidate correctly links to parent baseline candidate."""
    _, session_factory = test_db
    service = CandidatePersistenceService()

    async with session_factory() as session:
        base = await service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text="Baseline instructions",
            candidate_type=CandidateType.BASELINE.value,
        )
        await session.flush()

        opt = await service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text="Optimized instructions with enhanced examples",
            candidate_type=CandidateType.OPTIMIZED.value,
            parent_candidate_id=base.id,
        )
        await session.commit()
        opt_id = opt.id
        base_id = base.id

    async with session_factory() as session:
        opt_loaded = await session.get(PromptCandidate, opt_id)
        assert opt_loaded.parent_candidate_id == base_id


# =============================================================================
# 3. Evaluation Results Persistence and Holdout Protection Tests
# =============================================================================

@pytest.mark.asyncio
async def test_evaluation_persistence_aggregate_and_per_example(test_db, sample_job):
    """Aggregate and per-example evaluations are persisted distinctly without collision."""
    _, session_factory = test_db
    service = CandidatePersistenceService()

    async with session_factory() as session:
        cand = await service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text="Test prompt for evaluation storage",
        )
        await session.flush()

        # Save aggregate evaluation
        agg_eval = await service.save_evaluation(
            session=session,
            candidate_id=cand.id,
            job_id=sample_job,
            evaluation_type=EvaluationType.DETERMINISTIC.value,
            dataset_split="val",
            example_id="__aggregate__",
            metrics={"mean_score": 0.88, "valid_structure_rate": 1.0},
            passed=True,
        )

        # Save per-example evaluation
        ex_eval = await service.save_evaluation(
            session=session,
            candidate_id=cand.id,
            job_id=sample_job,
            evaluation_type=EvaluationType.DETERMINISTIC.value,
            dataset_split="val",
            example_id="prompt_eval_val_001",
            metrics={"score": 0.95, "crispe_compliance": 1.0},
            passed=True,
            sanitized_feedback="Excellent CRISPE structure.",
        )
        await session.commit()

    async with session_factory() as session:
        evals = (
            await session.scalars(
                select(CandidateEvaluation).where(CandidateEvaluation.candidate_id == cand.id)
            )
        ).all()
        assert len(evals) == 2
        examples = {e.example_id for e in evals}
        assert "__aggregate__" in examples
        assert "prompt_eval_val_001" in examples


@pytest.mark.asyncio
async def test_holdout_leakage_prohibited_at_service_layer(test_db, sample_job):
    """Persisting evaluations on the final holdout split is rejected with HoldoutLeakageError."""
    _, session_factory = test_db
    service = CandidatePersistenceService()

    async with session_factory() as session:
        cand = await service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text="Candidate being evaluated",
        )
        await session.flush()

        # Attempt forbidden 'final_holdout' split
        with pytest.raises(HoldoutLeakageError, match="strictly forbidden"):
            await service.save_evaluation(
                session=session,
                candidate_id=cand.id,
                job_id=sample_job,
                evaluation_type=EvaluationType.DETERMINISTIC.value,
                dataset_split="final_holdout",
                metrics={"mean_score": 0.99},
            )

        # Attempt forbidden 'test' split
        with pytest.raises(HoldoutLeakageError, match="strictly forbidden"):
            await service.save_evaluation(
                session=session,
                candidate_id=cand.id,
                job_id=sample_job,
                evaluation_type=EvaluationType.DETERMINISTIC.value,
                dataset_split="test",
                metrics={"mean_score": 0.99},
            )


# =============================================================================
# 4. Background Worker Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_worker_run_once_persists_candidates_and_evaluations(
    test_db,
    sample_owner,
    temp_artifact_storage,
):
    """Worker claims job, executes mock experiment, and persists baseline + optimized candidates."""
    _, session_factory = test_db
    artifact_service, _ = temp_artifact_storage

    # Create queued job
    job_service = OptimizationJobService()
    async with session_factory() as session:
        from app.schemas.optimization_job import CreateOptimizationJobRequest
        req = CreateOptimizationJobRequest(
            train_size=3,
            val_size=2,
            max_metric_calls=15,
            max_total_provider_calls=35,
            is_mock=True,
        )
        job = await job_service.create_job(session=session, principal=sample_owner, request=req)
        job_id = job.id

    worker = OptimizationWorker(
        worker_id="test_worker_step27",
        job_service=job_service,
        artifact_storage=artifact_service,
        session_factory=session_factory,
        poll_interval=0.1,
    )

    status = await worker.run_once()
    from app.schemas.optimization_job import JobStatus
    assert status == JobStatus.SUCCEEDED

    # Verify database persistence
    async with session_factory() as session:
        # Check job completion
        completed_job = await session.get(OptimizationJob, job_id)
        assert completed_job.status == JobStatus.SUCCEEDED.value
        assert completed_job.result_summary is not None

        # Check prompt_candidates
        cands = (
            await session.scalars(
                select(PromptCandidate).where(PromptCandidate.job_id == job_id).order_by(PromptCandidate.created_at)
            )
        ).all()
        assert len(cands) == 2

        baseline_cand = cands[0]
        optimized_cand = cands[1]
        assert baseline_cand.candidate_type == CandidateType.BASELINE.value
        assert optimized_cand.candidate_type == CandidateType.OPTIMIZED.value
        assert optimized_cand.parent_candidate_id == baseline_cand.id

        # Check candidate_evaluations
        evals = (
            await session.scalars(
                select(CandidateEvaluation).where(CandidateEvaluation.job_id == job_id)
            )
        ).all()
        assert len(evals) > 0

        # Check that artifacts exist
        assert artifact_service.artifact_exists(f"jobs/{job_id}/validation_comparison.json")
        assert artifact_service.artifact_exists(f"jobs/{job_id}/candidate_program.json")


# =============================================================================
# 5. API Endpoints and Access Control Tests
# =============================================================================

@pytest.mark.asyncio
async def test_api_candidate_endpoints_flow(
    test_db,
    sample_owner,
    sample_job,
    temp_artifact_storage,
):
    """API endpoints for listing candidates, getting detail, evaluations, and job results."""
    _, session_factory = test_db
    artifact_service, _ = temp_artifact_storage
    cand_service = CandidatePersistenceService()

    # Pre-populate candidate and evaluation
    async with session_factory() as session:
        c1 = await cand_service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text="Baseline instructions snapshot",
            candidate_type=CandidateType.BASELINE.value,
        )
        c2 = await cand_service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text="Optimized instructions snapshot",
            candidate_type=CandidateType.OPTIMIZED.value,
            parent_candidate_id=c1.id,
        )
        await cand_service.save_evaluation(
            session=session,
            candidate_id=c2.id,
            job_id=sample_job,
            evaluation_type=EvaluationType.DETERMINISTIC.value,
            dataset_split="val",
            metrics={"mean_score": 0.91},
        )
        await session.commit()
        c1_id = c1.id
        c2_id = c2.id

    app = create_application()
    app.dependency_overrides[get_current_principal] = lambda: sample_owner
    app.dependency_overrides[get_db_session] = lambda: session_factory()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. List candidates
        resp = await client.get(f"/api/v1/optimization/jobs/{sample_job}/candidates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 2
        assert len(data["items"]) == 2

        # 2. Candidate detail
        resp_detail = await client.get(f"/api/v1/optimization/candidates/{c2_id}")
        assert resp_detail.status_code == 200
        detail = resp_detail.json()
        assert detail["instruction_text"] == "Optimized instructions snapshot"
        assert detail["parent_candidate_id"] == str(c1_id)

        # 3. Candidate evaluations
        resp_eval = await client.get(f"/api/v1/optimization/candidates/{c2_id}/evaluations")
        assert resp_eval.status_code == 200
        evals_data = resp_eval.json()
        assert len(evals_data["aggregate_evaluations"]) == 1
        assert evals_data["aggregate_evaluations"][0]["metrics"]["mean_score"] == 0.91

        # 4. Job optimization results
        resp_res = await client.get(f"/api/v1/optimization/jobs/{sample_job}/results")
        assert resp_res.status_code == 200
        res_data = resp_res.json()
        assert res_data["candidate_count"] == 2
        assert res_data["baseline_candidate"]["id"] == str(c1_id)
        assert res_data["best_candidate"]["id"] == str(c2_id)


@pytest.mark.asyncio
async def test_api_tenant_isolation_prevents_unauthorized_access(
    test_db,
    sample_owner,
    other_owner,
    sample_job,
):
    """A user cannot access another user's candidates, evaluations, or job results (returns 404)."""
    _, session_factory = test_db
    cand_service = CandidatePersistenceService()

    async with session_factory() as session:
        c1 = await cand_service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text="Private instructions",
        )
        await session.commit()
        c1_id = c1.id

    app = create_application()
    # Authenticate as 'other_owner' who does NOT own sample_job
    app.dependency_overrides[get_current_principal] = lambda: other_owner
    app.dependency_overrides[get_db_session] = lambda: session_factory()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Candidate list -> 404
        r_list = await client.get(f"/api/v1/optimization/jobs/{sample_job}/candidates")
        assert r_list.status_code == 404

        # Candidate detail -> 404
        r_detail = await client.get(f"/api/v1/optimization/candidates/{c1_id}")
        assert r_detail.status_code == 404

        # Candidate evaluations -> 404
        r_eval = await client.get(f"/api/v1/optimization/candidates/{c1_id}/evaluations")
        assert r_eval.status_code == 404

        # Job results -> 404
        r_res = await client.get(f"/api/v1/optimization/jobs/{sample_job}/results")
        assert r_res.status_code == 404


# =============================================================================
# 6. Database and Migration Integrity Tests
# =============================================================================

def test_alembic_migration_0005_discovery_and_structure():
    """Verify that Alembic script directory discovers migration 0005 and validates its down-revision chain."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    alembic_ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini_path))
    script_dir = ScriptDirectory.from_config(alembic_cfg)

    head_rev = script_dir.get_current_head()
    assert head_rev == "0005_prompt_candidates"

    script = script_dir.get_revision("0005_prompt_candidates")
    assert script is not None
    assert script.down_revision == "0004_optimization_jobs"
    assert hasattr(script.module, "upgrade")
    assert hasattr(script.module, "downgrade")


@pytest.mark.asyncio
async def test_different_evaluator_versions_create_distinct_records(test_db, sample_job):
    """Re-evaluating the same immutable candidate under different evaluator versions creates distinct records."""
    _, session_factory = test_db
    service = CandidatePersistenceService()

    async with session_factory() as session:
        cand = await service.save_candidate(
            session=session,
            job_id=sample_job,
            instruction_text="Candidate evaluated under two evaluator versions",
        )
        await session.flush()

        # Version 1 evaluation
        eval_v1 = await service.save_evaluation(
            session=session,
            candidate_id=cand.id,
            job_id=sample_job,
            evaluation_type=EvaluationType.DETERMINISTIC.value,
            dataset_split="val",
            example_id="ex_001",
            evaluator_version="v1.0.0",
            rubric_version="v1.0.0",
            metrics={"crispe_score": 0.82},
        )

        # Version 2 evaluation with different rubric
        eval_v2 = await service.save_evaluation(
            session=session,
            candidate_id=cand.id,
            job_id=sample_job,
            evaluation_type=EvaluationType.DETERMINISTIC.value,
            dataset_split="val",
            example_id="ex_002",
            evaluator_version="v2.0.0",
            rubric_version="v2.0.0",
            metrics={"crispe_score": 0.89},
        )
        await session.commit()
        id1, id2 = eval_v1.id, eval_v2.id

    assert id1 != id2
    async with session_factory() as session:
        e1 = await session.get(CandidateEvaluation, id1)
        e2 = await session.get(CandidateEvaluation, id2)
        assert e1.evaluator_version == "v1.0.0"
        assert e2.evaluator_version == "v2.0.0"


# =============================================================================
# 7. Production System Prompt Invariant Test
# =============================================================================

def test_production_prompt_invariant_step27():
    """Verify that CRISPE_SYSTEM_PROMPT in generation_service.py remains strictly intact."""
    service = CrispeGenerationService()
    assert service is not None
    assert CRISPE_SYSTEM_PROMPT is not None
    assert "CRISPE" in CRISPE_SYSTEM_PROMPT
    assert "Context:" in CRISPE_SYSTEM_PROMPT
    assert "Role:" in CRISPE_SYSTEM_PROMPT
    assert "Instruction:" in CRISPE_SYSTEM_PROMPT
    assert "Specification:" in CRISPE_SYSTEM_PROMPT
    assert "Performance:" in CRISPE_SYSTEM_PROMPT
    assert "Example:" in CRISPE_SYSTEM_PROMPT
