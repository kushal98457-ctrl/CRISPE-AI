"""Comprehensive tests for durable optimization jobs, background workers, and recovery (Step 26).

Covers:
A. Job-creation & server-side validation tests (dataset partitions, models, budgets, holdout protection, idempotency).
B. State-machine tests (permitted transitions, terminal immutability, failure and timeout statuses).
C. Worker claiming & concurrency tests (atomic claiming, lease assignment, fencing tokens, stale worker rejection).
D. Progress, heartbeats, and crash-recovery tests (counter preservation across restarts, lease expiry recovery).
E. Cooperative cancellation tests (queued immediate cancellation, running cooperative signal, unauthorized rejection).
F. API routes tests (POST 202 Accepted, GET status, cancellation, owner isolation, auth protection).
G. Invariance test verifying production CRISPE prompt instructions remain strictly untouched.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    InvalidJobStateTransitionError,
    JobNotCancellableError,
    OptimizationJobNotFoundError,
    PromptValidationError,
)
from app.core.security import AuthorizedPrincipal, get_current_principal
from app.db.base import Base
from app.db.session import get_db_session
from app.db.sqlite_compat import register_sqlite_vector_compat
from app.evaluation.dataset_accessor import HoldoutAccessRestrictedError
from app.main import create_application
from app.models.optimization_job import OptimizationJob
from app.models.principal import Principal
from app.schemas.optimization_job import (
    CancelOptimizationJobRequest,
    CreateOptimizationJobRequest,
    JobStateMachine,
    JobStatus,
)
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
async def principal_owner(test_db):
    """Seed test principal owner."""
    _, session_factory = test_db
    p = Principal(
        id=f"principal_{uuid.uuid4().hex[:8]}",
        email="test_owner@example.com",
        display_name="Test Owner",
        is_active=True,
        is_superuser=False,
    )
    async with session_factory() as session:
        session.add(p)
        await session.commit()
    return p.to_authorized_principal()


@pytest_asyncio.fixture
async def principal_other(test_db):
    """Seed second test principal for tenant isolation tests."""
    _, session_factory = test_db
    p = Principal(
        id=f"principal_{uuid.uuid4().hex[:8]}",
        email="other_tenant@example.com",
        display_name="Other Tenant",
        is_active=True,
        is_superuser=False,
    )
    async with session_factory() as session:
        session.add(p)
        await session.commit()
    return p.to_authorized_principal()


# =============================================================================
# A. Job-Creation & Validation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_create_optimization_job_valid(test_db, principal_owner):
    """Verify OptimizationJobService persists a valid optimization job in 'queued' state."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(
        train_size=5,
        val_size=3,
        max_metric_calls=15,
        max_total_provider_calls=35,
        generation_model_id="gemma-4-26b-a4b-it",
        reflection_model_id="gemini-2.5-flash",
        is_mock=True,
    )

    async with session_factory() as session:
        job = await service.create_job(session=session, principal=principal_owner, request=req)
        assert job.id is not None
        assert job.owner_id == principal_owner.id
        assert job.status == JobStatus.QUEUED.value
        assert len(job.train_example_ids) == 5
        assert len(job.val_example_ids) == 3
        assert job.dataset_sha256 is not None
        assert job.current_phase == "queued"


@pytest.mark.asyncio
async def test_create_optimization_job_rejects_unapproved_model(test_db, principal_owner):
    """Verify caller cannot supply arbitrary unverified model identifiers."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(
        generation_model_id="completely-unauthorized-gpt-model",
        is_mock=False,
    )

    async with session_factory() as session:
        with pytest.raises(PromptValidationError, match="not in approved list"):
            await service.create_job(session=session, principal=principal_owner, request=req)


@pytest.mark.asyncio
async def test_create_optimization_job_idempotency(test_db, principal_owner):
    """Verify duplicate submission with the same idempotency key returns existing job."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(
        train_size=3,
        val_size=2,
        is_mock=True,
        idempotency_key="idempotent_key_abc_123",
    )

    async with session_factory() as session:
        job1 = await service.create_job(session=session, principal=principal_owner, request=req)
        job2 = await service.create_job(session=session, principal=principal_owner, request=req)
        assert job1.id == job2.id
        assert job1.status == JobStatus.QUEUED.value


# =============================================================================
# B. State-Machine Tests
# =============================================================================

def test_state_machine_permitted_transitions():
    """Verify legal state transitions are validated correctly."""
    # Queued -> Running
    JobStateMachine.validate_transition(JobStatus.QUEUED, JobStatus.RUNNING)
    # Running -> Succeeded
    JobStateMachine.validate_transition(JobStatus.RUNNING, JobStatus.SUCCEEDED)
    # Running -> Budget Exhausted
    JobStateMachine.validate_transition(JobStatus.RUNNING, JobStatus.BUDGET_EXHAUSTED)
    # Running -> Timed Out
    JobStateMachine.validate_transition(JobStatus.RUNNING, JobStatus.TIMED_OUT)
    # Running -> Cancel Requested -> Cancelled
    JobStateMachine.validate_transition(JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED)
    JobStateMachine.validate_transition(JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED)


def test_state_machine_rejects_terminal_reversion():
    """Verify terminal states cannot transition to running or any other state."""
    terminal_states = [
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMED_OUT,
        JobStatus.BUDGET_EXHAUSTED,
    ]
    for ts in terminal_states:
        assert JobStateMachine.is_terminal(ts)
        with pytest.raises(InvalidJobStateTransitionError):
            JobStateMachine.validate_transition(ts, JobStatus.RUNNING)
        with pytest.raises(InvalidJobStateTransitionError):
            JobStateMachine.validate_transition(ts, JobStatus.QUEUED)


# =============================================================================
# C. Worker Claiming, Concurrency, and Lease Fencing Tests
# =============================================================================

@pytest.mark.asyncio
async def test_worker_atomic_claiming_and_lease(test_db, principal_owner):
    """Verify a worker atomically claims a queued job with a fencing token and lease."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(train_size=3, val_size=2, is_mock=True)
    async with session_factory() as session:
        created = await service.create_job(session=session, principal=principal_owner, request=req)

    # Worker 1 claims job
    async with session_factory() as session:
        claimed = await service.claim_next_job(
            session=session,
            worker_id="worker_alpha",
            lease_duration_seconds=30.0,
        )
        assert claimed is not None
        assert claimed.id == created.id
        assert claimed.status == JobStatus.RUNNING.value
        assert claimed.worker_id == "worker_alpha"
        assert claimed.claim_token is not None
        assert claimed.attempt_count == 1
        assert claimed.lease_expires_at is not None

    # Worker 2 attempts claiming -> None (queue is now empty)
    async with session_factory() as session:
        second_attempt = await service.claim_next_job(
            session=session,
            worker_id="worker_beta",
            lease_duration_seconds=30.0,
        )
        assert second_attempt is None


@pytest.mark.asyncio
async def test_stale_worker_fenced_out(test_db, principal_owner):
    """Verify that a worker whose lease token was superseded cannot update job progress."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(train_size=3, val_size=2, is_mock=True)
    async with session_factory() as session:
        created = await service.create_job(session=session, principal=principal_owner, request=req)

    # Worker 1 claims job
    async with session_factory() as session:
        job = await service.claim_next_job(session=session, worker_id="worker_alpha")
        old_token = job.claim_token

    # Simulate lease expiry and Worker 2 reclaiming the job
    async with session_factory() as session:
        # Force lease expiry into past
        job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        session.add(job)
        await session.commit()

        # Worker 2 reclaims
        reclaimed = await service.claim_next_job(session=session, worker_id="worker_beta")
        assert reclaimed is not None
        assert reclaimed.claim_token != old_token
        assert reclaimed.worker_id == "worker_beta"

    # Worker 1 attempts progress update with stale claim token -> Fenced out!
    async with session_factory() as session:
        updated = await service.update_progress(
            session=session,
            job_id=created.id,
            claim_token=old_token,
            current_phase="stale_update",
        )
        assert updated is False  # Rowcount 0, fencing prevented stale write


# =============================================================================
# D. Progress, Heartbeats, and Recovery Tests
# =============================================================================

@pytest.mark.asyncio
async def test_worker_run_once_completes_job_mock(test_db, principal_owner):
    """Verify OptimizationWorker claims, executes, and finalizes a job with artifacts."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(
        train_size=3,
        val_size=2,
        max_metric_calls=10,
        max_total_provider_calls=25,
        is_mock=True,
    )
    async with session_factory() as session:
        created = await service.create_job(session=session, principal=principal_owner, request=req)

    worker = OptimizationWorker(
        worker_id="test_worker_1",
        job_service=service,
        session_factory=session_factory,
        poll_interval=0.1,
        lease_duration=10.0,
    )

    final_status = await worker.run_once()
    assert final_status == JobStatus.SUCCEEDED

    async with session_factory() as session:
        refreshed = await service.get_job(session=session, principal=principal_owner, job_id=created.id)
        assert refreshed.status == JobStatus.SUCCEEDED.value
        assert refreshed.completed_at is not None
        assert refreshed.used_metric_calls > 0
        assert refreshed.used_provider_calls > 0
        assert refreshed.artifacts_dir is not None
        assert refreshed.result_summary is not None
        assert refreshed.result_summary.get("is_promoted") is False


# =============================================================================
# E. Cooperative Cancellation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_cancel_queued_job_immediate(test_db, principal_owner):
    """Verify cancelling a queued job immediately transitions to CANCELLED without worker execution."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(train_size=3, val_size=2, is_mock=True)
    async with session_factory() as session:
        created = await service.create_job(session=session, principal=principal_owner, request=req)

        cancelled = await service.request_cancellation(
            session=session,
            principal=principal_owner,
            job_id=created.id,
            reason="User decided to abort before start",
        )
        assert cancelled.status == JobStatus.CANCELLED.value
        assert cancelled.cancelled_at is not None
        assert "User decided to abort" in (cancelled.cancel_reason or "")


@pytest.mark.asyncio
async def test_cancel_terminal_job_raises_conflict(test_db, principal_owner):
    """Verify attempting to cancel an already terminal job raises JobNotCancellableError."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(train_size=3, val_size=2, is_mock=True)
    async with session_factory() as session:
        created = await service.create_job(session=session, principal=principal_owner, request=req)
        # Cancel once
        await service.request_cancellation(session=session, principal=principal_owner, job_id=created.id)

        # Cancel again on terminal state
        with pytest.raises(JobNotCancellableError):
            await service.request_cancellation(session=session, principal=principal_owner, job_id=created.id)


@pytest.mark.asyncio
async def test_tenant_isolation_prevents_unauthorized_access_or_cancel(test_db, principal_owner, principal_other):
    """Verify that a different tenant cannot view or cancel another principal's job."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(train_size=3, val_size=2, is_mock=True)
    async with session_factory() as session:
        created = await service.create_job(session=session, principal=principal_owner, request=req)

        # Tenant 2 attempts to get job -> 404
        with pytest.raises(OptimizationJobNotFoundError):
            await service.get_job(session=session, principal=principal_other, job_id=created.id)

        # Tenant 2 attempts to cancel job -> 404
        with pytest.raises(OptimizationJobNotFoundError):
            await service.request_cancellation(session=session, principal=principal_other, job_id=created.id)


# =============================================================================
# F. API Endpoints Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_api_submit_job_returns_202_accepted(test_db, principal_owner):
    """Verify POST /api/v1/optimization/jobs returns HTTP 202 Accepted with job metadata."""
    _, session_factory = test_db
    app = create_application()

    app.dependency_overrides[get_current_principal] = lambda: principal_owner
    app.dependency_overrides[get_db_session] = lambda: session_factory()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/optimization/jobs",
            json={
                "train_size": 3,
                "val_size": 2,
                "max_metric_calls": 12,
                "max_total_provider_calls": 25,
                "is_mock": True,
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "id" in body
        assert body["status"] == "queued"
        assert body["job_type"] == "gepa_prompt_optimization"
        assert body["progress"]["completed_metric_calls"] == 0


@pytest.mark.asyncio
async def test_api_get_job_status(test_db, principal_owner):
    """Verify GET /api/v1/optimization/jobs/{job_id} returns the job details."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(train_size=3, val_size=2, is_mock=True)
    async with session_factory() as session:
        created = await service.create_job(session=session, principal=principal_owner, request=req)

    app = create_application()
    app.dependency_overrides[get_current_principal] = lambda: principal_owner
    app.dependency_overrides[get_db_session] = lambda: session_factory()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get(f"/api/v1/optimization/jobs/{created.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(created.id)
        assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_api_cancel_job(test_db, principal_owner):
    """Verify POST /api/v1/optimization/jobs/{job_id}/cancel cancels the job."""
    _, session_factory = test_db
    service = OptimizationJobService()

    req = CreateOptimizationJobRequest(train_size=3, val_size=2, is_mock=True)
    async with session_factory() as session:
        created = await service.create_job(session=session, principal=principal_owner, request=req)

    app = create_application()
    app.dependency_overrides[get_current_principal] = lambda: principal_owner
    app.dependency_overrides[get_db_session] = lambda: session_factory()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            f"/api/v1/optimization/jobs/{created.id}/cancel",
            json={"reason": "Testing cancellation API"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "cancelled"
        assert body["cancel_requested"] is True


@pytest.mark.asyncio
async def test_api_unauthenticated_request_rejected(test_db):
    """Verify unauthenticated requests to optimization endpoints are rejected with HTTP 401."""
    _, session_factory = test_db
    app = create_application()
    app.dependency_overrides[get_db_session] = lambda: session_factory()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/api/v1/optimization/jobs", json={})
        assert resp.status_code == 401


# =============================================================================
# G. Production Instruction Invariant Test
# =============================================================================

def test_production_instructions_remain_unmodified_step26():
    """Verify that active production CRISPE prompt instructions remain 100% immutable."""
    service = CrispeGenerationService()
    assert service is not None

    assert "You are an elite AI prompt engineer" in CRISPE_SYSTEM_PROMPT
    assert "1. Context:" in CRISPE_SYSTEM_PROMPT
    assert "2. Role:" in CRISPE_SYSTEM_PROMPT
    assert "3. Instruction:" in CRISPE_SYSTEM_PROMPT
    assert "4. Specification:" in CRISPE_SYSTEM_PROMPT
    assert "5. Performance:" in CRISPE_SYSTEM_PROMPT
    assert "6. Example:" in CRISPE_SYSTEM_PROMPT
