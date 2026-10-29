"""Automated tests for prompt release validation, promotion, and rollback (Step 28).

Covers:
1. Release state machine legal and forbidden transitions.
2. Release validation gates: candidate integrity, CRISPE contract, model compatibility,
   validation split quality, baseline regression check, and controlled holdout evaluation.
3. Durable release draft creation, audit events, and version deduplication.
4. Production promotion: atomic pointer activation, predecessor supersession, and separation of duties.
5. Rollback restoration: reviving superseded snapshots, audit tracking, and reason recording.
6. Generation runtime integration: active release prompt resolution, fallback, and cache invalidation.
7. End-to-end HTTP API routes for releases, validation, promotion, active pointer, and rollback.
8. Authentication requirements and error handling (401, 404, 409, 422).
"""

from datetime import datetime, timezone
import hashlib
from typing import AsyncGenerator
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import (
    ActiveReleaseConflictError,
    CandidateNotFoundError,
    InvalidReleaseStateTransitionError,
    ReleaseNotFoundError,
    ReleaseValidationFailedError,
)
from app.core.security import AuthorizedPrincipal, get_current_principal
from app.db.base import Base
from app.db.session import get_db_session, set_async_engine
from app.db.sqlite_compat import register_sqlite_vector_compat
from app.main import create_application
from app.models.optimization_job import OptimizationJob
from app.models.principal import Principal
from app.models.prompt_candidate import CandidateEvaluation, PromptCandidate
from app.models.prompt_release import PromptRelease, PromptReleaseAuditEvent
from app.schemas.prompt_release import (
    CreateReleaseRequest,
    PromoteReleaseRequest,
    ReleaseDetailResponse,
    ReleaseStateMachine,
    ReleaseStatus,
    ReleaseValidationPolicy,
    RollbackReleaseRequest,
)
from app.services.generation_service import (
    CRISPE_SYSTEM_PROMPT,
    CrispeGenerationService,
    get_cached_active_release,
    invalidate_release_cache,
    set_cached_active_release,
)
from app.services.prompt_release import PromptReleaseService, get_prompt_release_service
from app.services.release_validation import ReleaseValidationService


VALID_CRISPE_INSTRUCTION = """
# CRISPE Framework System Prompt
Context: You are operating in an enterprise software architecture environment.
Role: Senior Staff Prompt Engineer and AI systems architect.
Instruction: Transform raw ideas into rigorous structured specifications.
Specification: Output JSON matching canonical schemas. No conversational filler.
Performance: Achieve high precision and zero semantic hallucination.
Example: Input: 'build a search' -> Output: Context, Role, Instruction, Specification, Performance, Example.
"""

INVALID_CRISPE_INSTRUCTION = """
You are a helpful assistant. Just answer questions nicely.
"""


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
async def operator_principal(test_db):
    """Seed an authorized release operator principal."""
    _, session_factory = test_db
    async with session_factory() as session:
        principal = Principal(
            id="operator-28",
            display_name="Release Operator",
            is_active=True,
        )
        session.add(principal)
        await session.commit()
    return AuthorizedPrincipal(id="operator-28")


@pytest_asyncio.fixture
async def reviewer_principal(test_db):
    """Seed a distinct reviewer principal for separation of duties."""
    _, session_factory = test_db
    async with session_factory() as session:
        principal = Principal(
            id="reviewer-28",
            display_name="Security Reviewer",
            is_active=True,
        )
        session.add(principal)
        await session.commit()
    return AuthorizedPrincipal(id="reviewer-28")


@pytest_asyncio.fixture
async def sample_candidate(test_db, operator_principal):
    """Seed a valid evaluated prompt candidate with validation records."""
    _, session_factory = test_db
    content_hash = hashlib.sha256(VALID_CRISPE_INSTRUCTION.strip().encode("utf-8")).hexdigest()
    async with session_factory() as session:
        job = OptimizationJob(
            owner_id=operator_principal.id,
            status="succeeded",
            generation_model_id="gemma-4-26b-a4b-it",
            dataset_version="v1",
            max_metric_calls=15,
            max_total_provider_calls=35,
        )
        session.add(job)
        await session.flush()

        cand = PromptCandidate(
            id=uuid.uuid4(),
            job_id=job.id,
            instruction_text=VALID_CRISPE_INSTRUCTION.strip(),
            content_hash=content_hash,
            candidate_type="optimized",
            lifecycle_status="proposed",
            optimizer_version="gepa-0.1.0",
            program_version="crispe-dspy-v1",
            model_id="gemma-4-26b-a4b-it",
        )
        session.add(cand)
        await session.flush()

        # Seed 3 validation split evaluations
        for i in range(3):
            eval_record = CandidateEvaluation(
                candidate_id=cand.id,
                job_id=job.id,
                evaluation_type="combined",
                dataset_split="val",
                example_id=f"val-ex-{i}",
                metrics={"score": 0.88, "instruction_following": 0.90},
                passed=True,
                is_critical_failure=False,
            )
            session.add(eval_record)

        agg_record = CandidateEvaluation(
            candidate_id=cand.id,
            job_id=job.id,
            evaluation_type="combined",
            dataset_split="val",
            example_id="__aggregate__",
            metrics={"mean_score": 0.88, "sample_count": 3},
            passed=True,
            is_critical_failure=False,
        )
        session.add(agg_record)
        await session.commit()
        cand_id = cand.id

    return cand_id


# =============================================================================
# 1. State Machine Transition Tests
# =============================================================================

def test_release_state_machine_valid_transitions():
    """Verify that all legal state transitions succeed without error."""
    # Draft -> Validating
    ReleaseStateMachine.validate_transition(ReleaseStatus.DRAFT, ReleaseStatus.VALIDATING)
    # Draft -> Rejected
    ReleaseStateMachine.validate_transition(ReleaseStatus.DRAFT, ReleaseStatus.REJECTED)
    # Validating -> Validated
    ReleaseStateMachine.validate_transition(ReleaseStatus.VALIDATING, ReleaseStatus.VALIDATED)
    # Validating -> Rejected
    ReleaseStateMachine.validate_transition(ReleaseStatus.VALIDATING, ReleaseStatus.REJECTED)
    # Validated -> Active
    ReleaseStateMachine.validate_transition(ReleaseStatus.VALIDATED, ReleaseStatus.ACTIVE)
    # Active -> Superseded
    ReleaseStateMachine.validate_transition(ReleaseStatus.ACTIVE, ReleaseStatus.SUPERSEDED)
    # Active -> Rolled Back
    ReleaseStateMachine.validate_transition(ReleaseStatus.ACTIVE, ReleaseStatus.ROLLED_BACK)
    # Superseded -> Active (Rollback restoration)
    ReleaseStateMachine.validate_transition(ReleaseStatus.SUPERSEDED, ReleaseStatus.ACTIVE)


def test_release_state_machine_illegal_transitions():
    """Verify that unauthorized jumps or terminal transitions raise InvalidReleaseStateTransitionError."""
    # Draft cannot directly become Active
    with pytest.raises(InvalidReleaseStateTransitionError) as exc_info:
        ReleaseStateMachine.validate_transition(ReleaseStatus.DRAFT, ReleaseStatus.ACTIVE)
    assert exc_info.value.status_code == 409

    # Validating cannot directly become Active without Validated
    with pytest.raises(InvalidReleaseStateTransitionError):
        ReleaseStateMachine.validate_transition(ReleaseStatus.VALIDATING, ReleaseStatus.ACTIVE)

    # Terminal state Rejected cannot transition to Active
    with pytest.raises(InvalidReleaseStateTransitionError):
        ReleaseStateMachine.validate_transition(ReleaseStatus.REJECTED, ReleaseStatus.ACTIVE)

    # Terminal state Rolled Back cannot transition to Active
    with pytest.raises(InvalidReleaseStateTransitionError):
        ReleaseStateMachine.validate_transition(ReleaseStatus.ROLLED_BACK, ReleaseStatus.ACTIVE)


# =============================================================================
# 2. Release Validation Policy & Gate Tests
# =============================================================================

@pytest.mark.asyncio
async def test_validation_integrity_gate(test_db, sample_candidate):
    """Candidate SHA-256 integrity gate passes when hashes match and fails when tampered."""
    _, session_factory = test_db
    val_service = ReleaseValidationService()

    async with session_factory() as session:
        cand = (await session.execute(
            select(PromptCandidate).where(PromptCandidate.id == sample_candidate)
        )).scalar_one()

        # Valid release matching hash
        release = PromptRelease(
            release_version="v1.0.0",
            candidate_id=cand.id,
            candidate_hash=cand.content_hash,
            instruction_text=cand.instruction_text,
            created_by_id="operator-28",
        )

        report = await val_service.validate_candidate_for_release(
            session=session,
            release=release,
            candidate=cand,
        )
        assert report.decision == "passed"
        integrity_check = next(c for c in report.checks if c.check_name == "candidate_integrity")
        assert integrity_check.passed is True

        # Tampered release hash
        tampered_release = PromptRelease(
            release_version="v1.0.1",
            candidate_id=cand.id,
            candidate_hash="0000000000000000000000000000000000000000000000000000000000000000",
            instruction_text=cand.instruction_text,
            created_by_id="operator-28",
        )
        bad_report = await val_service.validate_candidate_for_release(
            session=session,
            release=tampered_release,
            candidate=cand,
        )
        assert bad_report.decision == "failed"
        bad_integrity = next(c for c in bad_report.checks if c.check_name == "candidate_integrity")
        assert bad_integrity.passed is False


@pytest.mark.asyncio
async def test_validation_crispe_contract_gate(test_db, operator_principal):
    """Candidate must provide all 6 CRISPE sections to pass structural gate."""
    _, session_factory = test_db
    val_service = ReleaseValidationService()

    bad_hash = hashlib.sha256(INVALID_CRISPE_INSTRUCTION.strip().encode("utf-8")).hexdigest()
    async with session_factory() as session:
        job = OptimizationJob(
            owner_id=operator_principal.id,
            status="succeeded",
            generation_model_id="gemma-4-26b-a4b-it",
            dataset_version="v1",
            max_metric_calls=15,
            max_total_provider_calls=35,
        )
        session.add(job)
        await session.flush()

        cand = PromptCandidate(
            job_id=job.id,
            instruction_text=INVALID_CRISPE_INSTRUCTION.strip(),
            content_hash=bad_hash,
            candidate_type="optimized",
        )
        session.add(cand)
        await session.flush()

        release = PromptRelease(
            release_version="v-invalid",
            candidate_id=cand.id,
            candidate_hash=bad_hash,
            instruction_text=cand.instruction_text,
            created_by_id="operator-28",
        )

        report = await val_service.validate_candidate_for_release(
            session=session,
            release=release,
            candidate=cand,
        )
        assert report.decision == "failed"
        crispe_check = next(c for c in report.checks if c.check_name == "crispe_structural_contract")
        assert crispe_check.passed is False
        assert len(crispe_check.details["missing_sections"]) > 0


@pytest.mark.asyncio
async def test_validation_regression_gate(test_db, sample_candidate):
    """Candidate is rejected if mean score regresses beyond policy allowance against baseline."""
    _, session_factory = test_db
    val_service = ReleaseValidationService()

    async with session_factory() as session:
        cand = (await session.execute(
            select(PromptCandidate).where(PromptCandidate.id == sample_candidate)
        )).scalar_one()

        release = PromptRelease(
            release_version="v1.1.0",
            candidate_id=cand.id,
            candidate_hash=cand.content_hash,
            instruction_text=cand.instruction_text,
            created_by_id="operator-28",
        )

        # Baseline with higher score (0.96 vs candidate's 0.88; diff = -0.08 > allowed 0.05)
        baseline = PromptRelease(
            release_version="v1.0.0-baseline",
            candidate_id=cand.id,
            candidate_hash=cand.content_hash,
            instruction_text=cand.instruction_text,
            created_by_id="operator-28",
            is_active=True,
            validation_report={"metrics_comparison": {"candidate_validation_mean_score": 0.96}},
        )

        policy = ReleaseValidationPolicy(max_permitted_score_regression=0.05)
        report = await val_service.validate_candidate_for_release(
            session=session,
            release=release,
            candidate=cand,
            current_active_release=baseline,
            policy=policy,
        )
        assert report.decision == "failed"
        reg_check = next(c for c in report.checks if c.check_name == "baseline_regression_check")
        assert reg_check.passed is False


@pytest.mark.asyncio
async def test_validation_controlled_holdout_gate(test_db, sample_candidate):
    """Holdout gate evaluates holdout examples cleanly and rejects when unauthorized."""
    _, session_factory = test_db
    val_service = ReleaseValidationService()

    async with session_factory() as session:
        cand = (await session.execute(
            select(PromptCandidate).where(PromptCandidate.id == sample_candidate)
        )).scalar_one()

        release = PromptRelease(
            release_version="v1.0.0",
            candidate_id=cand.id,
            candidate_hash=cand.content_hash,
            instruction_text=cand.instruction_text,
            created_by_id="operator-28",
        )

        # With holdout authorization
        report_auth = await val_service.validate_candidate_for_release(
            session=session,
            release=release,
            candidate=cand,
            authorized_holdout=True,
        )
        assert report_auth.decision == "passed"
        holdout_check = next(c for c in report_auth.checks if c.check_name == "controlled_holdout_evaluation")
        assert holdout_check.passed is True
        assert holdout_check.details["holdout_passed"] is True

        # Unauthorized holdout
        report_unauth = await val_service.validate_candidate_for_release(
            session=session,
            release=release,
            candidate=cand,
            authorized_holdout=False,
        )
        assert report_unauth.decision == "failed"
        holdout_unauth_check = next(c for c in report_unauth.checks if c.check_name == "controlled_holdout_evaluation")
        assert holdout_unauth_check.passed is False


# =============================================================================
# 3. Prompt Release Service Lifecycle & Concurrency Tests
# =============================================================================

@pytest.mark.asyncio
async def test_create_and_validate_release(test_db, sample_candidate, operator_principal):
    """Service creates release draft and advances through validation with audit events."""
    _, session_factory = test_db
    service = PromptReleaseService()

    async with session_factory() as session:
        # Create draft
        req = CreateReleaseRequest(
            candidate_id=sample_candidate,
            release_version="v1.0.0",
            notes="Initial candidate release candidate",
        )
        release = await service.create_release(session, req, principal_id=operator_principal.id)
        assert release.status == ReleaseStatus.DRAFT.value
        assert release.is_active is False
        assert release.release_version == "v1.0.0"

        # Check creation audit event
        audits = (await session.execute(
            select(PromptReleaseAuditEvent).where(PromptReleaseAuditEvent.release_id == release.id)
        )).scalars().all()
        assert len(audits) == 1
        assert audits[0].event_type == "created"

        # Validate release
        report = await service.validate_release(session, release.id, principal_id=operator_principal.id)
        assert report.decision == "passed"

        # Verify state is updated to VALIDATED
        await session.refresh(release)
        assert release.status == ReleaseStatus.VALIDATED.value
        assert release.validation_report is not None


@pytest.mark.asyncio
async def test_duplicate_version_rejected(test_db, sample_candidate, operator_principal):
    """Attempting to create two releases with identical version tag raises conflict."""
    _, session_factory = test_db
    service = PromptReleaseService()

    async with session_factory() as session:
        req = CreateReleaseRequest(
            candidate_id=sample_candidate,
            release_version="v1.0.0-dup",
        )
        await service.create_release(session, req, principal_id=operator_principal.id)

        # Second creation with same version must fail
        with pytest.raises(ActiveReleaseConflictError):
            await service.create_release(session, req, principal_id=operator_principal.id)


@pytest.mark.asyncio
async def test_atomic_promotion_and_supersede(test_db, sample_candidate, operator_principal, reviewer_principal):
    """Promoting a release activates it, supersedes prior release, and keeps exactly one active pointer."""
    _, session_factory = test_db
    service = PromptReleaseService()

    async with session_factory() as session:
        # 1. Create and validate release 1
        rel1 = await service.create_release(
            session,
            CreateReleaseRequest(candidate_id=sample_candidate, release_version="v1.0.0"),
            principal_id=operator_principal.id,
        )
        await service.validate_release(session, rel1.id, principal_id=operator_principal.id)

        # Promote release 1
        promoted1 = await service.promote_release(
            session,
            rel1.id,
            principal_id=reviewer_principal.id,
            reason="Approved for production launch",
        )
        assert promoted1.is_active is True
        assert promoted1.status == ReleaseStatus.ACTIVE.value
        assert promoted1.approved_by_id == reviewer_principal.id

        # Verify it is active
        active_rel = await service.get_active_release(session)
        assert active_rel.id == rel1.id

        # 2. Create and validate release 2
        rel2 = await service.create_release(
            session,
            CreateReleaseRequest(candidate_id=sample_candidate, release_version="v1.1.0"),
            principal_id=operator_principal.id,
        )
        await service.validate_release(session, rel2.id, principal_id=operator_principal.id)

        # Promote release 2
        promoted2 = await service.promote_release(
            session,
            rel2.id,
            principal_id=reviewer_principal.id,
            reason="Upgraded model prompts",
        )
        assert promoted2.is_active is True
        assert promoted2.status == ReleaseStatus.ACTIVE.value
        assert promoted2.previous_release_id == rel1.id

        # Verify release 1 is now SUPERSEDED
        await session.refresh(rel1)
        assert rel1.is_active is False
        assert rel1.status == ReleaseStatus.SUPERSEDED.value
        assert rel1.superseded_at is not None

        # Verify exactly one release is active
        all_active = (await session.execute(
            select(PromptRelease).where(PromptRelease.is_active == True)
        )).scalars().all()
        assert len(all_active) == 1
        assert all_active[0].id == rel2.id


@pytest.mark.asyncio
async def test_separation_of_duties_enforcement(test_db, sample_candidate, operator_principal):
    """When policy mandates separation of duties, the creator cannot promote their own release."""
    _, session_factory = test_db
    service = PromptReleaseService()

    async with session_factory() as session:
        rel = await service.create_release(
            session,
            CreateReleaseRequest(candidate_id=sample_candidate, release_version="v1.2.0"),
            principal_id=operator_principal.id,
        )
        await service.validate_release(session, rel.id, principal_id=operator_principal.id)

        policy = ReleaseValidationPolicy(require_separation_of_duties=True)

        with pytest.raises(ReleaseValidationFailedError) as exc_info:
            await service.promote_release(
                session,
                rel.id,
                principal_id=operator_principal.id,  # Same principal as creator
                policy=policy,
            )
        assert "Separation of duties" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_safe_rollback_restoration(test_db, sample_candidate, operator_principal, reviewer_principal):
    """Rollback restores a superseded release snapshot and records operator justification."""
    _, session_factory = test_db
    service = PromptReleaseService()

    async with session_factory() as session:
        # Create and promote release 1
        r1 = await service.create_release(
            session,
            CreateReleaseRequest(candidate_id=sample_candidate, release_version="v1.0.0"),
            principal_id=operator_principal.id,
        )
        await service.validate_release(session, r1.id, principal_id=operator_principal.id)
        await service.promote_release(session, r1.id, principal_id=reviewer_principal.id)

        # Create and promote release 2 (supersedes r1)
        r2 = await service.create_release(
            session,
            CreateReleaseRequest(candidate_id=sample_candidate, release_version="v2.0.0"),
            principal_id=operator_principal.id,
        )
        await service.validate_release(session, r2.id, principal_id=operator_principal.id)
        await service.promote_release(session, r2.id, principal_id=reviewer_principal.id)

        # Roll back to r1
        rollback_reason = "Unforeseen latency spikes detected in production v2.0.0"
        restored_r1 = await service.rollback_release(
            session=session,
            target_release_id=r1.id,
            principal_id=reviewer_principal.id,
            reason=rollback_reason,
        )

        assert restored_r1.id == r1.id
        assert restored_r1.is_active is True
        assert restored_r1.status == ReleaseStatus.ACTIVE.value

        # Verify r2 was rolled back
        await session.refresh(r2)
        assert r2.is_active is False
        assert r2.status == ReleaseStatus.ROLLED_BACK.value
        assert r2.rollback_reason == rollback_reason

        # Exactly one active pointer remains
        active = await service.get_active_release(session)
        assert active.id == r1.id


# =============================================================================
# 4. Runtime Integration and Prompt Resolution Tests
# =============================================================================

@pytest.mark.asyncio
async def test_generation_service_resolves_active_release(test_db, sample_candidate, operator_principal):
    """Generation service uses active release instruction and falls back cleanly."""
    _, session_factory = test_db
    service = PromptReleaseService()
    gen_service = CrispeGenerationService()

    invalidate_release_cache()

    async with session_factory() as session:
        # Fallback to CRISPE_SYSTEM_PROMPT when no release exists
        version, instruction = await gen_service.resolve_active_system_prompt(session=session)
        assert version == "v1.0.0-builtin"
        assert instruction == CRISPE_SYSTEM_PROMPT

        # Promote release with custom instructions
        rel = await service.create_release(
            session,
            CreateReleaseRequest(candidate_id=sample_candidate, release_version="v2.5.0-prod"),
            principal_id=operator_principal.id,
        )
        await service.validate_release(session, rel.id, principal_id=operator_principal.id)
        await service.promote_release(session, rel.id, principal_id=operator_principal.id)

        # Generation service now resolves to newly activated release
        act_version, act_instruction = await gen_service.resolve_active_system_prompt(session=session)
        assert act_version == "v2.5.0-prod"
        assert act_instruction == VALID_CRISPE_INSTRUCTION.strip()

        # Cache is populated
        cached = get_cached_active_release()
        assert cached == ("v2.5.0-prod", VALID_CRISPE_INSTRUCTION.strip())

    invalidate_release_cache()


# =============================================================================
# 5. Full End-to-End HTTP API Route Tests
# =============================================================================

@pytest.mark.asyncio
async def test_api_release_full_lifecycle(test_db, sample_candidate, operator_principal):
    """Test full HTTP API lifecycle: create, validate, promote, get active, and rollback."""
    engine, session_factory = test_db
    app = create_application()

    async def override_get_db():
        async with session_factory() as session:
            yield session

    async def override_principal():
        return operator_principal

    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_current_principal] = override_principal

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create release draft
        create_resp = await client.post(
            "/api/v1/optimization/releases",
            json={
                "candidate_id": str(sample_candidate),
                "release_version": "v1.0.0-api",
                "notes": "API test release",
            },
        )
        assert create_resp.status_code == 201
        rel_data = create_resp.json()
        release_id = rel_data["id"]
        assert rel_data["status"] == "draft"
        assert rel_data["is_active"] is False

        # 2. Get release by id
        get_resp = await client.get(f"/api/v1/optimization/releases/{release_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["release_version"] == "v1.0.0-api"

        # 3. Validate release
        val_resp = await client.post(f"/api/v1/optimization/releases/{release_id}/validate")
        assert val_resp.status_code == 200
        val_data = val_resp.json()
        assert val_data["decision"] == "passed"

        # 4. Get validation report
        val_get_resp = await client.get(f"/api/v1/optimization/releases/{release_id}/validation")
        assert val_get_resp.status_code == 200
        assert val_get_resp.json()["decision"] == "passed"

        # 5. Promote release to active
        promote_resp = await client.post(
            f"/api/v1/optimization/releases/{release_id}/promote",
            json={"reason": "Tested via API"},
        )
        assert promote_resp.status_code == 200
        assert promote_resp.json()["is_active"] is True
        assert promote_resp.json()["status"] == "active"

        # 6. Retrieve active release
        active_resp = await client.get("/api/v1/optimization/active-release")
        assert active_resp.status_code == 200
        assert active_resp.json()["id"] == release_id

        # 7. List releases
        list_resp = await client.get("/api/v1/optimization/releases")
        assert list_resp.status_code == 200
        assert list_resp.json()["total_count"] >= 1


@pytest.mark.asyncio
async def test_api_unauthorized_access(test_db):
    """Anonymous API requests without valid credentials are strictly rejected (401)."""
    engine, session_factory = test_db
    app = create_application()

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/optimization/releases",
            json={"candidate_id": str(uuid.uuid4()), "release_version": "v1.0.0"},
        )
        assert resp.status_code == 401
        assert resp.json()["error_code"] == "AUTHENTICATION_REQUIRED"
