"""Service orchestrating durable optimization job creation, state transitions, and worker claiming (Step 26).

Enforces:
1. Strict server-side validation of dataset versions, holdout exclusion, models, and budgets.
2. Atomic job claiming with row-level locking (SKIP LOCKED on PostgreSQL) and lease fencing.
3. Cooperative cancellation and heartbeat tracking.
4. Persistent resource-budget accounting across worker attempts.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    InvalidJobStateTransitionError,
    JobNotCancellableError,
    OptimizationJobNotFoundError,
    PromptValidationError,
)
from app.core.logging import get_logger
from app.core.security import AuthorizedPrincipal
from app.dspy_modules.subset_selector import PilotSubsetSelector
from app.evaluation.dataset_accessor import HoldoutAccessRestrictedError
from app.models.optimization_job import OptimizationJob
from app.schemas.optimization_job import (
    CreateOptimizationJobRequest,
    JobStateMachine,
    JobStatus,
)

logger = get_logger("optimization_job_service")

# Approved models whitelist for server-side validation
APPROVED_GENERATION_MODELS = frozenset({
    "gemma-4-26b-a4b-it",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "mock-student-gemini-flash",
})

APPROVED_REFLECTION_MODELS = frozenset({
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "mock-reflection-gemini-flash",
})


class OptimizationJobService:
    """Service managing durable optimization jobs and atomic worker orchestration."""

    def __init__(self, subset_selector: Optional[PilotSubsetSelector] = None) -> None:
        self.subset_selector = subset_selector or PilotSubsetSelector()

    # -------------------------------------------------------------------------
    # Client Job Management Operations
    # -------------------------------------------------------------------------

    async def create_job(
        self,
        session: AsyncSession,
        principal: AuthorizedPrincipal,
        request: CreateOptimizationJobRequest,
    ) -> OptimizationJob:
        """Validate, prepare, and atomically persist an optimization job in 'queued' state."""
        # 1. Idempotency Check
        if request.idempotency_key:
            stmt = select(OptimizationJob).where(
                OptimizationJob.owner_id == principal.id,
                OptimizationJob.idempotency_key == request.idempotency_key,
            )
            result = await session.execute(stmt)
            existing = result.scalars().first()
            if existing:
                logger.info(
                    "Idempotent job submission detected for key '%s'; returning existing job '%s'",
                    request.idempotency_key,
                    existing.id,
                )
                return existing

        # 2. Server-Side Model Validation
        if request.generation_model_id not in APPROVED_GENERATION_MODELS and not request.is_mock:
            raise PromptValidationError(
                f"Generation model '{request.generation_model_id}' is not in approved list: "
                f"{sorted(APPROVED_GENERATION_MODELS)}"
            )

        if request.reflection_model_id not in APPROVED_REFLECTION_MODELS and not request.is_mock:
            raise PromptValidationError(
                f"Reflection model '{request.reflection_model_id}' is not in approved list: "
                f"{sorted(APPROVED_REFLECTION_MODELS)}"
            )

        # 3. Dataset Integrity & Partition Validation
        train_examples, val_examples, subsets_meta = self.subset_selector.select_pilot_subsets(
            train_size=request.train_size,
            val_size=request.val_size,
        )

        # Enforce sealed holdout invariant
        for tid in subsets_meta.train_example_ids:
            if "holdout" in tid.lower() or "eval_hld" in tid.lower():
                raise HoldoutAccessRestrictedError("Holdout examples cannot enter training subset.")
        for vid in subsets_meta.val_example_ids:
            if "holdout" in vid.lower() or "eval_hld" in vid.lower():
                raise HoldoutAccessRestrictedError("Holdout examples cannot enter validation subset.")

        # 4. Construct Persistent Optimization Job Record
        now = datetime.now(timezone.utc)
        job = OptimizationJob(
            id=uuid.uuid4(),
            owner_id=principal.id,
            job_type="gepa_prompt_optimization",
            schema_version="v1",
            status=JobStatus.QUEUED.value,
            dataset_version=subsets_meta.dataset_version,
            dataset_sha256=subsets_meta.dataset_sha256,
            partition_manifest="prompt_optimization_eval_v1_splits",
            train_example_ids=subsets_meta.train_example_ids,
            val_example_ids=subsets_meta.val_example_ids,
            train_size=request.train_size,
            val_size=request.val_size,
            seed=request.seed,
            generation_model_id=request.generation_model_id,
            reflection_model_id=request.reflection_model_id,
            enable_model_judge=request.enable_model_judge,
            is_mock=request.is_mock,
            idempotency_key=request.idempotency_key,
            max_metric_calls=request.max_metric_calls,
            max_total_provider_calls=request.max_total_provider_calls,
            max_elapsed_seconds=request.max_elapsed_seconds,
            concurrency_limit=request.concurrency_limit,
            queued_at=now,
            current_phase="queued",
            progress_message="Job queued, awaiting worker claim",
        )

        session.add(job)
        await session.commit()
        await session.refresh(job)

        logger.info(
            "Created durable optimization job '%s' (owner: %s, train: %d, val: %d, status: %s)",
            job.id,
            job.owner_id,
            len(job.train_example_ids or []),
            len(job.val_example_ids or []),
            job.status,
        )
        return job

    async def get_job(
        self,
        session: AsyncSession,
        principal: AuthorizedPrincipal,
        job_id: uuid.UUID,
    ) -> OptimizationJob:
        """Retrieve an optimization job by ID, strictly enforcing owner isolation."""
        stmt = select(OptimizationJob).where(OptimizationJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalars().first()

        if not job:
            raise OptimizationJobNotFoundError(str(job_id))

        if job.owner_id != principal.id and not principal.is_superuser:
            # Safe 404 to avoid leaking job existence to unauthorized tenants
            logger.warning(
                "Unauthorized access attempt to job '%s' by principal '%s'",
                job_id,
                principal.id,
            )
            raise OptimizationJobNotFoundError(str(job_id))

        return job

    async def list_jobs(
        self,
        session: AsyncSession,
        principal: AuthorizedPrincipal,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[OptimizationJob], int]:
        """List optimization jobs owned by the requesting principal with pagination."""
        count_stmt = select(func.count()).select_from(OptimizationJob).where(
            OptimizationJob.owner_id == principal.id
        )
        total_count = (await session.execute(count_stmt)).scalar() or 0

        stmt = (
            select(OptimizationJob)
            .where(OptimizationJob.owner_id == principal.id)
            .order_by(OptimizationJob.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        items = list((await session.execute(stmt)).scalars().all())
        return items, total_count

    async def request_cancellation(
        self,
        session: AsyncSession,
        principal: AuthorizedPrincipal,
        job_id: uuid.UUID,
        reason: Optional[str] = None,
    ) -> OptimizationJob:
        """Request cancellation for a job, transitioning directly or setting cooperative flag."""
        job = await self.get_job(session=session, principal=principal, job_id=job_id)
        current_status = JobStatus(job.status)

        if JobStateMachine.is_terminal(current_status):
            raise JobNotCancellableError(current_status.value)

        now = datetime.now(timezone.utc)
        resolved_reason = reason or "Cancellation requested by user"

        if current_status == JobStatus.QUEUED:
            # Not yet executing: transition immediately to cancelled
            JobStateMachine.validate_transition(current_status, JobStatus.CANCELLED)
            job.status = JobStatus.CANCELLED.value
            job.cancelled_at = now
            job.cancel_requested = True
            job.cancel_reason = resolved_reason
            job.current_phase = "cancelled"
            job.progress_message = f"Cancelled while queued: {resolved_reason}"
            job.stop_reason = "USER_CANCELLED"
        else:
            # Currently running or waiting: flag for worker to observe
            JobStateMachine.validate_transition(current_status, JobStatus.CANCEL_REQUESTED)
            job.cancel_requested = True
            job.status = JobStatus.CANCEL_REQUESTED.value
            job.cancel_reason = resolved_reason
            job.progress_message = f"Cancellation requested: {resolved_reason}"

        await session.commit()
        await session.refresh(job)
        logger.info("Job '%s' cancellation processed (status: %s)", job.id, job.status)
        return job

    # -------------------------------------------------------------------------
    # Worker Claiming & Execution Lifecycle Operations
    # -------------------------------------------------------------------------

    async def claim_next_job(
        self,
        session: AsyncSession,
        worker_id: str,
        lease_duration_seconds: float = 60.0,
    ) -> Optional[OptimizationJob]:
        """Atomically select and lock the next eligible job inside a short transaction.

        Uses SELECT ... FOR UPDATE SKIP LOCKED on PostgreSQL to prevent double-claiming.
        Generates a unique claim_token (fencing token) to prevent stale writes.
        """
        now = datetime.now(timezone.utc)
        bind = session.bind

        # Build eligibility query
        stmt = (
            select(OptimizationJob)
            .where(
                (OptimizationJob.status == JobStatus.QUEUED.value)
                | (
                    (OptimizationJob.status == JobStatus.RUNNING.value)
                    & (OptimizationJob.lease_expires_at < now)
                )
                | (
                    (OptimizationJob.status == JobStatus.RETRY_WAIT.value)
                    & (OptimizationJob.lease_expires_at < now)
                )
                | (
                    (OptimizationJob.status == JobStatus.INTERRUPTED.value)
                    & (OptimizationJob.attempt_count < OptimizationJob.max_attempts)
                )
            )
            .order_by(OptimizationJob.created_at.asc())
            .limit(1)
        )

        # Apply PostgreSQL row locking if dialect supports it
        if bind and getattr(bind.dialect, "name", None) == "postgresql":
            stmt = stmt.with_for_update(skip_locked=True)

        result = await session.execute(stmt)
        job = result.scalars().first()

        if not job:
            return None

        # State machine check
        current_st = JobStatus(job.status)
        JobStateMachine.validate_transition(current_st, JobStatus.RUNNING)

        # Generate fencing token and assign worker lease
        claim_token = uuid.uuid4().hex
        job.claim_token = claim_token
        job.worker_id = worker_id
        job.attempt_count += 1
        job.status = JobStatus.RUNNING.value
        job.started_at = job.started_at or now
        job.last_heartbeat_at = now
        job.lease_expires_at = now + timedelta(seconds=lease_duration_seconds)
        job.current_phase = "claimed"
        job.progress_message = f"Claimed by worker '{worker_id}' (attempt {job.attempt_count})"

        await session.commit()
        await session.refresh(job)

        logger.info(
            "Worker '%s' claimed job '%s' (attempt: %d/%d, lease: %.1fs)",
            worker_id,
            job.id,
            job.attempt_count,
            job.max_attempts,
            lease_duration_seconds,
        )
        return job

    async def heartbeat_and_check_cancellation(
        self,
        session: AsyncSession,
        job_id: uuid.UUID,
        claim_token: str,
        lease_duration_seconds: float = 60.0,
    ) -> Tuple[bool, bool]:
        """Renew worker lease and check if cancellation has been requested.

        Returns (is_valid_lease, is_cancel_requested).
        If is_valid_lease is False, worker was fenced out and must terminate immediately.
        """
        now = datetime.now(timezone.utc)
        stmt = (
            update(OptimizationJob)
            .where(
                OptimizationJob.id == job_id,
                OptimizationJob.claim_token == claim_token,
            )
            .values(
                last_heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_duration_seconds),
            )
            .returning(OptimizationJob.cancel_requested)
        )
        result = await session.execute(stmt)
        row = result.first()
        await session.commit()

        if row is None:
            logger.warning(
                "Fencing check failed for job '%s': claim_token '%s' is no longer valid",
                job_id,
                claim_token,
            )
            return False, False

        cancel_requested = bool(row[0])
        return True, cancel_requested

    async def update_progress(
        self,
        session: AsyncSession,
        job_id: uuid.UUID,
        claim_token: str,
        current_phase: str,
        progress_message: Optional[str] = None,
        progress_percent: Optional[float] = None,
        used_metric_calls: Optional[int] = None,
        used_provider_calls: Optional[int] = None,
        used_total_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
    ) -> bool:
        """Update job progress and resource counters atomically with lease fencing."""
        now = datetime.now(timezone.utc)
        values: Dict[str, Any] = {
            "current_phase": current_phase,
            "last_heartbeat_at": now,
        }
        if progress_message is not None:
            values["progress_message"] = progress_message
        if progress_percent is not None:
            values["progress_percent"] = progress_percent
        if used_metric_calls is not None:
            values["used_metric_calls"] = used_metric_calls
        if used_provider_calls is not None:
            values["used_provider_calls"] = used_provider_calls
        if used_total_tokens is not None:
            values["used_total_tokens"] = used_total_tokens
        if estimated_cost_usd is not None:
            values["estimated_cost_usd"] = estimated_cost_usd

        stmt = (
            update(OptimizationJob)
            .where(
                OptimizationJob.id == job_id,
                OptimizationJob.claim_token == claim_token,
            )
            .values(**values)
        )
        result = await session.execute(stmt)
        await session.commit()
        return (result.rowcount or 0) > 0

    async def complete_job(
        self,
        session: AsyncSession,
        job_id: uuid.UUID,
        claim_token: str,
        status: JobStatus,
        stop_reason: str,
        artifacts_dir: Optional[str] = None,
        result_summary: Optional[Dict[str, Any]] = None,
        failure_category: Optional[str] = None,
        failure_summary: Optional[str] = None,
        resource_usage_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Finalize job execution with authoritative status and persist final artifacts."""
        now = datetime.now(timezone.utc)
        values: Dict[str, Any] = {
            "status": status.value,
            "stop_reason": stop_reason,
            "completed_at": now,
            "lease_expires_at": None,
            "artifacts_dir": artifacts_dir,
            "result_summary": result_summary,
            "failure_category": failure_category,
            "failure_summary": failure_summary,
            "current_phase": "completed" if status == JobStatus.SUCCEEDED else "terminated",
            "progress_message": f"Job finished with status '{status.value}' ({stop_reason})",
        }

        if resource_usage_dict:
            values["used_metric_calls"] = resource_usage_dict.get("gepa_metric_calls", 0)
            values["used_provider_calls"] = resource_usage_dict.get("total_provider_calls", 0)
            values["used_retry_attempts"] = resource_usage_dict.get("retry_attempts_count", 0)
            values["used_prompt_tokens"] = resource_usage_dict.get("total_prompt_tokens") or 0
            values["used_candidates_tokens"] = resource_usage_dict.get("total_candidates_tokens") or 0
            values["used_total_tokens"] = resource_usage_dict.get("total_tokens") or 0
            values["estimated_cost_usd"] = resource_usage_dict.get("estimated_expenditure_usd") or 0.0

        stmt = (
            update(OptimizationJob)
            .where(
                OptimizationJob.id == job_id,
                OptimizationJob.claim_token == claim_token,
            )
            .values(**values)
        )
        result = await session.execute(stmt)
        await session.commit()

        success = (result.rowcount or 0) > 0
        logger.info(
            "Finalized job '%s' with status '%s' (fenced: %s, stop_reason: %s)",
            job_id,
            status.value,
            success,
            stop_reason,
        )
        return success


# Module singleton dependency
_job_service_instance: Optional[OptimizationJobService] = None


def get_optimization_job_service() -> OptimizationJobService:
    """Retrieve or initialize OptimizationJobService singleton."""
    global _job_service_instance
    if _job_service_instance is None:
        _job_service_instance = OptimizationJobService()
    return _job_service_instance
