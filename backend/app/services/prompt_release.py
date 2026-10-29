"""Service orchestrating prompt release validation, promotion, and rollback lifecycles (Step 28).

Enforces:
1. Versioned prompt release lifecycle: Draft -> Validating -> Validated/Rejected -> Active -> Superseded/Rolled Back.
2. Single active production release pointer with partial unique constraints and row-level locking.
3. Separation of duties and strict policy validation before production activation.
4. Rollback restoration restoring an immutable past release snapshot without modifying prompt strings.
5. In-memory cache invalidation on promotion/rollback to ensure generation runtime consistency.
6. Comprehensive audit trail for all release actions.
"""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ActiveReleaseConflictError,
    CandidateNotFoundError,
    InvalidReleaseStateTransitionError,
    ReleaseNotFoundError,
    ReleaseValidationFailedError,
)
from app.core.logging import get_logger
from app.models.optimization_job import OptimizationJob
from app.models.prompt_candidate import PromptCandidate
from app.models.prompt_release import PromptRelease, PromptReleaseAuditEvent
from app.schemas.prompt_release import (
    CreateReleaseRequest,
    ReleaseDetailResponse,
    ReleaseStateMachine,
    ReleaseStatus,
    ReleaseSummaryResponse,
    ReleaseValidationPolicy,
    ReleaseValidationReport,
)
from app.services.generation_service import (
    CRISPE_SYSTEM_PROMPT,
    invalidate_release_cache,
    set_cached_active_release,
)
from app.services.release_validation import (
    ReleaseValidationService,
    get_release_validation_service,
)

logger = get_logger("prompt_release_service")


class PromptReleaseService:
    """Manages prompt candidate release packaging, policy validation, promotion, and rollbacks."""

    def __init__(
        self,
        validation_service: Optional[ReleaseValidationService] = None,
    ) -> None:
        self.validation_service = validation_service or get_release_validation_service()

    # -------------------------------------------------------------------------
    # 1. Release Creation (Draft)
    # -------------------------------------------------------------------------

    async def create_release(
        self,
        session: AsyncSession,
        request: CreateReleaseRequest,
        principal_id: str,
    ) -> PromptRelease:
        """Create a new prompt release draft from an immutable evaluated candidate."""
        # Check if version already exists
        version_stmt = select(PromptRelease).where(
            PromptRelease.release_version == request.release_version.strip()
        )
        existing_version = (await session.execute(version_stmt)).scalar_one_or_none()
        if existing_version is not None:
            raise ActiveReleaseConflictError(
                f"Release version '{request.release_version}' already exists."
            )

        # Retrieve target prompt candidate
        cand_stmt = select(PromptCandidate).where(PromptCandidate.id == request.candidate_id)
        candidate = (await session.execute(cand_stmt)).scalar_one_or_none()
        if candidate is None:
            raise CandidateNotFoundError(str(request.candidate_id))

        # Model identifier extraction from candidate
        model_id = getattr(candidate, "model_id", "gemma-4-26b-a4b-it")

        release = PromptRelease(
            release_version=request.release_version.strip(),
            candidate_id=candidate.id,
            candidate_hash=candidate.content_hash,
            job_id=candidate.job_id,
            instruction_text=candidate.instruction_text,
            runtime_compatibility=candidate.program_version or "crispe-dspy-v1",
            generation_model_id=model_id,
            provider_name="google-genai",
            status=ReleaseStatus.DRAFT.value,
            is_active=False,
            created_by_id=principal_id,
        )
        session.add(release)
        await session.flush()

        audit = PromptReleaseAuditEvent(
            release_id=release.id,
            event_type="created",
            acting_principal_id=principal_id,
            from_status=None,
            to_status=ReleaseStatus.DRAFT.value,
            details={
                "candidate_id": str(candidate.id),
                "release_version": release.release_version,
                "notes": request.notes,
            },
        )
        session.add(audit)
        await session.commit()
        await session.refresh(release)

        logger.info(
            "Created release draft %s (id=%s) for candidate %s by principal %s",
            release.release_version,
            release.id,
            candidate.id,
            principal_id,
        )
        return release

    # -------------------------------------------------------------------------
    # 2. Candidate Release Validation
    # -------------------------------------------------------------------------

    async def validate_release(
        self,
        session: AsyncSession,
        release_id: uuid.UUID,
        principal_id: str,
        policy: Optional[ReleaseValidationPolicy] = None,
        authorized_holdout: bool = True,
    ) -> ReleaseValidationReport:
        """Run all automated release validation gates against candidate and production baseline."""
        stmt = select(PromptRelease).where(PromptRelease.id == release_id)
        release = (await session.execute(stmt)).scalar_one_or_none()
        if release is None:
            raise ReleaseNotFoundError(str(release_id))

        # Validate legal transition to VALIDATING
        ReleaseStateMachine.validate_transition(
            ReleaseStatus(release.status), ReleaseStatus.VALIDATING
        )
        old_status = release.status
        release.status = ReleaseStatus.VALIDATING.value

        audit_validating = PromptReleaseAuditEvent(
            release_id=release.id,
            event_type="validating",
            acting_principal_id=principal_id,
            from_status=old_status,
            to_status=ReleaseStatus.VALIDATING.value,
        )
        session.add(audit_validating)
        await session.flush()

        # Load candidate
        cand_stmt = select(PromptCandidate).where(PromptCandidate.id == release.candidate_id)
        candidate = (await session.execute(cand_stmt)).scalar_one_or_none()
        if candidate is None:
            raise CandidateNotFoundError(str(release.candidate_id))

        # Load current active baseline if present
        act_stmt = select(PromptRelease).where(PromptRelease.is_active == True)
        current_active = (await session.execute(act_stmt)).scalar_one_or_none()

        # Execute policy evaluation gates
        active_policy = policy or ReleaseValidationPolicy()
        report = await self.validation_service.validate_candidate_for_release(
            session=session,
            release=release,
            candidate=candidate,
            current_active_release=current_active,
            policy=active_policy,
            authorized_holdout=authorized_holdout,
        )

        release.validation_policy_version = active_policy.policy_version
        release.validation_report = report.model_dump(mode="json")
        release.validated_at = datetime.now(timezone.utc)

        # Record isolated holdout summary if evaluated
        holdout_check = next(
            (c for c in report.checks if c.check_name == "controlled_holdout_evaluation"),
            None,
        )
        if holdout_check:
            release.holdout_evaluation_summary = holdout_check.details

        if report.decision == "passed":
            ReleaseStateMachine.validate_transition(
                ReleaseStatus.VALIDATING, ReleaseStatus.VALIDATED
            )
            release.status = ReleaseStatus.VALIDATED.value
            release.rejection_reason = None

            audit = PromptReleaseAuditEvent(
                release_id=release.id,
                event_type="validated",
                acting_principal_id=principal_id,
                from_status=ReleaseStatus.VALIDATING.value,
                to_status=ReleaseStatus.VALIDATED.value,
                details={"report_decision": "passed", "checks_count": len(report.checks)},
            )
            session.add(audit)
            logger.info("Release %s passed validation", release.release_version)
        else:
            ReleaseStateMachine.validate_transition(
                ReleaseStatus.VALIDATING, ReleaseStatus.REJECTED
            )
            release.status = ReleaseStatus.REJECTED.value
            release.rejection_reason = report.actionable_summary

            audit = PromptReleaseAuditEvent(
                release_id=release.id,
                event_type="rejected",
                acting_principal_id=principal_id,
                from_status=ReleaseStatus.VALIDATING.value,
                to_status=ReleaseStatus.REJECTED.value,
                details={"report_decision": "failed", "summary": report.actionable_summary},
                reason=report.actionable_summary,
            )
            session.add(audit)
            logger.warning(
                "Release %s failed validation: %s",
                release.release_version,
                report.actionable_summary,
            )

        await session.commit()
        await session.refresh(release)
        return report

    # -------------------------------------------------------------------------
    # 3. Production Promotion
    # -------------------------------------------------------------------------

    async def promote_release(
        self,
        session: AsyncSession,
        release_id: uuid.UUID,
        principal_id: str,
        reason: Optional[str] = None,
        policy: Optional[ReleaseValidationPolicy] = None,
    ) -> PromptRelease:
        """Promote a validated release to active production status atomically."""
        # Row lock target release if supported
        stmt = select(PromptRelease).where(PromptRelease.id == release_id)
        if session.bind and session.bind.dialect.name == "postgresql":
            stmt = stmt.with_for_update()

        release = (await session.execute(stmt)).scalar_one_or_none()
        if release is None:
            raise ReleaseNotFoundError(str(release_id))

        # Separation of duties check
        active_policy = policy or ReleaseValidationPolicy()
        if active_policy.require_separation_of_duties and release.created_by_id == principal_id:
            raise ReleaseValidationFailedError(
                "Separation of duties required: promoter cannot be the release creator."
            )

        # Enforce prerequisite validated status
        if release.status != ReleaseStatus.VALIDATED.value:
            raise InvalidReleaseStateTransitionError(
                current_status=release.status,
                requested_status=ReleaseStatus.ACTIVE.value,
            )

        ReleaseStateMachine.validate_transition(
            ReleaseStatus(release.status), ReleaseStatus.ACTIVE
        )

        # Lock and supersede current active release if present
        act_stmt = select(PromptRelease).where(PromptRelease.is_active == True)
        if session.bind and session.bind.dialect.name == "postgresql":
            act_stmt = act_stmt.with_for_update()

        current_active = (await session.execute(act_stmt)).scalar_one_or_none()
        now = datetime.now(timezone.utc)

        if current_active is not None and current_active.id != release.id:
            ReleaseStateMachine.validate_transition(
                ReleaseStatus(current_active.status), ReleaseStatus.SUPERSEDED
            )
            current_active.is_active = False
            current_active.status = ReleaseStatus.SUPERSEDED.value
            current_active.superseded_at = now

            audit_superseded = PromptReleaseAuditEvent(
                release_id=current_active.id,
                event_type="superseded",
                acting_principal_id=principal_id,
                from_status=ReleaseStatus.ACTIVE.value,
                to_status=ReleaseStatus.SUPERSEDED.value,
                details={"superseded_by_release_id": str(release.id)},
            )
            session.add(audit_superseded)
            release.previous_release_id = current_active.id

        # Activate target release
        release.is_active = True
        release.status = ReleaseStatus.ACTIVE.value
        release.promoted_at = now
        release.approved_by_id = principal_id

        audit_promoted = PromptReleaseAuditEvent(
            release_id=release.id,
            event_type="promoted",
            acting_principal_id=principal_id,
            from_status=ReleaseStatus.VALIDATED.value,
            to_status=ReleaseStatus.ACTIVE.value,
            reason=reason,
            details={
                "previous_release_id": str(current_active.id) if current_active else None
            },
        )
        session.add(audit_promoted)

        await session.commit()
        await session.refresh(release)

        # Invalidate runtime prompt cache so subsequent generation requests pick up new prompt
        invalidate_release_cache()
        set_cached_active_release(release.release_version, release.instruction_text)

        logger.info(
            "Promoted release %s (id=%s) to ACTIVE production pointer by %s",
            release.release_version,
            release.id,
            principal_id,
        )
        return release

    # -------------------------------------------------------------------------
    # 4. Rollback Restoration
    # -------------------------------------------------------------------------

    async def rollback_release(
        self,
        session: AsyncSession,
        target_release_id: uuid.UUID,
        principal_id: str,
        reason: str,
    ) -> PromptRelease:
        """Safely restore a previously approved release snapshot as the active production pointer."""
        if not reason or len(reason.strip()) < 5:
            raise ValueError("A clear, substantive rollback justification reason is required.")

        stmt = select(PromptRelease).where(PromptRelease.id == target_release_id)
        if session.bind and session.bind.dialect.name == "postgresql":
            stmt = stmt.with_for_update()

        target_release = (await session.execute(stmt)).scalar_one_or_none()
        if target_release is None:
            raise ReleaseNotFoundError(str(target_release_id))

        # Target must be an approved, superseded, or validated release (never draft or rejected)
        if target_release.status not in {
            ReleaseStatus.SUPERSEDED.value,
            ReleaseStatus.VALIDATED.value,
            ReleaseStatus.ACTIVE.value,
        }:
            raise InvalidReleaseStateTransitionError(
                current_status=target_release.status,
                requested_status=ReleaseStatus.ACTIVE.value,
            )

        if target_release.is_active:
            logger.info("Release %s is already active; rollback is a no-op", target_release.release_version)
            return target_release

        ReleaseStateMachine.validate_transition(
            ReleaseStatus(target_release.status), ReleaseStatus.ACTIVE
        )

        # Find current active release
        act_stmt = select(PromptRelease).where(PromptRelease.is_active == True)
        if session.bind and session.bind.dialect.name == "postgresql":
            act_stmt = act_stmt.with_for_update()

        current_active = (await session.execute(act_stmt)).scalar_one_or_none()
        now = datetime.now(timezone.utc)

        if current_active is not None and current_active.id != target_release.id:
            ReleaseStateMachine.validate_transition(
                ReleaseStatus(current_active.status), ReleaseStatus.ROLLED_BACK
            )
            current_active.is_active = False
            current_active.status = ReleaseStatus.ROLLED_BACK.value
            current_active.rolled_back_at = now
            current_active.rollback_reason = reason

            audit_rb = PromptReleaseAuditEvent(
                release_id=current_active.id,
                event_type="rolled_back",
                acting_principal_id=principal_id,
                from_status=ReleaseStatus.ACTIVE.value,
                to_status=ReleaseStatus.ROLLED_BACK.value,
                reason=reason,
                details={"rolled_back_in_favor_of": str(target_release.id)},
            )
            session.add(audit_rb)

        # Restore target release
        old_target_status = target_release.status
        target_release.is_active = True
        target_release.status = ReleaseStatus.ACTIVE.value
        target_release.promoted_at = now
        target_release.approved_by_id = principal_id

        audit_restore = PromptReleaseAuditEvent(
            release_id=target_release.id,
            event_type="promoted",
            acting_principal_id=principal_id,
            from_status=old_target_status,
            to_status=ReleaseStatus.ACTIVE.value,
            reason=f"Restored via rollback: {reason}",
            details={
                "restored_from_rollback_of": str(current_active.id) if current_active else None
            },
        )
        session.add(audit_restore)

        await session.commit()
        await session.refresh(target_release)

        # Invalidate generation cache
        invalidate_release_cache()
        set_cached_active_release(target_release.release_version, target_release.instruction_text)

        logger.info(
            "Rolled back production pointer to release %s (id=%s) by %s. Reason: %s",
            target_release.release_version,
            target_release.id,
            principal_id,
            reason,
        )
        return target_release

    # -------------------------------------------------------------------------
    # 5. Queries and Inspection
    # -------------------------------------------------------------------------

    async def get_active_release(self, session: AsyncSession) -> Optional[PromptRelease]:
        """Retrieve the currently active production prompt release, if any."""
        stmt = select(PromptRelease).where(PromptRelease.is_active == True)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def get_release(self, session: AsyncSession, release_id: uuid.UUID) -> PromptRelease:
        """Retrieve a specific release by its primary key identifier."""
        stmt = select(PromptRelease).where(PromptRelease.id == release_id)
        release = (await session.execute(stmt)).scalar_one_or_none()
        if release is None:
            raise ReleaseNotFoundError(str(release_id))
        return release

    async def list_releases(
        self,
        session: AsyncSession,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[PromptRelease], int]:
        """List versioned prompt releases with pagination and optional status filter."""
        count_stmt = select(func.count(PromptRelease.id))
        if status:
            count_stmt = count_stmt.where(PromptRelease.status == status)
        total_count = (await session.execute(count_stmt)).scalar() or 0

        stmt = select(PromptRelease)
        if status:
            stmt = stmt.where(PromptRelease.status == status)
        stmt = stmt.order_by(PromptRelease.created_at.desc()).limit(limit).offset(offset)

        releases = list((await session.execute(stmt)).scalars().all())
        return releases, total_count

    # -------------------------------------------------------------------------
    # 6. Bootstrap Initial Release
    # -------------------------------------------------------------------------

    async def bootstrap_initial_release(
        self,
        session: AsyncSession,
        principal_id: str,
    ) -> PromptRelease:
        """Ensure an active baseline production release exists for crisp system operations."""
        # 1. If an active release already exists, return it
        active = await self.get_active_release(session)
        if active is not None:
            return active

        # 2. Check if baseline release version exists
        v1_stmt = select(PromptRelease).where(PromptRelease.release_version == "v1.0.0-baseline")
        existing_v1 = (await session.execute(v1_stmt)).scalar_one_or_none()
        if existing_v1 is not None:
            existing_v1.is_active = True
            existing_v1.status = ReleaseStatus.ACTIVE.value
            existing_v1.promoted_at = datetime.now(timezone.utc)
            existing_v1.approved_by_id = principal_id
            await session.commit()
            await session.refresh(existing_v1)
            set_cached_active_release(existing_v1.release_version, existing_v1.instruction_text)
            return existing_v1

        # 3. Create baseline job, candidate, and release
        bootstrap_job = OptimizationJob(
            owner_id=principal_id,
            status="succeeded",
            generation_model_id="gemma-4-26b-a4b-it",
            dataset_version="v1",
            max_metric_calls=0,
            max_total_provider_calls=0,
        )
        session.add(bootstrap_job)
        await session.flush()

        content_hash = hashlib.sha256(CRISPE_SYSTEM_PROMPT.strip().encode("utf-8")).hexdigest()
        candidate = PromptCandidate(
            job_id=bootstrap_job.id,
            instruction_text=CRISPE_SYSTEM_PROMPT.strip(),
            content_hash=content_hash,
            candidate_type="baseline",
            lifecycle_status="approved",
            optimizer_version="baseline",
            program_version="crispe-baseline-v1",
            model_id="gemma-4-26b-a4b-it",
        )
        session.add(candidate)
        await session.flush()

        release = PromptRelease(
            release_version="v1.0.0-baseline",
            candidate_id=candidate.id,
            candidate_hash=content_hash,
            job_id=bootstrap_job.id,
            instruction_text=CRISPE_SYSTEM_PROMPT.strip(),
            runtime_compatibility="crispe-baseline-v1",
            generation_model_id="gemma-4-26b-a4b-it",
            provider_name="google-genai",
            status=ReleaseStatus.ACTIVE.value,
            is_active=True,
            created_by_id=principal_id,
            approved_by_id=principal_id,
            promoted_at=datetime.now(timezone.utc),
        )
        session.add(release)
        await session.flush()

        audit = PromptReleaseAuditEvent(
            release_id=release.id,
            event_type="promoted",
            acting_principal_id=principal_id,
            from_status=None,
            to_status=ReleaseStatus.ACTIVE.value,
            reason="Bootstrap initial production baseline release",
        )
        session.add(audit)
        await session.commit()
        await session.refresh(release)

        set_cached_active_release(release.release_version, release.instruction_text)
        return release


@lru_cache(maxsize=1)
def get_prompt_release_service() -> PromptReleaseService:
    """Provide a singleton instance of the PromptReleaseService."""
    return PromptReleaseService()
