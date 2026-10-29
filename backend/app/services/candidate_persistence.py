"""Service orchestrating prompt candidate versioning, evaluation records, and holdout protection (Step 27).

Enforces:
1. Immutable snapshot representation for candidate prompts with SHA-256 deduplication.
2. Separation of candidate definitions from evaluation records across authorized splits.
3. Strict holdout protection: absolute prohibition of evaluation or persistence against final holdout.
4. Tenant isolation: access to candidates and evaluations is strictly scoped to the job's owning principal.
5. Idempotent persistence across worker retries, crashes, or recovery runs.
"""

from functools import lru_cache
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import (
    CandidateNotFoundError,
    HoldoutLeakageError,
    OptimizationJobNotFoundError,
)
from app.core.logging import get_logger
from app.core.security import AuthorizedPrincipal
from app.dspy_modules.subset_selector import PilotSubsetSelector
from app.evaluation.split_manifest import PartitionName, SplitManifest
from app.models.optimization_job import OptimizationJob
from app.models.prompt_candidate import CandidateEvaluation, PromptCandidate
from app.schemas.prompt_candidate import (
    CandidateEvaluationsResponse,
    CandidateLifecycleStatus,
    CandidateSummaryResponse,
    CandidateType,
    EvaluationRecordResponse,
    JobOptimizationResultsResponse,
)

logger = get_logger("candidate_persistence_service")

# Forbidden dataset partitions for optimization and candidate ranking
FORBIDDEN_SPLITS = frozenset({"test", "holdout", "final_holdout"})


class CandidatePersistenceService:
    """Manages lifecycle, persistence, and audit retrieval of prompt candidates and evaluation records."""

    def __init__(self, subset_selector: Optional[PilotSubsetSelector] = None) -> None:
        self.subset_selector = subset_selector or PilotSubsetSelector()
        self._manifest: Optional[SplitManifest] = None

    def _get_manifest(self) -> SplitManifest:
        """Lazily cache the split manifest to verify example partition memberships."""
        if self._manifest is None:
            self._manifest = self.subset_selector.manifest
        return self._manifest

    # -------------------------------------------------------------------------
    # Holdout Validation
    # -------------------------------------------------------------------------

    def validate_split_and_example(
        self,
        dataset_split: str,
        example_id: Optional[str] = None,
    ) -> None:
        """Verify that an evaluation does not target the sealed final holdout partition."""
        norm_split = dataset_split.strip().lower()
        if norm_split in FORBIDDEN_SPLITS:
            logger.error("Forbidden evaluation split '%s' blocked from persistence", dataset_split)
            raise HoldoutLeakageError(
                f"Persisting candidate evaluations against the sealed '{dataset_split}' partition is strictly forbidden."
            )

        if example_id and example_id != "__aggregate__":
            manifest = self._get_manifest()
            holdout_partition = manifest.partitions.get(PartitionName.FINAL_HOLDOUT.value)
            if holdout_partition and example_id in holdout_partition.example_ids:
                logger.error("Holdout example '%s' detected during candidate evaluation persistence", example_id)
                raise HoldoutLeakageError(
                    f"Example '{example_id}' belongs to the sealed final holdout partition and cannot be persisted."
                )

    # -------------------------------------------------------------------------
    # Candidate Versioning & Mutation Lineage
    # -------------------------------------------------------------------------

    async def save_candidate(
        self,
        session: AsyncSession,
        job_id: uuid.UUID,
        instruction_text: str,
        candidate_type: str = CandidateType.OPTIMIZED.value,
        parent_candidate_id: Optional[uuid.UUID] = None,
        lifecycle_status: str = CandidateLifecycleStatus.PROPOSED.value,
        optimizer_version: str = "gepa-0.1.0",
        program_version: str = "crispe-dspy-v1",
        model_id: str = "gemma-4-26b-a4b-it",
        dataset_version: str = "v1",
        evaluation_config: Optional[Dict[str, Any]] = None,
        provenance_metadata: Optional[Dict[str, Any]] = None,
        artifact_ref: Optional[str] = None,
    ) -> PromptCandidate:
        """Record an immutable prompt candidate snapshot with deduplication by content hash."""
        clean_instructions = instruction_text.strip()
        content_hash = hashlib.sha256(clean_instructions.encode("utf-8")).hexdigest()

        # Idempotency check: see if this exact instruction has already been recorded for this job & type
        stmt = select(PromptCandidate).where(
            PromptCandidate.job_id == job_id,
            PromptCandidate.candidate_type == candidate_type,
            PromptCandidate.content_hash == content_hash,
        )
        result = await session.execute(stmt)
        existing = result.scalars().first()

        if existing:
            logger.info(
                "Candidate with content_hash '%s' already exists for job '%s' (id: %s). Reusing.",
                content_hash[:12],
                job_id,
                existing.id,
            )
            # Update artifact reference or status if more detail has become available
            if artifact_ref and not existing.artifact_ref:
                existing.artifact_ref = artifact_ref
            if lifecycle_status and existing.lifecycle_status != lifecycle_status:
                existing.lifecycle_status = lifecycle_status
            await session.flush()
            return existing

        candidate = PromptCandidate(
            job_id=job_id,
            parent_candidate_id=parent_candidate_id,
            candidate_type=candidate_type,
            instruction_text=clean_instructions,
            content_hash=content_hash,
            lifecycle_status=lifecycle_status,
            optimizer_version=optimizer_version,
            program_version=program_version,
            model_id=model_id,
            dataset_version=dataset_version,
            evaluation_config=evaluation_config or {},
            provenance_metadata=provenance_metadata or {},
            artifact_ref=artifact_ref,
        )
        session.add(candidate)
        await session.flush()
        logger.info(
            "Recorded candidate '%s' (type: %s, hash: %s) for job '%s'",
            candidate.id,
            candidate.candidate_type,
            content_hash[:12],
            job_id,
        )
        return candidate

    # -------------------------------------------------------------------------
    # Evaluation Persistence
    # -------------------------------------------------------------------------

    async def save_evaluation(
        self,
        session: AsyncSession,
        candidate_id: uuid.UUID,
        job_id: uuid.UUID,
        evaluation_type: str,
        dataset_split: str,
        dataset_version: str = "v1",
        example_id: Optional[str] = None,
        evaluator_version: str = "v1",
        rubric_version: str = "v1",
        metrics: Optional[Dict[str, Any]] = None,
        passed: Optional[bool] = None,
        is_critical_failure: bool = False,
        sanitized_feedback: Optional[str] = None,
        execution_status: str = "completed",
        artifact_ref: Optional[str] = None,
    ) -> CandidateEvaluation:
        """Persist a single candidate evaluation record with holdout leakage protection."""
        actual_example_id = example_id if example_id is not None else "__aggregate__"
        self.validate_split_and_example(dataset_split, actual_example_id)

        # Idempotency check: see if this exact evaluation has already been persisted
        stmt = select(CandidateEvaluation).where(
            CandidateEvaluation.candidate_id == candidate_id,
            CandidateEvaluation.evaluation_type == evaluation_type,
            CandidateEvaluation.dataset_split == dataset_split,
            CandidateEvaluation.example_id == actual_example_id,
        )
        result = await session.execute(stmt)
        existing = result.scalars().first()

        if existing:
            # Update metrics or artifact ref if refreshed
            if metrics:
                existing.metrics = metrics
            if artifact_ref:
                existing.artifact_ref = artifact_ref
            existing.execution_status = execution_status
            existing.is_critical_failure = is_critical_failure
            await session.flush()
            return existing

        eval_record = CandidateEvaluation(
            candidate_id=candidate_id,
            job_id=job_id,
            evaluation_type=evaluation_type,
            dataset_version=dataset_version,
            dataset_split=dataset_split,
            example_id=actual_example_id,
            evaluator_version=evaluator_version,
            rubric_version=rubric_version,
            metrics=metrics or {},
            passed=passed,
            is_critical_failure=is_critical_failure,
            sanitized_feedback=sanitized_feedback,
            execution_status=execution_status,
            artifact_ref=artifact_ref,
        )
        session.add(eval_record)
        await session.flush()
        return eval_record

    async def save_evaluations_batch(
        self,
        session: AsyncSession,
        evaluations_data: List[Dict[str, Any]],
    ) -> List[CandidateEvaluation]:
        """Atomically persist a batch of candidate evaluations after verifying holdout rules."""
        for item in evaluations_data:
            self.validate_split_and_example(
                dataset_split=item.get("dataset_split", ""),
                example_id=item.get("example_id"),
            )

        persisted: List[CandidateEvaluation] = []
        for item in evaluations_data:
            rec = await self.save_evaluation(
                session=session,
                candidate_id=item["candidate_id"],
                job_id=item["job_id"],
                evaluation_type=item["evaluation_type"],
                dataset_split=item["dataset_split"],
                dataset_version=item.get("dataset_version", "v1"),
                example_id=item.get("example_id"),
                evaluator_version=item.get("evaluator_version", "v1"),
                rubric_version=item.get("rubric_version", "v1"),
                metrics=item.get("metrics"),
                passed=item.get("passed"),
                is_critical_failure=item.get("is_critical_failure", False),
                sanitized_feedback=item.get("sanitized_feedback"),
                execution_status=item.get("execution_status", "completed"),
                artifact_ref=item.get("artifact_ref"),
            )
            persisted.append(rec)

        return persisted

    # -------------------------------------------------------------------------
    # Tenant Isolated Retrieval Services
    # -------------------------------------------------------------------------

    async def get_candidate(
        self,
        session: AsyncSession,
        principal: AuthorizedPrincipal,
        candidate_id: uuid.UUID,
    ) -> PromptCandidate:
        """Retrieve a specific prompt candidate, verifying principal ownership of the parent job."""
        stmt = (
            select(PromptCandidate)
            .join(OptimizationJob, PromptCandidate.job_id == OptimizationJob.id)
            .where(
                PromptCandidate.id == candidate_id,
                OptimizationJob.owner_id == principal.id,
            )
        )
        result = await session.execute(stmt)
        candidate = result.scalars().first()

        if not candidate:
            logger.warning(
                "Principal '%s' requested non-existent or unowned candidate '%s'",
                principal.id,
                candidate_id,
            )
            raise CandidateNotFoundError(str(candidate_id))

        return candidate

    async def list_candidates_for_job(
        self,
        session: AsyncSession,
        principal: AuthorizedPrincipal,
        job_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        candidate_type: Optional[str] = None,
    ) -> Tuple[List[PromptCandidate], int]:
        """List candidates produced by an optimization job with pagination and ownership check."""
        # Check job existence and ownership
        job_stmt = select(OptimizationJob).where(
            OptimizationJob.id == job_id,
            OptimizationJob.owner_id == principal.id,
        )
        job_res = await session.execute(job_stmt)
        if not job_res.scalars().first():
            raise OptimizationJobNotFoundError(str(job_id))

        # Query candidates count
        count_stmt = select(func.count(PromptCandidate.id)).where(PromptCandidate.job_id == job_id)
        if candidate_type:
            count_stmt = count_stmt.where(PromptCandidate.candidate_type == candidate_type)
        total_count = (await session.execute(count_stmt)).scalar() or 0

        # Query paginated candidates
        offset = max(0, (page - 1) * page_size)
        query = (
            select(PromptCandidate)
            .where(PromptCandidate.job_id == job_id)
            .order_by(PromptCandidate.created_at.asc())
            .offset(offset)
            .limit(page_size)
        )
        if candidate_type:
            query = query.where(PromptCandidate.candidate_type == candidate_type)

        items_res = await session.execute(query)
        candidates = list(items_res.scalars().all())
        return candidates, total_count

    async def get_candidate_evaluations(
        self,
        session: AsyncSession,
        principal: AuthorizedPrincipal,
        candidate_id: uuid.UUID,
        evaluation_type: Optional[str] = None,
        dataset_split: Optional[str] = None,
        include_per_example: bool = False,
    ) -> CandidateEvaluationsResponse:
        """Retrieve aggregate evaluations and optional per-example breakdowns for a candidate."""
        # Ensure candidate is accessible by requesting principal
        await self.get_candidate(session, principal, candidate_id)

        # Base query for candidate's evaluations
        query = select(CandidateEvaluation).where(CandidateEvaluation.candidate_id == candidate_id)
        if evaluation_type:
            query = query.where(CandidateEvaluation.evaluation_type == evaluation_type)
        if dataset_split:
            query = query.where(CandidateEvaluation.dataset_split == dataset_split)

        query = query.order_by(CandidateEvaluation.created_at.asc())
        result = await session.execute(query)
        all_evals = list(result.scalars().all())

        aggregates: List[EvaluationRecordResponse] = []
        per_examples: List[EvaluationRecordResponse] = []

        for ev in all_evals:
            dto = ev.to_response_dto()
            if ev.example_id == "__aggregate__":
                aggregates.append(dto)
            else:
                per_examples.append(dto)

        return CandidateEvaluationsResponse(
            candidate_id=candidate_id,
            aggregate_evaluations=aggregates,
            per_example_evaluations=per_examples if include_per_example else None,
        )

    async def get_job_optimization_summary(
        self,
        session: AsyncSession,
        principal: AuthorizedPrincipal,
        job_id: uuid.UUID,
    ) -> JobOptimizationResultsResponse:
        """Assemble complete optimization results summary for an owned job."""
        job_stmt = select(OptimizationJob).where(
            OptimizationJob.id == job_id,
            OptimizationJob.owner_id == principal.id,
        )
        job_res = await session.execute(job_stmt)
        job = job_res.scalars().first()
        if not job:
            raise OptimizationJobNotFoundError(str(job_id))

        # Retrieve all candidates for the job
        cand_stmt = (
            select(PromptCandidate)
            .where(PromptCandidate.job_id == job_id)
            .order_by(PromptCandidate.created_at.asc())
        )
        cand_res = await session.execute(cand_stmt)
        candidates = list(cand_res.scalars().all())

        baseline_cand: Optional[CandidateSummaryResponse] = None
        best_cand: Optional[CandidateSummaryResponse] = None

        for c in candidates:
            summary = c.to_summary_dto()
            if c.candidate_type == CandidateType.BASELINE.value and baseline_cand is None:
                baseline_cand = summary
            elif c.candidate_type == CandidateType.OPTIMIZED.value:
                best_cand = summary

        comparison = job.result_summary.get("comparison") if job.result_summary else None

        # Build safe artifact URNs
        artifacts: Dict[str, str] = {}
        if job.artifacts_dir:
            artifacts["summary"] = f"artifact://jobs/{job_id}/experiment_summary.json"
            artifacts["comparison"] = f"artifact://jobs/{job_id}/validation_comparison.json"
            artifacts["diff"] = f"artifact://jobs/{job_id}/instruction_diff.md"

        return JobOptimizationResultsResponse(
            job_id=job.id,
            status=job.status,
            stop_reason=job.stop_reason,
            baseline_candidate=baseline_cand,
            best_candidate=best_cand,
            comparison_metrics=comparison,
            candidate_count=len(candidates),
            artifacts=artifacts,
            completed_at=job.completed_at,
        )


@lru_cache(maxsize=1)
def get_candidate_persistence_service() -> CandidatePersistenceService:
    """Provide a singleton instance of the CandidatePersistenceService."""
    return CandidatePersistenceService()
