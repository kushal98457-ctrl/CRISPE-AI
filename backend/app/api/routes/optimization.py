"""API routes for durable prompt optimization job management (Step 26)."""

from typing import Optional
import uuid
from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ReleaseNotFoundError
from app.core.logging import get_logger
from app.core.security import AuthorizedPrincipal, get_current_principal
from app.db.session import get_db_session
from app.schemas.error import ErrorResponse
from app.schemas.optimization_job import (
    CancelOptimizationJobRequest,
    CreateOptimizationJobRequest,
    OptimizationJobListResponse,
    OptimizationJobResponse,
)
from app.schemas.prompt_candidate import (
    CandidateDetailResponse,
    CandidateEvaluationsResponse,
    CandidateSummaryResponse,
    JobOptimizationResultsResponse,
    PaginatedCandidatesResponse,
)
from app.schemas.prompt_release import (
    CreateReleaseRequest,
    PaginatedReleasesResponse,
    PromoteReleaseRequest,
    ReleaseDetailResponse,
    ReleaseSummaryResponse,
    ReleaseValidationPolicy,
    ReleaseValidationReport,
    RollbackReleaseRequest,
)
from app.services.candidate_persistence import (
    CandidatePersistenceService,
    get_candidate_persistence_service,
)
from app.services.optimization_job import (
    OptimizationJobService,
    get_optimization_job_service,
)
from app.services.prompt_release import (
    PromptReleaseService,
    get_prompt_release_service,
)

logger = get_logger("optimization_routes")

router = APIRouter(prefix="/optimization", tags=["Self-Evolving Optimization"])


@router.post(
    "/jobs",
    response_model=OptimizationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit and queue a durable prompt optimization job",
    description=(
        "Submits a validated GEPA prompt optimization job to the durable PostgreSQL queue. "
        "Returns HTTP 202 Accepted with the committed job identifier and initial queued status. "
        "Execution is detached from the HTTP request and processed by background workers."
    ),
    responses={
        status.HTTP_202_ACCEPTED: {
            "description": "Optimization job successfully queued.",
            "model": OptimizationJobResponse,
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Validation error on model, dataset, or resource budget limits.",
            "model": ErrorResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication is required to submit optimization jobs.",
            "model": ErrorResponse,
        },
        status.HTTP_403_FORBIDDEN: {
            "description": "Principal lacks permission to access holdout partition or submit jobs.",
            "model": ErrorResponse,
        },
    },
)
async def submit_optimization_job(
    request_data: CreateOptimizationJobRequest,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    job_service: OptimizationJobService = Depends(get_optimization_job_service),
) -> OptimizationJobResponse:
    """Queue a new prompt optimization job for asynchronous execution."""
    job = await job_service.create_job(
        session=session,
        principal=principal,
        request=request_data,
    )
    return job.to_response_dto()


@router.get(
    "/jobs/{job_id}",
    response_model=OptimizationJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Get optimization job status, progress, and telemetry",
    description="Retrieves the current execution state, telemetry, and output summary for an owned job.",
    responses={
        status.HTTP_200_OK: {
            "description": "Job status retrieved successfully.",
            "model": OptimizationJobResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required.",
            "model": ErrorResponse,
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Job not found or access denied.",
            "model": ErrorResponse,
        },
    },
)
async def get_optimization_job(
    job_id: uuid.UUID,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    job_service: OptimizationJobService = Depends(get_optimization_job_service),
) -> OptimizationJobResponse:
    """Retrieve the authoritative state of a single optimization job."""
    job = await job_service.get_job(
        session=session,
        principal=principal,
        job_id=job_id,
    )
    return job.to_response_dto()


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=OptimizationJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Request cooperative cancellation of an optimization job",
    description=(
        "Initiates cancellation for an active or queued job. If already queued, transitions "
        "immediately to cancelled. If currently running, signals the active worker to abort "
        "cooperatively at the next execution boundary."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Cancellation initiated or completed.",
            "model": OptimizationJobResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required.",
            "model": ErrorResponse,
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Job not found.",
            "model": ErrorResponse,
        },
        status.HTTP_409_CONFLICT: {
            "description": "Job is in a terminal state and cannot be cancelled.",
            "model": ErrorResponse,
        },
    },
)
async def cancel_optimization_job(
    job_id: uuid.UUID,
    cancel_payload: Optional[CancelOptimizationJobRequest] = None,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    job_service: OptimizationJobService = Depends(get_optimization_job_service),
) -> OptimizationJobResponse:
    """Cancel a queued or running optimization job."""
    reason = cancel_payload.reason if cancel_payload else "User requested cancellation"
    job = await job_service.request_cancellation(
        session=session,
        principal=principal,
        job_id=job_id,
        reason=reason,
    )
    return job.to_response_dto()


@router.get(
    "/jobs",
    response_model=OptimizationJobListResponse,
    status_code=status.HTTP_200_OK,
    summary="List optimization jobs owned by the requesting principal",
    description="Returns a paginated list of optimization jobs owned by the authenticated caller.",
    responses={
        status.HTTP_200_OK: {
            "description": "Jobs retrieved successfully.",
            "model": OptimizationJobListResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required.",
            "model": ErrorResponse,
        },
    },
)
async def list_optimization_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    job_service: OptimizationJobService = Depends(get_optimization_job_service),
) -> OptimizationJobListResponse:
    """List owned optimization jobs with pagination."""
    items, total = await job_service.list_jobs(
        session=session,
        principal=principal,
        limit=limit,
        offset=offset,
    )
    return OptimizationJobListResponse(
        items=[item.to_response_dto() for item in items],
        total_count=total,
    )


# =============================================================================
# Candidate Versioning and Evaluation Results Endpoints (Step 27)
# =============================================================================

@router.get(
    "/jobs/{job_id}/candidates",
    response_model=PaginatedCandidatesResponse,
    status_code=status.HTTP_200_OK,
    summary="List prompt candidates generated by an optimization job",
    description="Returns a paginated list of immutable prompt candidates produced or evaluated by an owned job.",
    responses={
        status.HTTP_200_OK: {
            "description": "Candidates retrieved successfully.",
            "model": PaginatedCandidatesResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required.",
            "model": ErrorResponse,
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Job not found or access denied.",
            "model": ErrorResponse,
        },
    },
)
async def list_job_candidates(
    job_id: uuid.UUID,
    page: int = Query(default=1, ge=1, description="Page number starting at 1"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page (max 100)"),
    candidate_type: Optional[str] = Query(default=None, description="Filter by candidate type (e.g. baseline, optimized)"),
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    candidate_service: CandidatePersistenceService = Depends(get_candidate_persistence_service),
) -> PaginatedCandidatesResponse:
    """Retrieve paginated candidates for an owned optimization job."""
    candidates, total = await candidate_service.list_candidates_for_job(
        session=session,
        principal=principal,
        job_id=job_id,
        page=page,
        page_size=page_size,
        candidate_type=candidate_type,
    )
    return PaginatedCandidatesResponse(
        items=[c.to_summary_dto() for c in candidates],
        total_count=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/candidates/{candidate_id}",
    response_model=CandidateDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get prompt candidate details and exact instruction snapshot",
    description="Retrieves the immutable instruction text, cryptographic content hash, and provenance metadata for a candidate.",
    responses={
        status.HTTP_200_OK: {
            "description": "Candidate details retrieved successfully.",
            "model": CandidateDetailResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required.",
            "model": ErrorResponse,
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Candidate not found or access denied.",
            "model": ErrorResponse,
        },
    },
)
async def get_candidate_detail(
    candidate_id: uuid.UUID,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    candidate_service: CandidatePersistenceService = Depends(get_candidate_persistence_service),
) -> CandidateDetailResponse:
    """Retrieve full immutable candidate details with instruction snapshot."""
    candidate = await candidate_service.get_candidate(
        session=session,
        principal=principal,
        candidate_id=candidate_id,
    )
    return candidate.to_detail_dto()


@router.get(
    "/candidates/{candidate_id}/evaluations",
    response_model=CandidateEvaluationsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get evaluation results and metrics for a prompt candidate",
    description="Retrieves aggregate metrics and optional per-example evaluation records across authorized dataset partitions.",
    responses={
        status.HTTP_200_OK: {
            "description": "Evaluation results retrieved successfully.",
            "model": CandidateEvaluationsResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required.",
            "model": ErrorResponse,
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Candidate not found or access denied.",
            "model": ErrorResponse,
        },
    },
)
async def get_candidate_evaluations(
    candidate_id: uuid.UUID,
    evaluation_type: Optional[str] = Query(default=None, description="Filter by evaluation modality (deterministic, combined)"),
    dataset_split: Optional[str] = Query(default=None, description="Filter by authorized split (val, train)"),
    include_per_example: bool = Query(default=False, description="Whether to include granular per-example evaluations"),
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    candidate_service: CandidatePersistenceService = Depends(get_candidate_persistence_service),
) -> CandidateEvaluationsResponse:
    """Retrieve evaluation records and metric scores for an accessible prompt candidate."""
    return await candidate_service.get_candidate_evaluations(
        session=session,
        principal=principal,
        candidate_id=candidate_id,
        evaluation_type=evaluation_type,
        dataset_split=dataset_split,
        include_per_example=include_per_example,
    )


@router.get(
    "/jobs/{job_id}/results",
    response_model=JobOptimizationResultsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get comprehensive optimization results and candidate comparison",
    description="Retrieves the final optimization outcome, baseline candidate, best candidate, comparative metrics, and artifact URNs.",
    responses={
        status.HTTP_200_OK: {
            "description": "Job optimization results retrieved successfully.",
            "model": JobOptimizationResultsResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required.",
            "model": ErrorResponse,
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Job not found or access denied.",
            "model": ErrorResponse,
        },
    },
)
async def get_job_optimization_results(
    job_id: uuid.UUID,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    candidate_service: CandidatePersistenceService = Depends(get_candidate_persistence_service),
) -> JobOptimizationResultsResponse:
    """Retrieve the authoritative results summary, baseline candidate, and best candidate for a job."""
    return await candidate_service.get_job_optimization_summary(
        session=session,
        principal=principal,
        job_id=job_id,
    )


# =============================================================================
# Prompt Release Management Endpoints (Step 28)
# =============================================================================

@router.post(
    "/releases",
    response_model=ReleaseDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new draft prompt release from an evaluated candidate",
    description="Packages an immutable candidate snapshot into a versioned release draft ready for validation.",
    responses={
        status.HTTP_201_CREATED: {
            "description": "Release draft created successfully.",
            "model": ReleaseDetailResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required.", "model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"description": "Prompt candidate not found.", "model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"description": "Release version already exists.", "model": ErrorResponse},
    },
)
async def create_release(
    request: CreateReleaseRequest,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    release_service: PromptReleaseService = Depends(get_prompt_release_service),
) -> ReleaseDetailResponse:
    """Create a new versioned prompt release draft."""
    release = await release_service.create_release(
        session=session,
        request=request,
        principal_id=principal.id,
    )
    return release.to_detail_dto()


@router.get(
    "/active-release",
    response_model=ReleaseDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current active production prompt release",
    description="Retrieves the single prompt release currently governing production CRISPE prompt generation.",
    responses={
        status.HTTP_200_OK: {"description": "Active release retrieved.", "model": ReleaseDetailResponse},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required.", "model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"description": "No active release found.", "model": ErrorResponse},
    },
)
async def get_active_release(
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    release_service: PromptReleaseService = Depends(get_prompt_release_service),
) -> ReleaseDetailResponse:
    """Retrieve the single active production prompt release."""
    release = await release_service.get_active_release(session=session)
    if release is None:
        raise ReleaseNotFoundError("active")
    return release.to_detail_dto()


@router.get(
    "/releases",
    response_model=PaginatedReleasesResponse,
    status_code=status.HTTP_200_OK,
    summary="List versioned prompt releases",
    description="Retrieves paginated release summaries with optional lifecycle status filtering.",
    responses={
        status.HTTP_200_OK: {"description": "Releases list retrieved.", "model": PaginatedReleasesResponse},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required.", "model": ErrorResponse},
    },
)
async def list_releases(
    status: Optional[str] = Query(default=None, description="Filter by lifecycle status (draft, validated, active, superseded, rolled_back)"),
    limit: int = Query(default=50, ge=1, le=100, description="Page limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    release_service: PromptReleaseService = Depends(get_prompt_release_service),
) -> PaginatedReleasesResponse:
    """List versioned prompt releases with pagination."""
    releases, total = await release_service.list_releases(
        session=session,
        status=status,
        limit=limit,
        offset=offset,
    )
    return PaginatedReleasesResponse(
        items=[r.to_summary_dto() for r in releases],
        total_count=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/releases/{release_id}",
    response_model=ReleaseDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get prompt release details and audit state",
    description="Retrieves complete release metadata, instruction text, and validation results.",
    responses={
        status.HTTP_200_OK: {"description": "Release details retrieved.", "model": ReleaseDetailResponse},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required.", "model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"description": "Release not found.", "model": ErrorResponse},
    },
)
async def get_release(
    release_id: uuid.UUID,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    release_service: PromptReleaseService = Depends(get_prompt_release_service),
) -> ReleaseDetailResponse:
    """Retrieve full details of a specific prompt release."""
    release = await release_service.get_release(session=session, release_id=release_id)
    return release.to_detail_dto()


@router.post(
    "/releases/{release_id}/validate",
    response_model=ReleaseValidationReport,
    status_code=status.HTTP_200_OK,
    summary="Run validation policy gates against a release candidate",
    description="Executes deterministic checks, CRISPE contracts, baseline regression checks, and holdout gates.",
    responses={
        status.HTTP_200_OK: {"description": "Validation completed.", "model": ReleaseValidationReport},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required.", "model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"description": "Release not found.", "model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"description": "Invalid release state transition.", "model": ErrorResponse},
    },
)
async def validate_release(
    release_id: uuid.UUID,
    policy: Optional[ReleaseValidationPolicy] = None,
    authorized_holdout: bool = Query(default=True, description="Whether operator authorizes holdout evaluation"),
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    release_service: PromptReleaseService = Depends(get_prompt_release_service),
) -> ReleaseValidationReport:
    """Validate a release draft against policies and production baseline."""
    return await release_service.validate_release(
        session=session,
        release_id=release_id,
        principal_id=principal.id,
        policy=policy,
        authorized_holdout=authorized_holdout,
    )


@router.get(
    "/releases/{release_id}/validation",
    response_model=ReleaseValidationReport,
    status_code=status.HTTP_200_OK,
    summary="Get validation report for a prompt release",
    description="Retrieves the recorded validation audit report for a release.",
    responses={
        status.HTTP_200_OK: {"description": "Validation report retrieved.", "model": ReleaseValidationReport},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required.", "model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"description": "Release or validation report not found.", "model": ErrorResponse},
    },
)
async def get_release_validation(
    release_id: uuid.UUID,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    release_service: PromptReleaseService = Depends(get_prompt_release_service),
) -> ReleaseValidationReport:
    """Retrieve recorded validation report for a release."""
    release = await release_service.get_release(session=session, release_id=release_id)
    if not release.validation_report:
        raise ReleaseNotFoundError(f"Validation report for release '{release_id}'")
    return ReleaseValidationReport.model_validate(release.validation_report)


@router.post(
    "/releases/{release_id}/promote",
    response_model=ReleaseDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Promote a validated prompt release to active production",
    description="Atomically promotes a validated release to ACTIVE, superseding the prior active release.",
    responses={
        status.HTTP_200_OK: {"description": "Release successfully promoted to active.", "model": ReleaseDetailResponse},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required.", "model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"description": "Release not found.", "model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"description": "Release cannot be promoted due to state or concurrency conflict.", "model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Separation of duties or validation gate failure.", "model": ErrorResponse},
    },
)
async def promote_release(
    release_id: uuid.UUID,
    request: Optional[PromoteReleaseRequest] = None,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    release_service: PromptReleaseService = Depends(get_prompt_release_service),
) -> ReleaseDetailResponse:
    """Promote a validated release to active production pointer."""
    reason = request.reason if request else None
    release = await release_service.promote_release(
        session=session,
        release_id=release_id,
        principal_id=principal.id,
        reason=reason,
    )
    return release.to_detail_dto()


@router.post(
    "/releases/{release_id}/rollback",
    response_model=ReleaseDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Safely roll back production to a prior approved release",
    description="Restores an approved prior release snapshot as the active production pointer.",
    responses={
        status.HTTP_200_OK: {"description": "Production successfully rolled back to target release.", "model": ReleaseDetailResponse},
        status.HTTP_401_UNAUTHORIZED: {"description": "Authentication required.", "model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"description": "Target release not found.", "model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"description": "Illegal rollback target status.", "model": ErrorResponse},
    },
)
async def rollback_release(
    release_id: uuid.UUID,
    request: RollbackReleaseRequest,
    principal: AuthorizedPrincipal = Depends(get_current_principal),
    session: AsyncSession = Depends(get_db_session),
    release_service: PromptReleaseService = Depends(get_prompt_release_service),
) -> ReleaseDetailResponse:
    """Roll back production pointer to a previously approved release snapshot."""
    release = await release_service.rollback_release(
        session=session,
        target_release_id=release_id,
        principal_id=principal.id,
        reason=request.reason,
    )
    return release.to_detail_dto()


