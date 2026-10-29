"""API routes for durable prompt optimization job management (Step 26)."""

from typing import Optional
import uuid
from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.services.optimization_job import (
    OptimizationJobService,
    get_optimization_job_service,
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
