"""Durable background worker for executing prompt optimization jobs (Step 26).

Coordinates:
1. Atomic job claiming via PostgreSQL row-level locks and lease fencing.
2. Independent asynchronous execution detached from HTTP request lifecycles.
3. Periodic heartbeat emission and cooperative cancellation checking.
4. Preserved resource accounting across worker retries and restarts.
5. Graceful shutdown on termination signals without leaving corrupted state.
"""

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.dspy_modules.budget_guard import (
    BudgetExceededError,
    ExperimentStopReason,
    OptimizationTimeoutError,
)
from app.dspy_modules.experiment_config import GepaExperimentConfig
from app.dspy_modules.experiment_runner import GepaExperimentResult, GepaExperimentRunner
from app.models.optimization_job import OptimizationJob
from app.schemas.optimization_job import JobStatus
from app.services.optimization_job import OptimizationJobService, get_optimization_job_service

logger = get_logger("optimization_worker")


class OptimizationWorker:
    """Worker process claiming and executing durable GEPA prompt optimization jobs."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        job_service: Optional[OptimizationJobService] = None,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        settings: Optional[Settings] = None,
        poll_interval: float = 2.0,
        lease_duration: float = 60.0,
        heartbeat_interval: float = 15.0,
    ) -> None:
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.job_service = job_service or get_optimization_job_service()
        self.session_factory = session_factory or get_session_factory()
        self.settings = settings or get_settings()
        self.poll_interval = poll_interval
        self.lease_duration = lease_duration
        self.heartbeat_interval = heartbeat_interval
        self._stop_requested = False

    def request_stop(self) -> None:
        """Signal worker to gracefully stop polling for new jobs."""
        logger.info("Worker '%s': graceful stop requested.", self.worker_id)
        self._stop_requested = True

    # -------------------------------------------------------------------------
    # Core Job Execution
    # -------------------------------------------------------------------------

    async def execute_claimed_job(self, job: OptimizationJob) -> JobStatus:
        """Execute a single claimed optimization job outside of a database transaction."""
        job_id = job.id
        claim_token = job.claim_token
        logger.info(
            "Worker '%s' executing job '%s' (claim_token: %s, attempt: %d)",
            self.worker_id,
            job_id,
            claim_token,
            job.attempt_count,
        )

        # 1. Setup Heartbeat and Cancellation Monitor Task
        stop_heartbeat = asyncio.Event()
        fencing_lost = asyncio.Event()
        cancellation_signaled = asyncio.Event()

        async def _heartbeat_loop():
            while not stop_heartbeat.is_set():
                try:
                    await asyncio.sleep(self.heartbeat_interval)
                    if stop_heartbeat.is_set():
                        break
                    async with self.session_factory() as session:
                        is_valid, is_cancel = await self.job_service.heartbeat_and_check_cancellation(
                            session=session,
                            job_id=job_id,
                            claim_token=claim_token,
                            lease_duration_seconds=self.lease_duration,
                        )
                        if not is_valid:
                            logger.error(
                                "Worker '%s' lost lease for job '%s' (fenced out)",
                                self.worker_id,
                                job_id,
                            )
                            fencing_lost.set()
                            break
                        if is_cancel:
                            logger.info(
                                "Worker '%s' observed cancellation request for job '%s'",
                                self.worker_id,
                                job_id,
                            )
                            cancellation_signaled.set()
                except Exception as exc:
                    logger.warning("Heartbeat error for job '%s': %s", job_id, exc)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

        # 2. Compute Remaining Resource Budget (Preserve Counters Across Attempts)
        remaining_metric_calls = max(2, job.max_metric_calls - job.used_metric_calls)
        remaining_provider_calls = max(2, job.max_total_provider_calls - job.used_provider_calls)

        config = GepaExperimentConfig(
            experiment_id=f"job_{job_id.hex[:12]}",
            train_size=job.train_size,
            val_size=job.val_size,
            max_metric_calls=remaining_metric_calls,
            max_total_provider_calls=remaining_provider_calls,
            max_elapsed_seconds=job.max_elapsed_seconds,
            concurrency_limit=job.concurrency_limit,
            seed=job.seed,
            enable_model_judge=job.enable_model_judge,
            is_mock=job.is_mock,
        )

        runner = GepaExperimentRunner(config=config, settings=self.settings)

        # Pre-seed guard with prior attempt counters if recovering
        if job.used_metric_calls > 0:
            runner.guard._metric_calls_count = job.used_metric_calls
        if job.used_provider_calls > 0:
            runner.guard._total_provider_calls = job.used_provider_calls

        final_status = JobStatus.FAILED
        stop_reason = "INTERNAL_FAILURE"
        error_category: Optional[str] = None
        error_summary: Optional[str] = None
        artifacts_dir: Optional[str] = None
        summary_payload: Optional[Dict[str, Any]] = None

        try:
            # Check pre-execution cancellation
            async with self.session_factory() as session:
                _, is_cancel = await self.job_service.heartbeat_and_check_cancellation(
                    session=session,
                    job_id=job_id,
                    claim_token=claim_token,
                )
                if is_cancel:
                    final_status = JobStatus.CANCELLED
                    stop_reason = "USER_CANCELLED"
                    logger.info("Job '%s' cancelled prior to optimizer launch.", job_id)
                    return final_status

            # Execute Experiment via GepaExperimentRunner
            # GepaExperimentRunner is synchronous; run in worker thread to prevent event loop block
            logger.info("Dispatching GEPA experiment runner for job '%s'...", job_id)
            exp_result: GepaExperimentResult = await asyncio.to_thread(
                runner.run_experiment,
                force_mock=job.is_mock,
            )

            artifacts_dir = str(config.results_dir / config.experiment_id)
            summary_payload = {
                "comparison": exp_result.comparison,
                "baseline_mean_score": exp_result.comparison.get("baseline_mean_score"),
                "candidate_mean_score": exp_result.comparison.get("candidate_mean_score"),
                "instructions_changed": exp_result.comparison.get("instructions_changed"),
                "candidate_status": "EXPERIMENTAL_CANDIDATE",
                "is_promoted": False,
            }

            if cancellation_signaled.is_set():
                final_status = JobStatus.CANCELLED
                stop_reason = "USER_CANCELLED"
            elif exp_result.status == "BUDGET_EXCEEDED":
                final_status = JobStatus.BUDGET_EXHAUSTED
                stop_reason = exp_result.resource_summary.budget_exceeded_reason or "BUDGET_EXHAUSTED"
            elif exp_result.status == "DEADLINE_EXCEEDED":
                final_status = JobStatus.TIMED_OUT
                stop_reason = exp_result.resource_summary.budget_exceeded_reason or "DEADLINE_EXCEEDED"
            elif exp_result.status == "FAILED":
                final_status = JobStatus.FAILED
                stop_reason = exp_result.error_message or "OPTIMIZATION_FAILED"
                error_category = "optimizer_failure"
                error_summary = exp_result.error_message
            else:
                final_status = JobStatus.SUCCEEDED
                stop_reason = "OPTIMIZATION_COMPLETED"

        except (BudgetExceededError, OptimizationTimeoutError) as exc:
            stop_reason = getattr(exc, "stop_reason", "BUDGET_EXCEEDED")
            final_status = (
                JobStatus.TIMED_OUT
                if stop_reason == "DEADLINE_EXCEEDED"
                else JobStatus.BUDGET_EXHAUSTED
            )
            error_category = "resource_limit_exceeded"
            error_summary = str(exc)
        except Exception as exc:
            final_status = JobStatus.FAILED
            stop_reason = "UNHANDLED_EXCEPTION"
            error_category = type(exc).__name__
            error_summary = str(exc)
            logger.error("Job '%s' execution threw unhandled error: %s", job_id, exc)

        finally:
            stop_heartbeat.set()
            try:
                await heartbeat_task
            except Exception:
                pass

            # Extract resource summary from guard
            res_summary = runner.guard.get_summary().to_dict()

            # Finalize Job Record in Database
            async with self.session_factory() as session:
                await self.job_service.complete_job(
                    session=session,
                    job_id=job_id,
                    claim_token=claim_token,
                    status=final_status,
                    stop_reason=stop_reason,
                    artifacts_dir=artifacts_dir,
                    result_summary=summary_payload,
                    failure_category=error_category,
                    failure_summary=error_summary,
                    resource_usage_dict=res_summary,
                )

        logger.info(
            "Worker '%s' completed job '%s' with outcome '%s' (%s)",
            self.worker_id,
            job_id,
            final_status.value,
            stop_reason,
        )
        return final_status

    # -------------------------------------------------------------------------
    # Polling & Dispatch Loop
    # -------------------------------------------------------------------------

    async def run_once(self) -> Optional[JobStatus]:
        """Claim and execute at most one pending job. Returns final status or None if no job claimed."""
        async with self.session_factory() as session:
            job = await self.job_service.claim_next_job(
                session=session,
                worker_id=self.worker_id,
                lease_duration_seconds=self.lease_duration,
            )

        if not job:
            return None

        return await self.execute_claimed_job(job)

    async def run_loop(self) -> None:
        """Continuously poll for and process eligible optimization jobs until stopped."""
        logger.info("Worker '%s' started polling loop (interval: %.1fs)", self.worker_id, self.poll_interval)
        while not self._stop_requested:
            try:
                status = await self.run_once()
                if status is None:
                    # No work available, sleep before next poll
                    await asyncio.sleep(self.poll_interval)
            except Exception as exc:
                logger.error("Worker '%s' encounter loop exception: %s", self.worker_id, exc)
                await asyncio.sleep(self.poll_interval)

        logger.info("Worker '%s' shutdown complete.", self.worker_id)


# =============================================================================
# Standalone CLI Entrypoint
# =============================================================================

def main() -> int:
    """CLI entrypoint for standalone background optimization worker."""
    import argparse

    parser = argparse.ArgumentParser(description="Run durable CRISPE optimization worker.")
    parser.add_argument("--worker-id", type=str, default=None, help="Explicit worker identifier.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds.")
    parser.add_argument("--lease-duration", type=float, default=60.0, help="Worker lease duration in seconds.")
    parser.add_argument("--run-once", action="store_true", help="Claim and run at most one job then exit.")

    args = parser.parse_args()

    worker = OptimizationWorker(
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
        lease_duration=args.lease_duration,
    )

    # Register OS signal handlers for graceful termination
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _sig_handler():
        logger.info("Signal caught; shutting down worker...")
        worker.request_stop()

    if sys.platform != "win32":
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, _sig_handler)

    try:
        if args.run_once:
            status = loop.run_until_complete(worker.run_once())
            print(f"Executed single job; final status: {status}")
            return 0 if status in (JobStatus.SUCCEEDED, None) else 1
        else:
            loop.run_until_complete(worker.run_loop())
            return 0
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; terminating worker.")
        worker.request_stop()
        return 0
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
