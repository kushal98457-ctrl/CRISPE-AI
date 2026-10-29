"""Reproducible baseline experiment runner for prompt evaluation (Step 19).

Coordinates the systematic execution of the CRISPE generation service across
specified evaluation partitions (defaulting to the validation set), capturing:
- Full structured CRISPE outputs
- Operational indicators (success rate, schema validity, token consumption, latency)
- Checkpointing for resilient resumption of interrupted runs
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.evaluation.dataset_accessor import EvaluationDatasetAccessor, get_dataset_accessor
from app.evaluation.schemas import EvaluationExample
from app.evaluation.split_manifest import PartitionName
from app.schemas.crispe import (
    CrispeGenerationRequest,
    CrispeGenerationResponse,
    CrispeSections,
    ModelUsageMetadata,
)
from app.services.generation_service import CrispeGenerationService, get_generation_service

logger = get_logger("baseline_runner")

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


class BaselineExampleResult(BaseModel):
    """Execution outcome and structured output for a single evaluation example."""

    model_config = ConfigDict(extra="ignore")

    experiment_id: str = Field(..., description="Unique run identifier.")
    example_id: str = Field(..., description="Evaluation dataset example identifier.")
    partition: str = Field(..., description="Partition name (e.g. validation, optimization).")
    task_type: str = Field(..., description="Domain category of the example.")
    difficulty: str = Field(..., description="Complexity tier of the example.")
    model_id: str = Field(..., description="Model identifier used for generation.")
    status: str = Field(..., description="Status: 'SUCCESS' or 'FAILED'.")
    error_category: Optional[str] = Field(default=None, description="Error classification if failed.")
    error_message: Optional[str] = Field(default=None, description="Masked error summary if failed.")
    latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Roundtrip call latency in milliseconds.")
    sections: Optional[CrispeSections] = Field(default=None, description="Generated structured 6-section CRISPE prompt.")
    formatted_prompt: Optional[str] = Field(default=None, description="Rendered Markdown canonical prompt.")
    schema_valid: bool = Field(default=False, description="Whether the output conformed to CrispeSections.")
    usage: Optional[ModelUsageMetadata] = Field(default=None, description="Token consumption metadata.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC completion timestamp.",
    )


class BaselineExperimentSummary(BaseModel):
    """Aggregate operational metrics and execution report for a baseline run."""

    model_config = ConfigDict(extra="ignore")

    experiment_id: str = Field(..., description="Unique experiment identifier.")
    dataset_version: str = Field(..., description="Evaluation dataset version.")
    dataset_sha256: str = Field(..., description="SHA-256 hash of dataset at run time.")
    partition: str = Field(..., description="Evaluated partition name.")
    model_id: str = Field(..., description="Model identifier evaluated.")
    total_attempted: int = Field(..., ge=0, description="Total examples attempted.")
    successful_responses: int = Field(..., ge=0, description="Successful model responses received.")
    failed_requests: int = Field(..., ge=0, description="Failed requests encountered.")
    valid_crispe_outputs: int = Field(..., ge=0, description="Outputs satisfying CrispeSections schema.")
    malformed_outputs: int = Field(..., ge=0, description="Incomplete or malformed outputs.")
    generation_success_rate: float = Field(..., ge=0.0, le=1.0, description="Successful responses / attempted.")
    schema_validity_rate: float = Field(..., ge=0.0, le=1.0, description="Valid CRISPE / successful.")
    overall_validity_rate: float = Field(..., ge=0.0, le=1.0, description="Valid CRISPE / total attempted.")
    mean_latency_ms: Optional[float] = Field(default=None, description="Mean latency in milliseconds.")
    p95_latency_ms: Optional[float] = Field(default=None, description="95th percentile latency in milliseconds.")
    min_latency_ms: Optional[float] = Field(default=None, description="Minimum latency in milliseconds.")
    max_latency_ms: Optional[float] = Field(default=None, description="Maximum latency in milliseconds.")
    total_prompt_tokens: int = Field(default=0, ge=0, description="Total prompt tokens consumed.")
    total_candidates_tokens: int = Field(default=0, ge=0, description="Total completion tokens consumed.")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed.")
    results_file: str = Field(..., description="Path to line-delimited JSON results file.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Summary generation timestamp.",
    )


class BaselineRunner:
    """Manages baseline generation execution with checkpointing and operational metrics."""

    def __init__(
        self,
        generation_service: Optional[CrispeGenerationService] = None,
        dataset_accessor: Optional[EvaluationDatasetAccessor] = None,
        results_dir: Optional[Path] = None,
    ) -> None:
        self.generation_service = generation_service or get_generation_service()
        self.accessor = dataset_accessor or get_dataset_accessor()
        self.results_dir = results_dir or DEFAULT_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _format_generator_input(self, example: EvaluationExample) -> str:
        """Format input for generation service without leaking evaluator criteria."""
        if example.task_context and example.task_context.strip().lower() != "none":
            return f"{example.user_idea}\n\nOperational Context: {example.task_context}"
        return example.user_idea

    def _load_existing_results(self, result_path: Path) -> Dict[str, BaselineExampleResult]:
        """Load already recorded results from checkpoint file for safe resumption."""
        completed: Dict[str, BaselineExampleResult] = {}
        if not result_path.is_file():
            return completed

        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    record = BaselineExampleResult.model_validate(data)
                    completed[record.example_id] = record
                except Exception as exc:
                    logger.warning("Skipping corrupted checkpoint record: %s", exc)

        return completed

    async def run_baseline(
        self,
        experiment_id: str,
        partition: Union[PartitionName, str] = PartitionName.VALIDATION,
        max_examples: Optional[int] = None,
        model_id: Optional[str] = None,
        allow_holdout: bool = False,
    ) -> Tuple[BaselineExperimentSummary, List[BaselineExampleResult]]:
        """Execute baseline prompt generation across the selected partition.

        Parameters:
            experiment_id: Unique identifier for this experiment run.
            partition: Target partition (default: validation).
            max_examples: Optional limit on number of examples to evaluate.
            model_id: Optional model override.
            allow_holdout: Must be explicitly set to True if running on final_holdout.

        Returns:
            Tuple of (BaselineExperimentSummary, list of BaselineExampleResult).
        """
        target_p = partition if isinstance(partition, PartitionName) else PartitionName(partition)
        examples = self.accessor.get_partition_examples(target_p, allow_holdout=allow_holdout)

        if max_examples is not None and max_examples > 0:
            examples = examples[:max_examples]

        result_path = self.results_dir / f"{experiment_id}.jsonl"
        completed = self._load_existing_results(result_path)

        results: List[BaselineExampleResult] = []
        effective_model_id = model_id or "default"

        logger.info(
            "Starting baseline run '%s' on partition '%s': %d examples (%d already in checkpoint)",
            experiment_id,
            target_p.value,
            len(examples),
            len(completed),
        )

        with open(result_path, "a", encoding="utf-8") as result_file:
            for example in examples:
                if example.example_id in completed:
                    results.append(completed[example.example_id])
                    continue

                prompt_input = self._format_generator_input(example)
                start_time = time.perf_counter()

                try:
                    request = CrispeGenerationRequest(
                        idea=prompt_input,
                        model_id=model_id,
                    )
                    gen_response: CrispeGenerationResponse = (
                        await self.generation_service.generate_crispe_prompt(request)
                    )
                    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                    effective_model_id = gen_response.model_id

                    # Verify sections
                    is_valid_schema = isinstance(gen_response.sections, CrispeSections)
                    rec = BaselineExampleResult(
                        experiment_id=experiment_id,
                        example_id=example.example_id,
                        partition=target_p.value,
                        task_type=example.task_type,
                        difficulty=example.difficulty,
                        model_id=gen_response.model_id,
                        status="SUCCESS",
                        latency_ms=elapsed_ms,
                        sections=gen_response.sections,
                        formatted_prompt=gen_response.formatted_prompt,
                        schema_valid=is_valid_schema,
                        usage=gen_response.usage,
                    )
                except Exception as exc:
                    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                    logger.error("Generation failed for example %s: %s", example.example_id, exc)
                    rec = BaselineExampleResult(
                        experiment_id=experiment_id,
                        example_id=example.example_id,
                        partition=target_p.value,
                        task_type=example.task_type,
                        difficulty=example.difficulty,
                        model_id=effective_model_id,
                        status="FAILED",
                        error_category=type(exc).__name__,
                        error_message=str(exc)[:300],
                        latency_ms=elapsed_ms,
                        schema_valid=False,
                    )

                result_file.write(rec.model_dump_json() + "\n")
                result_file.flush()
                results.append(rec)

        summary = self.calculate_summary(experiment_id, target_p.value, results, str(result_path))
        summary_path = self.results_dir / f"{experiment_id}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write(summary.model_dump_json(indent=2) + "\n")

        return summary, results

    def calculate_summary(
        self,
        experiment_id: str,
        partition: str,
        results: Sequence[BaselineExampleResult],
        results_file: str,
    ) -> BaselineExperimentSummary:
        """Calculate statistical summary from example results."""
        total = len(results)
        successful = sum(1 for r in results if r.status == "SUCCESS")
        failed = sum(1 for r in results if r.status == "FAILED")
        valid_schema = sum(1 for r in results if r.schema_valid)
        malformed = successful - valid_schema

        success_rate = (successful / total) if total > 0 else 0.0
        schema_valid_rate = (valid_schema / successful) if successful > 0 else 0.0
        overall_valid_rate = (valid_schema / total) if total > 0 else 0.0

        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        mean_lat = round(sum(latencies) / len(latencies), 2) if latencies else None
        min_lat = round(min(latencies), 2) if latencies else None
        max_lat = round(max(latencies), 2) if latencies else None

        p95_lat = None
        if latencies:
            sorted_lat = sorted(latencies)
            idx_95 = int(len(sorted_lat) * 0.95)
            p95_lat = sorted_lat[min(idx_95, len(sorted_lat) - 1)]

        prompt_tokens = sum(r.usage.prompt_tokens for r in results if r.usage)
        candidates_tokens = sum(r.usage.candidates_tokens for r in results if r.usage)
        total_tokens = sum(r.usage.total_tokens for r in results if r.usage)

        model_id = results[0].model_id if results else "unknown"

        manifest = self.accessor.manifest

        return BaselineExperimentSummary(
            experiment_id=experiment_id,
            dataset_version=manifest.dataset_version,
            dataset_sha256=manifest.dataset_sha256,
            partition=partition,
            model_id=model_id,
            total_attempted=total,
            successful_responses=successful,
            failed_requests=failed,
            valid_crispe_outputs=valid_schema,
            malformed_outputs=malformed,
            generation_success_rate=round(success_rate, 4),
            schema_validity_rate=round(schema_valid_rate, 4),
            overall_validity_rate=round(overall_valid_rate, 4),
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            total_prompt_tokens=prompt_tokens,
            total_candidates_tokens=candidates_tokens,
            total_tokens=total_tokens,
            results_file=results_file,
        )
