"""Configuration models and defaults for GEPA prompt optimization experiments (Step 24)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_EXPERIMENTS_DIR = (
    Path(__file__).resolve().parent.parent / "evaluation" / "experiments"
)


class GepaExperimentConfig(BaseModel):
    """Execution parameters and resource limits for a bounded GEPA optimization experiment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    experiment_id: str = Field(
        default_factory=lambda: f"gepa_pilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        description="Unique experiment execution identifier.",
    )
    train_size: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of representative training examples from optimization partition.",
    )
    val_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of validation examples for candidate evaluation.",
    )
    max_metric_calls: int = Field(
        default=15,
        ge=2,
        le=50,
        description="Strict ceiling on GEPA metric function invocations.",
    )
    max_total_provider_calls: int = Field(
        default=35,
        ge=2,
        le=100,
        description="Hard bound on cumulative external provider API requests.",
    )
    max_elapsed_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=1800.0,
        description="Maximum wall-clock execution time in seconds.",
    )
    concurrency_limit: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Maximum concurrent model requests during optimization.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility in optimizer candidate selection.",
    )
    enable_model_judge: bool = Field(
        default=False,
        description="Whether to execute LLM judge evaluation during metric scoring.",
    )
    max_total_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        description="Optional aggregate token ceiling for the optimization run.",
    )
    max_budget_usd: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional maximum budget allowed for the experiment in USD.",
    )
    pricing_version: str = Field(
        default="gemini-2025-standard-v1",
        description="Pricing table version used for cost calculations.",
    )
    max_retries_per_request: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts per provider request.",
    )
    is_mock: bool = Field(
        default=False,
        description="Whether to run using deterministic mock models (offline verification).",
    )
    results_dir: Path = Field(
        default=DEFAULT_EXPERIMENTS_DIR,
        description="Output directory where experiment reports and artifacts are persisted.",
    )

    @field_validator("max_total_provider_calls")
    @classmethod
    def validate_provider_ceiling(cls, v: int, info) -> int:
        metric_calls = info.data.get("max_metric_calls", 15)
        if v < metric_calls:
            raise ValueError(
                f"max_total_provider_calls ({v}) must be greater than or equal to max_metric_calls ({metric_calls})"
            )
        return v
