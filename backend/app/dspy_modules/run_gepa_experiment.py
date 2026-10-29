"""CLI command runner for GEPA prompt optimization experiments (Step 24).

Usage:
    python -m app.dspy_modules.run_gepa_experiment [--mock] [--train-size 5] [--val-size 3] [--max-calls 15]
"""

import argparse
import json
import sys

from app.core.logging import get_logger
from app.dspy_modules.experiment_config import GepaExperimentConfig
from app.dspy_modules.experiment_runner import GepaExperimentRunner

logger = get_logger("run_gepa_experiment")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded GEPA prompt optimization experiment.")
    parser.add_argument("--mock", action="store_true", help="Force execution using offline mock models.")
    parser.add_argument("--train-size", type=int, default=5, help="Number of training examples (default 5).")
    parser.add_argument("--val-size", type=int, default=3, help="Number of validation examples (default 3).")
    parser.add_argument("--max-calls", type=int, default=15, help="Maximum GEPA metric calls (default 15).")
    parser.add_argument("--max-provider-calls", type=int, default=35, help="Maximum total provider calls (default 35).")
    parser.add_argument("--timeout", type=float, default=300.0, help="Maximum execution timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Optimization seed (default 42).")

    args = parser.parse_args()

    config = GepaExperimentConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        max_metric_calls=args.max_calls,
        max_total_provider_calls=args.max_provider_calls,
        max_elapsed_seconds=args.timeout,
        seed=args.seed,
        is_mock=args.mock,
    )

    runner = GepaExperimentRunner(config=config)
    result = runner.run_experiment(force_mock=args.mock)

    comp = result.comparison
    res_usage = result.resource_summary

    print("\n" + "=" * 80)
    print(f" GEPA Prompt Optimization Experiment Results: {result.config.experiment_id}")
    print("=" * 80)
    print(f"Status:                    {result.status}")
    print(f"Execution Mode:            {'Deterministic Mock Double' if result.config.is_mock else 'Live Model Execution'}")
    print(f"Dataset Version:           {result.subsets_metadata.dataset_version}")
    print(f"Training Examples:         {len(result.subsets_metadata.train_example_ids)} {result.subsets_metadata.train_example_ids}")
    print(f"Validation Examples:       {len(result.subsets_metadata.val_example_ids)} {result.subsets_metadata.val_example_ids}")
    print(f"Holdout Excluded:          {result.subsets_metadata.holdout_excluded} (100% sealed)")
    print("-" * 80)
    print(f"Baseline Validation Score: {comp['baseline_mean_score']:.4f}")
    print(f"Candidate Validation Score:{comp['candidate_mean_score']:.4f}")
    print(f"Absolute Score Delta:      {comp['absolute_score_diff']:+.4f}")
    print(f"Relative Change:           {comp['relative_change_pct']:+.2f}%")
    print(f"Instructions Changed:      {comp['instructions_changed']}")
    print("-" * 80)
    print(f"GEPA Metric Calls Used:    {res_usage.gepa_metric_calls}/{result.config.max_metric_calls}")
    print(f"Total Provider Calls Used: {res_usage.total_provider_calls}/{result.config.max_total_provider_calls}")
    if res_usage.total_tokens is not None:
        print(f"Total Tokens Reported:     {res_usage.total_tokens}")
    if res_usage.estimated_expenditure_usd is not None:
        print(f"Estimated Cost (USD):      ${res_usage.estimated_expenditure_usd:.4f} (estimate: {res_usage.pricing_version})")
    print(f"Elapsed Wall-Clock Time:   {res_usage.elapsed_seconds:.2f}s")
    print("-" * 80)
    print("Persisted Artifacts:")
    for path in result.artifacts_saved:
        print(f"  - {path}")
    print("=" * 80 + "\n")

    return 0 if result.status in ("SUCCESS", "NO_IMPROVEMENT") else 1


if __name__ == "__main__":
    sys.exit(main())
