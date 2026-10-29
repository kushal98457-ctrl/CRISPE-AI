"""Baseline artifact evaluation adapter and reporting workflow (Steps 20 & 21).

Evaluates saved baseline prompt-generation outputs from Step 19 without issuing
live model API calls (deterministic) or optionally integrating the LLM-as-a-judge
(Step 21), verifying structure, requirements, semantic quality, and producing
unified evaluation reports.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.logging import get_logger
from app.evaluation.baseline_runner import BaselineExampleResult, DEFAULT_RESULTS_DIR
from app.evaluation.check_schemas import (
    EvaluationExampleResult,
    EvaluationSummaryReport,
)
from app.evaluation.dataset_accessor import (
    EvaluationDatasetAccessor,
    HoldoutAccessRestrictedError,
    get_dataset_accessor,
)
from app.evaluation.judge_schemas import CombinedEvaluationResult
from app.evaluation.model_evaluator import ModelPromptEvaluator
from app.evaluation.schemas import EvaluationExample
from app.evaluation.scoring_engine import DeterministicScorer, ScoringPolicy
from app.evaluation.split_manifest import PartitionName

logger = get_logger("baseline_evaluator")


class BaselineEvaluator:
    """Evaluates previously generated baseline artifacts using deterministic rules and optional AI judge."""

    def __init__(
        self,
        dataset_accessor: Optional[EvaluationDatasetAccessor] = None,
        scorer: Optional[DeterministicScorer] = None,
        model_evaluator: Optional[ModelPromptEvaluator] = None,
        results_dir: Optional[Path] = None,
    ) -> None:
        self.accessor = dataset_accessor or get_dataset_accessor()
        self.scorer = scorer or DeterministicScorer()
        self.model_evaluator = model_evaluator
        self.results_dir = results_dir or DEFAULT_RESULTS_DIR

    def load_baseline_results(self, results_path: Path) -> List[BaselineExampleResult]:
        """Load and parse baseline results from a JSONL file."""
        if not results_path.is_file():
            raise FileNotFoundError(f"Baseline results file not found at: {results_path}")

        results: List[BaselineExampleResult] = []
        with open(results_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    record = BaselineExampleResult.model_validate(data)
                    results.append(record)
                except Exception as exc:
                    logger.warning("Line %d in %s is malformed: %s", line_no, results_path, exc)

        return results

    def evaluate_results(
        self,
        baseline_results: Sequence[BaselineExampleResult],
        experiment_id: str,
        allow_holdout: bool = False,
    ) -> Tuple[EvaluationSummaryReport, List[EvaluationExampleResult]]:
        """Evaluate a sequence of baseline results against their benchmark criteria deterministically."""
        # Pre-check partition and holdout access
        partitions_in_results = {r.partition for r in baseline_results}
        if PartitionName.FINAL_HOLDOUT.value in partitions_in_results and not allow_holdout:
            logger.warning("Attempted to evaluate final_holdout baseline results without authorization.")
            raise HoldoutAccessRestrictedError(
                "Cannot evaluate final holdout results without explicit authorization (allow_holdout=True)."
            )

        evaluated_examples: List[EvaluationExampleResult] = []
        dataset_version = self.accessor.manifest.dataset_version
        partition_name = next(iter(partitions_in_results)) if partitions_in_results else "unknown"

        for b_res in baseline_results:
            example = self.accessor.get_example_by_id(b_res.example_id, allow_holdout=allow_holdout)
            if not example:
                logger.error("Example '%s' from baseline result not found in dataset.", b_res.example_id)
                continue

            eval_res = self.scorer.evaluate_example(
                example=example,
                sections=b_res.sections,
                raw_text=b_res.formatted_prompt,
                generation_status=b_res.status,
                generation_error=b_res.error_message,
            )
            evaluated_examples.append(eval_res)

        summary = self.scorer.aggregate_results(
            results=evaluated_examples,
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            partition=partition_name,
        )

        return summary, evaluated_examples

    async def evaluate_results_combined(
        self,
        baseline_results: Sequence[BaselineExampleResult],
        experiment_id: str,
        allow_holdout: bool = False,
        enable_model_judge: bool = True,
    ) -> Tuple[EvaluationSummaryReport, List[CombinedEvaluationResult]]:
        """Evaluate baseline results using both deterministic rules and model-assisted judging."""
        # Pre-check partition and holdout access
        partitions_in_results = {r.partition for r in baseline_results}
        if PartitionName.FINAL_HOLDOUT.value in partitions_in_results and not allow_holdout:
            logger.warning("Attempted to evaluate final_holdout baseline results without authorization.")
            raise HoldoutAccessRestrictedError(
                "Cannot evaluate final holdout results without explicit authorization (allow_holdout=True)."
            )

        judge = self.model_evaluator or ModelPromptEvaluator()
        combined_evaluations: List[CombinedEvaluationResult] = []
        det_evaluations: List[EvaluationExampleResult] = []
        dataset_version = self.accessor.manifest.dataset_version
        partition_name = next(iter(partitions_in_results)) if partitions_in_results else "unknown"

        for b_res in baseline_results:
            example = self.accessor.get_example_by_id(b_res.example_id, allow_holdout=allow_holdout)
            if not example:
                continue

            comb_res = await judge.evaluate_combined(
                example=example,
                sections=b_res.sections,
                raw_text=b_res.formatted_prompt,
                deterministic_scorer=self.scorer,
                enable_model_judge=enable_model_judge,
            )
            combined_evaluations.append(comb_res)
            det_evaluations.append(comb_res.deterministic_result)

        summary = self.scorer.aggregate_results(
            results=det_evaluations,
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            partition=partition_name,
        )

        return summary, combined_evaluations

    def evaluate_file(
        self,
        results_path: Path,
        experiment_id: Optional[str] = None,
        allow_holdout: bool = False,
        save_report: bool = True,
    ) -> Tuple[EvaluationSummaryReport, List[EvaluationExampleResult]]:
        """Evaluate a baseline results JSONL file deterministically."""
        baseline_results = self.load_baseline_results(results_path)
        exp_id = experiment_id or results_path.stem

        summary, evaluations = self.evaluate_results(
            baseline_results=baseline_results,
            experiment_id=exp_id,
            allow_holdout=allow_holdout,
        )

        if save_report:
            out_eval_path = results_path.parent / f"{exp_id}_eval_details.jsonl"
            out_sum_path = results_path.parent / f"{exp_id}_eval_summary.json"

            with open(out_eval_path, "w", encoding="utf-8") as f:
                for ev in evaluations:
                    f.write(ev.model_dump_json() + "\n")

            with open(out_sum_path, "w", encoding="utf-8") as f:
                f.write(summary.model_dump_json(indent=2) + "\n")

            logger.info("Saved evaluation report to %s and %s", out_sum_path, out_eval_path)

        return summary, evaluations


def main() -> None:
    """CLI entrypoint for evaluating baseline generation output files."""
    parser = argparse.ArgumentParser(description="Evaluate baseline prompt generation outputs.")
    parser.add_argument("--results-file", type=str, required=True, help="Path to baseline JSONL file.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID override.")
    parser.add_argument("--allow-holdout", action="store_true", help="Authorize evaluation on final holdout.")
    parser.add_argument("--with-model-judge", action="store_true", help="Include model-assisted LLM judge evaluation.")
    args = parser.parse_args()

    results_path = Path(args.results_file).resolve()
    evaluator = BaselineEvaluator()

    if args.with_model_judge:
        baseline_results = evaluator.load_baseline_results(results_path)
        exp_id = args.experiment_id or results_path.stem
        summary, comb_evals = asyncio.run(
            evaluator.evaluate_results_combined(
                baseline_results=baseline_results,
                experiment_id=exp_id,
                allow_holdout=args.allow_holdout,
                enable_model_judge=True,
            )
        )
        print("=================================================================")
        print(f"COMBINED EVALUATION SUMMARY: {summary.experiment_id}")
        print("=================================================================")
        print(f"Total Examples:            {summary.total_examples}")
        print(f"Deterministic Score Mean:  {summary.mean_composite_score:.4f}")
        model_scores = [c.model_assisted_score for c in comb_evals if c.model_assisted_score is not None]
        avg_model = sum(model_scores) / len(model_scores) if model_scores else 0.0
        print(f"Model-Assisted Score Mean: {avg_model:.4f}")
        composite_mean = sum(c.composite_score for c in comb_evals) / len(comb_evals) if comb_evals else 0.0
        print(f"Composite Score Mean:      {composite_mean:.4f}")
        disagreements_count = sum(len(c.disagreements) for c in comb_evals)
        print(f"Total Disagreements:       {disagreements_count}")
        print("=================================================================")
    else:
        summary, evals = evaluator.evaluate_file(
            results_path=results_path,
            experiment_id=args.experiment_id,
            allow_holdout=args.allow_holdout,
        )

        print("=================================================================")
        print(f"DETERMINISTIC EVALUATION SUMMARY: {summary.experiment_id}")
        print("=================================================================")
        print(f"Partition:                 {summary.partition}")
        print(f"Dataset Version:           {summary.dataset_version}")
        print(f"Total Examples:            {summary.total_examples}")
        print(f"Valid CRISPE Outputs:      {summary.valid_crispe_outputs} / {summary.total_examples}")
        print(f"Mean Composite Score:      {summary.mean_composite_score:.4f}")
        print(f"Required Check Pass Rate:  {summary.required_check_pass_rate:.4f}")
        print(f"Critical Failure Rate:     {summary.critical_failure_rate:.4f}")
        print("Dimensional Mean Scores:")
        for dim, score in summary.dimension_mean_scores.items():
            print(f"  - {dim:<15}: {score:.4f}")
        print("=================================================================")


if __name__ == "__main__":
    main()
