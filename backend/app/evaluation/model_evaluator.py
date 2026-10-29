"""Model-assisted prompt evaluation service using LLM-as-a-judge (Step 21).

Coordinates structured judge requests, Pydantic response validation, bounded retries,
and combination with deterministic checks into unified evaluation reports.
"""

import asyncio
from datetime import datetime, timezone
import json
import re
import time
from typing import Dict, List, Optional, Tuple
import uuid

from app.adapters.gemini_adapter import (
    BaseModelAdapter,
    ModelGenerationRequest,
    ModelGenerationResponse,
    get_model_adapter,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ProviderAuthenticationError,
    ProviderConfigError,
    ProviderPermissionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.core.logging import get_logger
from app.evaluation.check_schemas import (
    CheckOutcome,
    EvaluationDimension,
    EvaluationExampleResult,
)
from app.evaluation.judge_prompt import (
    JUDGE_INSTRUCTION_VERSION,
    JUDGE_SYSTEM_INSTRUCTION,
    build_judge_user_prompt,
)
from app.evaluation.judge_rubric import (
    DEFAULT_RUBRIC_V1,
    EvaluationRubric,
    ModelJudgeDimension,
)
from app.evaluation.judge_schemas import (
    CombinedEvaluationResult,
    DimensionEvaluation,
    ModelEvaluationResult,
    ModelJudgeResponse,
)
from app.evaluation.schemas import EvaluationExample
from app.evaluation.scoring_engine import DeterministicScorer
from app.schemas.crispe import CrispeSections

logger = get_logger("model_evaluator")


class ModelPromptEvaluator:
    """Evaluates generated CRISPE prompts using an independent LLM judge with structured outputs."""

    def __init__(
        self,
        adapter: Optional[BaseModelAdapter] = None,
        settings: Optional[Settings] = None,
        rubric: Optional[EvaluationRubric] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.adapter = adapter or get_model_adapter()
        self.rubric = rubric or DEFAULT_RUBRIC_V1
        self.model_id = model_id or self.settings.EVALUATOR_MODEL
        self.timeout_seconds = self.settings.EVALUATOR_TIMEOUT_SECONDS
        self.max_retries = self.settings.EVALUATOR_MAX_RETRIES
        self.temperature = self.settings.EVALUATOR_TEMPERATURE

    def _parse_judge_response(self, raw_content: str, parsed_obj: Optional[Any]) -> ModelJudgeResponse:
        """Extract and validate the ModelJudgeResponse instance from provider output."""
        if isinstance(parsed_obj, ModelJudgeResponse):
            return parsed_obj

        if isinstance(parsed_obj, dict):
            return ModelJudgeResponse.model_validate(parsed_obj)

        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        data = json.loads(cleaned)
        return ModelJudgeResponse.model_validate(data)

    async def evaluate_prompt(
        self,
        example: EvaluationExample,
        sections: CrispeSections,
    ) -> ModelEvaluationResult:
        """Execute model-assisted evaluation of a CRISPE prompt using structured judge output.

        Parameters:
            example: Benchmark record from evaluation dataset.
            sections: The 6-section CRISPE prompt to evaluate.

        Returns:
            ModelEvaluationResult containing dimensional ratings and actionable feedback.
        """
        eval_id = f"judge_{uuid.uuid4().hex[:12]}"
        user_prompt = build_judge_user_prompt(example=example, sections=sections, rubric=self.rubric)

        request = ModelGenerationRequest(
            prompt=user_prompt,
            system_instruction=JUDGE_SYSTEM_INSTRUCTION,
            model_id=self.model_id,
            temperature=self.temperature,
            timeout=self.timeout_seconds,
            response_schema=ModelJudgeResponse,
            response_mime_type="application/json",
        )

        attempts = 0
        last_exception: Optional[Exception] = None
        start_time = time.perf_counter()

        while attempts <= self.max_retries:
            attempts += 1
            try:
                logger.info(
                    "Executing model judge evaluation for '%s' (attempt %d/%d, model=%s)",
                    example.example_id,
                    attempts,
                    self.max_retries + 1,
                    self.model_id,
                )
                response: ModelGenerationResponse = await self.adapter.generate_content(request)
                latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

                judge_response = self._parse_judge_response(response.content, response.parsed)

                # Compute mean score across applicable dimensions
                applicable_dims = [d for d in judge_response.dimensions if d.applicable and d.score is not None]
                if applicable_dims:
                    mean_score = sum(d.score for d in applicable_dims) / len(applicable_dims)
                else:
                    mean_score = 0.0

                normalized_score = round(mean_score / 4.0, 4)

                dims_map: Dict[str, DimensionEvaluation] = {
                    d.dimension.value if isinstance(d.dimension, ModelJudgeDimension) else str(d.dimension): d
                    for d in judge_response.dimensions
                }

                return ModelEvaluationResult(
                    evaluation_id=eval_id,
                    example_id=example.example_id,
                    dataset_version=example.example_version,
                    rubric_version=self.rubric.rubric_version,
                    judge_instruction_version=JUDGE_INSTRUCTION_VERSION,
                    evaluator_model_id=response.model_id or self.model_id,
                    status="SUCCESS",
                    dimensions=dims_map,
                    mean_dimension_score=round(mean_score, 4),
                    normalized_model_score=normalized_score,
                    critical_issues=judge_response.critical_issues,
                    summary_weaknesses=judge_response.summary_weaknesses,
                    suggested_improvements=judge_response.suggested_improvements,
                    usage=response.usage,
                    latency_ms=latency_ms,
                    evaluated_at=datetime.now(timezone.utc),
                )

            except (ProviderAuthenticationError, ProviderPermissionError, ProviderConfigError) as exc:
                # Permanent credential or configuration errors fail immediately without retry
                logger.error("Permanent provider error during judge evaluation: %s", exc)
                last_exception = exc
                break

            except (ProviderRateLimitError, ProviderTimeoutError, ProviderUnavailableError) as exc:
                # Transient network, timeout, or quota errors retry with exponential backoff
                last_exception = exc
                logger.warning(
                    "Transient provider error during judge evaluation (attempt %d/%d): %s",
                    attempts,
                    self.max_retries + 1,
                    exc,
                )
                if attempts <= self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** (attempts - 1)))

            except Exception as exc:
                # Unexpected parsing or schema failure
                logger.error("Unexpected error during judge evaluation: %s", exc)
                last_exception = exc
                break

        # If loop exits without returning, evaluation failed
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        err_msg = str(last_exception) if last_exception else "Judge evaluation failed."
        logger.error("Model judge evaluation failed for '%s': %s", example.example_id, err_msg)

        return ModelEvaluationResult(
            evaluation_id=eval_id,
            example_id=example.example_id,
            dataset_version=example.example_version,
            rubric_version=self.rubric.rubric_version,
            judge_instruction_version=JUDGE_INSTRUCTION_VERSION,
            evaluator_model_id=self.model_id,
            status="EVALUATOR_FAILED",
            dimensions={},
            mean_dimension_score=0.0,
            normalized_model_score=0.0,
            critical_issues=[f"Evaluator error: {err_msg}"],
            summary_weaknesses=[],
            suggested_improvements=[],
            error_message=err_msg,
            latency_ms=latency_ms,
            evaluated_at=datetime.now(timezone.utc),
        )

    def detect_disagreements(
        self,
        det_res: EvaluationExampleResult,
        model_res: ModelEvaluationResult,
    ) -> List[str]:
        """Detect and describe discrepancies between deterministic rules and model judgments."""
        disagreements: List[str] = []

        if model_res.status != "SUCCESS":
            return disagreements

        dims = model_res.dimensions

        # 1. Deterministic requirements failure vs High judge requirement score
        failed_req_checks = [c for c in det_res.checks if c.required and c.outcome == CheckOutcome.FAILED]
        req_dim = dims.get(ModelJudgeDimension.REQUIREMENT_AND_CONSTRAINT_COVERAGE.value)
        if failed_req_checks and req_dim and req_dim.score is not None and req_dim.score >= 3:
            check_ids = [c.check_id for c in failed_req_checks]
            disagreements.append(
                f"Disagreement on Requirements: Deterministic checks failed ({check_ids}), "
                f"but model judge awarded high rating ({req_dim.score}/4) on requirement coverage."
            )

        # 2. Deterministic high score vs Judge FAIL / Low score
        if det_res.composite_score >= 0.85 and model_res.normalized_model_score < 0.50:
            disagreements.append(
                f"Disagreement on Overall Quality: Deterministic score is high ({det_res.composite_score:.2f}), "
                f"but model judge assigned low normalized score ({model_res.normalized_model_score:.2f})."
            )

        # 3. Deterministic low score vs Judge Strong rating
        if det_res.composite_score < 0.50 and model_res.normalized_model_score >= 0.85:
            disagreements.append(
                f"Disagreement on Overall Quality: Deterministic score is low ({det_res.composite_score:.2f}), "
                f"but model judge awarded strong rating ({model_res.normalized_model_score:.2f})."
            )

        # 4. Prohibitions divergence
        pro_dim = dims.get(ModelJudgeDimension.INTERNAL_CONSISTENCY.value)
        pro_checks_failed = [c for c in det_res.checks if c.dimension == EvaluationDimension.PROHIBITIONS and c.outcome == CheckOutcome.FAILED]
        if pro_checks_failed and pro_dim and pro_dim.score is not None and pro_dim.score == 4:
            disagreements.append(
                "Disagreement on Prohibitions: Deterministic checks detected prohibited pattern, "
                "but model judge gave perfect score on internal consistency."
            )

        return disagreements

    async def evaluate_combined(
        self,
        example: EvaluationExample,
        sections: Optional[CrispeSections],
        raw_text: Optional[str] = None,
        deterministic_result: Optional[EvaluationExampleResult] = None,
        deterministic_scorer: Optional[DeterministicScorer] = None,
        enable_model_judge: bool = True,
    ) -> CombinedEvaluationResult:
        """Run complete dual evaluation combining deterministic checks and model-assisted judging.

        Scoring & Penalty Policy:
        - When both evaluators run: composite = 0.40 * det_score + 0.60 * model_score.
        - If deterministic critical failure occurs: composite capped at 0.40 (or 0.20 for schema invalid).
        - If model judge flags critical issues: composite capped at 0.40.
        - If generation failed: composite is strictly 0.00.
        """
        eval_id = f"comb_{uuid.uuid4().hex[:12]}"
        scorer = deterministic_scorer or DeterministicScorer()

        # 1. Deterministic Evaluation
        det_res = deterministic_result or scorer.evaluate_example(
            example=example,
            sections=sections,
            raw_text=raw_text,
        )

        model_res: Optional[ModelEvaluationResult] = None
        disagreements: List[str] = []
        critical_reasons = list(det_res.critical_failure_reasons)
        has_critical = det_res.critical_failure

        # 2. Model-Assisted Evaluation (if enabled, sections exist, and not generation failure)
        if enable_model_judge and sections is not None and det_res.status != "GENERATION_FAILED":
            model_res = await self.evaluate_prompt(example=example, sections=sections)
            if model_res.status == "SUCCESS":
                disagreements = self.detect_disagreements(det_res, model_res)
                if model_res.critical_issues:
                    has_critical = True
                    critical_reasons.extend(model_res.critical_issues)

        # 3. Calculate Composite Score
        det_score = det_res.composite_score
        model_score: Optional[float] = None
        if model_res and model_res.status == "SUCCESS":
            model_score = model_res.normalized_model_score
            # Weighted average: 40% deterministic structural/lexical checks + 60% semantic judge
            uncapped_composite = (0.40 * det_score) + (0.60 * model_score)
        else:
            uncapped_composite = det_score

        composite = uncapped_composite

        # 4. Apply Critical Penalty Caps
        if has_critical:
            composite = min(composite, 0.40)

        if not det_res.crispe_structure_valid:
            composite = min(composite, 0.20)

        if det_res.status == "GENERATION_FAILED":
            composite = 0.00

        composite = round(max(0.0, min(1.0, composite)), 4)

        # Status determination
        if det_res.status == "GENERATION_FAILED":
            comb_status = "FAILED"
        elif model_res is None or model_res.status == "SKIPPED":
            comb_status = "PARTIAL_SUCCESS"  # Deterministic only
        elif model_res.status == "EVALUATOR_FAILED":
            comb_status = "PARTIAL_SUCCESS"  # Deterministic preserved despite judge failure
        else:
            comb_status = "SUCCESS"

        return CombinedEvaluationResult(
            evaluation_id=eval_id,
            example_id=example.example_id,
            dataset_version=example.example_version,
            combined_status=comb_status,
            deterministic_result=det_res,
            model_result=model_res,
            deterministic_score=det_score,
            model_assisted_score=model_score,
            composite_score=composite,
            critical_failure=has_critical,
            critical_failure_reasons=critical_reasons,
            disagreements=disagreements,
            evaluated_at=datetime.now(timezone.utc),
        )
