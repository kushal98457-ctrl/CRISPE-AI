# DSPy & GEPA CRISPE Prompt Generation & Evaluation Integration

## Overview

The `app.dspy_modules` package provides a programmable, typed DSPy program and GEPA optimization boundary for generating canonical CRISPE prompts. It establishes the foundational integration layer for **Feature A (Self-Evolving Prompt Engine)** while maintaining strict security, concurrency isolation, and dataset partition protection.

---

## 1. Module Structure & Architecture

| Module | Responsibility |
| :--- | :--- |
| [`signature.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/signature.py) | Defines [`CrispeGenerationSignature`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/signature.py#L9-L60) with 4 structured inputs and the 6 canonical CRISPE output fields. |
| [`program.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/program.py) | Encapsulates [`CrispeGenerationProgram`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/program.py#L24-L125) (`dspy.Module`), predictor execution, and canonical [`CrispeSections`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/schemas/crispe.py#L8-L83) validation. |
| [`model_client.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/model_client.py) | Configures DSPy Language Models (`dspy.LM`), model identifier normalization, reflection model creation (`create_dspy_reflection_lm`), and context isolation via `dspy.context(lm=...)`. |
| [`dataset_converter.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/dataset_converter.py) | Converts [`EvaluationExample`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/schemas.py#L34-L170) records into `dspy.Example` instances, strictly designating inputs and enforcing sealed holdout partition protection. |
| [`gepa_metric.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/gepa_metric.py) | Implements [`CrispeGepaMetric`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/gepa_metric.py#L42-L260) satisfying the 5-arg `GEPAFeedbackMetric` protocol, returning `ScoreWithFeedback` with composite scores, diagnostic reflection feedback, and critical failure caps. |
| [`service.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/service.py) | Provides [`DspyCrispeGenerationService`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/service.py#L35-L135), providing functional parity with [`CrispeGenerationService`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/generation_service.py#L51-L157) without altering existing endpoints. |
| [`evaluation_adapter.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/evaluation_adapter.py) | Connects DSPy predictions to Steps 20 & 21 evaluators while strictly separating generator inputs from hidden evaluation criteria. |
| [`verify_gepa_compat.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/verify_gepa_compat.py) | Standalone verification script executing 14 diagnostic compatibility checks across runtime, imports, models, evaluators, and GEPA. |

---

## 2. Verified Runtime & Dependency Compatibility Matrix

| Package / Runtime | Declared Constraint | Installed Version | Status | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Python** | `>=3.12` | **3.14.5** | Supported | Runtime execution environment |
| **dspy / dspy-ai** | `>=2.5.0` | **3.4.1** | Supported | Prompt programming & GEPA teleprompter |
| **gepa** | `>=0.1.4` | **0.1.4** (and `dspy.GEPA`) | Supported | Reflective instruction optimization |
| **litellm** | `>=1.40.0` | **1.106.5** | Supported | Multi-provider routing adapter |
| **google-genai** | `>=2.12.0` | **2.33.0** | Supported | Google AI Studio Gemini API client |
| **pydantic** | `>=2.8.0,<3.0.0` | **2.14.1** | Supported | Schema validation & typed models |
| **fastapi** | `>=0.115.0,<1.0.0` | **0.162.0** | Supported | Backend REST API framework |
| **pytest** | `>=8.0.0` | **9.2.0** | Supported | Automated test execution |

---

## 3. Gemini Model Identifiers & Reflection Separation

| Role | Application Config Key | Default Identifier | DSPy / LiteLLM Provider Routing | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Generator** | `PRIMARY_MODEL` / `GEMINI_MODEL_ID` | `gemma-4-26b-a4b-it` (or `gemini-2.5-flash`) | `gemini/gemma-4-26b-a4b-it` | Configurable via environment override. |
| **Production Fallback** | `FALLBACK_MODEL` / `GEMINI_FALLBACK_MODEL_ID` | `gemini-2.5-flash` | `gemini/gemini-2.5-flash` | Production backup model. |
| **GEPA Reflection LM** | `REFLECTION_MODEL` | `gemini-2.5-flash` | `gemini/gemini-2.5-flash` | Used by GEPA to propose prompt mutations (temperature=0.0). |
| **LLM Judge Evaluator** | `EVALUATOR_MODEL` / `JUDGE_MODEL_ID` | `gemini-2.5-flash` | Native `google-genai` SDK | Independent judge for semantic evaluation. |

> [!NOTE]
> When `PRIMARY_MODEL` is set to `gemma-4-26b-a4b-it`, the prompt generation model is Gemma-based, while the evaluator and reflection models use `gemini-2.5-flash`. To unify generation on Gemini Flash, set `PRIMARY_MODEL=gemini-2.5-flash` in `.env`.

---

## 4. GEPA Metric Contract (`CrispeGepaMetric`)

In DSPy 3.4.1, `dspy.GEPA` enforces the `GEPAFeedbackMetric` protocol:
```python
def __call__(
    self,
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None,
    program_trace: Optional[Any] = None,
    **kwargs: Any,
) -> ScoreWithFeedback:
    ...
```

### Key Behaviors:
1. **5-Argument Binding:** Validated by `inspect.signature(metric).bind(None, None, None, None, None)`.
2. **Deterministic Checks:** Evaluates all machine-checkable rules from Step 20 (`DeterministicScorer`).
3. **Model-Assisted Rubric:** Evaluates 8 orthogonal dimensions when `enable_model_judge=True` (`ModelPromptEvaluator`).
4. **Critical Failure Penalty:** Caps composite score at $\le 0.40$ (or $\le 0.20$ for malformed schema) if critical safety or constraint violations are detected.
5. **Actionable Feedback:** Prioritizes critical failures, failed deterministic checks, and low-scoring judge dimensions into structured reflection feedback for GEPA's proposer.
6. **Return Type:** Returns `ScoreWithFeedback(score=composite_score, feedback=diagnostic_feedback, objective_scores={"deterministic": det_score, "model": model_score})`.

---

## 5. Dataset Conversion & Partition Isolation

[`app.dspy_modules.dataset_converter`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/dataset_converter.py) converts dataset records into `dspy.Example` instances:

1. **Input Designation:** Strictly marks generation inputs using `.with_inputs("idea", "task_context", "constraints", "output_format_preference")`.
2. **Criteria Protection:** Hidden criteria (`acceptance_criteria`, `critical_requirements`, `prohibited_behaviors`, `grading_notes`) are stored as metadata on `example.evaluation_example` for the metric, but are completely hidden from `example.inputs()`.
3. **Sealed Holdout Protection:** `load_holdout_dspy_examples(authorized=False)` raises `HoldoutAccessRestrictedError` to prevent final holdout leakage into training loops.

---

## 6. Running Compatibility Verifications & Tests

### Standalone 14-Check Diagnostic Script (Zero Live API Calls):
```powershell
python -m app.dspy_modules.verify_gepa_compat
```
Outputs the 14-point compatibility table with PASS/FAIL/BLOCKED status.

### Automated Unit and Integration Tests:
```powershell
pytest tests/test_gepa_compatibility.py -v
pytest tests/test_dspy_program.py -v
```

### Full Backend Test Suite:
```powershell
pytest tests -q
```
*(308 tests passing, 6 skipped)*

### Opt-in Live Model Smoke Test (Requires GEMINI_API_KEY):
```powershell
$env:RUN_LIVE_TESTS="1"
$env:GEMINI_API_KEY="your-api-key"
pytest tests/test_live_smoke.py -k "test_live_gepa_metric_smoke" -v
```

---

## 7. Operational Safety and Prohibitions

> [!IMPORTANT]
> **No optimization execution occurred in Step 23.**
> `optimizer.compile()` was strictly NOT called. No candidate prompts were proposed or promoted to production. All verification and automated test runs consumed zero live model credits. Step 24 will perform the first controlled, bounded optimization experiment.
