# DSPy & GEPA CRISPE Prompt Generation & Optimization Engine

## Overview

The `app.dspy_modules` package provides a programmable, typed DSPy program, GEPA reflective optimization pipeline, and multi-tiered resource guards for generating and optimizing canonical CRISPE prompts. It establishes the complete integration layer for **Feature A (Self-Evolving Prompt Engine)** while maintaining strict security, concurrency isolation, and dataset partition protection.

---

## 1. Module Structure & Architecture

| Module | Responsibility |
| :--- | :--- |
| [`signature.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/signature.py) | Defines [`CrispeGenerationSignature`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/signature.py#L9-L60) with 4 structured inputs and the 6 canonical CRISPE output fields. |
| [`program.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/program.py) | Encapsulates [`CrispeGenerationProgram`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/program.py#L24-L125) (`dspy.Module`), predictor execution, and canonical [`CrispeSections`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/schemas/crispe.py#L8-L83) validation. |
| [`model_client.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/model_client.py) | Configures DSPy Language Models (`dspy.LM`), model identifier normalization, reflection model creation (`create_dspy_reflection_lm`), and context isolation via `dspy.context(lm=...)`. |
| [`dataset_converter.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/dataset_converter.py) | Converts [`EvaluationExample`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/schemas.py#L34-L170) records into `dspy.Example` instances, strictly designating inputs and enforcing sealed holdout partition protection. |
| [`gepa_metric.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/gepa_metric.py) | Implements [`CrispeGepaMetric`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/gepa_metric.py#L42-L260) satisfying the 5-arg `GEPAFeedbackMetric` protocol, returning `ScoreWithFeedback` with composite scores, diagnostic reflection feedback, and critical failure caps. |
| [`budget_guard.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/budget_guard.py) | Thread-safe [`ResourceBudgetGuard`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/budget_guard.py) enforcing metric-call bounds, cumulative provider request ceilings, wall-clock timeouts, and concurrency limits. |
| [`subset_selector.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/subset_selector.py) | [`PilotSubsetSelector`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/subset_selector.py) selecting balanced, diverse training/validation subsets with dataset SHA-256 verification and zero holdout leakage. |
| [`experiment_config.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_config.py) | Typed Pydantic configuration [`GepaExperimentConfig`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_config.py) governing pilot parameters, sample counts, and budget limits. |
| [`experiment_runner.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_runner.py) | Complete experiment orchestrator [`GepaExperimentRunner`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_runner.py) executing baseline evaluation, reflective optimization, candidate comparison, and artifact persistence. |
| [`run_gepa_experiment.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/run_gepa_experiment.py) | CLI entrypoint for bounded prompt optimization experiments (`python -m app.dspy_modules.run_gepa_experiment`). |
| [`service.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/service.py) | Provides [`DspyCrispeGenerationService`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/service.py#L35-L135), providing functional parity with [`CrispeGenerationService`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/generation_service.py#L51-L157) without altering existing endpoints. |
| [`evaluation_adapter.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/evaluation_adapter.py) | Connects DSPy predictions to Steps 20 & 21 evaluators while strictly separating generator inputs from hidden evaluation criteria. |
| [`verify_gepa_compat.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/verify_gepa_compat.py) | Standalone verification script executing 14 diagnostic compatibility checks across runtime, imports, models, evaluators, and GEPA. |

---

## 2. Resource Budget Guards & Operational Limits (Step 24)

GEPA experiments involve multiple recursive operations: program rollouts, metric evaluations, reflection mutations, and validation scorings. To prevent runaway costs, the [`ResourceBudgetGuard`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/budget_guard.py) enforces three independent ceilings:

1. **GEPA Metric Invocations:** Hard bound on metric evaluations (default: 15 calls for pilots). Reaching this limit gracefully terminates candidate rollouts via `BudgetExceededError`.
2. **Total Provider API Requests:** Hard cumulative ceiling across generation, reflection, and judge calls (default: 35 calls).
3. **Execution Timeout:** Strict wall-clock deadline (default: 300.0s). Exceeding it triggers an `OptimizationTimeoutError`.
4. **Concurrency:** Default concurrency limit = 1 for pilot experiments.

---

## 3. Leakage-Aware Dataset Selection & Integrity

The [`PilotSubsetSelector`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/subset_selector.py) enforces strict experimental hygiene:

- **Cryptographic Verification:** Verifies the dataset file SHA-256 (`9157053e31d52931dbe49182fc1f79bfd8a085d1aaeef0876793b176fbb85573`) against the split manifest.
- **Partition Invariant:** Training examples come strictly from `optimization` (30 records), validation examples strictly from `validation` (10 records).
- **Holdout Partition Sealed:** `load_holdout_dspy_examples(authorized=False)` raises `HoldoutAccessRestrictedError`. The final holdout partition (~20%, 10 records) is 100% excluded from optimization and evaluation.
- **Domain Diversity:** Ensures selected pilot examples span multiple task families (software engineering, AI/ML, product design, edge cases, technical research).

---

## 4. Experiment Workflow & Baseline Comparison

The [`GepaExperimentRunner`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_runner.py) coordinates the optimization lifecycle:

```
+-------------------------------------------------------------------+
| 1. Select Balanced Subsets (Train=5, Val=3) with Sealed Holdout   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 2. Evaluate Original Baseline Program on Validation Subset        |
|    - Record mean score, critical failures, and schema validity    |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 3. Execute Guarded GEPA Optimizer (dspy.GEPA.compile)             |
|    - Reflection LM proposes targeted instruction mutations        |
|    - Metric evaluates rollouts with prioritized feedback          |
|    - ResourceBudgetGuard terminates when metric limit is reached  |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 4. Evaluate Optimized Candidate on Same Validation Subset         |
|    - Identical evaluation rubric and scoring policy               |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 5. Comparative Analysis & Artifact Persistence                    |
|    - Absolute delta & relative change (%)                         |
|    - Unified instruction diff (difflib)                           |
|    - Save experiment_summary.json, validation_comparison.json,    |
|      candidate_program.json, and instruction_diff.md              |
+-------------------------------------------------------------------+
```

---

## 5. Saved Artifacts & Inspection

Experiment runs persist four machine-readable and human-readable artifacts to `backend/app/evaluation/experiments/<experiment_id>/`:

1. `experiment_summary.json`: High-level run metadata, budget limits, actual resource usage, baseline vs. candidate scores, status (`SUCCESS`, `NO_IMPROVEMENT`, `BUDGET_EXCEEDED`, `BLOCKED`).
2. `validation_comparison.json`: Detailed per-example scores and diagnostic check feedbacks comparing baseline vs. candidate on every validation sample.
3. `instruction_diff.md`: Markdown document displaying a unified diff (`difflib`) between the original baseline instruction and the candidate prompt instruction.
4. `candidate_program.json`: Serialized DSPy program state (`dump_state()`) containing predictor signatures, parameters, and instructions.

---

## 6. Strict Production Protection Invariant

> [!IMPORTANT]
> **Experimental Candidates are NEVER Automatically Promoted:**
> All generated candidate programs are strictly stamped `is_promoted: false` and `candidate_status: "EXPERIMENTAL_CANDIDATE"`.
> The active production generation service (`CrispeGenerationService`) and production system prompts in `app.services.generation_service` remain **100% unmodified**. Production release and rollback workflows belong to Steps 27–28.

---

## 7. Running Experiments & Tests

### Execute Pilot Experiment via CLI:
```powershell
python -m app.dspy_modules.run_gepa_experiment --mock --train-size 5 --val-size 3 --max-calls 15
```

### Automated Unit & Integration Tests (Zero Real API Credits):
```powershell
pytest tests/test_gepa_experiment.py -v
pytest tests/test_gepa_compatibility.py -v
```

### Full Backend Test Suite:
```powershell
pytest tests -q
```
*(319 tests passing, 6 skipped)*
