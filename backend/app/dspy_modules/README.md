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
| [`budget_guard.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/budget_guard.py) | Backward-compatible [`ResourceBudgetGuard`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/budget_guard.py) wrapping [`ResourceBudgetManager`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/resource_budget.py) (Steps 24-25). |
| [`resource_budget.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/resource_budget.py) | Centralized [`ResourceBudgetManager`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/resource_budget.py) enforcing pre-dispatch reservation, retries, concurrency limits, wall-clock deadlines, token accumulation, and cost estimates. |
| [`subset_selector.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/subset_selector.py) | [`PilotSubsetSelector`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/subset_selector.py) selecting balanced, diverse training/validation subsets with dataset SHA-256 verification and zero holdout leakage. |
| [`experiment_config.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_config.py) | Typed Pydantic configuration [`GepaExperimentConfig`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_config.py) governing pilot parameters, sample counts, and budget limits. |
| [`experiment_runner.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_runner.py) | Complete experiment orchestrator [`GepaExperimentRunner`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/experiment_runner.py) executing baseline evaluation, reflective optimization, candidate comparison, and artifact persistence. |
| [`run_gepa_experiment.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/run_gepa_experiment.py) | CLI entrypoint for bounded prompt optimization experiments (`python -m app.dspy_modules.run_gepa_experiment`). |
| [`service.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/service.py) | Provides [`DspyCrispeGenerationService`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/service.py#L35-L135), providing functional parity with [`CrispeGenerationService`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/generation_service.py#L51-L157) without altering existing endpoints. |
| [`evaluation_adapter.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/evaluation_adapter.py) | Connects DSPy predictions to Steps 20 & 21 evaluators while strictly separating generator inputs from hidden evaluation criteria. |
| [`verify_gepa_compat.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/verify_gepa_compat.py) | Standalone verification script executing 14 diagnostic compatibility checks across runtime, imports, models, evaluators, and GEPA. |
| [`optimization_worker.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/workers/optimization_worker.py) | Durable background worker executing prompt optimization jobs detached from HTTP requests with atomic claiming and lease fencing (Step 26). |
| [`optimization_job.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/optimization_job.py) | Job lifecycle service managing job creation, state transitions, cooperative cancellation, and persistent usage recovery (Step 26). |
| [`candidate_persistence.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/candidate_persistence.py) | Persistent candidate versioning, evaluation records, deduplication by content hash, and holdout protection (Step 27). |
| [`artifact_storage.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/artifact_storage.py) | Atomic, traversal-safe filesystem artifact persistence under an isolated root directory with relative URNs (Step 27). |

---

## 2. AI Resource Budgeting & Provider-Call Accounting (Steps 24-25)

GEPA experiments involve multiple recursive operations: program rollouts, metric evaluations, reflection mutations, and validation scorings. To ensure complete cost predictability and operational safety, the centralized [`ResourceBudgetManager`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/resource_budget.py) and [`ResourceBudgetGuard`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/budget_guard.py) enforce multi-tiered resource boundaries:

1. **Metric vs. Provider Discrepancy Separation:**
   - GEPA metric evaluations are tracked independently from physical external provider requests.
   - A single metric evaluation can invoke both student generation and LLM-as-a-judge models.
   - Retries and reflection operations consume additional provider API calls without incrementing the GEPA metric counter.

2. **Pre-Dispatch Capacity Reservation:**
   - Every provider operation must reserve capacity via `reserve_provider_call` before any network request is dispatched.
   - If the request budget or wall-clock deadline is exhausted, the request fails closed immediately (`BudgetExceededError`, `OptimizationTimeoutError`).

3. **Attempt & Retry Accounting:**
   - Transient failure retries inside provider adapters (`GeminiModelAdapter`) consume additional request capacity (`is_retry=True`).
   - Retries are bounded by `max_retries_per_request` and cannot bypass the cumulative experiment request ceiling.

4. **Wall-Clock Deadlines & Concurrency Limits:**
   - Wall-clock deadlines are strictly enforced against monotonic time (`OptimizationTimeoutError`).
   - Process-local concurrency is bounded via `threading.Semaphore`, preventing connection pool starvation or rate limit spikes.
   - Concurrency slots are unconditionally released in `finally` blocks upon completion, timeout, cancellation, or error.

5. **Token Telemetry & Expenditure Estimation:**
   - Provider-reported token usage (`prompt_tokens`, `candidates_tokens`, `total_tokens`) is recorded without fabricating missing values.
   - If token metadata is unavailable, `has_missing_usage=True` is flagged rather than coercing `None` to zero.
   - Estimated expenditure in USD is computed using the verified rate table (`gemini-2025-standard-v1`: \$0.075/1M input, \$0.30/1M output for Gemini 2.5 Flash).
   - Expenditure numbers are explicitly labeled as **estimates**. Unknown model rates return `None`.

6. **Ambient Context Propagation:**
   - Active experiment budgets propagate automatically through task contexts using `active_budget_manager` (`ContextVar`), binding adapter calls down the call stack without manual parameter threading.

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
pytest tests/test_candidate_persistence.py -v
pytest tests/test_optimization_job.py -v
pytest tests/test_resource_budget.py -v
pytest tests/test_gepa_experiment.py -v
pytest tests/test_gepa_compatibility.py -v
```

### Full Backend Test Suite:
```powershell
pytest tests -q
```
*(374 tests passing, 6 skipped)*
