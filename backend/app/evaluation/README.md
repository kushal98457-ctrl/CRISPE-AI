# CRISPE-AI Prompt Evaluation & Benchmarking System

## Overview

The `app.evaluation` package provides reproducible evaluation datasets, partitioned benchmark manifests, baseline generation execution, and a deterministic prompt-quality evaluation engine for the **Self-Evolving Prompt Engine** (Feature A).

Deterministic evaluation measures objectively verifiable, machine-checkable syntactic, structural, and lexical constraints of generated CRISPE prompts before subsequent model-assisted evaluation (Step 21) and DSPy/GEPA optimization (Steps 25–28) are executed.

---

## 1. Architecture & Module Responsibilities

| Module | Responsibility |
| :--- | :--- |
| [`check_schemas.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/check_schemas.py) | Pydantic data models for check definitions, check outcomes, evaluation example results, and aggregate summary reports. |
| [`rule_engine.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/rule_engine.py) | Pure rule execution engine with ReDoS-safe regex validation, canonical 6-section CRISPE validation, text normalization, and negation awareness. |
| [`rule_builder.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/rule_builder.py) | Translates structured [`EvaluationExample`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/schemas.py) records into deterministic check suites. |
| [`scoring_engine.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/scoring_engine.py) | Implements multi-dimensional scoring policies, critical failure caps, schema failure caps, and aggregate metric calculations with explicit denominators. |
| [`baseline_evaluator.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/baseline_evaluator.py) | Adapter that evaluates saved baseline generation JSONL artifacts without issuing live LLM API requests. |
| [`baseline_runner.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/baseline_runner.py) | Coordinates generation baselines across dataset partitions with checkpoint resumption and token/latency metrics. |
| [`dataset_accessor.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/dataset_accessor.py) | Provides partitioned access to benchmark records with sealed holdout protection ([`HoldoutAccessRestrictedError`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/dataset_accessor.py)). |
| [`split_manifest.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/split_manifest.py) | Stratified, group-aware dataset partition allocations (60% optimization, 20% validation, 20% holdout) with SHA-256 verification. |

---

## 2. Supported Deterministic Check Types

The engine supports 9 explicit, machine-checkable check types:

1. **`crispe_schema_valid`**:
   - Verifies that the prompt satisfies canonical [`CrispeSections`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/schemas/crispe.py) schema (all 6 sections present, non-empty, and valid strings).
2. **`required_section_present`**:
   - Verifies that a specific target section (`context`, `role`, `instruction`, `specification`, `performance`, or `example`) has at least $N$ characters (default 5–10).
3. **`minimum_text_length`**:
   - Asserts that a target section meets a minimum character or word count threshold.
4. **`required_text`**:
   - Tests for exact phrase containment with configurable case-sensitivity and whitespace normalization.
5. **`required_term_group`**:
   - Verifies that at least `min_matches` (default 1) of a list of technical synonyms or phrases appear in the target section.
6. **`forbidden_text`**:
   - Detects prohibited phrases or antipatterns. Employs negation-context detection: if a term appears within a negative boundary statement (e.g. *"Do not use eval()"*, *"avoid MD5"*), it is recognized as constraint enforcement rather than a violation.
7. **`regex_constraint`**:
   - Executes pre-validated regular expressions. Enforces ReDoS protection by rejecting nested repetitions (e.g. `(a+)+`) and patterns longer than 500 characters. Arbitrary code execution (`eval`/`exec`) is strictly prohibited.
8. **`constraint_preserved`**:
   - Asserts that explicit technology locks, limitations, or negative boundaries defined in the dataset example are preserved in the prompt.
9. **`output_format_requirement`**:
   - Verifies that format indicators (e.g., Markdown headers, JSON schema, tables, step lists) are explicitly declared in the prompt.

---

## 3. Representation of Rules in the Dataset

Rules are derived and represented through two complementary mechanisms:

1. **Dynamic Check Suite Generation ([`rule_builder.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/rule_builder.py)):**
   - Automatically builds a complete check suite from structured fields of an [`EvaluationExample`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/schemas.py):
     - `critical_requirements` $\to$ `REQUIRED_TERM_GROUP` with `is_critical=True`, `dimension=requirements`
     - `required_behaviors` $\to$ `REQUIRED_TERM_GROUP` with `dimension=requirements`
     - `constraints` $\to$ `CONSTRAINT_PRESERVED` with `dimension=constraints`
     - `prohibited_behaviors` $\to$ `FORBIDDEN_TEXT` with `dimension=prohibitions`
     - `expected_output_format` $\to$ `OUTPUT_FORMAT_REQUIREMENT` with `dimension=output_format`
     - Canonical CRISPE schema $\to$ `CRISPE_SCHEMA_VALID` with `dimension=structure`
2. **Explicit Check Definitions (`deterministic_checks` field):**
   - [`EvaluationExample`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/schemas.py) supports an optional `deterministic_checks: Optional[List[DeterministicCheckDefinition]] = None` field, allowing dataset authors to attach exact machine-checkable rules without modifying legacy records.

---

## 4. Multi-Dimensional Scoring Dimensions & Formulas

The evaluation engine computes five orthogonal dimensional scores rather than an opaque single score:

- **Structure ($W_1 = 0.25$):** Canonical 6-section CRISPE schema validity and section completeness.
- **Requirements ($W_2 = 0.35$):** Explicit requirement coverage, critical requirements, and technical term groups.
- **Constraints ($W_3 = 0.20$):** Preservation of explicit technological boundaries and limitations.
- **Output Format ($W_4 = 0.10$):** Adherence to declared output formatting indicators.
- **Prohibitions ($W_5 = 0.10$):** Absence of prohibited patterns and antipatterns.

### Dimensional Score Formula

For each dimension $d$:
$$s_d = \frac{\sum_{c \in \text{passed}(d)} w_c}{\sum_{c \in \text{applicable}(d)} w_c}$$

Where:
- $\text{applicable}(d)$ includes all executed checks whose outcome is `passed`, `failed`, or `error` (`not_applicable` checks are excluded).
- If a dimension has 0 applicable checks, $s_d = 1.0$ (neutral).

### Uncapped Composite Score

$$S_{\text{uncapped}} = \frac{\sum_{d \in \text{active}} W_d \cdot s_d}{\sum_{d \in \text{active}} W_d}$$

Active weights are dynamically normalized to sum to 1.0.

---

## 5. Critical Failure Penalty Policy

To prevent a prompt with missing critical requirements from disguising its failure behind high formatting or structure scores:

1. **Critical Requirement Failure Cap:**
   - If any check marked `is_critical=True` fails (or errors), the composite score is strictly capped at **0.40**:
     $$S = \min(S_{\text{uncapped}}, 0.40)$$
2. **Canonical Schema Invalidation Cap:**
   - If the prompt fails canonical CRISPE schema validation (missing sections, whitespace-only fields), the composite score is strictly capped at **0.20**:
     $$S = \min(S, 0.20)$$
3. **Upstream Generation Failure:**
   - If the generative model fails or times out before producing output, $S = 0.0$ and status is set to `GENERATION_FAILED`.

---

## 6. Aggregate Metrics & Explicit Denominators

The aggregation layer ([`EvaluationSummaryReport`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/check_schemas.py)) calculates metrics with mathematically explicit denominators:

| Metric | Formula | Denominator Definition |
| :--- | :--- | :--- |
| **`required_check_pass_rate`** | $\frac{N_{\text{passed\_required}}}{N_{\text{applicable\_required}}}$ | Total required checks with outcome `passed`, `failed`, or `error`. Returns 1.0 if denominator is 0. |
| **`critical_failure_rate`** | $\frac{N_{\text{critical\_failures}}}{N_{\text{total\_examples}}}$ | Total examples attempted. |
| **`mean_composite_score`** | $\frac{\sum_{i=1}^N S_i}{N_{\text{total\_examples}}}$ | Total examples evaluated. Returns 0.0 if empty dataset. |
| **`generation_success_rate`** | $\frac{N_{\text{successful}}}{N_{\text{total\_attempted}}}$ | Total baseline attempts. |

---

## 7. Evaluating Stored Baseline Artifacts

Baseline outputs generated in Step 19 can be evaluated directly from disk without network calls:

```powershell
python -m app.evaluation.baseline_evaluator --results-file backend/app/evaluation/results/baseline_validation_v1.jsonl
```

The adapter:
1. Loads [`BaselineExampleResult`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/baseline_runner.py) records from the JSONL file.
2. Resolves each example from the evaluation dataset accessor.
3. Enforces holdout protections (raises [`HoldoutAccessRestrictedError`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/evaluation/dataset_accessor.py) if `final_holdout` partition is evaluated without `--allow-holdout`).
4. Executes deterministic checks and produces `[exp_id]_eval_details.jsonl` and `[exp_id]_eval_summary.json`.

---

## 8. What Deterministic Evaluation Can and Cannot Establish

### What It Can Establish:
- Structural adherence to the 6-section CRISPE framework.
- Presence of non-empty, non-trivial content in all sections.
- Explicit textual preservation of required technologies, algorithms, and libraries.
- Adherence to length thresholds and declared output formats.
- Absence of explicitly prohibited keywords.

### What It Cannot Establish (Boundaries for Step 21):
- **Semantic Understanding:** A prompt may mention `"HMAC SHA-256"` but describe an insecure or nonsensical implementation.
- **Instruction Quality & Tone:** Whether the prompt instructions are clear, coherent, and actionable for downstream LLMs.
- **Subtle Constraint Violations:** Logical contradictions or nuanced constraints that cannot be expressed as substring matches or regular expressions.
- **Downstream Output Correctness:** It evaluates the generated prompt itself, not downstream code execution.

Step 21 introduces the **Model-Assisted Prompt-Quality Evaluation Engine** (LLM-as-a-judge with structured rubrics) to assess these semantic qualities.
