# DSPy CRISPE Prompt Generation & Evaluation Integration

## Overview

The `app.dspy_modules` package introduces a programmable, typed DSPy program for generating canonical CRISPE prompts. It establishes the foundation for subsequent **Self-Evolving Prompt Engine** capabilities (DSPy teleprompting and GEPA reflective optimization in Steps 25–28) while preserving complete functional parity with the original production generation service.

---

## 1. Module Structure & Architecture

| Module | Responsibility |
| :--- | :--- |
| [`signature.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/signature.py) | Defines [`CrispeGenerationSignature`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/signature.py#L9-L60) with 4 structured inputs and the 6 canonical CRISPE output fields. |
| [`program.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/program.py) | Encapsulates [`CrispeGenerationProgram`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/program.py#L24-L125) (`dspy.Module`), predictor execution, and canonical [`CrispeSections`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/schemas/crispe.py#L8-L83) validation. |
| [`model_client.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/model_client.py) | Configures the DSPy Language Model (`dspy.LM`), model identifier normalization, and concurrency isolation via `dspy.context(lm=...)`. |
| [`service.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/service.py) | Provides [`DspyCrispeGenerationService`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/service.py#L35-L135), providing functional parity with [`CrispeGenerationService`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/generation_service.py#L51-L157) without altering existing endpoints. |
| [`evaluation_adapter.py`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/evaluation_adapter.py) | Connects DSPy predictions to Steps 20 & 21 evaluators while strictly separating generator inputs from hidden evaluation criteria. |

---

## 2. DSPy Signature (`CrispeGenerationSignature`)

```python
class CrispeGenerationSignature(dspy.Signature):
    # Inputs (Permissible user-provided inputs)
    idea: str = dspy.InputField(desc="The core task, raw prompt idea, or initial instruction.")
    task_context: str = dspy.InputField(default="", desc="Operational domain background and environment.")
    constraints: str = dspy.InputField(default="", desc="Explicit technical boundaries and negative rules.")
    output_format_preference: str = dspy.InputField(default="", desc="Structural format or schema desired.")

    # Outputs (Canonical 6-section CRISPE framework)
    context: str = dspy.OutputField(desc="C - Context: Operational background and business setting.")
    role: str = dspy.OutputField(desc="R - Role: Persona, authority, perspective, and identity.")
    instruction: str = dspy.OutputField(desc="I - Instruction: Step-by-step actions and methodologies.")
    specification: str = dspy.OutputField(desc="S - Specification: Constraints, formatting rules, negative boundaries.")
    performance: str = dspy.OutputField(desc="P - Performance: Evaluation criteria and success metrics.")
    example: str = dspy.OutputField(desc="E - Example: Concrete illustrative demonstration pattern.")
```

### Security & Boundary Protection:
Hidden evaluator fields (`acceptance_criteria`, `critical_requirements`, `prohibited_behaviors`, `grading_notes`, and holdout answers) are **never** passed to the Signature inputs.

---

## 3. Concurrency and Context Isolation Strategy

In a production FastAPI backend serving concurrent requests, process-wide mutation via `dspy.configure()` causes race conditions and cross-request setting leakage.

To solve this:
1. **Request-Scoped Contexts:** Every generation and evaluation request executes inside an isolated context manager:
   ```python
   with dspy.context(lm=lm):
       sections, pred = await program.execute_async(idea=request.idea)
   ```
2. **Contextvars Propagation:** Python's `contextvars` guarantee that asynchronous tasks and worker threads (`asyncio.to_thread`) inherit their own isolated LM reference without mutating global process state.
3. **Verified in Tests:** Tested under high concurrency with simultaneous workers using different mock models (`test_concurrent_request_isolation`).

---

## 4. Model Adapter & Configuration Integration

- **Installed Version:** DSPy `3.4.1` with LiteLLM integration.
- **Model Normalization:** Model identifiers from `app.core.config.Settings` (e.g. `PRIMARY_MODEL = "gemma-4-26b-a4b-it"` or `"gemini-2.5-flash"`) are automatically mapped to LiteLLM's `gemini/<model_name>` routing syntax.
- **Freshness & Cache:** `cache=False` by default to ensure evaluation reproducibility.
- **Secure Credentials:** API keys are resolved via `settings.get_gemini_api_key()` and never printed or exposed in logs.

---

## 5. Output Validation Against Canonical CRISPE Schema

After execution, the raw `dspy.Prediction` is validated by [`validate_and_extract_sections`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/dspy_modules/program.py#L52-L95):
1. Verifies that all 6 sections exist.
2. Asserts each section is a string and non-empty (rejects whitespace-only outputs).
3. Constructs and validates canonical [`CrispeSections`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/schemas/crispe.py#L8-L83).
4. Raises [`PromptValidationError`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/core/exceptions.py) if any section is missing or malformed, preventing silent error masking.

---

## 6. Running Tests

### Automated Unit and Integration Tests (Zero Live Credits Consumed):
```powershell
pytest tests/test_dspy_program.py -v
```

### Opt-in Live Model Smoke Test:
```powershell
$env:RUN_LIVE_TESTS="1"
$env:GEMINI_API_KEY="your-api-key"
pytest tests/test_live_smoke.py -k "test_live_dspy_generation_smoke" -v
```

---

## 7. Status and Next Steps for Optimization

- **Production Generation Parity:** The existing `CrispeGenerationService` and FastAPI routes remain completely intact and active.
- **Programmable Ready:** The DSPy module is now typed, isolated, and callable as an independent generation program.
- **Next Steps:** Step 23 (Optimization Feedback Schemas & Mutation Guidance) and Step 25+ (DSPy Teleprompting and GEPA Optimization).
