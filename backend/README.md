# CRISPE-AI Backend Service

FastAPI backend service supporting CRISPE Prompt Forge, Semantic Prompt Memory, and Self-Evolving Engine.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python ≥ 3.12 (Host verified: `Python 3.14.5`)
- Virtual environment at `backend/.venv`

### 2. Activate Virtual Environment

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment settings:
- `GEMINI_API_KEY`: Google AI Studio API key (alias: `GOOGLE_API_KEY`)
- `GEMINI_MODEL_ID`: Default generation model (defaults to `gemma-4-26b-a4b-it`)
- `GEMINI_FALLBACK_MODEL_ID`: Fallback generation model (defaults to `gemini-2.5-flash`)
- `DATABASE_URL`: PostgreSQL connection string with pgvector
- `CORS_ORIGINS`: Allowed frontend origins (defaults to `http://localhost:5173`)

### 4. Run Development Server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Liveness Probe:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or `/api/v1/health`
- **CRISPE Prompt Generation:** `POST /api/v1/prompts/generate`

---

## 📡 API Reference

### Generate Structured CRISPE Prompt
- **Method & Path:** `POST /api/v1/prompts/generate`
- **Content-Type:** `application/json`

#### Request Payload:
```json
{
  "idea": "Build a secure webhook handler with signature verification",
  "model_id": "gemma-4-26b-a4b-it",
  "temperature": 0.4,
  "thinking_budget": 1024
}
```

#### Successful Response (`200 OK`):
```json
{
  "idea": "Build a secure webhook handler with signature verification",
  "sections": {
    "context": "Cloud-native payment webhook ingestion endpoint receiving third-party notifications.",
    "role": "Principal Security Engineer and API Designer.",
    "instruction": "Design an asynchronous webhook receiver that verifies HMAC-SHA256 signatures, deduplicates events, and stores raw payloads.",
    "specification": "Provide TypeScript code and a Markdown specification. Reject unsigned requests with 401. Enforce replay attack window of 300 seconds.",
    "performance": "Verification must complete in < 2ms without blocking the Node.js event loop.",
    "example": "Input: Stripe webhook payload with X-Signature header. Output: Validated event dispatches to internal queue."
  },
  "formatted_prompt": "# C - Context\nCloud-native payment webhook...\n\n# R - Role\nPrincipal Security Engineer...\n\n# I - Instruction\nDesign an asynchronous...\n\n# S - Specification\nProvide TypeScript...\n\n# P - Performance\nVerification must...\n\n# E - Example\nInput: Stripe webhook...",
  "model_id": "gemma-4-26b-a4b-it",
  "usage": {
    "prompt_tokens": 85,
    "candidates_tokens": 240,
    "total_tokens": 325,
    "thoughts_tokens": null
  },
  "finish_reason": "STOP",
  "request_id": "req-3b48f98a",
  "schema_version": "1.0",
  "created_at": "2026-10-28T13:15:00Z"
}
```

#### Error Response Format (`4xx` / `5xx`):
```json
{
  "status_code": 422,
  "error_code": "VALIDATION_ERROR",
  "message": "Request input validation failed.",
  "request_id": "req-3b48f98a",
  "details": [
    {
      "loc": ["body", "idea"],
      "msg": "Prompt idea cannot be empty or whitespace.",
      "type": "value_error"
    }
  ]
}
```

---

## 🧪 Testing

### Running Unit & Route Tests (Offline / Zero Credentials)
All regular tests use mock adapters and test fakes. No Google API key is required:

```bash
pytest backend/tests -v
```

### Running the Optional Live Smoke Test
An opt-in smoke test verifies real end-to-end communication against Google AI Studio:

```bash
# Set your local key and enable live tests
$env:GEMINI_API_KEY="AIzaSyYourActualKey"
$env:RUN_LIVE_TESTS="1"
pytest backend/tests/test_live_smoke.py -v
```
*(When `RUN_LIVE_TESTS` is unset, the test is automatically skipped.)*

---

## 🛠️ Architecture & Modules

```
backend/
├── app/
│   ├── main.py                     # Application factory, lifespan, CORS, and error handlers
│   ├── adapters/                   # AI model provider adapters
│   │   ├── __init__.py             # Adapter exports
│   │   └── gemini_adapter.py       # Google GenAI SDK adapter with retry & error translation
│   ├── core/
│   │   ├── config.py               # Pydantic Settings (.env management)
│   │   ├── exceptions.py           # Domain exceptions (ProviderError, PromptGenerationError)
│   │   └── logging.py              # Structured logging and secret redaction filter
│   ├── middleware/
│   │   └── request_context.py      # Request ID and latency tracking
│   ├── schemas/
│   │   ├── crispe.py               # CrispeSections, CrispeGenerationRequest/Response
│   │   ├── error.py                # Standardized ErrorResponse model
│   │   └── health.py               # HealthResponse model
│   ├── api/
│   │   ├── router.py               # Root /api/v1 router aggregator
│   │   └── routes/
│   │       ├── health.py           # /health endpoint implementation
│   │       └── prompts.py          # /prompts/generate endpoint implementation (Step 8)
│   └── services/
│       ├── __init__.py             # Service exports
│       └── generation_service.py   # CRISPE prompt generation service
└── tests/
    ├── conftest.py                 # Pytest async client fixtures
    ├── test_configuration.py       # Settings validation tests
    ├── test_environment.py         # Runtime dependency smoke tests
    ├── test_errors_and_cors.py     # CORS & error handler tests
    ├── test_generation_service.py  # CRISPE generation service tests
    ├── test_health.py              # Health check tests
    ├── test_live_smoke.py          # Opt-in live smoke test
    ├── test_model_adapter.py       # Gemini model adapter tests
    └── test_prompts_route.py       # API route tests for /prompts/generate (Step 8)
```

---

## 🤖 Model Integration & Reliability Strategy

- **SDK:** Official `google-genai` Python SDK (v2.33.0).
- **Asynchronous Execution:** Generation is dispatched using `client.aio.models.generate_content(...)`.
- **Structured Output:** Enforced via `response_schema=CrispeSections` and `response_mime_type="application/json"` with automatic JSON code block fallback cleanup.
- **CRISPE Headings:** Renders standard Markdown matching the existing UI contract:
  - `# C - Context`
  - `# R - Role`
  - `# I - Instruction`
  - `# S - Specification`
  - `# P - Performance`
  - `# E - Example`
- **Error Mapping & Resilience:**
  - `401 / API_KEY_INVALID` ➔ `ProviderAuthenticationError` (Non-retryable)
  - `403 / PERMISSION_DENIED` ➔ `ProviderPermissionError` (Non-retryable)
  - `404 / NOT_FOUND` ➔ `ModelNotFoundError` (Non-retryable)
  - `429 / RESOURCE_EXHAUSTED` ➔ `ProviderRateLimitError` (Retryable with exponential backoff & jitter)
  - `408 / 504 / Timeout` ➔ `ProviderTimeoutError` (Retryable)
  - `500 / 502 / 503` ➔ `ProviderUnavailableError` (Retryable)
