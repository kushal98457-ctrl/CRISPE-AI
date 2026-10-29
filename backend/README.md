# CRISPE-AI Backend Service

FastAPI backend service supporting CRISPE Prompt Forge, Semantic Prompt Memory, and Self-Evolving Engine.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python ≥ 3.12 (Host verified: `Python 3.14.5`)
- Virtual environment at `backend/.venv`
- PostgreSQL 16 with pgvector (optional for offline testing, required for persistence)

### 2. Activate Virtual Environment

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Start Local PostgreSQL + pgvector (Docker)

```bash
docker compose up -d
```
*(Starts `pgvector/pgvector:pg16` on port `5432` with database `crispe_ai`)*

### 4. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment settings:
- `GEMINI_API_KEY`: Google AI Studio API key (alias: `GOOGLE_API_KEY`)
- `GEMINI_MODEL_ID`: Default generation model (defaults to `gemma-4-26b-a4b-it`)
- `GEMINI_FALLBACK_MODEL_ID`: Fallback generation model (defaults to `gemini-2.5-flash`)
- `DATABASE_URL`: `postgresql+psycopg://postgres:postgres@localhost:5432/crispe_ai`
- `CORS_ORIGINS`: Allowed frontend origins (defaults to `http://localhost:5173`)

### 5. Run Development Server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Liveness Probe:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or `/api/v1/health`
- **Database Readiness Probe:** [http://127.0.0.1:8000/health/ready](http://127.0.0.1:8000/health/ready)
- **CRISPE Prompt Generation:** `POST /api/v1/prompts/generate`

---

## 🗄️ Database & Migrations

### Architecture
- **ORM:** SQLAlchemy 2.x declarative models via [`app.db.base.Base`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/db/base.py).
- **Driver:** `psycopg` (v3.3.6) asynchronous driver (`postgresql+psycopg://`).
- **Session Lifecycle:** Asynchronous dependency injection via [`get_db_session()`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/db/session.py) with automatic rollback on error.
- **Engine Lifecycle:** Asynchronous connection pool disposal on FastAPI lifespan shutdown.
- **Migrations:** Alembic configured in asynchronous mode (`backend/alembic/`).

### Common Alembic Commands
```bash
# Check current migration revision
alembic current

# Generate a new migration revision
alembic revision --autogenerate -m "create_prompt_memory_table"

# Apply pending migrations
alembic upgrade head

# Rollback one migration revision
alembic downgrade -1
```

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

---

## 🧪 Testing

### Running Unit, Route, and Database Tests (Offline / Zero External Dependencies)
All unit and route tests use mock adapters, dependency overrides, and in-memory async SQLite. No Google API key or PostgreSQL service is required:

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
├── alembic/                        # Database migration scripts (Step 9)
│   ├── env.py                      # Asynchronous migration runner connected to Base.metadata
│   ├── script.py.mako              # Migration file template
│   └── versions/                   # Migration revision history
├── alembic.ini                     # Alembic configuration
├── app/
│   ├── main.py                     # Application factory, lifespan, CORS, and error handlers
│   ├── adapters/                   # AI model provider adapters
│   │   ├── __init__.py             # Adapter exports
│   │   └── gemini_adapter.py       # Google GenAI SDK adapter with retry & error translation
│   ├── core/
│   │   ├── config.py               # Pydantic Settings (.env management)
│   │   ├── exceptions.py           # Domain exceptions (DatabaseError, ProviderError, etc.)
│   │   └── logging.py              # Structured logging and secret redaction filter
│   ├── db/                         # Database infrastructure (Step 9)
│   │   ├── __init__.py             # DB module exports
│   │   ├── base.py                 # SQLAlchemy 2.0 DeclarativeBase and TimestampMixin
│   │   └── session.py              # AsyncEngine, sessionmaker, and get_db_session
│   ├── middleware/
│   │   └── request_context.py      # Request ID and latency tracking
│   ├── schemas/
│   │   ├── crispe.py               # CrispeSections, CrispeGenerationRequest/Response
│   │   ├── error.py                # Standardized ErrorResponse model
│   │   └── health.py               # HealthResponse and ReadinessResponse models
│   ├── api/
│   │   ├── router.py               # Root /api/v1 router aggregator
│   │   └── routes/
│   │       ├── health.py           # /health and /health/ready probe implementations
│   │       └── prompts.py          # /prompts/generate endpoint implementation
│   └── services/
│       ├── __init__.py             # Service exports
│       └── generation_service.py   # CRISPE prompt generation service
└── tests/
    ├── conftest.py                 # Pytest async client fixtures
    ├── test_configuration.py       # Settings validation tests
    ├── test_database.py            # Database engine, session, & readiness tests (Step 9)
    ├── test_environment.py         # Runtime dependency smoke tests
    ├── test_errors_and_cors.py     # CORS & error handler tests
    ├── test_generation_service.py  # CRISPE generation service tests
    ├── test_health.py              # Health check tests
    ├── test_live_smoke.py          # Opt-in live smoke test
    ├── test_model_adapter.py       # Gemini model adapter tests
    └── test_prompts_route.py       # API route tests for /prompts/generate
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

---

## 🛡️ AI Resource Budgeting & Provider-Call Accounting (Step 25)

The centralized [`ResourceBudgetManager`](file:///C:/Users/kushl/.gemini/antigravity/scratch/CRISPE-AI/backend/app/services/resource_budget.py) provides multi-tiered execution safety and auditability across all AI model generation, evaluation, reflection, and optimization operations:

- **Pre-Dispatch Capacity Reservation:** Every provider call must reserve capacity before dispatch. If the request budget or wall-clock deadline is exhausted, the call fails closed immediately.
- **Metric vs. Provider Separation:** Distinguishes GEPA feedback metric invocations from physical external API requests. A single metric invocation can trigger student generation, LLM-as-a-judge evaluation, and reflection proposer requests.
- **Attempt & Retry Accounting:** Adapter retries on transient errors consume request capacity (`is_retry=True`) and cannot bypass the experiment request ceiling.
- **Execution Limits:**
  - Wall-clock deadlines enforced via monotonic clock (`OptimizationTimeoutError`).
  - Request concurrency strictly bounded via in-process semaphore.
  - Concurrency capacity is unconditionally released in `finally` blocks upon completion, timeout, or error.
- **Token & Cost Accounting:**
  - Provider-reported token counts (`prompt_tokens`, `candidates_tokens`, `total_tokens`) are recorded cleanly.
  - Missing token metadata is explicitly flagged (`has_missing_usage=True`) without coercing `None` to zero.
  - Estimated expenditure in USD is calculated using verified pricing tables (`gemini-2025-standard-v1`).
- **Ambient Context Propagation:** The active budget manager binds to async task contexts via `active_budget_manager` (`ContextVar`), intercepting calls down the stack without parameter leakage.

### Run Resource Budgeting Tests:
```powershell
pytest tests/test_resource_budget.py -v
```
*(27 passed in ~2.1s)*

---

## ⚙️ Durable Optimization Jobs & Background Workers (Step 26)

CRISPE-AI's Self-Evolving Prompt Engine executes independently of HTTP request lifecycles using a resilient, PostgreSQL-backed durable queue:

- **Transactional Queue (`optimization_jobs`):** Authoritative state storage in PostgreSQL. Jobs are claimed in short transactions, committed immediately, and executed outside database transaction boundaries.
- **Atomic Worker Claiming & Fencing:** Workers atomically claim jobs using row-level locking (`FOR UPDATE SKIP LOCKED`). Each claim issues an ephemeral fencing token (`claim_token`); any stale worker whose lease expired is fenced out from corrupting subsequent attempts.
- **Heartbeat & Lease Refresh:** Active workers refresh leases at regular intervals (`heartbeat_interval`). If a worker crashes, its lease expires, allowing recovery mechanisms to detect and safely re-queue the job.
- **Persistent Resource Accounting Across Restarts:** Persisted counters (`used_metric_calls`, `used_provider_calls`, `used_total_tokens`, `estimated_cost_usd`) survive worker restarts. Recovered attempts resume with `remaining_budget = max_budget - previously_used`, preventing quota resets.
- **Cooperative Cancellation:** `POST /api/v1/optimization/jobs/{id}/cancel` cancels queued jobs immediately, or signals running workers to halt cooperatively at the next iteration boundary while preserving completed artifacts.
- **Tenant Isolation:** All job operations strictly enforce principal ownership. Unauthorized callers receive HTTP 404 to avoid leaking job existence.

### Run Background Optimization Worker:
```powershell
# Continuous polling loop
python -m app.workers.optimization_worker --poll-interval 2.0 --lease-duration 60.0

# Process single job and exit
python -m app.workers.optimization_worker --run-once
```

### Run Optimization Job System Tests:
```powershell
pytest tests/test_optimization_job.py -v
```
*(16 passed in ~16s)*

---

## 🧬 Persistent Candidate Versioning & Optimization Results (Step 27)

CRISPE-AI provides durable storage for prompt candidate lineage, evaluation records, and artifacts produced by the Self-Evolving Prompt Engine:

- **Immutable Prompt Candidates (`prompt_candidates`):** Preserves exact instruction text snapshots, parent-child mutation lineage (`parent_candidate_id`), target model checkpoint, and cryptographic SHA-256 digests (`content_hash`) preventing duplicate insertions across retries.
- **Separate Evaluation Persistence (`candidate_evaluations`):** Evaluates candidates independently across authorized dataset splits (`train`, `val`). Persists both overall aggregate metrics and detailed per-example scoring with pass/fail indicators and sanitized judge feedback.
- **Strict Final Holdout Protection:** Database `CHECK` constraint (`dataset_split NOT IN ('test', 'holdout', 'final_holdout')`) and service-layer validation strictly prohibit evaluating or persisting candidate results against the sealed benchmark partition.
- **Safe Artifact Storage (`ArtifactStorageService`):** Atomically stores serialized program states, validation comparisons, and instruction diffs under an isolated root directory with path-traversal blocking and uniform relative URNs (`artifact://...`).
- **Tenant Isolation:** Access to candidate snapshots, evaluations, and job result summaries is strictly scoped to the job's owning security principal. Cross-tenant queries return 404.
- **Read-Only API Endpoints:**
  - `GET /api/v1/optimization/jobs/{job_id}/candidates` (Paginated candidate list)
  - `GET /api/v1/optimization/candidates/{candidate_id}` (Candidate detail & instruction snapshot)
  - `GET /api/v1/optimization/candidates/{candidate_id}/evaluations` (Evaluation records & metrics)
  - `GET /api/v1/optimization/jobs/{job_id}/results` (Optimization summary & candidate comparison)

### Run Candidate Persistence Tests:
```powershell
pytest tests/test_candidate_persistence.py -v
```
*(12 passed in ~16s)*
