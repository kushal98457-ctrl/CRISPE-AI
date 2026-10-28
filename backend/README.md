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

Ensure `.env` contains:
- `GEMINI_API_KEY`: Google AI Studio API key
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

---

## 🧪 Testing

Run backend test suite:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

---

## 🛠️ Architecture & Modules

```
backend/
├── app/
│   ├── main.py                     # Application factory, lifespan, CORS, and error handlers
│   ├── core/
│   │   ├── config.py               # Pydantic Settings (.env management)
│   │   └── logging.py              # Structured logging and secret redaction filter
│   ├── middleware/
│   │   └── request_context.py      # Request ID and latency tracking
│   ├── schemas/
│   │   ├── health.py               # HealthResponse model
│   │   └── error.py                # Standardized ErrorResponse model
│   ├── api/
│   │   ├── router.py               # Root /api/v1 router aggregator
│   │   └── routes/
│   │       └── health.py           # /health endpoint implementation
│   └── services/                   # Placeholder for upcoming services (Step 6+)
└── tests/
    ├── conftest.py                 # Pytest async client fixtures
    ├── test_configuration.py       # Settings validation tests
    ├── test_environment.py         # Runtime dependency smoke tests
    ├── test_errors_and_cors.py     # CORS & error handler tests
    └── test_health.py              # Health check tests
```

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `CORS Error` in browser | Request from origin not in `CORS_ORIGINS` | Add frontend origin to `CORS_ORIGINS` in `backend/.env`. |
| `ModuleNotFoundError: app` | Running pytest from wrong working directory | Run pytest using `backend/.venv/Scripts/pytest.exe` pointing to `backend/tests`. |
| Port 8000 already in use | Another process running on port 8000 | Specify a different port: `--port 8001` or terminate conflicting process. |
