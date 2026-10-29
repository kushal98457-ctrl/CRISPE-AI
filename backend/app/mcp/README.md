# CRISPE-AI Model Context Protocol (MCP) Server Foundation (Step 29)

## 🌐 Overview

The `app.mcp` package establishes the official Model Context Protocol (MCP) server foundation for **Feature B (CRISPE MCP Server)**. It provides a standardized protocol interface enabling AI assistant clients (Claude Desktop, IDE agents, autonomous CLI tools) to access CRISPE-AI's prompt engineering, semantic prompt memory, and self-evolving prompt optimization engines.

The implementation builds on the official Anthropic Model Context Protocol Python SDK (`mcp` v2.3.0) with clean separation of transport, capabilities, schemas, and security controls.

---

## 📦 SDK Selection & Compatibility

- **Package:** `mcp` (Official Anthropic Model Context Protocol Python SDK)
- **Version Range:** `mcp>=2.0.0,<3.0.0` (Currently verified: `v2.3.0`)
- **Compatibility:** Python ≥ 3.12 (Host verified on `Python 3.14.5` under Windows 11)
- **Architecture:** Uses the official `MCPServer` high-level server class and `mcp.client.Client` testing client introduced in MCP v2.

### Why MCP v2?
In MCP v2, Anthropic consolidated server APIs, renaming the legacy standalone `FastMCP` class to `from mcp.server.mcpserver import MCPServer` directly in the core `mcp` distribution. This removes third-party dependencies, standardizes Pydantic serialization, and provides in-memory protocol client testing without subprocess spawning.

---

## 🏗️ Module Architecture

```
backend/app/mcp/
├── __init__.py           # Public exports (create_mcp_server, MCPServerConfig, etc.)
├── config.py             # Server configuration contract (MCPServerConfig)
├── errors.py             # Domain error taxonomy and credential/path sanitization
├── schemas.py            # Protocol response DTOs (PingResponse, ServerMetadataResponse)
├── server.py             # Server factory, lifecycle, and stdio entrypoint
├── services/
│   ├── __init__.py
│   └── context.py        # Dependency injection container (MCPAppContext)
└── tools/
    ├── __init__.py
    └── smoke.py          # Harmless smoke-test capabilities (ping, server_info)
```

### Architectural Separation
1. **Server Construction:** `create_mcp_server(config, context)` is a pure factory function. Importing or constructing the server does **not** open network ports, connect to PostgreSQL, execute Alembic migrations, or call the Gemini API.
2. **Service Access via Dependency Injection:** `MCPAppContext` defers resolution of domain services (`CrispeGenerationService`, `SemanticSearchService`, `OptimizationJobService`, `PromptReleaseService`) until a tool actively requests them, preventing import-time side effects.
3. **Strict Stdio Stream Isolation:** When running over standard I/O, `configure_stdio_logging()` routes all logging strictly to `sys.stderr`. Writing to `sys.stdout` is completely avoided to prevent corrupting MCP JSON-RPC protocol frames.
4. **Credential & Path Redaction:** `sanitize_error_message()` scrubs connection strings (`postgresql://...`), API keys (`AIzaSy...`, `sk-...`), query credentials, and absolute filesystem paths (`C:\Users\...`, `/home/...`) before any error is returned to an MCP client.

---

## 🚀 Installation & Running

### 1. Install MCP Dependencies
```powershell
# From backend directory with virtual environment activated:
pip install -e .[mcp]
```

### 2. Run Local Stdio Server
```powershell
# Run from backend directory:
python -m app.mcp.server
```

### 3. Connect via Client (e.g., Claude Desktop Configuration)
```json
{
  "mcpServers": {
    "crispe-ai": {
      "command": "C:\\Users\\kushl\\.gemini\\antigravity\\scratch\\CRISPE-AI\\backend\\.venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "app.mcp.server"
      ],
      "cwd": "C:\\Users\\kushl\\.gemini\\antigravity\\scratch\\CRISPE-AI\\backend"
    }
  }
}
```

---

## 🧪 Automated Testing

Run the dedicated MCP test suite using `pytest`:

```powershell
pytest tests/mcp/test_server_foundation.py -v
```

Tests verify:
- Factory default construction and custom configuration overrides.
- In-memory protocol initialization handshake using `mcp.client.Client`.
- Tool discovery and schema publication (`ping`, `server_info`).
- Successful smoke tool execution with structured JSON outputs.
- Error redaction for database URIs, API keys, and local file paths.
- Stdio logging redirection (stderr only, stdout remains pristine).
- Application isolation (existing FastAPI application and domain services remain unaffected).

---

## 🧭 Capabilities Roadmap

| Step | Scope | Status | Description |
| :--- | :--- | :--- | :--- |
| **Step 29** | **MCP Foundation** | **Completed** | Server factory, stdio entrypoint, smoke tools, error redaction, tests. |
| **Step 30** | **Generation & Memory Tools** | Deferred | `crispe_generate`, `memory_search`, `memory_save`. |
| **Step 31** | **Optimization Tools** | Deferred | `optimization_submit`, `optimization_status`, `candidate_inspect`. |
| **Step 32** | **Transport Hardening & Auth** | Deferred | Streamable HTTP transport, token authentication, production CORS. |

> [!NOTE]
> Remote exposure (Streamable HTTP, SSE, Bearer authentication) is intentionally deferred to Step 32 in accordance with the 46-step roadmap.
