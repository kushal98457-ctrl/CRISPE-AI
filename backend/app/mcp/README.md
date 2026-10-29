# CRISPE-AI Model Context Protocol (MCP) Server (Steps 29 & 30)

## 🌐 Overview

The `app.mcp` package provides the official Model Context Protocol (MCP) server integration for **Feature B (CRISPE MCP Server)**. It exposes CRISPE prompt synthesis, semantic prompt memory search, and persistent prompt indexing directly to compatible AI assistant clients (Claude Desktop, IDE agent extensions, autonomous CLI runners).

The implementation builds on the official Anthropic Model Context Protocol Python SDK (`mcp` v2.3.0) with clean separation of transport, capabilities, schemas, and security controls.

---

## 📦 SDK Selection & Compatibility

- **Package:** `mcp` (Official Anthropic Model Context Protocol Python SDK)
- **Version Range:** `mcp>=2.0.0,<3.0.0` (Currently verified: `v2.3.0`)
- **Compatibility:** Python ≥ 3.12 (Host verified on `Python 3.14.5` under Windows 11)
- **Architecture:** Uses the official `MCPServer` high-level server class and `mcp.client.Client` testing client introduced in MCP v2.

---

## 🏗️ Module Architecture

```
backend/app/mcp/
├── __init__.py           # Public exports (create_mcp_server, MCPServerConfig, schemas, etc.)
├── config.py             # Server configuration contract (MCPServerConfig)
├── errors.py             # Domain error taxonomy (ToolError) and credential/path sanitization
├── schemas.py            # Protocol request/response DTOs for generation, memory, and smoke tools
├── server.py             # Server factory, lifecycle, tool registration, and stdio entrypoint
├── services/
│   ├── __init__.py
│   └── context.py        # Dependency injection container (MCPAppContext)
└── tools/
    ├── __init__.py
    ├── generation.py     # generate_crispe_prompt capability
    ├── memory.py         # search_prompt_memory and save_prompt_memory capabilities
    └── smoke.py          # Minimal smoke-test capabilities (ping, server_info)
```

---

## 🛠️ Registered MCP Tools

### 1. `generate_crispe_prompt` (Step 30)
- **Description:** Transforms a raw task description or prompt idea into a production-grade, structured prompt strictly conforming to the CRISPE framework (Context, Role, Instruction, Specification, Performance, Example).
- **Parameters:**
  - `idea` (string, optional): Primary task description or core prompt concept.
  - `context_section`, `role_section`, `instruction_section`, `specification_section`, `performance_section`, `example_section` (strings, optional): Explicit section hints or constraints preserved into the prompt.
  - `model_id` (string, optional): Model override validated against security whitelist (`gemma-4-26b-a4b-it`, `gemini-2.5-flash`, `gemini-1.5-flash`, `mock-student-gemini-flash`).
  - `temperature` (float, optional, 0.0-2.0): Sampling temperature override.
  - `thinking_budget` (int, optional, 0-32768): Reasoning token budget.
- **Output:** Structured CRISPE response containing all 6 canonical sections, formatted Markdown, model ID, and token usage metrics.

### 2. `search_prompt_memory` (Step 30)
- **Description:** Searches the user's persistent CRISPE prompt memory using high-dimensional vector similarity (768-dim embeddings) and multi-signal ranking.
- **Security Boundary:** Requires a verified `trusted_principal` in `MCPAppContext`. Fails closed (`UNAUTHORIZED`) if unauthenticated or principal is suspended. Client cannot spoof arbitrary tenant or owner IDs.
- **Parameters:**
  - `query` (string, required): Natural language search query.
  - `top_k` (int, default 5, 1-10): Maximum nearest neighbor prompts to retrieve.
  - `task_type` (string, optional): Category or domain tag filter.
  - `min_similarity` (float, optional, 0.0-1.0): Minimum cosine similarity cutoff.
- **Output:** List of `PromptMemoryMatchItem` objects with similarity scores, formatted Markdown, original ideas, and creation timestamps.

### 3. `save_prompt_memory` (Step 30)
- **Description:** Explicitly persists a validated CRISPE prompt to the user's private semantic memory and automatically indexes its 768-dimensional vector embedding.
- **Security Boundary:** Requires an active `trusted_principal`. Prompts are strictly owned by the caller's principal.
- **Idempotency:** Re-saving an identical prompt for the same principal returns `status="already_exists"` and the existing prompt record ID.
- **Parameters:**
  - `idea` (string, required): Original prompt idea.
  - `context`, `role`, `instruction`, `specification`, `performance`, `example` (strings, required): The 6 validated CRISPE sections.
  - `task_type` (string, optional): Category tag.
  - `model_id` (string, optional): Generator model identifier.
- **Output:** `SavePromptMemoryOutput` with `prompt_id`, `status` (`created` or `already_exists`), `owner_id`, and timestamp.

### 4. `ping` & `server_info` (Step 29)
- Harmless diagnostic capabilities confirming protocol readiness, version metadata, and supported tool features without side effects.

---

## 🔒 Security Architecture

1. **Owner Scoping & Isolation:** Memory operations derive ownership exclusively from `app_context.get_trusted_principal()`. Callers cannot supply or override tenant IDs.
2. **Model Whitelisting:** Both generation and memory tools enforce a strict whitelist (`ALLOWED_MODELS`), preventing model injection attacks.
3. **Data Sanitization:** All error messages are passed through `sanitize_error_message()`, which scrubs database URIs, Google AI API keys (`AIzaSy...`), bearer tokens, and local filesystem paths before returning responses.
4. **ToolError Conformance:** `MCPError` subclasses `mcp.server.mcpserver.exceptions.ToolError`, ensuring protocol-level `is_error=True` reporting without unhandled server crashes.
5. **Stdio Stream Isolation:** All logs are strictly emitted to `sys.stderr`. Writing to `sys.stdout` is completely avoided to prevent corrupting JSON-RPC protocol messages.

---

## 🧪 Automated Testing

Run the full MCP test suite using `pytest`:

```powershell
pytest tests/mcp/ -v
```

Test coverage includes:
- `tests/mcp/test_server_foundation.py` (13 tests): Factory configuration, discovery, smoke tools, logging isolation, error redaction, FastAPI coexistence.
- `tests/mcp/test_generation_tool.py` (7 tests): Discovery, CRISPE synthesis, section hint synthesis, model whitelisting, empty input rejection, provider error sanitization, feature flags.
- `tests/mcp/test_memory_tools.py` (11 tests): Discovery, unauthorized failsafe, suspended principal rejection, vector retrieval, multi-signal ranking, idempotent saving, model whitelisting.

---

## 🧭 Capabilities Roadmap

| Step | Scope | Status | Description |
| :--- | :--- | :--- | :--- |
| **Step 29** | **MCP Foundation** | **Completed** | Server factory, stdio entrypoint, smoke tools, error redaction, tests. |
| **Step 30** | **Generation & Memory Tools** | **Completed** | `generate_crispe_prompt`, `search_prompt_memory`, `save_prompt_memory`. |
| **Step 31** | **Optimization Tools** | Next | `optimization_submit`, `optimization_status`, `candidate_inspect`. |
| **Step 32** | **Transport Hardening & Auth** | Planned | Streamable HTTP transport, token authentication, production CORS. |
