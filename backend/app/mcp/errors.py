"""Structured errors and security sanitization for CRISPE-AI MCP Server (Step 29).

Enforces:
1. Distinct error categories (validation, resource not found, internal).
2. Strict redaction of secrets, database connection strings, API keys, and local file paths.
3. Safe error reporting preventing stack traces from leaking to external MCP clients.
"""

import re
from typing import Any, Dict, Optional


# Regex patterns for redacting sensitive values
_CONNECTION_STRING_PATTERN = re.compile(
    r"[a-zA-Z0-9+_-]+://[^:\s]+:[^@\s]+@[^\s/]+(?:\S*)",
    re.IGNORECASE,
)
_API_KEY_PATTERN = re.compile(
    r"\b(?:AIza[0-9A-Za-z-_]{35}|sk-[a-zA-Z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._~+/-]{15,})\b",
    re.IGNORECASE,
)
_PASSWORD_PARAM_PATTERN = re.compile(
    r"(password|secret|token|api_key|credential)=([^\s&,;]+)",
    re.IGNORECASE,
)
_WINDOWS_PATH_PATTERN = re.compile(
    r"[a-zA-Z]:\\(?:Users|home|Program Files|var|tmp|private)[^\s\"',;>]*",
    re.IGNORECASE,
)
_UNIX_PATH_PATTERN = re.compile(
    r"/(?:Users|home|root|var|etc|tmp)/[^\s\"',;>]*",
    re.IGNORECASE,
)


def sanitize_error_message(message: str) -> str:
    """Scrub sensitive credentials, connection strings, API keys, and local filesystem paths.

    Ensures that unexpected exceptions do not expose infrastructure details to MCP clients.
    """
    if not message:
        return "An internal error occurred."

    sanitized = str(message)
    # Redact connection strings (e.g. postgresql://user:pass@host:5432/db)
    sanitized = _CONNECTION_STRING_PATTERN.sub("<REDACTED_DB_URI>", sanitized)
    # Redact API keys and authorization tokens
    sanitized = _API_KEY_PATTERN.sub("<REDACTED_CREDENTIAL>", sanitized)
    # Redact query parameters with passwords
    sanitized = _PASSWORD_PARAM_PATTERN.sub(r"\1=<REDACTED>", sanitized)
    # Redact local filesystem paths
    sanitized = _WINDOWS_PATH_PATTERN.sub("<REDACTED_PATH>", sanitized)
    sanitized = _UNIX_PATH_PATTERN.sub("<REDACTED_PATH>", sanitized)

    return sanitized


class MCPError(Exception):
    """Base exception for all CRISPE-AI MCP integration errors."""

    def __init__(
        self,
        message: str,
        code: str = "MCP_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = sanitize_error_message(message)
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to a structured dictionary safe for MCP responses."""
        return {
            "error": True,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class MCPInvalidInputError(MCPError):
    """Raised when client input fails schema validation or semantic constraints."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            code="INVALID_INPUT",
            details=details,
        )


class MCPResourceNotFoundError(MCPError):
    """Raised when an identified resource (prompt, candidate, job) is not found."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            code="RESOURCE_NOT_FOUND",
            details=details,
        )


class MCPInternalError(MCPError):
    """Raised for unexpected internal failures, masking sensitive stack information."""

    def __init__(
        self,
        message: str = "An unexpected internal server error occurred while processing the MCP request.",
        original_error: Optional[Exception] = None,
    ) -> None:
        safe_msg = sanitize_error_message(str(original_error)) if original_error else message
        super().__init__(
            message=safe_msg,
            code="INTERNAL_ERROR",
            details={"type": type(original_error).__name__} if original_error else {},
        )
        self.original_error = original_error
