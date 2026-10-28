"""Domain exceptions for CRISPE-AI backend services and model adapters."""

from typing import Optional


class CRISPEError(Exception):
    """Base exception for all domain errors in CRISPE-AI."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(CRISPEError):
    """Raised when configuration or credentials are missing or invalid."""
    pass


class MissingProviderCredentialError(ConfigurationError):
    """Raised when an operation requires an API credential that is not configured."""
    pass


class DatabaseError(CRISPEError):
    """Base exception for database infrastructure failures."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "DATABASE_ERROR",
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.original_error = original_error


class DatabaseConfigError(DatabaseError):
    """Raised when database connection URL or configuration is missing or invalid."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_CONFIG_ERROR",
            original_error=original_error,
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when connection to PostgreSQL fails."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_code="DATABASE_CONNECTION_ERROR",
            original_error=original_error,
        )


class DatabaseUnavailableError(DatabaseError):
    """Raised when database is unreachable or offline."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_code="DATABASE_UNAVAILABLE",
            original_error=original_error,
        )


class ProviderError(CRISPEError):
    """Base exception for AI model provider failures."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_code: str = "PROVIDER_ERROR",
        is_retryable: bool = False,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.is_retryable = is_retryable
        self.original_error = original_error


class ProviderConfigError(ProviderError):
    """Raised when provider configuration or client initialization fails."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="PROVIDER_CONFIG_ERROR",
            is_retryable=False,
            original_error=original_error,
        )


class ProviderAuthenticationError(ProviderError):
    """Raised when authentication with the model provider fails (HTTP 401)."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=401,
            error_code="PROVIDER_AUTH_ERROR",
            is_retryable=False,
            original_error=original_error,
        )


class ProviderPermissionError(ProviderError):
    """Raised when access is forbidden by the provider (HTTP 403)."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=403,
            error_code="PROVIDER_PERMISSION_DENIED",
            is_retryable=False,
            original_error=original_error,
        )


class ModelNotFoundError(ProviderError):
    """Raised when the specified model identifier cannot be found (HTTP 404)."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=404,
            error_code="MODEL_NOT_FOUND",
            is_retryable=False,
            original_error=original_error,
        )


class ProviderRateLimitError(ProviderError):
    """Raised when rate limits or quotas are exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=429,
            error_code="PROVIDER_RATE_LIMIT",
            is_retryable=True,
            original_error=original_error,
        )
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Raised when a request to the provider times out (HTTP 408 / 504)."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=504,
            error_code="PROVIDER_TIMEOUT",
            is_retryable=True,
            original_error=original_error,
        )


class ProviderUnavailableError(ProviderError):
    """Raised when the provider service is unavailable or returns 5xx."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_code="PROVIDER_UNAVAILABLE",
            is_retryable=True,
            original_error=original_error,
        )


class InvalidProviderResponseError(ProviderError):
    """Raised when the provider output is empty, malformed, or cannot be parsed."""

    def __init__(self, message: str, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message=message,
            status_code=502,
            error_code="INVALID_PROVIDER_RESPONSE",
            is_retryable=False,
            original_error=original_error,
        )


class ProviderAPIError(ProviderError):
    """Raised for general upstream provider API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_code: str = "PROVIDER_API_ERROR",
        is_retryable: bool = False,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=error_code,
            is_retryable=is_retryable,
            original_error=original_error,
        )


class PromptGenerationError(CRISPEError):
    """Base exception for prompt generation service workflow errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 422,
        error_code: str = "PROMPT_GENERATION_ERROR",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class PromptValidationError(PromptGenerationError):
    """Raised when prompt generation input or structured output validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="PROMPT_VALIDATION_ERROR",
        )


class EmptyPromptError(PromptGenerationError):
    """Raised when an empty or whitespace-only prompt input is provided."""

    def __init__(self, message: str = "Prompt input text cannot be empty or whitespace.") -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="EMPTY_PROMPT_ERROR",
        )


class SecurityError(CRISPEError):
    """Base exception for authorization, identity, and data ownership violations."""

    def __init__(
        self,
        message: str,
        status_code: int = 403,
        error_code: str = "SECURITY_ERROR",
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.original_error = original_error


class AuthenticationRequiredError(SecurityError):
    """Raised when an operation requires an authenticated security principal (HTTP 401)."""

    def __init__(
        self,
        message: str = "Authentication is required to access this resource.",
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_REQUIRED",
            original_error=original_error,
        )


class OwnershipViolationError(SecurityError):
    """Raised when a principal attempts to access or modify a resource owned by another principal (HTTP 403)."""

    def __init__(
        self,
        message: str = "You do not have permission to access or modify this resource.",
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=403,
            error_code="OWNERSHIP_VIOLATION",
            original_error=original_error,
        )


class InactivePrincipalError(SecurityError):
    """Raised when a deactivated security principal attempts to perform operations (HTTP 403)."""

    def __init__(
        self,
        message: str = "The security principal is inactive and cannot perform operations.",
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=403,
            error_code="INACTIVE_PRINCIPAL",
            original_error=original_error,
        )


class OwnerImmutableError(SecurityError):
    """Raised when an operation attempts to reassign or change the owner of an existing record (HTTP 400)."""

    def __init__(
        self,
        message: str = "Record ownership cannot be transferred or reassigned.",
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_code="OWNER_IMMUTABLE",
            original_error=original_error,
        )


class PrincipalNotFoundError(SecurityError):
    """Raised when an expected principal cannot be found in the identity store (HTTP 404)."""

    def __init__(
        self,
        message: str = "Security principal was not found.",
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=404,
            error_code="PRINCIPAL_NOT_FOUND",
            original_error=original_error,
        )
