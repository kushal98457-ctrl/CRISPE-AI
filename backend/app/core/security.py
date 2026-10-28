"""Security principal contracts, context models, and authorization boundaries."""

from dataclasses import dataclass
from typing import Optional

from app.core.exceptions import (
    AuthenticationRequiredError,
    InactivePrincipalError,
)
from app.core.logging import get_logger

logger = get_logger("security")


@dataclass(frozen=True)
class AuthorizedPrincipal:
    """Represents an authenticated, backend-verified security principal.

    This context object must only be constructed by trusted backend authentication
    providers (such as validated OAuth2 JWTs, verified session tokens, or internal
    system workers). It is NEVER instantiated directly from untrusted client headers
    or arbitrary HTTP request parameters.
    """

    id: str
    is_active: bool = True
    is_superuser: bool = False
    email: Optional[str] = None
    display_name: Optional[str] = None

    def ensure_active(self) -> None:
        """Enforce that the security principal is active and permitted to perform actions.

        Raises:
            InactivePrincipalError: If the principal has been deactivated or suspended.
        """
        if not self.is_active:
            logger.warning("Access rejected for deactivated principal: %s", self.id)
            raise InactivePrincipalError(
                f"Security principal '{self.id}' is deactivated and cannot perform operations."
            )


async def get_current_principal() -> AuthorizedPrincipal:
    """FastAPI dependency stub resolving the active security principal.

    In the current architecture phase, no trusted authentication provider
    (OAuth2, JWT bearer, or session cookie) has been integrated yet.
    To prevent spoofing or direct-object-reference vulnerabilities:
    - This dependency rejects public requests with HTTP 401 AuthenticationRequiredError.
    - It intentionally refuses to trust client-supplied headers (e.g., X-User-Id).
    - Future authentication steps will implement cryptographic token validation
      and map the subject claim to an AuthorizedPrincipal.
    - Automated tests can safely override this dependency via FastAPI's
      `app.dependency_overrides[get_current_principal]`.
    """
    raise AuthenticationRequiredError(
        "Authentication is required. No trusted authentication provider is currently configured."
    )
