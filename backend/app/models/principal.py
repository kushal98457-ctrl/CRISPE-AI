"""SQLAlchemy ORM model for security principals and ownership anchors."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional
import uuid

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.security import AuthorizedPrincipal
from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.prompt_record import PromptRecord


class Principal(Base, TimestampMixin):
    """Represents an authenticated user account, service principal, or ownership anchor.

    This entity provides referential integrity and data isolation for all user-owned
    resources (such as persistent CRISPE prompts, semantic memory embeddings, and
    optimization histories). It does NOT store plaintext passwords or manage public
    login forms; instead, it anchors external upstream identities (e.g. Google OAuth,
    JWT subjects) to an immutable database identifier.
    """

    __tablename__ = "principals"

    id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        default=lambda: f"principal_{uuid.uuid4().hex}",
        doc="Stable primary key identifier for the security principal.",
    )
    email: Mapped[Optional[str]] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        doc="Optional unique verified email address associated with the principal.",
    )
    display_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="Optional human-readable display name or organization label.",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether the principal is active. Deactivated principals cannot access resources.",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Platform administrator flag (never silently bypasses tenant isolation).",
    )

    # 1-to-many relationship with owned prompt records
    prompts: Mapped[List["PromptRecord"]] = relationship(
        "PromptRecord",
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Principal, generating defaults for in-memory instances."""
        if "id" not in kwargs or kwargs["id"] is None:
            kwargs["id"] = f"principal_{uuid.uuid4().hex}"
        if "is_active" not in kwargs or kwargs["is_active"] is None:
            kwargs["is_active"] = True
        if "is_superuser" not in kwargs or kwargs["is_superuser"] is None:
            kwargs["is_superuser"] = False
        super().__init__(**kwargs)

    def to_authorized_principal(self) -> AuthorizedPrincipal:
        """Convert the persistent database record into an immutable AuthorizedPrincipal context."""
        return AuthorizedPrincipal(
            id=self.id,
            is_active=self.is_active,
            is_superuser=self.is_superuser,
            email=self.email,
            display_name=self.display_name,
        )
