"""SQLAlchemy ORM model for version-aware persistent prompt vector embeddings."""

from datetime import datetime, timezone
import hashlib
from typing import TYPE_CHECKING, Any, List, Optional
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    UUID,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.prompt_record import PromptRecord

# Pinned dimension established in Steps 12 & 13 for Google text-embedding-004
EMBEDDING_DIMENSION = 768


class PromptEmbedding(Base, TimestampMixin):
    """Stores high-dimensional vector embeddings associated with CRISPE prompt records.

    Each embedding is associated with a specific prompt record, model checkpoint,
    and configuration version. The source_content_hash allows detecting stale embeddings
    when prompt contents are updated without altering or dropping the embedding record.
    """

    __tablename__ = "prompt_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "prompt_id",
            "config_version",
            name="uq_prompt_embeddings_prompt_config",
        ),
        Index("ix_prompt_embeddings_prompt_id", "prompt_id"),
        Index("ix_prompt_embeddings_config_version", "config_version"),
        Index("ix_prompt_embeddings_source_hash", "source_content_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique primary key identifier for the embedding record.",
    )
    prompt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prompt_records.id", ondelete="CASCADE"),
        nullable=False,
        doc="Foreign key reference to the parent prompt record.",
    )
    embedding: Mapped[List[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
        doc="High-dimensional vector embedding (768 float32 dimensions).",
    )
    provider: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="google-genai",
        doc="Embedding provider identifier (e.g. google-genai).",
    )
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="text-embedding-004",
        doc="Specific model checkpoint that synthesized the vector.",
    )
    config_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="v1",
        doc="Configuration version distinguishing incompatible vector spaces.",
    )
    source_content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 digest of the exact canonical prompt text embedded.",
    )

    # Relationship to parent prompt record
    prompt: Mapped["PromptRecord"] = relationship(
        "PromptRecord",
        back_populates="embeddings",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize PromptEmbedding, generating defaults for in-memory instances."""
        if "id" not in kwargs or kwargs["id"] is None:
            kwargs["id"] = uuid.uuid4()
        if "provider" not in kwargs or kwargs["provider"] is None:
            kwargs["provider"] = "google-genai"
        if "model_id" not in kwargs or kwargs["model_id"] is None:
            kwargs["model_id"] = "text-embedding-004"
        if "config_version" not in kwargs or kwargs["config_version"] is None:
            kwargs["config_version"] = "v1"
        super().__init__(**kwargs)

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute the canonical SHA-256 hex digest for prompt text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
