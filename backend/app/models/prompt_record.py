"""SQLAlchemy ORM model for persistent CRISPE prompt records."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
import uuid
from sqlalchemy import ForeignKey, Index, Integer, String, Text, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.schemas.crispe import (
    CrispeGenerationResponse,
    CrispeSections,
    ModelUsageMetadata,
)

if TYPE_CHECKING:
    from app.models.principal import Principal


class PromptRecord(Base, TimestampMixin):
    """Persistent storage for generated and user-saved CRISPE prompts."""

    __tablename__ = "prompt_records"
    __table_args__ = (
        Index("ix_prompt_records_owner_created", "owner_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique primary key identifier for the prompt record.",
    )
    original_idea: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The raw task description or user idea input.",
    )

    # Six core CRISPE framework sections
    context: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Background situation, domain knowledge, and setting.",
    )
    role: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="AI persona, authority, perspective, and tone.",
    )
    instruction: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Step-by-step task commands and methodologies.",
    )
    specification: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Output format rules, constraints, and negative boundaries.",
    )
    performance: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Quality metrics, benchmarks, and criteria for success.",
    )
    example: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Illustrative demonstrations and few-shot samples.",
    )

    formatted_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Rendered canonical Markdown prompt matching UI headers.",
    )
    model_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="gemma-4-26b-a4b-it",
        index=True,
        doc="The specific model identifier used to synthesize the prompt.",
    )
    task_type: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="Optional classification or domain category tag.",
    )
    owner_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        ForeignKey("principals.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        doc="Foreign key reference to the owning Principal.",
    )

    owner: Mapped[Optional["Principal"]] = relationship(
        "Principal",
        back_populates="prompts",
    )

    # Token consumption telemetry
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    candidates_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    finish_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize PromptRecord, providing defaults for non-persisted in-memory instances."""
        if "id" not in kwargs or kwargs["id"] is None:
            kwargs["id"] = uuid.uuid4()
        super().__init__(**kwargs)

    def to_sections(self) -> CrispeSections:
        """Reconstitute the six structured sections into a validated CrispeSections model."""
        return CrispeSections(
            context=self.context,
            role=self.role,
            instruction=self.instruction,
            specification=self.specification,
            performance=self.performance,
            example=self.example,
        )

    def to_response(self, request_id: Optional[str] = None) -> CrispeGenerationResponse:
        """Map the persistent record into the application API response contract."""
        usage = None
        if self.total_tokens is not None:
            usage = ModelUsageMetadata(
                prompt_tokens=self.prompt_tokens or 0,
                candidates_tokens=self.candidates_tokens or 0,
                total_tokens=self.total_tokens or 0,
            )

        return CrispeGenerationResponse(
            idea=self.original_idea,
            sections=self.to_sections(),
            formatted_prompt=self.formatted_prompt,
            model_id=self.model_id,
            usage=usage,
            finish_reason=self.finish_reason,
            request_id=request_id,
            schema_version="1.0",
            created_at=self.created_at or datetime.now(timezone.utc),
        )
