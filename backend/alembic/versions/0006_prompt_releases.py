"""create prompt_releases and prompt_release_audit_events tables (Step 28)

Revision ID: 0006_prompt_releases
Revises: 0005_prompt_candidates
Create Date: 2026-10-29 21:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0006_prompt_releases'
down_revision: Union[str, None] = '0005_prompt_candidates'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB(astext_type=sa.Text()) if is_postgres else sa.JSON()

    # 1. Create prompt_releases table
    op.create_table(
        'prompt_releases',
        sa.Column('id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('release_version', sa.String(length=50), nullable=False),
        sa.Column('candidate_id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('candidate_hash', sa.String(length=64), nullable=False),
        sa.Column('job_id', sa.UUID(as_uuid=True), nullable=True),
        sa.Column('instruction_text', sa.Text(), nullable=False),
        sa.Column('runtime_compatibility', sa.String(length=50), server_default='crispe-dspy-v1', nullable=False),
        sa.Column('generation_model_id', sa.String(length=100), server_default='gemma-4-26b-a4b-it', nullable=False),
        sa.Column('provider_name', sa.String(length=50), server_default='google-genai', nullable=False),
        sa.Column('status', sa.String(length=50), server_default='draft', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('previous_release_id', sa.UUID(as_uuid=True), nullable=True),
        sa.Column('validation_policy_version', sa.String(length=50), server_default='release_policy_v1', nullable=False),
        sa.Column('validation_report', json_type, nullable=True),
        sa.Column('holdout_evaluation_summary', json_type, nullable=True),
        sa.Column('created_by_id', sa.String(length=128), nullable=False),
        sa.Column('approved_by_id', sa.String(length=128), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('validated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('promoted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('superseded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rolled_back_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('rollback_reason', sa.Text(), nullable=True),
        sa.Column('audit_event_id', sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(['candidate_id'], ['prompt_candidates.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['job_id'], ['optimization_jobs.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['previous_release_id'], ['prompt_releases.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['created_by_id'], ['principals.id']),
        sa.ForeignKeyConstraint(['approved_by_id'], ['principals.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('release_version', name='uq_prompt_releases_version'),
    )
    op.create_index('ix_prompt_releases_status', 'prompt_releases', ['status'], unique=False)
    op.create_index('ix_prompt_releases_active', 'prompt_releases', ['is_active'], unique=False)
    op.create_index('ix_prompt_releases_candidate_id', 'prompt_releases', ['candidate_id'], unique=False)

    # Partial unique index ensuring at most one active release
    if is_postgres:
        op.create_index(
            'uq_prompt_releases_single_active',
            'prompt_releases',
            ['is_active'],
            unique=True,
            postgresql_where=sa.text('is_active = TRUE'),
        )
    else:
        op.create_index(
            'uq_prompt_releases_single_active',
            'prompt_releases',
            ['is_active'],
            unique=True,
            sqlite_where=sa.text('is_active = 1'),
        )

    # 2. Create prompt_release_audit_events table
    op.create_table(
        'prompt_release_audit_events',
        sa.Column('id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('release_id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('acting_principal_id', sa.String(length=128), nullable=False),
        sa.Column('from_status', sa.String(length=50), nullable=True),
        sa.Column('to_status', sa.String(length=50), nullable=False),
        sa.Column('details', json_type, nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['release_id'], ['prompt_releases.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['acting_principal_id'], ['principals.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_prompt_release_audit_release', 'prompt_release_audit_events', ['release_id', 'created_at'], unique=False)
    op.create_index('ix_prompt_release_audit_principal', 'prompt_release_audit_events', ['acting_principal_id'], unique=False)


def downgrade() -> None:
    op.drop_table('prompt_release_audit_events')
    op.drop_table('prompt_releases')
