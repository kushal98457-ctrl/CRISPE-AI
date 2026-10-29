"""create prompt_candidates and candidate_evaluations tables (Step 27)

Revision ID: 0005_prompt_candidates
Revises: 0004_optimization_jobs
Create Date: 2026-10-29 20:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0005_prompt_candidates'
down_revision: Union[str, None] = '0004_optimization_jobs'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB(astext_type=sa.Text()) if is_postgres else sa.JSON()

    # 1. Create prompt_candidates table
    op.create_table(
        'prompt_candidates',
        sa.Column('id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('parent_candidate_id', sa.UUID(as_uuid=True), nullable=True),
        sa.Column('candidate_type', sa.String(length=50), server_default='optimized', nullable=False),
        sa.Column('instruction_text', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('lifecycle_status', sa.String(length=50), server_default='proposed', nullable=False),
        sa.Column('optimizer_version', sa.String(length=50), server_default='gepa-0.1.0', nullable=False),
        sa.Column('program_version', sa.String(length=50), server_default='crispe-dspy-v1', nullable=False),
        sa.Column('model_id', sa.String(length=100), server_default='gemma-4-26b-a4b-it', nullable=False),
        sa.Column('dataset_version', sa.String(length=50), server_default='v1', nullable=False),
        sa.Column('evaluation_config', json_type, nullable=True),
        sa.Column('provenance_metadata', json_type, nullable=True),
        sa.Column('artifact_ref', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['optimization_jobs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_candidate_id'], ['prompt_candidates.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id', 'candidate_type', 'content_hash', name='uq_prompt_candidates_job_type_content'),
    )
    op.create_index('ix_prompt_candidates_job_id', 'prompt_candidates', ['job_id'], unique=False)
    op.create_index('ix_prompt_candidates_parent_id', 'prompt_candidates', ['parent_candidate_id'], unique=False)
    op.create_index('ix_prompt_candidates_content_hash', 'prompt_candidates', ['content_hash'], unique=False)
    op.create_index('ix_prompt_candidates_type', 'prompt_candidates', ['candidate_type'], unique=False)
    op.create_index('ix_prompt_candidates_lifecycle', 'prompt_candidates', ['lifecycle_status'], unique=False)
    op.create_index('ix_prompt_candidates_job_created', 'prompt_candidates', ['job_id', 'created_at'], unique=False)

    # 2. Create candidate_evaluations table
    op.create_table(
        'candidate_evaluations',
        sa.Column('id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('candidate_id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('evaluation_type', sa.String(length=50), nullable=False),
        sa.Column('dataset_version', sa.String(length=50), server_default='v1', nullable=False),
        sa.Column('dataset_split', sa.String(length=50), nullable=False),
        sa.Column('example_id', sa.String(length=100), server_default='__aggregate__', nullable=False),
        sa.Column('evaluator_version', sa.String(length=50), server_default='v1', nullable=False),
        sa.Column('rubric_version', sa.String(length=50), server_default='v1', nullable=False),
        sa.Column('metrics', json_type, nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=True),
        sa.Column('is_critical_failure', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('sanitized_feedback', sa.Text(), nullable=True),
        sa.Column('execution_status', sa.String(length=50), server_default='completed', nullable=False),
        sa.Column('artifact_ref', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.CheckConstraint("dataset_split NOT IN ('test', 'holdout', 'final_holdout')", name='ck_candidate_evaluations_no_holdout'),
        sa.ForeignKeyConstraint(['candidate_id'], ['prompt_candidates.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['job_id'], ['optimization_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('candidate_id', 'evaluation_type', 'dataset_split', 'example_id', name='uq_candidate_evaluations_identity'),
    )
    op.create_index('ix_candidate_evaluations_candidate_id', 'candidate_evaluations', ['candidate_id'], unique=False)
    op.create_index('ix_candidate_evaluations_job_id', 'candidate_evaluations', ['job_id'], unique=False)
    op.create_index('ix_candidate_evaluations_type_split', 'candidate_evaluations', ['evaluation_type', 'dataset_split'], unique=False)
    op.create_index('ix_candidate_evaluations_job_created', 'candidate_evaluations', ['job_id', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_table('candidate_evaluations')
    op.drop_table('prompt_candidates')
