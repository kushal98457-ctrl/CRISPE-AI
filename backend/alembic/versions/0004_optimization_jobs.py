"""create optimization_jobs table for durable background optimization (Step 26)

Revision ID: 0004_optimization_jobs
Revises: 0003_prompt_embeddings
Create Date: 2026-10-29 20:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0004_optimization_jobs'
down_revision: Union[str, None] = '0003_prompt_embeddings'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check dialect for JSON / JSONB support
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB(astext_type=sa.Text()) if is_postgres else sa.JSON()

    op.create_table(
        'optimization_jobs',
        sa.Column('id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('owner_id', sa.String(length=128), nullable=False),
        sa.Column('job_type', sa.String(length=50), server_default='gepa_prompt_optimization', nullable=False),
        sa.Column('schema_version', sa.String(length=20), server_default='v1', nullable=False),
        sa.Column('status', sa.String(length=50), server_default='queued', nullable=False),

        # Dataset & Partition boundaries
        sa.Column('dataset_version', sa.String(length=50), server_default='v1', nullable=False),
        sa.Column('dataset_sha256', sa.String(length=64), nullable=True),
        sa.Column('partition_manifest', sa.String(length=100), server_default='prompt_optimization_eval_v1_splits', nullable=False),
        sa.Column('train_example_ids', json_type, nullable=True),
        sa.Column('val_example_ids', json_type, nullable=True),
        sa.Column('train_size', sa.Integer(), server_default='5', nullable=False),
        sa.Column('val_size', sa.Integer(), server_default='3', nullable=False),
        sa.Column('seed', sa.Integer(), server_default='42', nullable=False),

        # Model configuration
        sa.Column('generation_model_id', sa.String(length=100), server_default='gemma-4-26b-a4b-it', nullable=False),
        sa.Column('reflection_model_id', sa.String(length=100), server_default='gemini-2.5-flash', nullable=False),
        sa.Column('enable_model_judge', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('is_mock', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('idempotency_key', sa.String(length=64), nullable=True),

        # Budget limits
        sa.Column('max_metric_calls', sa.Integer(), server_default='15', nullable=False),
        sa.Column('max_total_provider_calls', sa.Integer(), server_default='35', nullable=False),
        sa.Column('max_elapsed_seconds', sa.Float(), server_default='300.0', nullable=False),
        sa.Column('concurrency_limit', sa.Integer(), server_default='1', nullable=False),
        sa.Column('max_retries_per_request', sa.Integer(), server_default='3', nullable=False),
        sa.Column('max_total_tokens', sa.Integer(), nullable=True),
        sa.Column('max_budget_usd', sa.Float(), nullable=True),
        sa.Column('pricing_version', sa.String(length=50), server_default='gemini-2025-standard-v1', nullable=False),

        # Persistent usage counters
        sa.Column('used_metric_calls', sa.Integer(), server_default='0', nullable=False),
        sa.Column('used_provider_calls', sa.Integer(), server_default='0', nullable=False),
        sa.Column('used_retry_attempts', sa.Integer(), server_default='0', nullable=False),
        sa.Column('used_prompt_tokens', sa.Integer(), server_default='0', nullable=False),
        sa.Column('used_candidates_tokens', sa.Integer(), server_default='0', nullable=False),
        sa.Column('used_total_tokens', sa.Integer(), server_default='0', nullable=False),
        sa.Column('estimated_cost_usd', sa.Float(), server_default='0.0', nullable=False),

        # Worker lease & claim fencing
        sa.Column('worker_id', sa.String(length=100), nullable=True),
        sa.Column('claim_token', sa.String(length=64), nullable=True),
        sa.Column('attempt_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('max_attempts', sa.Integer(), server_default='2', nullable=False),
        sa.Column('lease_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_heartbeat_at', sa.DateTime(timezone=True), nullable=True),

        # Lifecycle timestamps
        sa.Column('queued_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(timezone=True), nullable=True),

        # Progress & cancellation
        sa.Column('current_phase', sa.String(length=50), server_default='queued', nullable=False),
        sa.Column('progress_percent', sa.Float(), nullable=True),
        sa.Column('progress_message', sa.String(length=255), nullable=True),
        sa.Column('cancel_requested', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('cancel_reason', sa.String(length=255), nullable=True),

        # Outcome & artifacts
        sa.Column('stop_reason', sa.String(length=50), nullable=True),
        sa.Column('failure_category', sa.String(length=50), nullable=True),
        sa.Column('failure_summary', sa.Text(), nullable=True),
        sa.Column('artifacts_dir', sa.String(length=500), nullable=True),
        sa.Column('result_summary', json_type, nullable=True),

        # Base timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),

        # Constraints
        sa.ForeignKeyConstraint(['owner_id'], ['principals.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Indexes
    op.create_index('ix_optimization_jobs_owner_id', 'optimization_jobs', ['owner_id'], unique=False)
    op.create_index('ix_optimization_jobs_status', 'optimization_jobs', ['status'], unique=False)
    op.create_index('ix_optimization_jobs_worker_id', 'optimization_jobs', ['worker_id'], unique=False)
    op.create_index('ix_optimization_jobs_status_lease', 'optimization_jobs', ['status', 'lease_expires_at'], unique=False)
    op.create_index('ix_optimization_jobs_owner_created', 'optimization_jobs', ['owner_id', 'created_at'], unique=False)
    op.create_index('ix_optimization_jobs_idempotency', 'optimization_jobs', ['owner_id', 'idempotency_key'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_optimization_jobs_idempotency', table_name='optimization_jobs')
    op.drop_index('ix_optimization_jobs_owner_created', table_name='optimization_jobs')
    op.drop_index('ix_optimization_jobs_status_lease', table_name='optimization_jobs')
    op.drop_index('ix_optimization_jobs_worker_id', table_name='optimization_jobs')
    op.drop_index('ix_optimization_jobs_status', table_name='optimization_jobs')
    op.drop_index('ix_optimization_jobs_owner_id', table_name='optimization_jobs')
    op.drop_table('optimization_jobs')
