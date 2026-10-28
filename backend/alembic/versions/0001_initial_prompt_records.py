"""create initial prompt_records table

Revision ID: 0001_prompt_records
Revises: 
Create Date: 2026-10-28 13:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001_prompt_records'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'prompt_records',
        sa.Column('id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('original_idea', sa.Text(), nullable=False),
        sa.Column('context', sa.Text(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('instruction', sa.Text(), nullable=False),
        sa.Column('specification', sa.Text(), nullable=False),
        sa.Column('performance', sa.Text(), nullable=False),
        sa.Column('example', sa.Text(), nullable=False),
        sa.Column('formatted_prompt', sa.Text(), nullable=False),
        sa.Column('model_id', sa.String(length=128), server_default='gemma-4-26b-a4b-it', nullable=False),
        sa.Column('task_type', sa.String(length=64), nullable=True),
        sa.Column('owner_id', sa.String(length=128), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('candidates_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('finish_reason', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_prompt_records_model_id', 'prompt_records', ['model_id'], unique=False)
    op.create_index('ix_prompt_records_task_type', 'prompt_records', ['task_type'], unique=False)
    op.create_index('ix_prompt_records_owner_id', 'prompt_records', ['owner_id'], unique=False)
    op.create_index('ix_prompt_records_created_at', 'prompt_records', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_prompt_records_created_at', table_name='prompt_records')
    op.drop_index('ix_prompt_records_owner_id', table_name='prompt_records')
    op.drop_index('ix_prompt_records_task_type', table_name='prompt_records')
    op.drop_index('ix_prompt_records_model_id', table_name='prompt_records')
    op.drop_table('prompt_records')
