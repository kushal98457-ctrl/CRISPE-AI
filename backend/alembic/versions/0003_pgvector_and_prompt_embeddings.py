"""create pgvector extension and prompt_embeddings table

Revision ID: 0003_prompt_embeddings
Revises: 0002_principal_and_prompt_ownership
Create Date: 2026-10-29 01:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003_prompt_embeddings'
down_revision: Union[str, None] = '0002_principal_and_prompt_ownership'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable pgvector extension on PostgreSQL
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Create prompt_embeddings table
    op.create_table(
        'prompt_embeddings',
        sa.Column('id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('prompt_id', sa.UUID(as_uuid=True), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(768), nullable=False),
        sa.Column('provider', sa.String(length=64), server_default='google-genai', nullable=False),
        sa.Column('model_id', sa.String(length=128), server_default='text-embedding-004', nullable=False),
        sa.Column('config_version', sa.String(length=32), server_default='v1', nullable=False),
        sa.Column('source_content_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['prompt_id'], ['prompt_records.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('prompt_id', 'config_version', name='uq_prompt_embeddings_prompt_config'),
    )

    # 3. Create indexes
    op.create_index('ix_prompt_embeddings_prompt_id', 'prompt_embeddings', ['prompt_id'], unique=False)
    op.create_index('ix_prompt_embeddings_config_version', 'prompt_embeddings', ['config_version'], unique=False)
    op.create_index('ix_prompt_embeddings_source_hash', 'prompt_embeddings', ['source_content_hash'], unique=False)


def downgrade() -> None:
    # 1. Drop indexes and table
    op.drop_index('ix_prompt_embeddings_source_hash', table_name='prompt_embeddings')
    op.drop_index('ix_prompt_embeddings_config_version', table_name='prompt_embeddings')
    op.drop_index('ix_prompt_embeddings_prompt_id', table_name='prompt_embeddings')
    op.drop_table('prompt_embeddings')

    # Note: Extension vector is intentionally preserved on downgrade to avoid dropping
    # shared server-level extensions needed by other database objects.
