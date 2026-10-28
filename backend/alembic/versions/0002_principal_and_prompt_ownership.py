"""create principals table and prompt ownership foreign key

Revision ID: 0002_principal_and_prompt_ownership
Revises: 0001_prompt_records
Create Date: 2026-10-28 13:48:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002_principal_and_prompt_ownership'
down_revision: Union[str, None] = '0001_prompt_records'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create principals table (root identity/ownership anchor)
    op.create_table(
        'principals',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('display_name', sa.String(length=128), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column('is_superuser', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_principals_email', 'principals', ['email'], unique=True)

    # 2. Add foreign key constraint and composite index to prompt_records
    with op.batch_alter_table('prompt_records') as batch_op:
        batch_op.create_foreign_key(
            'fk_prompt_records_owner_id_principals',
            'principals',
            ['owner_id'],
            ['id'],
            ondelete='CASCADE',
        )
        batch_op.create_index(
            'ix_prompt_records_owner_created',
            ['owner_id', 'created_at'],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table('prompt_records') as batch_op:
        batch_op.drop_index('ix_prompt_records_owner_created')
        batch_op.drop_constraint('fk_prompt_records_owner_id_principals', type_='foreignkey')

    op.drop_index('ix_principals_email', table_name='principals')
    op.drop_table('principals')
