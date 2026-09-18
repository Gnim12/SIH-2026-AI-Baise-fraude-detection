"""officers and officer_sessions

Revision ID: a1b2c3d4e5f6
Revises: c6dde9793ad6
Create Date: 2026-09-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'c6dde9793ad6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'officers',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('officer_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False, server_default='officer'),
        sa.Column('active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_officers_officer_id', 'officers', ['officer_id'], unique=True)

    op.create_table(
        'officer_sessions',
        sa.Column('token', sa.String(), primary_key=True),
        sa.Column('officer_id', sa.String(), sa.ForeignKey('officers.officer_id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_officer_sessions_officer_id', 'officer_sessions', ['officer_id'])
    op.create_index('ix_officer_sessions_expires_at', 'officer_sessions', ['expires_at'])


def downgrade() -> None:
    op.drop_index('ix_officer_sessions_expires_at', table_name='officer_sessions')
    op.drop_index('ix_officer_sessions_officer_id', table_name='officer_sessions')
    op.drop_table('officer_sessions')
    op.drop_index('ix_officers_officer_id', table_name='officers')
    op.drop_table('officers')
