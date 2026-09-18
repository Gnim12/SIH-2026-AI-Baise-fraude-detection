"""password reset requests

Revision ID: b7c8d9e0f1a2
Revises: 9d17cd0a007b
Create Date: 2026-09-01 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7c8d9e0f1a2'
down_revision: Union[str, Sequence[str], None] = '9d17cd0a007b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'password_reset_requests',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('officer_id', sa.String(), sa.ForeignKey('officers.officer_id', ondelete='CASCADE'), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('status', sa.String(), nullable=False, server_default='pending'),
        sa.Column('reference_code', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_by', sa.String(), sa.ForeignKey('officers.officer_id', ondelete='SET NULL'), nullable=True),
    )
    op.create_index('ix_password_reset_requests_officer_id', 'password_reset_requests', ['officer_id'])
    op.create_index(
        'ix_password_reset_requests_reference_code', 'password_reset_requests', ['reference_code'], unique=True,
    )


def downgrade() -> None:
    op.drop_index('ix_password_reset_requests_reference_code', table_name='password_reset_requests')
    op.drop_index('ix_password_reset_requests_officer_id', table_name='password_reset_requests')
    op.drop_table('password_reset_requests')
