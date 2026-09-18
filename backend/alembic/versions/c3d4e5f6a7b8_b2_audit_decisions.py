"""b2 audit chain, decisions, config audit (audit_entry, decision,
config_change, threshold_value)

Milestone B2: decision submission, the real SHA-256 hash-linked audit chain,
and the config audit trail. See app/storage/b2_models.py's module docstring
for the append-only rationale, and app/audit/chain.py for the chain itself.

Revision ID: c3d4e5f6a7b8
Revises: f2a913e6d4b0
Create Date: 2026-09-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'f2a913e6d4b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'b2_audit_entry',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('terminal_id', sa.String(), nullable=False),
        sa.Column('sequence', sa.Integer(), nullable=False),
        sa.Column('entry_type', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('run_id', sa.String(), nullable=True),
        sa.Column('previous_hash', sa.String(), nullable=False),
        sa.Column('entry_hash', sa.String(), nullable=False, unique=True),
        sa.Column('sealed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('officer_id', sa.String(), nullable=False),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.UniqueConstraint('terminal_id', 'sequence', name='uq_b2_audit_entry_terminal_seq'),
    )
    op.create_index('ix_b2_audit_entry_terminal_id', 'b2_audit_entry', ['terminal_id'])
    op.create_index('ix_b2_audit_entry_session_id', 'b2_audit_entry', ['session_id'])
    op.create_index('ix_b2_audit_entry_entry_hash', 'b2_audit_entry', ['entry_hash'])

    op.create_table(
        'b2_decision',
        sa.Column('session_id', sa.String(), sa.ForeignKey('b1_session.id'), primary_key=True),
        sa.Column('case_id', sa.String(), nullable=False, unique=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('b1_run.id'), nullable=False),
        sa.Column('audit_entry_id', sa.String(), sa.ForeignKey('b2_audit_entry.id'), nullable=False),
        sa.Column('officer_id', sa.String(), nullable=False),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('recommendation', sa.String(), nullable=False),
        sa.Column('divergence', sa.String(), nullable=False),
        sa.Column('divergence_reason', sa.String(), nullable=True),
        sa.Column('notes', sa.String(), nullable=False),
        sa.Column('attestation', sa.Boolean(), nullable=False),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finding_dispositions_json', sa.JSON(), nullable=False),
    )
    op.create_index('ix_b2_decision_case_id', 'b2_decision', ['case_id'])

    op.create_table(
        'b2_config_change',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('audit_entry_id', sa.String(), sa.ForeignKey('b2_audit_entry.id'), nullable=False),
        sa.Column('officer_id', sa.String(), nullable=False),
        sa.Column('changed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('reason', sa.String(), nullable=False),
        sa.Column('changes_json', sa.JSON(), nullable=False),
    )

    op.create_table(
        'b2_threshold_value',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('value', sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('b2_threshold_value')
    op.drop_table('b2_config_change')
    op.drop_index('ix_b2_decision_case_id', table_name='b2_decision')
    op.drop_table('b2_decision')
    op.drop_index('ix_b2_audit_entry_entry_hash', table_name='b2_audit_entry')
    op.drop_index('ix_b2_audit_entry_session_id', table_name='b2_audit_entry')
    op.drop_index('ix_b2_audit_entry_terminal_id', table_name='b2_audit_entry')
    op.drop_table('b2_audit_entry')
