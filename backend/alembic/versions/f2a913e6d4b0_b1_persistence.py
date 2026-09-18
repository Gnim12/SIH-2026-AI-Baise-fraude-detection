"""b1 persistence (session, artefact, run, stage_event, signal, finding,
finding_signal, case)

Milestone B1b: the new stage-registry pipeline (app/pipeline/registry.py,
app/api/b1_*.py) needs its own tables, prefixed `b1_` to avoid colliding
with the older BACKEND_BRIEF.md `sessions` table (a single-JSONB-payload
shape, see 9d17cd0a007b) -- these are genuinely different systems sharing
one database. See app/storage/b1_models.py's module docstring for the full
rationale (SQLAlchemy over SQLModel, table naming, document-number masking).

Revision ID: f2a913e6d4b0
Revises: b7c8d9e0f1a2
Create Date: 2026-09-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2a913e6d4b0'
down_revision: Union[str, Sequence[str], None] = 'b7c8d9e0f1a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'b1_session',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('document_class', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
    )

    op.create_table(
        'b1_artefact',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('session_id', sa.String(), sa.ForeignKey('b1_session.id'), nullable=False),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('content_type', sa.String(), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('sha256', sa.String(), nullable=False),
        sa.Column('stored_path', sa.String(), nullable=False),
        sa.Column('captured_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('frame_count', sa.Integer(), nullable=True),
        sa.Column('angular_coverage_deg', sa.Float(), nullable=True),
        sa.UniqueConstraint('session_id', 'kind', name='uq_b1_artefact_session_kind'),
    )
    op.create_index('ix_b1_artefact_session_id', 'b1_artefact', ['session_id'])
    op.create_index('ix_b1_artefact_kind', 'b1_artefact', ['kind'])

    op.create_table(
        'b1_run',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('session_id', sa.String(), sa.ForeignKey('b1_session.id'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('verdict', sa.String(), nullable=True),
        sa.Column('coverage_complete', sa.Boolean(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('result_json', sa.JSON(), nullable=True),
    )
    op.create_index('ix_b1_run_session_id', 'b1_run', ['session_id'])

    op.create_table(
        'b1_stage_event',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('b1_run.id'), nullable=False),
        sa.Column('stage_id', sa.String(), nullable=False),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('detail', sa.String(), nullable=True),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sequence', sa.Integer(), nullable=False),
        sa.UniqueConstraint('run_id', 'sequence', name='uq_b1_stage_event_run_seq'),
    )
    op.create_index('ix_b1_stage_event_run_id', 'b1_stage_event', ['run_id'])

    op.create_table(
        'b1_signal',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('b1_run.id'), nullable=False),
        sa.Column('stage_id', sa.String(), nullable=False),
        sa.Column('signal_id', sa.String(), nullable=False),
        sa.Column('modality', sa.String(), nullable=False),
        sa.Column('label', sa.String(), nullable=False),
        sa.Column('detail', sa.String(), nullable=False),
        sa.Column('region_json', sa.JSON(), nullable=True),
        sa.Column('model_pin', sa.String(), nullable=False),
        sa.Column('emitted_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_b1_signal_run_id', 'b1_signal', ['run_id'])

    op.create_table(
        'b1_finding',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('run_id', sa.String(), sa.ForeignKey('b1_run.id'), nullable=False),
        sa.Column('finding_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('hypothesis', sa.String(), nullable=False),
        sa.Column('severity', sa.String(), nullable=False),
        sa.Column('region_json', sa.JSON(), nullable=True),
    )
    op.create_index('ix_b1_finding_run_id', 'b1_finding', ['run_id'])

    op.create_table(
        'b1_finding_signal',
        sa.Column('finding_id', sa.String(), sa.ForeignKey('b1_finding.id'), primary_key=True),
        sa.Column('signal_id', sa.String(), sa.ForeignKey('b1_signal.id'), primary_key=True),
    )

    op.create_table(
        'b1_case',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('session_id', sa.String(), sa.ForeignKey('b1_session.id'), nullable=False),
        sa.Column('run_id', sa.String(), sa.ForeignKey('b1_run.id'), nullable=False),
        sa.Column('case_id', sa.String(), nullable=False),
        sa.Column('screened_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('document_class', sa.String(), nullable=True),
        sa.Column('issuing_state', sa.String(), nullable=True),
        sa.Column('document_number_masked', sa.String(), nullable=True),
        sa.UniqueConstraint('case_id', name='uq_b1_case_case_id'),
    )
    op.create_index('ix_b1_case_session_id', 'b1_case', ['session_id'])


def downgrade() -> None:
    op.drop_table('b1_case')
    op.drop_table('b1_finding_signal')
    op.drop_table('b1_finding')
    op.drop_table('b1_signal')
    op.drop_table('b1_stage_event')
    op.drop_table('b1_run')
    op.drop_table('b1_artefact')
    op.drop_table('b1_session')
