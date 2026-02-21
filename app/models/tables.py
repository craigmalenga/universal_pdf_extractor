"""
SQLAlchemy ORM models.
Column names match the PostgreSQL DDL in BATON_PASS_BUILD_SPEC_v1.md Part 4 exactly.
"""

import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    BigInteger,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, ENUM
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


# ────────────────────────────────────────────────────────────
# DOCUMENTS
# ────────────────────────────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    doc_hash: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    doc_family: Mapped[str] = mapped_column(
        ENUM('BANK_STATEMENT', 'MOTOR_FINANCE', 'UNKNOWN', name='doc_family_enum', create_type=False),
        nullable=False, default="UNKNOWN", server_default="UNKNOWN"
    )
    doc_family_confidence: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    provider_guess: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provider_confidence: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, default="GBP", server_default="GBP"
    )
    raw_file_uri: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        ENUM('UPLOADED', 'QUEUED', 'RENDERING', 'PREPROCESSING', 'SEGMENTING', 'EXTRACTING', 'VALIDATING', 'COMPLETED', 'NEEDS_REVIEW', 'FAILED', name='doc_status_enum', create_type=False),
        nullable=False, default="UPLOADED", server_default="UPLOADED"
    )
    callback_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5, server_default="5"
    )
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    account_holder_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    account_holder_address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    account_holder_postcode: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    segments = relationship("DocumentSegment", back_populates="document", cascade="all, delete-orphan")
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    extraction_runs = relationship("ExtractionRun", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_documents_hash", "doc_hash"),
        Index("idx_documents_status", "status"),
        Index("idx_documents_created", "created_at"),
    )


# ────────────────────────────────────────────────────────────
# DOCUMENT SEGMENTS
# ────────────────────────────────────────────────────────────
class DocumentSegment(Base):
    __tablename__ = "document_segments"

    segment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_page: Mapped[int] = mapped_column(Integer, nullable=False)
    end_page: Mapped[int] = mapped_column(Integer, nullable=False)
    statement_period_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    statement_period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    opening_balance: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    closing_balance: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    computed_net_movement: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    reconciliation_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    document = relationship("Document", back_populates="segments")
    pages = relationship("Page", back_populates="segment")
    transactions = relationship("Transaction", back_populates="segment")

    __table_args__ = (
        UniqueConstraint("doc_id", "segment_index", name="uq_segment_doc_index"),
        Index("idx_segments_doc", "doc_id"),
    )


# ────────────────────────────────────────────────────────────
# PAGES
# ────────────────────────────────────────────────────────────
class Page(Base):
    __tablename__ = "pages"

    page_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    page_index: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_segments.segment_id"), nullable=True
    )
    width_px: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height_px: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dpi: Mapped[int] = mapped_column(Integer, nullable=False, default=300, server_default="300")
    orientation_detected: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0"
    )
    skew_degrees: Mapped[Decimal] = mapped_column(
        Numeric(6, 3), default=0, server_default="0.000"
    )
    preprocessing_profile: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_layer_quality: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ocr_engine_used: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ocr_mean_confidence: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    raw_render_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    normalized_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extraction_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transform_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    document = relationship("Document", back_populates="pages")
    segment = relationship("DocumentSegment", back_populates="pages")

    __table_args__ = (
        UniqueConstraint("doc_id", "page_index", name="uq_page_doc_index"),
        Index("idx_pages_doc", "doc_id"),
        Index("idx_pages_segment", "segment_id"),
    )


# ────────────────────────────────────────────────────────────
# EXTRACTION RUNS
# ────────────────────────────────────────────────────────────
class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    segment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_segments.segment_id"), nullable=True
    )
    pipeline_version: Mapped[str] = mapped_column(Text, nullable=False)
    engine_versions_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    preprocessing_profile: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_latest: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        ENUM('RUNNING', 'COMPLETED', 'FAILED', name='extraction_status_enum', create_type=False),
        nullable=False, default="RUNNING", server_default="RUNNING"
    )
    validation_status: Mapped[Optional[str]] = mapped_column(
        ENUM('PASS', 'PASS_WITH_WARNINGS', 'NEEDS_REVIEW', 'FAIL', name='validation_status_enum', create_type=False),
        nullable=True
    )
    confidence_document: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    reconciliation_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rows_with_balance_confirmation: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    error_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    diagnostics_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    cost_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    document = relationship("Document", back_populates="extraction_runs")
    detected_tables = relationship("DetectedTable", back_populates="extraction_run", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="extraction_run", cascade="all, delete-orphan")
    review_items = relationship("ReviewQueueItem", back_populates="extraction_run", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_runs_doc", "doc_id"),
        Index("idx_runs_segment", "segment_id"),
        Index("idx_runs_latest", "doc_id", "is_latest", postgresql_where=text("is_latest = TRUE")),
    )


# ────────────────────────────────────────────────────────────
# DETECTED TABLES
# ────────────────────────────────────────────────────────────
class DetectedTable(Base):
    __tablename__ = "detected_tables"

    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extraction_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    page_index: Mapped[int] = mapped_column(Integer, nullable=False)
    table_index: Mapped[int] = mapped_column(Integer, nullable=False)
    table_type: Mapped[str] = mapped_column(
        ENUM('TRANSACTION_TABLE', 'PAYMENT_SCHEDULE', 'PAYMENTS_RECEIVED', 'FEES_CHARGES', 'SUMMARY', 'INTEREST', 'ARREARS', 'SETTLEMENT', 'OTHER', 'UNKNOWN', name='table_type_enum', create_type=False),
        nullable=False, default="UNKNOWN", server_default="UNKNOWN"
    )
    table_type_confidence: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    bbox_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    header_row_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    column_mapping_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    raw_cells_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    extraction_run = relationship("ExtractionRun", back_populates="detected_tables")
    transactions = relationship("Transaction", back_populates="detected_table")

    __table_args__ = (
        UniqueConstraint("run_id", "page_index", "table_index", name="uq_table_run_page_index"),
        Index("idx_detected_tables_run", "run_id"),
        Index("idx_detected_tables_doc", "doc_id"),
    )


# ────────────────────────────────────────────────────────────
# TRANSACTIONS
# ────────────────────────────────────────────────────────────
class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extraction_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    segment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_segments.segment_id"), nullable=True
    )
    table_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("detected_tables.table_id"), nullable=True
    )
    page_index: Mapped[int] = mapped_column(Integer, nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    posted_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    value_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    description_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description_clean: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, default="GBP", server_default="GBP"
    )
    direction: Mapped[str] = mapped_column(
        ENUM('DEBIT', 'CREDIT', 'UNKNOWN', name='tx_direction_enum', create_type=False),
        nullable=False, default="UNKNOWN", server_default="UNKNOWN"
    )
    direction_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    running_balance: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    balance_confirmed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    balance_tolerance_used: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 4), nullable=True
    )
    reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transaction_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category_hint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_balance_marker: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    confidence_overall: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    confidence_amount: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    confidence_direction: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    confidence_date: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    confidence_description: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    confidence_balance: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    confidence_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    extraction_run = relationship("ExtractionRun", back_populates="transactions")
    segment = relationship("DocumentSegment", back_populates="transactions")
    detected_table = relationship("DetectedTable", back_populates="transactions")
    evidence = relationship("TransactionEvidence", back_populates="transaction", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("run_id", "page_index", "row_index", name="uq_tx_run_page_row"),
        Index("idx_tx_run", "run_id"),
        Index("idx_tx_doc", "doc_id"),
        Index("idx_tx_segment", "segment_id"),
        Index("idx_tx_date", "posted_date"),
        Index("idx_tx_direction", "direction"),
        Index("idx_tx_confidence", "confidence_overall"),
    )


# ────────────────────────────────────────────────────────────
# TRANSACTION EVIDENCE
# ────────────────────────────────────────────────────────────
class TransactionEvidence(Base):
    __tablename__ = "transaction_evidence"

    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transactions.transaction_id", ondelete="CASCADE"),
        nullable=False
    )
    field_name: Mapped[str] = mapped_column(Text, nullable=False)
    page_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    correction_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    engine_name: Mapped[str] = mapped_column(Text, nullable=False)
    engine_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_confidence: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    transaction = relationship("Transaction", back_populates="evidence")

    __table_args__ = (
        Index("idx_evidence_tx", "transaction_id"),
        Index("idx_evidence_field", "field_name"),
    )


# ────────────────────────────────────────────────────────────
# TEMPLATES
# ────────────────────────────────────────────────────────────
class Template(Base):
    __tablename__ = "templates"

    template_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    template_name: Mapped[str] = mapped_column(Text, nullable=False)
    provider_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    doc_family: Mapped[str] = mapped_column(
        ENUM('BANK_STATEMENT', 'MOTOR_FINANCE', 'UNKNOWN', name='doc_family_enum', create_type=False),
        nullable=False, default="BANK_STATEMENT", server_default="BANK_STATEMENT"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    versions = relationship("TemplateVersion", back_populates="template", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_templates_provider", "provider_name"),
        Index("idx_templates_active", "is_active", postgresql_where=text("is_active = TRUE")),
    )


# ────────────────────────────────────────────────────────────
# TEMPLATE VERSIONS
# ────────────────────────────────────────────────────────────
class TemplateVersion(Base):
    __tablename__ = "template_versions"

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    template_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("templates.template_id", ondelete="CASCADE"),
        nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    fingerprint_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    column_mapping_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    region_hints_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    quirks_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    accuracy_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    # Relationships
    template = relationship("Template", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("template_id", "version_number", name="uq_template_version"),
        Index("idx_template_versions_tid", "template_id"),
    )


# ────────────────────────────────────────────────────────────
# REVIEW QUEUE
# ────────────────────────────────────────────────────────────
class ReviewQueueItem(Base):
    __tablename__ = "review_queue"

    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extraction_runs.run_id", ondelete="CASCADE"),
        nullable=False
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False
    )
    segment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_segments.segment_id"), nullable=True
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    reason_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5, server_default="5")
    assigned_to: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        ENUM('PENDING', 'IN_REVIEW', 'RESOLVED', 'SKIPPED', name='review_status_enum', create_type=False),
        nullable=False, default="PENDING", server_default="PENDING"
    )
    resolution_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    extraction_run = relationship("ExtractionRun", back_populates="review_items")

    __table_args__ = (
        Index("idx_review_status", "status", "priority"),
        Index("idx_review_doc", "doc_id"),
        Index("idx_review_pending", "status", postgresql_where=text("status = 'PENDING'")),
    )


# ────────────────────────────────────────────────────────────
# GOLDEN FIXTURES
# ────────────────────────────────────────────────────────────
class GoldenFixture(Base):
    __tablename__ = "golden_fixtures"

    fixture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    file_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False)
    bank_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    doc_family: Mapped[str] = mapped_column(
        ENUM('BANK_STATEMENT', 'MOTOR_FINANCE', 'UNKNOWN', name='doc_family_enum', create_type=False),
        nullable=False
    )
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_transaction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )


# ────────────────────────────────────────────────────────────
# COST EVENTS
# ────────────────────────────────────────────────────────────
class CostEvent(Base):
    __tablename__ = "cost_events"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
        server_default=text("gen_random_uuid()")
    )
    run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extraction_runs.run_id", ondelete="SET NULL"),
        nullable=True
    )
    doc_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="SET NULL"),
        nullable=True
    )
    engine_name: Mapped[str] = mapped_column(Text, nullable=False)
    operation: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6), nullable=True)
    cost_gbp: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6), nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    __table_args__ = (
        Index("idx_cost_run", "run_id"),
        Index("idx_cost_engine", "engine_name"),
        Index("idx_cost_created", "created_at"),
    )