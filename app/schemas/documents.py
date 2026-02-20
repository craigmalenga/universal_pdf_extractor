"""
Pydantic request/response schemas for the /api/v1/documents endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from decimal import Decimal


# ── Request Schemas ──────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""
    doc_id: str
    file_name: str
    file_size_bytes: int
    doc_hash: str
    status: str
    message: str = "Document uploaded successfully. Processing queued."


class DocumentListParams(BaseModel):
    """Query parameters for listing documents."""
    status: Optional[str] = None
    doc_family: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    order_by: str = "created_at"
    order_dir: str = "desc"


# ── Response Schemas ─────────────────────────────────────────

class DocumentSummary(BaseModel):
    """Lightweight document summary for list endpoints."""
    doc_id: str
    file_name: str
    file_size_bytes: int
    doc_family: str
    provider_guess: Optional[str] = None
    status: str
    page_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated document list response."""
    documents: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class SegmentSummary(BaseModel):
    """Segment info in document detail response."""
    segment_id: str
    segment_index: int
    start_page: int
    end_page: int
    statement_period_start: Optional[str] = None
    statement_period_end: Optional[str] = None
    opening_balance: Optional[Decimal] = None
    closing_balance: Optional[Decimal] = None
    reconciliation_status: Optional[str] = None
    confidence: Optional[float] = None
    transaction_count: int = 0

    model_config = {"from_attributes": True}


class ExtractionRunSummary(BaseModel):
    """Extraction run info in document detail response."""
    run_id: str
    pipeline_version: str
    is_latest: bool
    status: str
    validation_status: Optional[str] = None
    confidence_document: Optional[float] = None
    reconciliation_rate: Optional[float] = None
    row_count: Optional[int] = None
    duration_ms: Optional[int] = None
    started_at: datetime
    ended_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentDetail(BaseModel):
    """Full document detail response."""
    doc_id: str
    doc_hash: str
    file_name: str
    file_size_bytes: int
    mime_type: str
    page_count: Optional[int] = None
    doc_family: str
    doc_family_confidence: Optional[float] = None
    provider_guess: Optional[str] = None
    provider_confidence: Optional[float] = None
    currency: str
    status: str
    priority: int
    metadata_json: Optional[dict] = None
    created_at: datetime
    updated_at: datetime
    segments: list[SegmentSummary] = []
    extraction_runs: list[ExtractionRunSummary] = []

    model_config = {"from_attributes": True}


class DocumentReprocessRequest(BaseModel):
    """Request to reprocess a document."""
    force: bool = False
    preprocessing_profile: Optional[str] = None
    engine_override: Optional[str] = None