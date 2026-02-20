"""
Canonical output schemas.
These are the final API response shapes for extracted documents.
"""

from pydantic import BaseModel
from typing import Optional
from decimal import Decimal


class CanonicalTransaction(BaseModel):
    """The final output format for a single transaction."""
    transaction_id: str
    page_index: int
    row_index: int
    posted_date: Optional[str] = None       # ISO format YYYY-MM-DD
    value_date: Optional[str] = None
    description_raw: str
    description_clean: Optional[str] = None
    amount: Optional[Decimal] = None
    currency: str = "GBP"
    direction: str = "UNKNOWN"              # DEBIT, CREDIT, UNKNOWN
    direction_source: str = "UNKNOWN"       # BALANCE_SOLVER, COLUMN, SIGN, HEADER, UNKNOWN
    running_balance: Optional[Decimal] = None
    balance_confirmed: bool = False
    reference: Optional[str] = None
    transaction_type: Optional[str] = None  # DD, SO, FP, BACS, CHAPS, TRANSFER, ATM, etc
    is_balance_marker: bool = False
    confidence_overall: float
    confidence_amount: float
    confidence_direction: float
    confidence_date: float
    evidence: list[dict] = []


class CanonicalSegmentOutput(BaseModel):
    """Output for one statement segment."""
    segment_index: int
    statement_period_start: Optional[str] = None
    statement_period_end: Optional[str] = None
    opening_balance: Optional[Decimal] = None
    closing_balance: Optional[Decimal] = None
    computed_net_movement: Optional[Decimal] = None
    reconciliation_status: str              # PASS, PASS_WITH_WARNINGS, NEEDS_REVIEW, FAIL
    confidence: float
    transactions: list[CanonicalTransaction]


class CanonicalDocumentOutput(BaseModel):
    """The complete output for a processed document."""
    doc_id: str
    file_name: str
    doc_family: str
    provider_guess: Optional[str] = None
    provider_confidence: Optional[float] = None
    page_count: int
    pipeline_version: str
    engine_versions: dict
    validation_status: str
    confidence_document: float
    reconciliation_rate: Optional[float] = None
    total_transaction_count: int
    segments: list[CanonicalSegmentOutput]
    warnings: list[str] = []
    errors: list[str] = []
    processing_time_ms: int
    cost_usd: Optional[float] = None