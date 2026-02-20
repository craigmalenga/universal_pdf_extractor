"""
Pydantic transaction output schemas for API responses.
"""

from pydantic import BaseModel
from typing import Optional
from decimal import Decimal
from datetime import datetime


class TransactionResponse(BaseModel):
    """Single transaction in API response."""
    transaction_id: str
    page_index: int
    row_index: int
    posted_date: Optional[str] = None
    value_date: Optional[str] = None
    description_raw: Optional[str] = None
    description_clean: Optional[str] = None
    amount: Optional[Decimal] = None
    currency: str = "GBP"
    direction: str = "UNKNOWN"
    direction_source: Optional[str] = None
    running_balance: Optional[Decimal] = None
    balance_confirmed: bool = False
    reference: Optional[str] = None
    transaction_type: Optional[str] = None
    is_balance_marker: bool = False
    confidence_overall: Optional[float] = None
    confidence_amount: Optional[float] = None
    confidence_direction: Optional[float] = None
    confidence_date: Optional[float] = None

    model_config = {"from_attributes": True}


class TransactionListResponse(BaseModel):
    """Paginated transaction list."""
    transactions: list[TransactionResponse]
    total: int
    run_id: str
    doc_id: str


class TransactionEvidenceResponse(BaseModel):
    """Evidence record for a transaction field."""
    evidence_id: str
    field_name: str
    page_index: int
    bbox_json: dict
    extracted_text: str
    corrected_text: Optional[str] = None
    correction_reason: Optional[str] = None
    engine_name: str
    token_confidence: Optional[float] = None

    model_config = {"from_attributes": True}