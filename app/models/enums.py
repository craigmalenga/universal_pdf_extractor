"""
Python enums matching PostgreSQL enum types.
Names and values MUST match the DB DDL exactly.
"""

from enum import Enum


class DocFamily(str, Enum):
    BANK_STATEMENT = "BANK_STATEMENT"
    MOTOR_FINANCE = "MOTOR_FINANCE"
    UNKNOWN = "UNKNOWN"


class DocStatus(str, Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    RENDERING = "RENDERING"
    PREPROCESSING = "PREPROCESSING"
    SEGMENTING = "SEGMENTING"
    EXTRACTING = "EXTRACTING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


class ExtractionStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAIL = "FAIL"


class TxDirection(str, Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"
    UNKNOWN = "UNKNOWN"


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"
    SKIPPED = "SKIPPED"


class TableType(str, Enum):
    TRANSACTION_TABLE = "TRANSACTION_TABLE"
    PAYMENT_SCHEDULE = "PAYMENT_SCHEDULE"
    PAYMENTS_RECEIVED = "PAYMENTS_RECEIVED"
    FEES_CHARGES = "FEES_CHARGES"
    SUMMARY = "SUMMARY"
    INTEREST = "INTEREST"
    ARREARS = "ARREARS"
    SETTLEMENT = "SETTLEMENT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class PipelineStage(str, Enum):
    """Pipeline checkpoint stages for resume-on-failure."""
    LOADED = "LOADED"
    RENDERED = "RENDERED"
    PREPROCESSED = "PREPROCESSED"
    EXTRACTED = "EXTRACTED"
    SEGMENTED = "SEGMENTED"
    ANALYSED = "ANALYSED"
    VALIDATED = "VALIDATED"
    PERSISTED = "PERSISTED"
    FAILED = "FAILED"


class ExtractionPath(str, Enum):
    PDF_TEXT = "PDF_TEXT"
    OCR = "OCR"
    HYBRID = "HYBRID"


class TextLayerQuality(str, Enum):
    GOOD = "GOOD"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


class DirectionSource(str, Enum):
    BALANCE_SOLVER = "BALANCE_SOLVER"
    COLUMN = "COLUMN"
    SIGN = "SIGN"
    HEADER = "HEADER"
    UNKNOWN = "UNKNOWN"