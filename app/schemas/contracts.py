"""
Core extraction contracts.
NormalizedPageExtraction is THE central schema.
Both PDF text and OCR paths MUST produce this structure.
All downstream pipeline code operates on this - never on engine-specific output.
"""

from pydantic import BaseModel, Field
from typing import Optional


class BBox(BaseModel):
    """Bounding box, normalised to page dimensions (0.0 to 1.0)."""
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)


class ExtractedToken(BaseModel):
    """A single word/token with position and confidence."""
    text: str
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0, default=0.95)
    source: str  # 'pdfplumber', 'docai', 'tesseract'


class ExtractedLine(BaseModel):
    """A line of text (ordered tokens on same y-axis)."""
    tokens: list[ExtractedToken]
    bbox: BBox
    text: str


class ExtractedCell(BaseModel):
    """A table cell."""
    text: str
    bbox: BBox
    row_index: int
    col_index: int
    confidence: float = Field(ge=0.0, le=1.0, default=0.95)
    is_header: bool = False


class ExtractedTable(BaseModel):
    """A table detected on the page."""
    cells: list[ExtractedCell]
    bbox: BBox
    row_count: int
    col_count: int
    has_grid_lines: bool = False
    header_row_index: Optional[int] = 0
    confidence: float = Field(ge=0.0, le=1.0, default=0.80)


class ExtractedKeyValue(BaseModel):
    """A key-value pair detected on the page (e.g., 'Sort Code: 12-34-56')."""
    key: str
    value: str
    key_bbox: BBox
    value_bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0, default=0.80)


class PageMetrics(BaseModel):
    """Quality metrics for the page extraction."""
    dpi: int = 300
    orientation_degrees: int = 0
    skew_degrees: float = 0.0
    preprocessing_profile: Optional[str] = None
    mean_token_confidence: float = Field(ge=0.0, le=1.0, default=0.95)
    token_count: int = 0
    text_density: float = 0.0
    gibberish_ratio: float = 0.0
    extraction_path: str = "PDF_TEXT"
    engine_name: str = ""
    engine_version: str = ""
    processing_time_ms: int = 0


class NormalizedPageExtraction(BaseModel):
    """
    THE CORE CONTRACT.

    Both PDF text and OCR extraction paths MUST produce this structure.
    All downstream pipeline code operates on this - never on engine-specific output.

    Invariants:
    - tokens are ordered by y0 then x0 (top-to-bottom, left-to-right)
    - lines are ordered by y0 (top-to-bottom)
    - lines.text == ' '.join(token.text for token in matching tokens)
    - tables.cells are ordered by row_index then col_index
    - all bbox values are normalised to [0.0, 1.0] relative to page dimensions
    - confidence values are [0.0, 1.0] where 1.0 = highest confidence
    - for PDF text extraction: confidence defaults to 0.95 (high but not certain)
    - for OCR: confidence comes from engine
    """
    doc_id: str
    page_index: int
    tokens: list[ExtractedToken]
    lines: list[ExtractedLine]
    tables: list[ExtractedTable]
    key_values: list[ExtractedKeyValue] = []
    metrics: PageMetrics
    raw_text: str = ""


class NormalizedDocumentExtraction(BaseModel):
    """Full document extraction - list of page extractions."""
    doc_id: str
    page_count: int
    pages: list[NormalizedPageExtraction]
    extraction_path: str
    engine_name: str
    engine_version: str
    total_processing_time_ms: int = 0