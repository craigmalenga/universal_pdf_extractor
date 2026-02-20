"""
Table extraction: column detection, row reconstruction, cell assignment.
Part 12 of the spec.

Works on NormalizedPageExtraction tokens/lines to find table structure.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

from app.schemas.contracts import ExtractedLine, ExtractedToken, BBox
from app.pipeline.date_parser import is_date_like
from app.pipeline.amount_parser import is_amount_like

logger = structlog.get_logger(__name__)


# ─── Data Structures ─────────────────────────────────────────

@dataclass
class ColumnRegion:
    column_index: int
    x_start: float
    x_end: float
    role: str = "UNKNOWN"  # Assigned later by semantic mapper


@dataclass
class CellValue:
    text: str
    column_index: int
    bbox: Optional[BBox] = None
    confidence: float = 0.0


@dataclass
class ExtractedRow:
    line_indices: list[int] = field(default_factory=list)
    cells: list[CellValue] = field(default_factory=list)
    is_balance_marker: bool = False
    raw_text: str = ""


# ─── Balance Marker Detection ────────────────────────────────

BALANCE_MARKER_PATTERNS = [
    r"(balance\s+)?(carried|brought)\s+(forward|fwd|f/?wd)",
    r"\bb/?f\b",
    r"\bc/?f\b",
    r"balance\s+(at|on)\s+(start|end|close)",
    r"(opening|closing)\s+balance",
    r"total\s+balance\s+(carried|brought)",
    r"continued\s+(on|over)",
    r"statement\s+continued",
]

# Patterns for summary/header rows that should NOT be treated as transactions
SUMMARY_ROW_PATTERNS = [
    r"personal\s+account\s*(balance|statement)",
    r"(total|net)\s+(balance|outgoings|deposits|income|payments|in|out)",
    r"balance\s+in\s+pots?",
    r"(including|excluding)\s+(all\s+)?pots?",
    r"(regular|savings)\s+pots?\s+(with|provided)",
    r"sort\s*code",
    r"account\s*number",
    r"\biban\b",
    r"\bbic\b",
    r"\bswift\b",
    r"statement\s+period",
    r"(from|to)\s+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
    r"(financial\s+services|compensation\s+scheme|fscs)",
    r"(authorised|regulated)\s+by",
    r"registered\s+(office|in\s+england)",
    r"company\s+(registered|number|no)",
    r"monzo\s+bank\s+limited",
    r"pot\s+(type|name|balance|statement)",
    r"this\s+pot\s+was\s+(closed|opened)",
    r"(important\s+information|compensation\s+arrangements)",
    r"(page|sheet)\s+\d+\s+(of|/)\s+\d+",
]


def is_balance_marker(text: str) -> bool:
    """Detect carried/brought forward balance markers."""
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in BALANCE_MARKER_PATTERNS)


def is_summary_row(text: str) -> bool:
    """Detect summary/header rows that should not be treated as transactions."""
    text_lower = text.lower().strip()
    if not text_lower:
        return False
    # Check balance markers first
    if is_balance_marker(text_lower):
        return True
    # Check summary patterns
    return any(re.search(p, text_lower) for p in SUMMARY_ROW_PATTERNS)


# ─── Column Detection ────────────────────────────────────────

def detect_columns(
    lines: list[ExtractedLine],
    min_column_occupancy: float = 0.08,
    n_bins: int = 120,
) -> list[ColumnRegion]:
    """
    Detect column boundaries from token x-coordinates using histogram analysis.
    Uses a low occupancy threshold to catch sparse amount columns.
    Returns list of ColumnRegion sorted left-to-right.
    """
    if not lines:
        return []

    # Collect all token x0 positions
    x_positions = []
    for line in lines:
        for token in line.tokens:
            x_positions.append(token.bbox.x0)

    if len(x_positions) < 5:
        return []

    x_arr = np.array(x_positions)

    # Build histogram
    hist, bin_edges = np.histogram(x_arr, bins=n_bins, range=(0.0, 1.0))

    # Smooth to merge nearby peaks
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(hist.astype(float), sigma=1.5)

    # Find peaks - use progressively lower thresholds
    from scipy.signal import find_peaks
    peaks = []
    for occupancy in [min_column_occupancy, 0.05, 0.03]:
        threshold = max(len(lines) * occupancy, 2.0)
        peaks, _ = find_peaks(smoothed, height=threshold, distance=4)
        if len(peaks) >= 3:  # Bank statements need at least 3 columns (date, desc, amount)
            break

    if len(peaks) == 0:
        # Fallback: single column spanning entire width
        return [ColumnRegion(column_index=0, x_start=0.0, x_end=1.0)]

    # Build column regions from peaks
    columns = []
    for i, peak in enumerate(peaks):
        x_start = bin_edges[peak]
        if i + 1 < len(peaks):
            # Column ends at midpoint between this peak and next
            next_peak = peaks[i + 1]
            x_end = (bin_edges[peak] + bin_edges[next_peak]) / 2.0
        else:
            x_end = 1.0

        columns.append(ColumnRegion(
            column_index=i,
            x_start=max(0.0, x_start - 0.01),  # Small margin
            x_end=min(1.0, x_end),
        ))

    # Adjust first column to start at 0
    if columns:
        columns[0].x_start = 0.0

    logger.info("columns_detected", count=len(columns),
                positions=[(round(c.x_start, 3), round(c.x_end, 3)) for c in columns])

    return columns


# ─── Token to Column Assignment ──────────────────────────────

def assign_token_to_column(token: ExtractedToken, columns: list[ColumnRegion]) -> int:
    """Find which column a token belongs to based on its x-center."""
    if not columns:
        return 0
    x_center = (token.bbox.x0 + token.bbox.x1) / 2.0
    for col in columns:
        if col.x_start <= x_center <= col.x_end:
            return col.column_index
    # Fallback: nearest column
    distances = [abs(x_center - (c.x_start + c.x_end) / 2.0) for c in columns]
    return columns[distances.index(min(distances))].column_index


def assign_line_to_cells(line: ExtractedLine, columns: list[ColumnRegion]) -> list[CellValue]:
    """Assign tokens in a line to column cells."""
    cell_tokens: dict[int, list[ExtractedToken]] = {}
    for token in line.tokens:
        col_idx = assign_token_to_column(token, columns)
        cell_tokens.setdefault(col_idx, []).append(token)

    cells = []
    for col_idx in sorted(cell_tokens.keys()):
        tokens = sorted(cell_tokens[col_idx], key=lambda t: t.bbox.x0)
        text = " ".join(t.text for t in tokens)
        mean_conf = sum(t.confidence for t in tokens) / len(tokens)
        bbox = BBox(
            x0=min(t.bbox.x0 for t in tokens),
            y0=min(t.bbox.y0 for t in tokens),
            x1=max(t.bbox.x1 for t in tokens),
            y1=max(t.bbox.y1 for t in tokens),
        )
        cells.append(CellValue(text=text, column_index=col_idx, bbox=bbox, confidence=mean_conf))

    return cells


# ─── Row Reconstruction ──────────────────────────────────────

def _has_date_in_column(cells: list[CellValue], date_col_idx: int) -> bool:
    """Check if any cell in the date column contains a date-like value."""
    for cell in cells:
        if cell.column_index == date_col_idx and is_date_like(cell.text):
            return True
    return False


def _has_amount_in_columns(cells: list[CellValue], amount_col_indices: list[int]) -> bool:
    """Check if any cell in the amount columns contains an amount-like value."""
    for cell in cells:
        if cell.column_index in amount_col_indices and is_amount_like(cell.text):
            return True
    return False


def _is_continuation(prev_line: ExtractedLine, curr_line: ExtractedLine) -> bool:
    """Check if current line is a continuation of previous (close vertically)."""
    gap = curr_line.bbox.y0 - prev_line.bbox.y1
    typical_height = prev_line.bbox.y1 - prev_line.bbox.y0
    if typical_height <= 0:
        return gap < 0.02  # ~2% of page
    return gap <= typical_height * 1.8


def reconstruct_rows(
    lines: list[ExtractedLine],
    columns: list[ColumnRegion],
    date_column_index: int = 0,
    amount_column_indices: Optional[list[int]] = None,
) -> list[ExtractedRow]:
    """
    Reconstruct transaction rows from text lines.

    Rules:
    1. New row starts when date token appears in date column
    2. Amount on next line with no date → merge into current row
    3. No date + no amount → description continuation
    4. Balance markers → standalone row
    """
    if not lines or not columns:
        return []

    if amount_column_indices is None:
        # Default: assume last 1-3 columns are amounts
        amount_column_indices = [c.column_index for c in columns if c.column_index > 0]

    rows: list[ExtractedRow] = []
    current_row: Optional[ExtractedRow] = None

    for i, line in enumerate(lines):
        cells = assign_line_to_cells(line, columns)

        # Check for balance marker
        if is_balance_marker(line.text):
            if current_row:
                rows.append(current_row)
                current_row = None
            rows.append(ExtractedRow(
                line_indices=[i],
                cells=cells,
                is_balance_marker=True,
                raw_text=line.text,
            ))
            continue

        has_date = _has_date_in_column(cells, date_column_index)
        has_amount = _has_amount_in_columns(cells, amount_column_indices)

        if has_date:
            # Start new row
            if current_row:
                rows.append(current_row)
            current_row = ExtractedRow(
                line_indices=[i],
                cells=cells,
                raw_text=line.text,
            )
        elif has_amount and not has_date and current_row:
            # Amount on continuation line — merge
            current_row.line_indices.append(i)
            current_row.cells.extend(cells)
            current_row.raw_text += " " + line.text
        elif current_row:
            # Description continuation — check vertical proximity
            prev_line = lines[current_row.line_indices[-1]]
            if _is_continuation(prev_line, line):
                current_row.line_indices.append(i)
                current_row.cells.extend(cells)
                current_row.raw_text += " " + line.text
            else:
                rows.append(current_row)
                current_row = None
        # else: orphan line (header/footer/noise) → skip

    if current_row:
        rows.append(current_row)

    logger.debug("rows_reconstructed",
                 total=len(rows),
                 transactions=sum(1 for r in rows if not r.is_balance_marker),
                 balance_markers=sum(1 for r in rows if r.is_balance_marker))

    return rows


# ─── Header Detection ────────────────────────────────────────

HEADER_KEYWORDS = {
    "date", "description", "details", "particulars", "narrative",
    "debit", "credit", "paid out", "paid in", "money out", "money in",
    "withdrawal", "deposit", "balance", "amount", "reference", "type",
    "dr", "cr", "running balance", "closing balance", "transaction",
}


def detect_header_line(lines: list[ExtractedLine], max_lines: int = 10) -> Optional[int]:
    """
    Find the header line of the transaction table.
    Returns the line index, or None if not found.
    """
    for i, line in enumerate(lines[:max_lines]):
        text_lower = line.text.lower()
        matches = sum(1 for kw in HEADER_KEYWORDS if kw in text_lower)
        if matches >= 2:
            return i
    return None


def extract_header_texts(line: ExtractedLine, columns: list[ColumnRegion]) -> list[str]:
    """Extract header text per column from the header line."""
    cells = assign_line_to_cells(line, columns)
    header = [""] * len(columns)
    for cell in cells:
        if cell.column_index < len(header):
            header[cell.column_index] = cell.text.strip()
    return header