"""
Semantic mapping: column role assignment.
Part 13 of the spec.

Assigns meaning to detected columns: DATE, DESCRIPTION, DEBIT, CREDIT, BALANCE, etc.
Two-pass strategy: header-first, statistics-second.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import structlog

from app.pipeline.table_extractor import ColumnRegion, ExtractedRow, CellValue
from app.pipeline.date_parser import is_date_like
from app.pipeline.amount_parser import is_amount_like

logger = structlog.get_logger(__name__)


class ColumnRole(str, Enum):
    DATE = "DATE"
    VALUE_DATE = "VALUE_DATE"
    DESCRIPTION = "DESCRIPTION"
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"
    SINGLE_AMOUNT = "SINGLE_AMOUNT"
    BALANCE = "BALANCE"
    REFERENCE = "REFERENCE"
    TYPE = "TYPE"
    UNKNOWN = "UNKNOWN"


@dataclass
class ColumnStats:
    date_rate: float = 0.0
    amount_rate: float = 0.0
    empty_rate: float = 0.0
    text_rate: float = 0.0
    mean_length: float = 0.0
    unique_ratio: float = 0.0
    monotonic_score: float = 0.0
    position: float = 0.5


# ─── Column Value Extraction ─────────────────────────────────

def _get_column_values(rows: list[ExtractedRow], col_idx: int) -> list[str]:
    """Get all cell values for a specific column across rows."""
    values = []
    for row in rows:
        if row.is_balance_marker:
            continue
        found = False
        for cell in row.cells:
            if cell.column_index == col_idx:
                values.append(cell.text.strip())
                found = True
                break
        if not found:
            values.append("")
    return values


# ─── Column Statistics ────────────────────────────────────────

def _compute_stats(values: list[str], col: ColumnRegion) -> ColumnStats:
    """Compute statistical properties of column values."""
    if not values:
        return ColumnStats(position=(col.x_start + col.x_end) / 2.0)

    total = len(values)
    non_empty = [v for v in values if v]
    empty_count = total - len(non_empty)

    date_count = sum(1 for v in non_empty if is_date_like(v))
    amount_count = sum(1 for v in non_empty if is_amount_like(v))
    text_count = sum(1 for v in non_empty if re.match(r'^[a-zA-Z\s\-\.]+$', v) and len(v) > 3)

    lengths = [len(v) for v in non_empty]
    mean_len = sum(lengths) / len(lengths) if lengths else 0

    unique_count = len(set(non_empty))
    unique_ratio = unique_count / len(non_empty) if non_empty else 0

    # Monotonic score for balance detection
    monotonic_score = 0.0
    numeric_values = []
    for v in non_empty:
        cleaned = re.sub(r'[£$€,\s]', '', v)
        try:
            numeric_values.append(float(cleaned))
        except ValueError:
            pass

    if len(numeric_values) >= 3:
        diffs = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
        if diffs:
            # A balance column changes direction frequently
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
            monotonic_score = sign_changes / len(diffs) if diffs else 0
            # High sign_changes = NOT monotonic (it's a balance that goes up and down)
            # Low sign_changes = monotonic (unlikely to be balance)
            # Invert: for balance detection we want columns that vary
            monotonic_score = 1.0 - monotonic_score if len(diffs) > 2 else 0.0

    n = len(non_empty) if non_empty else 1
    return ColumnStats(
        date_rate=date_count / n,
        amount_rate=amount_count / n,
        empty_rate=empty_count / total,
        text_rate=text_count / n,
        mean_length=mean_len,
        unique_ratio=unique_ratio,
        monotonic_score=monotonic_score,
        position=(col.x_start + col.x_end) / 2.0,
    )


# ─── Header-based Assignment ─────────────────────────────────

HEADER_MAP = {
    ColumnRole.DATE: ["date", "when", "posted"],
    ColumnRole.VALUE_DATE: ["value date", "value"],
    ColumnRole.DESCRIPTION: ["description", "details", "particulars", "narrative", "transaction"],
    ColumnRole.DEBIT: ["debit", "paid out", "money out", "withdrawal", "payments", "dr"],
    ColumnRole.CREDIT: ["credit", "paid in", "money in", "deposit", "receipts", "cr"],
    ColumnRole.SINGLE_AMOUNT: ["amount"],
    ColumnRole.BALANCE: ["balance", "running", "closing"],
    ColumnRole.REFERENCE: ["ref", "reference", "cheque"],
    ColumnRole.TYPE: ["type", "code"],
}


def _match_header(header_text: str) -> Optional[ColumnRole]:
    """Match a header text to a column role. Strips currency prefixes first."""
    h = header_text.lower().strip()
    if not h:
        return None

    # Strip currency prefixes: "(GBP) Amount" -> "amount", "(gbp) balance" -> "balance"
    h = re.sub(r'\(\s*(?:gbp|eur|usd|currency)\s*\)\s*', '', h, flags=re.IGNORECASE).strip()
    # Also strip standalone currency prefix without parens: "GBP Amount" -> "amount"
    h = re.sub(r'^(?:gbp|eur|usd)\s+', '', h, flags=re.IGNORECASE).strip()

    if not h:
        return None

    # Check most specific multi-word matches first
    if "value" in h and "date" in h:
        return ColumnRole.VALUE_DATE
    if "paid out" in h or "money out" in h:
        return ColumnRole.DEBIT
    if "paid in" in h or "money in" in h:
        return ColumnRole.CREDIT

    for role, keywords in HEADER_MAP.items():
        if any(kw in h for kw in keywords):
            return role
    return None


# ─── Main Assignment Function ─────────────────────────────────

def assign_column_roles(
    columns: list[ColumnRegion],
    header_texts: Optional[list[str]],
    sample_rows: list[ExtractedRow],
) -> dict[int, ColumnRole]:
    """
    Assign semantic roles to columns.
    Strategy: header-first, statistics-second, ambiguity resolution third.
    """
    roles: dict[int, ColumnRole] = {}

    # ── PASS 1: Header-based ──
    if header_texts:
        for i, header in enumerate(header_texts):
            if i >= len(columns):
                break
            role = _match_header(header)
            if role:
                # Avoid duplicate DATE — second one becomes VALUE_DATE
                if role == ColumnRole.DATE and ColumnRole.DATE in roles.values():
                    role = ColumnRole.VALUE_DATE
                roles[i] = role

    # ── PASS 2: Statistical for unassigned ──
    for i, col in enumerate(columns):
        if i in roles:
            continue

        values = _get_column_values(sample_rows, i)
        stats = _compute_stats(values, col)

        # Date column
        if stats.date_rate > 0.5 and ColumnRole.DATE not in roles.values():
            roles[i] = ColumnRole.DATE
        elif stats.date_rate > 0.3:
            roles[i] = ColumnRole.VALUE_DATE

        # Amount columns
        elif stats.amount_rate > 0.3:
            if stats.empty_rate > 0.3:
                # High empty = probably debit or credit (not every row has both)
                if ColumnRole.DEBIT not in roles.values():
                    roles[i] = ColumnRole.DEBIT
                elif ColumnRole.CREDIT not in roles.values():
                    roles[i] = ColumnRole.CREDIT
                else:
                    roles[i] = ColumnRole.SINGLE_AMOUNT
            elif stats.monotonic_score > 0.6:
                roles[i] = ColumnRole.BALANCE
            else:
                roles[i] = ColumnRole.SINGLE_AMOUNT

        # Description: long text, high uniqueness
        elif stats.text_rate > 0.4 and stats.mean_length > 8:
            roles[i] = ColumnRole.DESCRIPTION

        # Reference: short, unique, alphanumeric
        elif stats.unique_ratio > 0.7 and stats.mean_length < 12:
            roles[i] = ColumnRole.REFERENCE

        else:
            roles[i] = ColumnRole.UNKNOWN

    # ── PASS 3: Resolve ambiguities ──
    assigned_roles = set(roles.values())

    # If we have SINGLE_AMOUNT but no DEBIT/CREDIT, that's fine (balance solver will infer)
    # If we have nothing that looks like an amount, it's an error
    amount_roles = assigned_roles & {
        ColumnRole.DEBIT, ColumnRole.CREDIT, ColumnRole.SINGLE_AMOUNT, ColumnRole.BALANCE
    }

    if not amount_roles:
        logger.warning("no_amount_column_detected",
                       columns=len(columns),
                       roles=roles)

    # If no DATE found, try the leftmost UNKNOWN column
    if ColumnRole.DATE not in assigned_roles:
        for i in sorted(roles.keys()):
            if roles[i] == ColumnRole.UNKNOWN:
                roles[i] = ColumnRole.DATE
                break

    # If no DESCRIPTION found, assign the widest UNKNOWN column
    if ColumnRole.DESCRIPTION not in set(roles.values()):
        unknown_cols = [(i, columns[i]) for i in roles if roles[i] == ColumnRole.UNKNOWN and i < len(columns)]
        if unknown_cols:
            widest = max(unknown_cols, key=lambda x: x[1].x_end - x[1].x_start)
            roles[widest[0]] = ColumnRole.DESCRIPTION

    # Update column objects
    for i, role in roles.items():
        if i < len(columns):
            columns[i].role = role.value

    logger.debug("column_roles_assigned", roles={i: r.value for i, r in roles.items()})

    return roles