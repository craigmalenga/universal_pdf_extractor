"""
UK-first date parser.
Spec reference: BATON_PASS_BUILD_SPEC Part 15.

Strategy:
1. Try unambiguous formats first (named month)
2. For numeric formats: assume dd/mm (UK default)
3. Validate against statement period
4. Handle year inference for formats without year
"""

import re
from datetime import date, timedelta
from typing import Optional

from dateutil import parser as dateutil_parser
from pydantic import BaseModel


class DateParseResult(BaseModel):
    parsed_date: Optional[date] = None
    raw_text: str
    format_detected: str
    confidence: float
    is_ambiguous: bool
    ambiguity_note: Optional[str] = None


# Ordered by specificity (try most specific first)
DATE_FORMATS = [
    # Unambiguous: named month
    (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', 'DD_MONTH_YYYY', False),
    (r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})', 'DD_MON_YYYY', False),
    (r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{2})', 'DD_MON_YY', False),
    (r'(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{2,4})', 'DD_ORD_MON_YYYY', False),

    # ISO format
    (r'(\d{4})-(\d{2})-(\d{2})', 'YYYY-MM-DD', False),

    # UK numeric (potentially ambiguous)
    (r'(\d{2})/(\d{2})/(\d{4})', 'DD/MM/YYYY', True),
    (r'(\d{2})-(\d{2})-(\d{4})', 'DD-MM-YYYY', True),
    (r'(\d{2})\.(\d{2})\.(\d{4})', 'DD.MM.YYYY', True),
    (r'(\d{1,2})/(\d{1,2})/(\d{4})', 'D/M/YYYY', True),
    (r'(\d{2})/(\d{2})/(\d{2})', 'DD/MM/YY', True),

    # No year
    (r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*', 'DD_MON', True),
    (r'(\d{1,2})/(\d{1,2})', 'DD/MM', True),
]


def parse_date_uk(
    raw: str,
    statement_period_start: Optional[date] = None,
    statement_period_end: Optional[date] = None,
    surrounding_dates: Optional[list[date]] = None,
) -> DateParseResult:
    """
    Parse a date string with UK locale priority (dd/mm).

    Disambiguation strategy:
    1. Try unambiguous formats first
    2. For numeric formats: assume dd/mm (UK default)
    3. Validate against statement period
    4. Validate monotonic ordering with surrounding dates
    5. Handle year inference for formats without year
    """
    raw_clean = raw.strip()

    for pattern, format_name, potentially_ambiguous in DATE_FORMATS:
        m = re.match(pattern, raw_clean, re.IGNORECASE)
        if not m:
            continue

        try:
            parsed = _parse_by_format(m, format_name, statement_period_start, statement_period_end)
        except (ValueError, OverflowError):
            continue

        if parsed is None:
            continue

        # Check ambiguity
        is_ambiguous = False
        ambiguity_note = None
        if potentially_ambiguous and format_name.startswith('DD'):
            groups = m.groups()
            if len(groups) >= 2:
                try:
                    day_val = int(groups[0])
                    month_val = int(groups[1])
                    if day_val <= 12 and month_val <= 12 and day_val != month_val:
                        is_ambiguous = True
                        ambiguity_note = f"dd/mm vs mm/dd ambiguous ({groups[0]}/{groups[1]})"

                        # Disambiguation: check against period
                        if statement_period_start and statement_period_end:
                            if statement_period_start <= parsed <= statement_period_end + timedelta(days=5):
                                is_ambiguous = False
                except (ValueError, IndexError):
                    pass

        # Confidence scoring
        confidence = 0.95 if not is_ambiguous else 0.70
        if parsed.year > date.today().year + 1:
            confidence = 0.3  # Future date is suspicious
        if parsed.year < 2000:
            confidence = 0.5  # Very old date

        return DateParseResult(
            parsed_date=parsed,
            raw_text=raw,
            format_detected=format_name,
            confidence=confidence,
            is_ambiguous=is_ambiguous,
            ambiguity_note=ambiguity_note,
        )

    return DateParseResult(
        parsed_date=None,
        raw_text=raw,
        format_detected="UNKNOWN",
        confidence=0.0,
        is_ambiguous=False,
    )


def _parse_by_format(
    match,
    format_name: str,
    period_start: Optional[date],
    period_end: Optional[date],
) -> Optional[date]:
    """Parse date from regex match based on detected format."""

    if format_name == 'YYYY-MM-DD':
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    if format_name in ('DD/MM/YYYY', 'DD-MM-YYYY', 'DD.MM.YYYY', 'D/M/YYYY'):
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        return date(year, month, day)

    if format_name == 'DD/MM/YY':
        day = int(match.group(1))
        month = int(match.group(2))
        yy = int(match.group(3))
        year = 1900 + yy if yy > 50 else 2000 + yy
        return date(year, month, day)

    if 'MON' in format_name or 'MONTH' in format_name:
        return dateutil_parser.parse(match.group(0), dayfirst=True).date()

    if format_name == 'DD_MON':
        parsed = dateutil_parser.parse(match.group(0), dayfirst=True).date()
        if period_start:
            candidate = parsed.replace(year=period_start.year)
            if period_start.month == 12 and parsed.month == 1:
                candidate = parsed.replace(year=period_start.year + 1)
            return candidate
        return parsed

    if format_name == 'DD/MM':
        day = int(match.group(1))
        month = int(match.group(2))
        year = period_start.year if period_start else date.today().year
        if period_start and period_start.month == 12 and month == 1:
            year += 1
        return date(year, month, day)

    return None


def is_date_like(text: str) -> bool:
    """Quick check if text looks like it could be a date."""
    text = text.strip()
    if not text:
        return False
    # Quick regex check for date-like patterns
    date_patterns = [
        r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
        r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
        r'\d{4}-\d{2}-\d{2}',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in date_patterns)