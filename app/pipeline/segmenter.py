"""
Multi-statement PDF segmentation.
Spec reference: BATON_PASS_BUILD_SPEC Part 10.
Decision D-005: Segmentation before full extraction.
"""

import re
from typing import Optional

from pydantic import BaseModel

from app.schemas.contracts import NormalizedPageExtraction


class SegmentBoundary(BaseModel):
    page_index: int
    confidence: float
    signals: list[str]


# ── Pattern Constants ────────────────────────────────────────

STATEMENT_PERIOD_PATTERNS = [
    r'statement\s+period\s*[:\-]\s*\d',
    r'from\s+\d{1,2}[\s/\-]\w+[\s/\-]\d{2,4}\s+(to|until)',
    r'statement\s+date\s*[:\-]',
    r'period\s+ending\s*[:\-]',
    r'date\s+range\s*[:\-]',
]

OPENING_BALANCE_PATTERNS = [
    r'(opening|brought?\s+forward|b/f)\s+(balance|bal)',
    r'balance\s+(brought|carried)\s+forward',
    r'previous\s+balance',
    r'balance\s+at\s+start',
]

ACCOUNT_HEADER_PATTERNS = [
    r'(account\s+(number|no)|sort\s+code|a/c\s+no)',
    r'\d{2}[\-\s]\d{2}[\-\s]\d{2}\s+\d{6,8}',
]

PAGE_NUMBER_PATTERNS = [
    r'page\s+1\s+of\s+\d+',
    r'page\s+1\b',
]


def detect_segment_boundaries(
    extractions: list[NormalizedPageExtraction],
) -> list[SegmentBoundary]:
    """
    Detect where one statement ends and another begins
    within a multi-statement PDF.

    Uses cheap header scan: examine only the top 15% of each page.
    """
    boundaries = [
        SegmentBoundary(page_index=0, confidence=1.0, signals=["FIRST_PAGE"])
    ]

    for i in range(1, len(extractions)):
        curr = extractions[i]
        score = 0.0
        signals = []

        # Get text from top 15% of page
        top_text = _get_top_region_text(curr, y_threshold=0.15)

        # STRONG SIGNALS (1.0 each)
        if _has_pattern_match(top_text, STATEMENT_PERIOD_PATTERNS):
            score += 1.0
            signals.append("STATEMENT_PERIOD_TEXT")

        if _has_pattern_match(top_text, OPENING_BALANCE_PATTERNS):
            score += 1.0
            signals.append("OPENING_BALANCE_TEXT")

        if _has_pattern_match(top_text, ACCOUNT_HEADER_PATTERNS):
            score += 1.0
            signals.append("ACCOUNT_HEADER_REPEAT")

        # MODERATE SIGNALS (0.4 each)
        if _has_pattern_match(top_text, PAGE_NUMBER_PATTERNS):
            score += 0.4
            signals.append("PAGE_NUMBER_RESET")

        # THRESHOLD
        if score >= 0.8:
            boundaries.append(SegmentBoundary(
                page_index=i,
                confidence=min(score / 2.0, 1.0),
                signals=signals,
            ))

    return boundaries


def build_segments(
    boundaries: list[SegmentBoundary],
    total_pages: int,
) -> list[dict]:
    """Convert boundaries into page ranges."""
    segments = []
    for i, boundary in enumerate(boundaries):
        start_page = boundary.page_index
        end_page = (
            boundaries[i + 1].page_index - 1
            if i + 1 < len(boundaries)
            else total_pages - 1
        )
        segments.append({
            "segment_index": i,
            "start_page": start_page,
            "end_page": end_page,
            "boundary_confidence": boundary.confidence,
            "boundary_signals": boundary.signals,
        })
    return segments


def _get_top_region_text(
    extraction: NormalizedPageExtraction,
    y_threshold: float = 0.15,
) -> str:
    """Extract text from the top region of a page."""
    top_tokens = [
        token.text
        for token in extraction.tokens
        if token.bbox.y0 < y_threshold
    ]
    return " ".join(top_tokens).lower()


def _has_pattern_match(text: str, patterns: list[str]) -> bool:
    """Check if any pattern matches the text."""
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)