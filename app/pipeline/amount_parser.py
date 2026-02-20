"""
UK amount parser.
Spec reference: BATON_PASS_BUILD_SPEC Part 16.

Handles all UK bank statement amount conventions:
- GBP1,234.56 / 1,234.56 / 1234.56
- (1,234.56)        -> negative (parentheses)
- 1,234.56 DR       -> negative (DR/CR suffix)
- 1,234.56 CR       -> positive
- -1,234.56         -> negative (leading minus)
- 1,234.56-         -> negative (trailing minus)
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Optional

from pydantic import BaseModel


class AmountParseResult(BaseModel):
    amount: Optional[Decimal] = None
    raw_text: str
    is_negative: bool = False
    sign_convention: Optional[str] = None  # PARENTHESES, DR_CR, MINUS, NONE
    confidence: float = 0.0


def parse_amount_uk(raw: str) -> AmountParseResult:
    """
    Parse a monetary amount from UK bank statement conventions.
    """
    s = raw.strip()

    if not s or s in ('-', '--', '---'):
        return AmountParseResult(amount=None, raw_text=raw, confidence=0.0)

    # Remove currency symbols
    s = s.replace('GBP', '').replace('gbp', '')
    s = s.replace('$', '').replace('EUR', '').replace('eur', '')

    # Handle pound sign - both regular and unicode variants
    for char in ('\\u00a3', '\\u20a4'):
        s = s.replace(char, '')
    s = s.replace(chr(163), '')  # pound sign

    s = s.strip()

    if not s:
        return AmountParseResult(amount=None, raw_text=raw, confidence=0.0)

    # Detect sign convention
    is_negative = False
    sign_convention = 'NONE'

    # Parentheses: (100.00) -> negative
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1].strip()
        is_negative = True
        sign_convention = 'PARENTHESES'

    # DR/CR suffix: 100.00DR -> negative
    m = re.match(r'^(.+?)\s*(DR|CR|D|C)$', s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
        suffix = m.group(2).upper()
        if suffix in ('DR', 'D'):
            is_negative = True
            sign_convention = 'DR_CR'
        elif suffix in ('CR', 'C'):
            is_negative = False
            sign_convention = 'DR_CR'

    # Trailing minus: 100.00-
    if not is_negative and s.endswith('-'):
        s = s[:-1].strip()
        is_negative = True
        sign_convention = 'MINUS'

    # Leading minus: -100.00
    if not is_negative and (s.startswith('-') or s.startswith(chr(8722))):
        s = s[1:].strip()
        is_negative = True
        sign_convention = 'MINUS'

    # Remove thousand separators (commas in UK)
    s = s.replace(',', '')

    # Remove any remaining spaces
    s = s.replace(' ', '')

    # Parse the numeric value
    try:
        amount = Decimal(s)
        if is_negative:
            amount = amount * Decimal('-1')

        # Confidence based on parse quality
        confidence = 0.95
        if sign_convention == 'PARENTHESES':
            confidence = 0.95
        elif sign_convention == 'DR_CR':
            confidence = 0.90
        elif sign_convention == 'MINUS':
            confidence = 0.90

        # Sanity checks
        abs_amount = abs(amount)
        if abs_amount > Decimal('10000000'):
            confidence = 0.5  # Suspiciously large
        if abs_amount == Decimal('0'):
            confidence = 0.80  # Zero amount - plausible but unusual

        return AmountParseResult(
            amount=amount,
            raw_text=raw,
            is_negative=is_negative,
            sign_convention=sign_convention,
            confidence=confidence,
        )

    except (InvalidOperation, ValueError):
        return AmountParseResult(
            amount=None,
            raw_text=raw,
            confidence=0.0,
        )


def is_amount_like(text: str) -> bool:
    """Quick check if text looks like it could be a monetary amount."""
    text = text.strip()
    if not text:
        return False
    # Remove currency symbols for check
    cleaned = text.replace(chr(163), '').replace('GBP', '').replace(',', '').strip()
    cleaned = re.sub(r'\s*(DR|CR|D|C)$', '', cleaned, flags=re.IGNORECASE).strip()
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith('-') or cleaned.startswith(chr(8722)):
        cleaned = cleaned[1:].strip()
    if cleaned.endswith('-'):
        cleaned = cleaned[:-1].strip()
    # Should be a number
    try:
        Decimal(cleaned.replace(',', '').replace(' ', ''))
        return True
    except (InvalidOperation, ValueError):
        return False