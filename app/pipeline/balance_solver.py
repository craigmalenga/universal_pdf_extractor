"""
Balance solver - direction inference and reconciliation.
Spec reference: BATON_PASS_BUILD_SPEC Part 14.

Decision D-007: NEVER guess direction. Use UNKNOWN if uncertain.

Four cases:
1. Separate debit/credit columns -> direction from column
2. Single amount with sign convention -> direction from sign
3. Single amount + balance column -> infer from balance chain
4. Single amount, no sign, no balance -> header/description hints only
"""

import re
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel

from app.models.enums import DirectionSource


class SolverResult(BaseModel):
    direction: str = "UNKNOWN"  # DEBIT, CREDIT, UNKNOWN
    direction_source: str = "UNKNOWN"  # DirectionSource value
    confidence: float = 0.0
    balance_confirmed: bool = False
    tolerance_used: Decimal = Decimal("0.00")
    correction_applied: Optional[str] = None


# Tolerance ladder for balance matching
TOLERANCES = [
    Decimal("0.00"),    # Exact match
    Decimal("0.01"),    # One penny (common rounding)
    Decimal("0.02"),    # Two pence
    Decimal("0.05"),    # Minor OCR error
    Decimal("1.00"),    # Last-resort poor OCR
]


def solve_directions(
    rows: list[dict],
    opening_balance: Optional[Decimal],
    closing_balance: Optional[Decimal],
    column_roles: dict,
) -> list[SolverResult]:
    """
    Master direction solver.
    Routes to the appropriate strategy based on column structure.
    """
    role_values = [r for r in column_roles.values()]

    has_separate_columns = ("DEBIT" in role_values and "CREDIT" in role_values)
    has_single_amount = "SINGLE_AMOUNT" in role_values
    has_balance = "BALANCE" in role_values

    if has_separate_columns:
        results = solve_case1_separate_columns(rows)
    elif has_single_amount and _rows_have_sign_convention(rows):
        results = solve_case2_signed_amounts(rows)
    elif has_single_amount and has_balance:
        results = solve_case3_balance_inference(rows, opening_balance)
    elif has_single_amount:
        results = solve_case4_no_balance(rows, column_roles)
    else:
        results = [SolverResult(
            direction="UNKNOWN",
            direction_source=DirectionSource.UNKNOWN.value,
            confidence=0.0,
            balance_confirmed=False,
            tolerance_used=Decimal("0.00"),
        )] * len(rows)

    # Validate with balance chain if available
    if has_balance and opening_balance is not None:
        results = _validate_with_balance_chain(rows, results, opening_balance)

    return results


def solve_case1_separate_columns(rows: list[dict]) -> list[SolverResult]:
    """
    Case 1: Separate debit and credit columns.
    Direction determined by which column is populated.
    """
    results = []
    for row in rows:
        debit_val = row.get("debit_amount")
        credit_val = row.get("credit_amount")

        if debit_val is not None and credit_val is not None:
            # EDGE CASE 1a: Both populated - flag for review
            results.append(SolverResult(
                direction="UNKNOWN",
                direction_source=DirectionSource.COLUMN.value,
                confidence=0.3,
                balance_confirmed=False,
                tolerance_used=Decimal("0.00"),
                correction_applied="BOTH_COLUMNS_POPULATED",
            ))
        elif debit_val is not None:
            results.append(SolverResult(
                direction="DEBIT",
                direction_source=DirectionSource.COLUMN.value,
                confidence=0.90,
                balance_confirmed=False,
                tolerance_used=Decimal("0.00"),
            ))
        elif credit_val is not None:
            results.append(SolverResult(
                direction="CREDIT",
                direction_source=DirectionSource.COLUMN.value,
                confidence=0.90,
                balance_confirmed=False,
                tolerance_used=Decimal("0.00"),
            ))
        else:
            # EDGE CASE 1b: Neither populated
            results.append(SolverResult(
                direction="UNKNOWN",
                direction_source=DirectionSource.COLUMN.value,
                confidence=0.2,
                balance_confirmed=False,
                tolerance_used=Decimal("0.00"),
                correction_applied="NO_AMOUNT_IN_EITHER_COLUMN",
            ))

    return results


def solve_case2_signed_amounts(rows: list[dict]) -> list[SolverResult]:
    """
    Case 2: Single amount column with sign convention.
    Parse sign, parentheses, DR/CR suffix.
    """
    results = []
    for row in rows:
        amount_raw = row.get("amount_raw", "")
        parsed = _parse_signed_amount(amount_raw)

        if parsed is None:
            results.append(SolverResult(
                direction="UNKNOWN",
                direction_source=DirectionSource.SIGN.value,
                confidence=0.0,
                balance_confirmed=False,
                tolerance_used=Decimal("0.00"),
            ))
            continue

        amount_value, sign_source = parsed

        if amount_value < 0:
            direction = "DEBIT"
        elif amount_value > 0:
            direction = "CREDIT"
        else:
            direction = "UNKNOWN"

        results.append(SolverResult(
            direction=direction,
            direction_source=DirectionSource.SIGN.value,
            confidence=0.85 if sign_source in ("PARENTHESES", "DR_CR") else 0.75,
            balance_confirmed=False,
            tolerance_used=Decimal("0.00"),
        ))

    return results


def solve_case3_balance_inference(
    rows: list[dict],
    opening_balance: Optional[Decimal],
) -> list[SolverResult]:
    """
    Case 3: Single amount column, no sign, but balance column available.
    Use sequential balance fitting.
    """
    results = []
    current_balance = opening_balance

    for row in rows:
        amount = row.get("amount")
        reported_balance = row.get("running_balance")

        if amount is None or current_balance is None:
            results.append(SolverResult(
                direction="UNKNOWN",
                direction_source=DirectionSource.UNKNOWN.value,
                confidence=0.0,
                balance_confirmed=False,
                tolerance_used=Decimal("0.00"),
            ))
            if reported_balance is not None:
                current_balance = reported_balance
            continue

        # Test both hypotheses
        hypothesis_debit = current_balance - amount
        hypothesis_credit = current_balance + amount

        if reported_balance is not None:
            debit_match = _find_best_tolerance(hypothesis_debit, reported_balance)
            credit_match = _find_best_tolerance(hypothesis_credit, reported_balance)

            if debit_match is not None and credit_match is None:
                direction = "DEBIT"
                tolerance = debit_match
                confidence = _tolerance_to_confidence(tolerance)
            elif credit_match is not None and debit_match is None:
                direction = "CREDIT"
                tolerance = credit_match
                confidence = _tolerance_to_confidence(tolerance)
            elif debit_match is not None and credit_match is not None:
                # Both match (rare - only if amount == 0)
                direction = "UNKNOWN"
                tolerance = min(debit_match, credit_match)
                confidence = 0.3
            else:
                # Neither matches - try OCR correction
                direction, tolerance, confidence = _attempt_balance_correction(
                    current_balance, amount, reported_balance
                )

            results.append(SolverResult(
                direction=direction,
                direction_source=DirectionSource.BALANCE_SOLVER.value,
                confidence=confidence,
                balance_confirmed=(direction != "UNKNOWN"),
                tolerance_used=tolerance or Decimal("0.00"),
            ))

            if reported_balance is not None:
                current_balance = reported_balance
        else:
            results.append(SolverResult(
                direction="UNKNOWN",
                direction_source=DirectionSource.BALANCE_SOLVER.value,
                confidence=0.2,
                balance_confirmed=False,
                tolerance_used=Decimal("0.00"),
            ))

    return results


def solve_case4_no_balance(
    rows: list[dict],
    column_roles: dict,
) -> list[SolverResult]:
    """
    Case 4: Single amount column, no sign, no balance.
    Fall back to header keywords and column position.
    """
    results = []
    for row in rows:
        desc = row.get("description_raw", "").upper()
        if " DR" in desc or "DEBIT" in desc:
            direction = "DEBIT"
            source = DirectionSource.HEADER.value
            confidence = 0.6
        elif " CR" in desc or "CREDIT" in desc:
            direction = "CREDIT"
            source = DirectionSource.HEADER.value
            confidence = 0.6
        else:
            direction = "UNKNOWN"
            source = DirectionSource.UNKNOWN.value
            confidence = 0.0

        results.append(SolverResult(
            direction=direction,
            direction_source=source,
            confidence=confidence,
            balance_confirmed=False,
            tolerance_used=Decimal("0.00"),
        ))

    return results


# ── Helper Functions ─────────────────────────────────────────

def _find_best_tolerance(computed: Decimal, reported: Decimal) -> Optional[Decimal]:
    """Find the tightest tolerance that matches."""
    diff = abs(computed - reported)
    for tolerance in TOLERANCES:
        if diff <= tolerance:
            return tolerance
    return None


def _tolerance_to_confidence(tolerance: Decimal) -> float:
    """Map tolerance to confidence score."""
    mapping = {
        Decimal("0.00"): 0.98,
        Decimal("0.01"): 0.95,
        Decimal("0.02"): 0.90,
        Decimal("0.05"): 0.80,
        Decimal("1.00"): 0.60,
    }
    return mapping.get(tolerance, 0.50)


def _attempt_balance_correction(
    prev_balance: Decimal,
    amount: Decimal,
    reported_balance: Decimal,
) -> tuple:
    """
    Try common OCR single-digit corrections on the reported balance.
    Common OCR errors: 0<->O, 1<->l/I, 5<->S, 8<->B
    """
    reported_str = str(reported_balance)

    OCR_SUBSTITUTIONS = {
        "0": ["O", "o", "Q"],
        "1": ["l", "I", "7"],
        "5": ["S", "s"],
        "8": ["B", "3"],
        "6": ["G", "b"],
        "9": ["g", "q"],
    }

    for i, char in enumerate(reported_str):
        for digit, subs in OCR_SUBSTITUTIONS.items():
            if char in subs:
                corrected_str = reported_str[:i] + digit + reported_str[i + 1:]
                try:
                    corrected = Decimal(corrected_str)
                    hypothesis_debit = prev_balance - amount
                    hypothesis_credit = prev_balance + amount
                    if abs(hypothesis_debit - corrected) <= Decimal("0.01"):
                        return ("DEBIT", Decimal("0.01"), 0.70)
                    if abs(hypothesis_credit - corrected) <= Decimal("0.01"):
                        return ("CREDIT", Decimal("0.01"), 0.70)
                except (InvalidOperation, ValueError):
                    continue

    return ("UNKNOWN", None, 0.0)


def _parse_signed_amount(raw: str) -> Optional[tuple]:
    """Parse a signed amount from various UK conventions."""
    s = raw.strip().replace(chr(163), "").replace(",", "").replace(" ", "")

    # Parentheses
    m = re.match(r'^\((\d+\.?\d*)\)$', s)
    if m:
        return (Decimal(m.group(1)) * Decimal("-1"), "PARENTHESES")

    # DR/CR suffix
    m = re.match(r'^(\d+\.?\d*)\s*(DR|CR)$', s, re.IGNORECASE)
    if m:
        val = Decimal(m.group(1))
        if m.group(2).upper() == "DR":
            return (val * Decimal("-1"), "DR_CR")
        else:
            return (val, "DR_CR")

    # Trailing minus
    m = re.match(r'^(\d+\.?\d*)\-$', s)
    if m:
        return (Decimal(m.group(1)) * Decimal("-1"), "TRAILING_MINUS")

    # Leading minus
    m = re.match(r'^[\-\u2212](\d+\.?\d*)$', s)
    if m:
        return (Decimal(m.group(1)) * Decimal("-1"), "LEADING_MINUS")

    # Positive number
    m = re.match(r'^(\d+\.?\d*)$', s)
    if m:
        return (Decimal(m.group(1)), "UNSIGNED")

    return None


def _rows_have_sign_convention(rows: list[dict]) -> bool:
    """Check if rows use a sign convention (parentheses, DR/CR, etc)."""
    sign_count = 0
    for row in rows:
        raw = row.get("amount_raw", "")
        if re.search(r'[\(\)]|DR|CR|\-$|^\-', raw, re.IGNORECASE):
            sign_count += 1
    return sign_count > len(rows) * 0.3


def _validate_with_balance_chain(
    rows: list[dict],
    results: list[SolverResult],
    opening_balance: Decimal,
) -> list[SolverResult]:
    """
    Walk the balance chain and confirm/upgrade confidence where
    the running balance matches.
    """
    current = opening_balance

    for i, (row, result) in enumerate(zip(rows, results)):
        amount = row.get("amount")
        reported_balance = row.get("running_balance")

        if amount is None or result.direction == "UNKNOWN":
            if reported_balance is not None:
                current = reported_balance
            continue

        # Compute expected balance
        if result.direction == "DEBIT":
            expected = current - amount
        elif result.direction == "CREDIT":
            expected = current + amount
        else:
            if reported_balance is not None:
                current = reported_balance
            continue

        if reported_balance is not None:
            tolerance = _find_best_tolerance(expected, reported_balance)
            if tolerance is not None:
                result.balance_confirmed = True
                result.tolerance_used = tolerance
                # Upgrade confidence if balance confirms
                result.confidence = max(result.confidence, _tolerance_to_confidence(tolerance))
            current = reported_balance
        else:
            current = expected

    return results