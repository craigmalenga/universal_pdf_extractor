"""
Confidence scoring - weighted model with hard gates.
Spec reference: BATON_PASS_BUILD_SPEC Part 20.
Decision D-006: Hard gates override weighted scoring for safety.
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel

from app.config import settings


class ConfidenceResult(BaseModel):
    """Confidence scores for a document/segment."""
    document_confidence: float = 0.0
    reconciliation_rate: float = 0.0
    validation_status: str = "NEEDS_REVIEW"  # PASS, PASS_WITH_WARNINGS, NEEDS_REVIEW, FAIL
    hard_gate_failures: list[str] = []
    warnings: list[str] = []
    field_scores: dict = {}


# ── Weights for document-level confidence ────────────────────
DOCUMENT_WEIGHTS = {
    "reconciliation_rate": 0.35,
    "mean_balance_confidence": 0.25,
    "mean_direction_confidence": 0.20,
    "mean_amount_confidence": 0.10,
    "mean_date_confidence": 0.10,
}


def score_document(
    transactions: list[dict],
    segment_reconciliation_status: Optional[str] = None,
    opening_balance: Optional[Decimal] = None,
    closing_balance: Optional[Decimal] = None,
) -> ConfidenceResult:
    """
    Score document confidence with weighted model and hard gates.
    """
    if not transactions:
        return ConfidenceResult(
            document_confidence=0.0,
            reconciliation_rate=0.0,
            validation_status="FAIL",
            hard_gate_failures=["NO_TRANSACTIONS"],
        )

    # ── Calculate field averages ─────────────────────────────
    n = len(transactions)
    mean_amount = sum(t.get("confidence_amount", 0.0) for t in transactions) / n
    mean_direction = sum(t.get("confidence_direction", 0.0) for t in transactions) / n
    mean_date = sum(t.get("confidence_date", 0.0) for t in transactions) / n
    mean_balance = sum(t.get("confidence_balance", 0.0) for t in transactions) / n

    # Reconciliation rate: % of transactions with confirmed balance
    confirmed = sum(1 for t in transactions if t.get("balance_confirmed", False))
    recon_rate = confirmed / n if n > 0 else 0.0

    # ── Weighted score ───────────────────────────────────────
    weighted = (
        DOCUMENT_WEIGHTS["reconciliation_rate"] * recon_rate
        + DOCUMENT_WEIGHTS["mean_balance_confidence"] * mean_balance
        + DOCUMENT_WEIGHTS["mean_direction_confidence"] * mean_direction
        + DOCUMENT_WEIGHTS["mean_amount_confidence"] * mean_amount
        + DOCUMENT_WEIGHTS["mean_date_confidence"] * mean_date
    )

    # ── Hard gates (Decision D-006) ──────────────────────────
    hard_gate_failures = []
    warnings = []

    # Gate 1: Zero transactions extracted
    if n == 0:
        hard_gate_failures.append("HARD_GATE_NO_TRANSACTIONS")

    # Gate 2: All directions UNKNOWN
    unknown_count = sum(1 for t in transactions if t.get("direction") == "UNKNOWN")
    if unknown_count == n:
        hard_gate_failures.append("HARD_GATE_ALL_DIRECTIONS_UNKNOWN")

    # Gate 3: Reconciliation rate below threshold
    if recon_rate < 0.5 and n > 5:
        hard_gate_failures.append("HARD_GATE_LOW_RECONCILIATION")

    # Gate 4: Mean amount confidence too low
    if mean_amount < 0.5:
        hard_gate_failures.append("HARD_GATE_LOW_AMOUNT_CONFIDENCE")

    # Gate 5: Opening/closing balance mismatch
    if opening_balance is not None and closing_balance is not None:
        total_debits = sum(
            abs(t.get("amount", Decimal("0")))
            for t in transactions
            if t.get("direction") == "DEBIT" and t.get("amount") is not None
        )
        total_credits = sum(
            abs(t.get("amount", Decimal("0")))
            for t in transactions
            if t.get("direction") == "CREDIT" and t.get("amount") is not None
        )
        expected_closing = opening_balance + total_credits - total_debits
        balance_diff = abs(expected_closing - closing_balance)
        if balance_diff > Decimal("5.00"):
            hard_gate_failures.append(f"HARD_GATE_BALANCE_MISMATCH_{balance_diff}")

    # ── Warnings ─────────────────────────────────────────────
    if unknown_count > 0 and unknown_count < n:
        warnings.append(f"WARN_{unknown_count}_UNKNOWN_DIRECTIONS")

    if mean_date < 0.7:
        warnings.append("WARN_LOW_DATE_CONFIDENCE")

    if recon_rate < 0.8 and recon_rate >= 0.5:
        warnings.append("WARN_MODERATE_RECONCILIATION")

    # ── Determine validation status ──────────────────────────
    if hard_gate_failures:
        if any("BALANCE_MISMATCH" in g for g in hard_gate_failures):
            validation_status = "NEEDS_REVIEW"
        else:
            validation_status = "FAIL"
    elif weighted >= settings.CONFIDENCE_PASS_THRESHOLD and not warnings:
        validation_status = "PASS"
    elif weighted >= settings.CONFIDENCE_WARN_THRESHOLD:
        validation_status = "PASS_WITH_WARNINGS"
    elif weighted >= settings.CONFIDENCE_FAIL_THRESHOLD:
        validation_status = "NEEDS_REVIEW"
    else:
        validation_status = "FAIL"

    return ConfidenceResult(
        document_confidence=round(weighted, 4),
        reconciliation_rate=round(recon_rate, 4),
        validation_status=validation_status,
        hard_gate_failures=hard_gate_failures,
        warnings=warnings,
        field_scores={
            "mean_amount_confidence": round(mean_amount, 4),
            "mean_direction_confidence": round(mean_direction, 4),
            "mean_date_confidence": round(mean_date, 4),
            "mean_balance_confidence": round(mean_balance, 4),
            "reconciliation_rate": round(recon_rate, 4),
        },
    )