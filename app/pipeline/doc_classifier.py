"""
Document classification - bank statement vs motor finance.
Spec reference: BATON_PASS_BUILD_SPEC Part 18.
Decision D-008: Motor finance Phase 1 minimum viable.
"""

import re
from typing import Optional

from pydantic import BaseModel

from app.models.enums import DocFamily


class ClassificationResult(BaseModel):
    doc_family: str = DocFamily.UNKNOWN.value
    confidence: float = 0.0
    signals: list[str] = []


# Keywords strongly indicating motor finance
MOTOR_FINANCE_KEYWORDS = [
    r"hire\s+purchase",
    r"conditional\s+sale",
    r"personal\s+contract\s+(purchase|plan|hire)",
    r"\bpcp\b",
    r"\bhp\b(?!\s*(sauce|printer))",  # HP but not HP sauce or HP printer
    r"finance\s+agreement",
    r"vehicle\s+registration",
    r"settlement\s+figure",
    r"balloon\s+payment",
    r"guaranteed\s+minimum\s+future\s+value",
    r"optional\s+final\s+payment",
    r"total\s+amount\s+payable",
    r"annual\s+percentage\s+rate",
    r"\bapr\b\s*[\d%]",
    r"motor\s+finance",
    r"vehicle\s+finance",
    r"car\s+finance",
]

# Keywords strongly indicating bank statement
BANK_STATEMENT_KEYWORDS = [
    r"bank\s+statement",
    r"current\s+account",
    r"savings\s+account",
    r"sort\s+code",
    r"account\s+number",
    r"direct\s+debit",
    r"standing\s+order",
    r"faster\s+payment",
    r"\bbacs\b",
    r"\bchaps\b",
    r"overdraft",
    r"brought\s+forward",
    r"carried\s+forward",
    r"opening\s+balance",
    r"closing\s+balance",
]


def classify_document(page_texts: list[str]) -> ClassificationResult:
    """
    Classify document as bank statement, motor finance, or unknown.
    Examines all page text for classification signals.
    """
    combined_text = " ".join(page_texts).lower()
    signals = []

    # Score motor finance keywords
    mf_score = 0.0
    for pattern in MOTOR_FINANCE_KEYWORDS:
        if re.search(pattern, combined_text, re.IGNORECASE):
            mf_score += 0.15
            signals.append(f"MOTOR_FINANCE:{pattern[:30]}")

    # Score bank statement keywords
    bs_score = 0.0
    for pattern in BANK_STATEMENT_KEYWORDS:
        if re.search(pattern, combined_text, re.IGNORECASE):
            bs_score += 0.12
            signals.append(f"BANK_STATEMENT:{pattern[:30]}")

    # Cap scores
    mf_score = min(mf_score, 1.0)
    bs_score = min(bs_score, 1.0)

    # Decision
    if bs_score > mf_score and bs_score >= 0.3:
        return ClassificationResult(
            doc_family=DocFamily.BANK_STATEMENT.value,
            confidence=bs_score,
            signals=signals,
        )
    elif mf_score > bs_score and mf_score >= 0.3:
        return ClassificationResult(
            doc_family=DocFamily.MOTOR_FINANCE.value,
            confidence=mf_score,
            signals=signals,
        )
    else:
        return ClassificationResult(
            doc_family=DocFamily.UNKNOWN.value,
            confidence=max(bs_score, mf_score),
            signals=signals,
        )