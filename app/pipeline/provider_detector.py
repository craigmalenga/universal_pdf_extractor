"""
Provider detection - bank/lender identification.
Spec reference: BATON_PASS_BUILD_SPEC Part 17.
"""

import re
from typing import Optional

from pydantic import BaseModel


class ProviderResult(BaseModel):
    provider_name: Optional[str] = None
    confidence: float = 0.0
    signals: list[str] = []


# Known UK bank/lender patterns
PROVIDER_PATTERNS = {
    "Barclays": [
        r"barclays",
        r"barclays\s+bank",
        r"sort\s+code\s*:\s*20[\-\s]\d{2}[\-\s]\d{2}",
    ],
    "HSBC": [
        r"hsbc",
        r"hsbc\s+uk",
        r"sort\s+code\s*:\s*40[\-\s]\d{2}[\-\s]\d{2}",
    ],
    "Lloyds": [
        r"lloyds",
        r"lloyds\s+bank",
        r"lloyds\s+banking\s+group",
        r"sort\s+code\s*:\s*30[\-\s]\d{2}[\-\s]\d{2}",
    ],
    "NatWest": [
        r"natwest",
        r"national\s+westminster",
        r"sort\s+code\s*:\s*60[\-\s]\d{2}[\-\s]\d{2}",
    ],
    "RBS": [
        r"\brbs\b",
        r"royal\s+bank\s+of\s+scotland",
        r"sort\s+code\s*:\s*83[\-\s]\d{2}[\-\s]\d{2}",
    ],
    "Santander": [
        r"santander",
        r"sort\s+code\s*:\s*09[\-\s]\d{2}[\-\s]\d{2}",
    ],
    "Halifax": [
        r"halifax",
        r"sort\s+code\s*:\s*11[\-\s]\d{2}[\-\s]\d{2}",
    ],
    "Nationwide": [
        r"nationwide",
        r"nationwide\s+building\s+society",
        r"sort\s+code\s*:\s*07[\-\s]\d{2}[\-\s]\d{2}",
    ],
    "TSB": [
        r"\btsb\b",
        r"tsb\s+bank",
    ],
    "Metro Bank": [
        r"metro\s+bank",
        r"sort\s+code\s*:\s*23[\-\s]05[\-\s]\d{2}",
    ],
    "Monzo": [
        r"monzo",
        r"monzo\s+bank",
        r"sort\s+code\s*:\s*04[\-\s]00[\-\s]04",
    ],
    "Starling": [
        r"starling",
        r"starling\s+bank",
        r"sort\s+code\s*:\s*60[\-\s]83[\-\s]71",
    ],
    "Revolut": [
        r"revolut",
    ],
    "Allied Irish": [
        r"allied\s+irish",
        r"\baib\b",
    ],
    "Bank of Ireland": [
        r"bank\s+of\s+ireland",
        r"\bboi\b",
    ],
    "Clydesdale": [
        r"clydesdale",
        r"virgin\s+money",
    ],
    "Co-operative Bank": [
        r"co[\-\s]?operative\s+bank",
        r"the\s+co[\-\s]?op\s+bank",
    ],
}


def detect_provider(page_texts: list[str]) -> ProviderResult:
    """
    Detect the bank/lender from page text content.
    Examines first 2-3 pages for strongest signal.
    """
    combined_text = " ".join(page_texts[:3]).lower()
    best_match: Optional[str] = None
    best_score = 0.0
    all_signals = []

    for provider, patterns in PROVIDER_PATTERNS.items():
        match_count = 0
        signals = []
        for pattern in patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                match_count += 1
                signals.append(f"{provider}:{pattern[:30]}")

        if match_count > 0:
            score = min(match_count * 0.4, 1.0)
            if score > best_score:
                best_score = score
                best_match = provider
                all_signals = signals

    return ProviderResult(
        provider_name=best_match,
        confidence=best_score,
        signals=all_signals,
    )