"""
Shared test fixtures.
"""

import pytest


@pytest.fixture
def sample_date_strings():
    """Common UK date string samples for testing."""
    return [
        ("01/02/2024", "2024-02-01"),      # DD/MM/YYYY UK
        ("15 Jan 2024", "2024-01-15"),       # DD MON YYYY
        ("5 February 2024", "2024-02-05"),   # D MONTH YYYY
        ("2024-03-15", "2024-03-15"),        # ISO
        ("01/02/24", "2024-02-01"),          # DD/MM/YY
        ("1st Jan 2024", "2024-01-01"),      # Ordinal
    ]


@pytest.fixture
def sample_amounts():
    """Common UK amount string samples for testing."""
    return [
        ("1,234.56", "1234.56", False),
        ("(500.00)", "-500.00", True),
        ("100.00 DR", "-100.00", True),
        ("250.00 CR", "250.00", False),
        ("-75.50", "-75.50", True),
        ("75.50-", "-75.50", True),
        ("0.01", "0.01", False),
        ("10000", "10000", False),
    ]


@pytest.fixture
def sample_balance_chain():
    """Sample balance chain for solver testing."""
    return {
        "opening_balance": "1000.00",
        "rows": [
            {"amount": "50.00", "running_balance": "950.00", "expected_direction": "DEBIT"},
            {"amount": "200.00", "running_balance": "1150.00", "expected_direction": "CREDIT"},
            {"amount": "75.25", "running_balance": "1074.75", "expected_direction": "DEBIT"},
        ],
    }