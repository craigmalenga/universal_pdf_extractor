"""
Tests for balance solver.
"""

from decimal import Decimal

import pytest

from app.pipeline.balance_solver import (
    solve_directions,
    solve_case1_separate_columns,
    solve_case3_balance_inference,
    _find_best_tolerance,
    _tolerance_to_confidence,
)


class TestSolveCase1SeparateColumns:
    """Test Case 1: Separate debit/credit columns."""

    def test_debit_populated(self):
        rows = [{"debit_amount": Decimal("50.00"), "credit_amount": None}]
        results = solve_case1_separate_columns(rows)
        assert results[0].direction == "DEBIT"
        assert results[0].confidence >= 0.90

    def test_credit_populated(self):
        rows = [{"debit_amount": None, "credit_amount": Decimal("200.00")}]
        results = solve_case1_separate_columns(rows)
        assert results[0].direction == "CREDIT"
        assert results[0].confidence >= 0.90

    def test_both_populated_unknown(self):
        rows = [{"debit_amount": Decimal("50.00"), "credit_amount": Decimal("50.00")}]
        results = solve_case1_separate_columns(rows)
        assert results[0].direction == "UNKNOWN"
        assert results[0].correction_applied == "BOTH_COLUMNS_POPULATED"

    def test_neither_populated(self):
        rows = [{"debit_amount": None, "credit_amount": None}]
        results = solve_case1_separate_columns(rows)
        assert results[0].direction == "UNKNOWN"


class TestSolveCase3BalanceInference:
    """Test Case 3: Balance inference from running balance."""

    def test_simple_debit(self):
        rows = [{"amount": Decimal("50.00"), "running_balance": Decimal("950.00")}]
        results = solve_case3_balance_inference(rows, opening_balance=Decimal("1000.00"))
        assert results[0].direction == "DEBIT"
        assert results[0].balance_confirmed

    def test_simple_credit(self):
        rows = [{"amount": Decimal("200.00"), "running_balance": Decimal("1200.00")}]
        results = solve_case3_balance_inference(rows, opening_balance=Decimal("1000.00"))
        assert results[0].direction == "CREDIT"
        assert results[0].balance_confirmed

    def test_chain_of_transactions(self):
        rows = [
            {"amount": Decimal("50.00"), "running_balance": Decimal("950.00")},
            {"amount": Decimal("200.00"), "running_balance": Decimal("1150.00")},
            {"amount": Decimal("75.25"), "running_balance": Decimal("1074.75")},
        ]
        results = solve_case3_balance_inference(rows, opening_balance=Decimal("1000.00"))
        assert results[0].direction == "DEBIT"
        assert results[1].direction == "CREDIT"
        assert results[2].direction == "DEBIT"
        assert all(r.balance_confirmed for r in results)

    def test_penny_tolerance(self):
        rows = [{"amount": Decimal("50.00"), "running_balance": Decimal("950.01")}]
        results = solve_case3_balance_inference(rows, opening_balance=Decimal("1000.00"))
        assert results[0].direction == "DEBIT"
        assert results[0].tolerance_used == Decimal("0.01")

    def test_no_opening_balance(self):
        rows = [{"amount": Decimal("50.00"), "running_balance": Decimal("950.00")}]
        results = solve_case3_balance_inference(rows, opening_balance=None)
        assert results[0].direction == "UNKNOWN"


class TestFindBestTolerance:
    """Test tolerance matching."""

    def test_exact_match(self):
        assert _find_best_tolerance(Decimal("100.00"), Decimal("100.00")) == Decimal("0.00")

    def test_penny_diff(self):
        assert _find_best_tolerance(Decimal("100.00"), Decimal("100.01")) == Decimal("0.01")

    def test_no_match(self):
        assert _find_best_tolerance(Decimal("100.00"), Decimal("200.00")) is None


class TestToleranceToConfidence:
    """Test tolerance to confidence mapping."""

    def test_exact(self):
        assert _tolerance_to_confidence(Decimal("0.00")) == 0.98

    def test_penny(self):
        assert _tolerance_to_confidence(Decimal("0.01")) == 0.95

    def test_last_resort(self):
        assert _tolerance_to_confidence(Decimal("1.00")) == 0.60