"""
Tests for UK amount parser.
"""

from decimal import Decimal

import pytest

from app.pipeline.amount_parser import parse_amount_uk, is_amount_like


class TestParseAmountUK:
    """Test UK amount parsing."""

    def test_simple_amount(self):
        result = parse_amount_uk("1234.56")
        assert result.amount == Decimal("1234.56")
        assert not result.is_negative

    def test_with_commas(self):
        result = parse_amount_uk("1,234.56")
        assert result.amount == Decimal("1234.56")

    def test_with_pound_sign(self):
        result = parse_amount_uk(chr(163) + "500.00")
        assert result.amount == Decimal("500.00")

    def test_parentheses_negative(self):
        result = parse_amount_uk("(500.00)")
        assert result.amount == Decimal("-500.00")
        assert result.is_negative
        assert result.sign_convention == "PARENTHESES"

    def test_dr_suffix(self):
        result = parse_amount_uk("100.00 DR")
        assert result.amount == Decimal("-100.00")
        assert result.is_negative
        assert result.sign_convention == "DR_CR"

    def test_cr_suffix(self):
        result = parse_amount_uk("250.00 CR")
        assert result.amount == Decimal("250.00")
        assert not result.is_negative
        assert result.sign_convention == "DR_CR"

    def test_leading_minus(self):
        result = parse_amount_uk("-75.50")
        assert result.amount == Decimal("-75.50")
        assert result.is_negative

    def test_trailing_minus(self):
        result = parse_amount_uk("75.50-")
        assert result.amount == Decimal("-75.50")
        assert result.is_negative

    def test_zero(self):
        result = parse_amount_uk("0.00")
        assert result.amount == Decimal("0.00")

    def test_integer_amount(self):
        result = parse_amount_uk("500")
        assert result.amount == Decimal("500")

    def test_empty_string(self):
        result = parse_amount_uk("")
        assert result.amount is None

    def test_dash_only(self):
        result = parse_amount_uk("-")
        assert result.amount is None

    def test_large_amount(self):
        result = parse_amount_uk("1,234,567.89")
        assert result.amount == Decimal("1234567.89")

    def test_small_penny_amount(self):
        result = parse_amount_uk("0.01")
        assert result.amount == Decimal("0.01")


class TestIsAmountLike:
    """Test quick amount pattern check."""

    def test_simple_number(self):
        assert is_amount_like("1234.56")

    def test_with_pound(self):
        assert is_amount_like(chr(163) + "500")

    def test_with_dr(self):
        assert is_amount_like("100.00 DR")

    def test_parentheses(self):
        assert is_amount_like("(500.00)")

    def test_not_amount(self):
        assert not is_amount_like("hello world")

    def test_empty(self):
        assert not is_amount_like("")