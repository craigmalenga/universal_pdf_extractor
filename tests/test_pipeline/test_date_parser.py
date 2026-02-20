"""
Tests for UK date parser.
"""

from datetime import date

import pytest

from app.pipeline.date_parser import parse_date_uk, is_date_like


class TestParseDateUK:
    """Test UK-first date parsing."""

    def test_dd_mm_yyyy_slash(self):
        result = parse_date_uk("01/02/2024")
        assert result.parsed_date == date(2024, 2, 1)  # UK: day first
        assert result.confidence >= 0.70

    def test_dd_mon_yyyy(self):
        result = parse_date_uk("15 Jan 2024")
        assert result.parsed_date == date(2024, 1, 15)
        assert result.confidence >= 0.90
        assert not result.is_ambiguous

    def test_dd_month_yyyy_full(self):
        result = parse_date_uk("5 February 2024")
        assert result.parsed_date == date(2024, 2, 5)
        assert result.confidence >= 0.90

    def test_iso_format(self):
        result = parse_date_uk("2024-03-15")
        assert result.parsed_date == date(2024, 3, 15)
        assert result.confidence >= 0.90

    def test_dd_mm_yy(self):
        result = parse_date_uk("01/02/24")
        assert result.parsed_date == date(2024, 2, 1)

    def test_ordinal_date(self):
        result = parse_date_uk("1st Jan 2024")
        assert result.parsed_date == date(2024, 1, 1)

    def test_ambiguous_date_flagged(self):
        result = parse_date_uk("05/06/2024")
        # 5 June or June 5 - both day and month <=12
        assert result.parsed_date == date(2024, 6, 5)  # UK default
        assert result.is_ambiguous  # But flagged

    def test_unambiguous_date_not_flagged(self):
        result = parse_date_uk("25/06/2024")
        assert result.parsed_date == date(2024, 6, 25)
        assert not result.is_ambiguous  # 25 can only be day

    def test_unparseable_returns_none(self):
        result = parse_date_uk("not a date")
        assert result.parsed_date is None
        assert result.confidence == 0.0

    def test_empty_string(self):
        result = parse_date_uk("")
        assert result.parsed_date is None

    def test_period_disambiguation(self):
        period_start = date(2024, 1, 1)
        period_end = date(2024, 1, 31)
        result = parse_date_uk("05/01/2024", period_start, period_end)
        assert result.parsed_date == date(2024, 1, 5)
        # Within period should reduce ambiguity
        assert not result.is_ambiguous


class TestIsDateLike:
    """Test quick date pattern check."""

    def test_dd_mm_yyyy(self):
        assert is_date_like("01/02/2024")

    def test_named_month(self):
        assert is_date_like("15 Jan 2024")

    def test_iso(self):
        assert is_date_like("2024-01-15")

    def test_not_date(self):
        assert not is_date_like("hello world")

    def test_empty(self):
        assert not is_date_like("")