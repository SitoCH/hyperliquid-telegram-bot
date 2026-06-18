import pytest
from unittest.mock import patch, MagicMock
import os
import time
import asyncio
from utils import fmt_price, fmt, log_execution_time, OPERATION_CANCELLED


class TestFmtPrice:
    def test_above_1000(self):
        assert fmt_price(1500) == "1,500"

    def test_above_100(self):
        assert fmt_price(150.5) == "150.50"

    def test_above_1(self):
        assert fmt_price(1.2345) == "1.234"

    def test_above_0_1(self):
        assert fmt_price(0.12345) == "0.1235"

    def test_above_0_01(self):
        assert fmt_price(0.012345) == "0.01235"

    def test_above_0_001(self):
        assert fmt_price(0.0012345) == "0.001234"

    def test_below_0_001(self):
        assert fmt_price(0.00012345) == "0.0001234"

    def test_exact_boundaries(self):
        assert fmt_price(1000) == "1,000.00"
        assert fmt_price(100) == "100.000"
        assert fmt_price(1) == "1.0000"
        assert fmt_price(0.1) == "0.10000"
        assert fmt_price(0.01) == "0.010000"
        assert fmt_price(0.001) == "0.0010000"

    def test_zero(self):
        assert fmt_price(0) == "0.0000000"

    def test_negative_falls_to_max_precision(self):
        assert fmt_price(-1500) == "-1,500.0000000"

    def test_large_number_rounds_to_integer(self):
        assert fmt_price(999_999.99) == "1,000,000"

    def test_small_negative_falls_to_max_precision(self):
        assert fmt_price(-0.5) == "-0.5000000"


class TestFmt:
    def test_basic_formatting(self):
        assert fmt(1234.5) == "1,234.50"

    def test_zero(self):
        assert fmt(0) == "0.00"

    def test_negative(self):
        assert fmt(-500.5) == "-500.50"

    def test_large_number(self):
        assert fmt(1_000_000.123) == "1,000,000.12"

    def test_small_number(self):
        assert fmt(0.001) == "0.00"


class TestLogExecutionTime:
    def test_sync_function(self):
        mock_logger = MagicMock()

        @log_execution_time
        def sync_func() -> int:
            return 42

        with patch('utils.logger', mock_logger):
            result = sync_func()

        assert result == 42
        assert mock_logger.info.called
        args = mock_logger.info.call_args[0][0]
        assert "sync_func execution time:" in args
        assert "seconds" in args

    @pytest.mark.asyncio
    async def test_async_function(self):
        mock_logger = MagicMock()

        @log_execution_time
        async def async_func() -> int:
            return 99

        with patch('utils.logger', mock_logger):
            result = await async_func()

        assert result == 99
        assert mock_logger.info.called
        args = mock_logger.info.call_args[0][0]
        assert "async_func execution time:" in args
        assert "seconds" in args

    def test_preserves_function_name(self):
        @log_execution_time
        def some_test_func() -> None:
            pass

        assert some_test_func.__name__ == "some_test_func"

    def test_sync_with_args(self):
        mock_logger = MagicMock()

        @log_execution_time
        def add(a: int, b: int) -> int:
            return a + b

        with patch('utils.logger', mock_logger):
            assert add(3, 4) == 7

    @pytest.mark.asyncio
    async def test_async_with_args(self):
        mock_logger = MagicMock()

        @log_execution_time
        async def multiply(a: int, b: int) -> int:
            return a * b

        with patch('utils.logger', mock_logger):
            result = await multiply(6, 7)
            assert result == 42


class TestOperationCancelled:
    def test_constant(self):
        assert OPERATION_CANCELLED == "Operation cancelled"


class TestExchangeEnabled:
    def test_exchange_disabled_by_default(self):
        assert not os.environ.get("HTB_KEY_FILE")
