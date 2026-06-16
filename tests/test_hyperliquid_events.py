import pytest
from unittest.mock import patch, MagicMock
from typing import Any, Dict

from hyperliquid_events import (
    get_fill_icon,
    get_fill_description,
    process_fill,
    on_user_events,
)


class TestGetFillIcon:
    def test_positive_pnl(self):
        assert get_fill_icon(100.0) == "🟢"

    def test_negative_pnl(self):
        assert get_fill_icon(-50.0) == "🔴"

    def test_zero_pnl(self):
        assert get_fill_icon(0) == "🔴"

    def test_small_positive_pnl(self):
        assert get_fill_icon(0.01) == "🟢"


class TestGetFillDescription:
    def test_basic_no_optional(self):
        mock_tg = MagicMock()
        mock_tg.get_link.side_effect = lambda text, action: text
        with patch('hyperliquid_events.telegram_utils', mock_tg):
            result = get_fill_description("🔵 Opened long:", "BTC", "1.5", 0.1, "USDC")
        assert "🔵 Opened long:" in result
        assert "Coin: BTC" in result
        assert "Size: 1.5" in result
        assert "Fee: 0.10 USDC" in result
        assert "Amount:" not in result
        assert "Profit:" not in result

    def test_with_amount(self):
        mock_tg = MagicMock()
        mock_tg.get_link.side_effect = lambda text, action: text
        with patch('hyperliquid_events.telegram_utils', mock_tg):
            result = get_fill_description("🔵 Bought spot:", "ETH", "10.0", 0.05, "USDC", amount=5000.0)
        assert "🔵 Bought spot:" in result
        assert "Amount: 5,000.00 USDC" in result
        assert "Profit:" not in result

    def test_with_closed_pnl(self):
        mock_tg = MagicMock()
        mock_tg.get_link.side_effect = lambda text, action: text
        with patch('hyperliquid_events.telegram_utils', mock_tg):
            result = get_fill_description("🟢 Closed long:", "BTC", "1.0", 0.2, "USDC", closed_pnl=150.0)
        assert "Profit: 150.00 USDC" in result
        assert "Amount:" not in result

    def test_with_both_amount_and_pnl(self):
        mock_tg = MagicMock()
        mock_tg.get_link.side_effect = lambda text, action: text
        with patch('hyperliquid_events.telegram_utils', mock_tg):
            result = get_fill_description(
                "🟢 Closed short:", "SOL", "50.0", 0.3, "USDC", amount=6000.0, closed_pnl=200.0
            )
        assert "Amount: 6,000.00 USDC" in result
        assert "Profit: 200.00 USDC" in result

    def test_zero_fee(self):
        mock_tg = MagicMock()
        mock_tg.get_link.side_effect = lambda text, action: text
        with patch('hyperliquid_events.telegram_utils', mock_tg):
            result = get_fill_description("🔵 Opened long:", "BTC", "1.0", 0.0, "USDC")
        assert "Fee: 0.00 USDC" in result

    def test_small_amount(self):
        mock_tg = MagicMock()
        mock_tg.get_link.side_effect = lambda text, action: text
        with patch('hyperliquid_events.telegram_utils', mock_tg):
            result = get_fill_description("Dust conversion:", "SHIB", "1000", 0.001, "USDC", amount=0.05)
        assert "Amount: 0.05 USDC" in result


def make_fill(dir: str, coin: str = "BTC", sz: str = "1.5", px: str = "40000",
              fee: str = "0.1", fee_token: str = "USDC", closed_pnl: str = "0") -> Dict[str, Any]:
    return {
        "coin": coin,
        "px": px,
        "sz": sz,
        "side": "B",
        "time": 1700000000000,
        "startPosition": "0",
        "dir": dir,
        "closedPnl": closed_pnl,
        "hash": "0xabc",
        "oid": 1,
        "crossed": True,
        "fee": fee,
        "tid": 1,
        "feeToken": fee_token,
    }


class TestProcessFill:
    @pytest.mark.parametrize("dir,expected_prefix", [
        ("Open Long", "🔵 Opened long:"),
        ("Open Short", "🔵 Opened short:"),
        ("Buy", "🔵 Bought spot:"),
        ("Sell", "🔵 Sold spot:"),
        ("Spot Dust Conversion", "🧹 Dust conversion:"),
    ])
    def test_fill_directions_with_amount(self, dir, expected_prefix):
        fill = make_fill(dir)
        with patch('hyperliquid_events.telegram_utils') as mock_tg:
            process_fill(fill)
            assert mock_tg.queue_send.called
            message = mock_tg.queue_send.call_args[0][0]
            assert expected_prefix in message

    @pytest.mark.parametrize("dir,expected_prefix", [
        ("Close Long", "Closed long:"),
        ("Close Short", "Closed short:"),
        ("Liquidated Isolated Long", "Liquidated isolated long:"),
        ("Long > Short", "Long -> short:"),
    ])
    def test_fill_directions_with_pnl(self, dir, expected_prefix):
        fill = make_fill(dir, closed_pnl="100.0")
        with patch('hyperliquid_events.telegram_utils') as mock_tg:
            process_fill(fill)
            assert mock_tg.queue_send.called
            message = mock_tg.queue_send.call_args[0][0]
            assert expected_prefix in message
            assert "Profit: 100.00 USDC" in message

    def test_fill_positive_pnl_icon(self):
        fill = make_fill("Close Long", closed_pnl="200.0")
        with patch('hyperliquid_events.telegram_utils') as mock_tg:
            process_fill(fill)
            message = mock_tg.queue_send.call_args[0][0]
            assert "🟢" in message

    def test_fill_negative_pnl_icon(self):
        fill = make_fill("Close Short", closed_pnl="-50.0")
        with patch('hyperliquid_events.telegram_utils') as mock_tg:
            process_fill(fill)
            message = mock_tg.queue_send.call_args[0][0]
            assert "🔴" in message

    def test_unknown_dir_falls_back_to_json(self):
        fill = make_fill("SomeUnknownDirection")
        with patch('hyperliquid_events.telegram_utils') as mock_tg:
            process_fill(fill)
            assert mock_tg.queue_send.called
            message = mock_tg.queue_send.call_args[0][0]
            assert "SomeUnknownDirection" in message
            assert "coin" in message

    def test_fill_calculates_amount_correctly(self):
        fill = make_fill("Open Long", px="50000", sz="2.0")
        with patch('hyperliquid_events.telegram_utils') as mock_tg:
            process_fill(fill)
            message = mock_tg.queue_send.call_args[0][0]
            assert "100,000.00 USDC" in message

    def test_fill_empty_size(self):
        fill = make_fill("Buy", sz="0")
        with patch('hyperliquid_events.telegram_utils') as mock_tg:
            process_fill(fill)
            assert mock_tg.queue_send.called


class TestOnUserEvents:
    def test_processes_all_fills(self):
        fills = [
            make_fill("Open Long", coin="BTC"),
            make_fill("Open Short", coin="ETH"),
            make_fill("Close Long", coin="SOL", closed_pnl="50.0"),
        ]
        user_events = {
            "channel": "user",
            "data": {"fills": fills},
        }
        with patch('hyperliquid_events.process_fill') as mock_process_fill:
            on_user_events(user_events)
            assert mock_process_fill.call_count == 3

    def test_no_fills_key(self):
        user_events = {"channel": "user", "data": {}}
        with patch('hyperliquid_events.process_fill') as mock_process_fill:
            on_user_events(user_events)
            mock_process_fill.assert_not_called()

    def test_empty_fills_list(self):
        user_events = {"channel": "user", "data": {"fills": []}}
        with patch('hyperliquid_events.process_fill') as mock_process_fill:
            on_user_events(user_events)
            mock_process_fill.assert_not_called()
