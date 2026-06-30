import pytest
import os
from unittest.mock import patch, MagicMock, PropertyMock
import requests


# Must set env var before importing the module
os.environ.setdefault("HTB_USER_WALLET", "0x0000000000000000000000000000000000000000")


class TestHyperliquidUtilsInit:
    def test_init_success(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.address == "0x0000000000000000000000000000000000000000"
            # Info is not called until .info is accessed
            assert mock_info.call_count == 0
            _ = instance.info
            mock_info.assert_called_once()

    def test_init_with_vault(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch.dict(os.environ, {"HTB_USER_VAULT": "0xVAULT"}):
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.address == "0xVAULT"

    def test_init_missing_wallet_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="HTB_USER_WALLET environment variable is required"):
                from hyperliquid_utils.utils import HyperliquidUtils
                HyperliquidUtils()


class TestHyperliquidUtilsWebsocket:
    def test_init_websocket(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance.init_websocket()
            # Only called once because of lazy property
            assert mock_info.call_count == 1

    def test_on_websocket_error(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch('hyperliquid_utils.utils.telegram_utils') as mock_tg:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance._reconnecting = False
            instance._on_websocket_error(None, "test error")
            assert instance._reconnecting is True
            mock_tg.queue_send.assert_called_once_with("⚠️ WebSocket connection error — reconnecting...")

    def test_on_websocket_close(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch('hyperliquid_utils.utils.telegram_utils') as mock_tg:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance._reconnecting = False
            instance._on_websocket_close(None, 0, "closed")
            assert instance._reconnecting is True
            mock_tg.queue_send.assert_called_once_with("🔌 WebSocket disconnected — reconnecting...")

    def test_reconnect_debounce(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch('hyperliquid_utils.utils.telegram_utils') as mock_tg:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance._reconnecting = False
            # First call should proceed
            instance._on_websocket_close(None, 0, "closed")
            assert mock_tg.queue_send.call_count == 1
            # Second call should be ignored (debounced)
            instance._on_websocket_close(None, 0, "closed again")
            assert mock_tg.queue_send.call_count == 1


class TestHyperliquidUtilsExchange:
    def test_get_exchange_with_key_file(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch.dict(os.environ, {"HTB_KEY_FILE": "/tmp/fake_key", "HTB_USER_WALLET": "0xwallet"}), \
                patch('os.path.isfile', return_value=True), \
                patch('builtins.open', MagicMock()), \
                patch('eth_account.Account.from_key') as mock_from_key:
            mock_account = MagicMock()
            mock_from_key.return_value = mock_account
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            with patch('hyperliquid_utils.utils.Exchange') as mock_exchange:
                result = instance.get_exchange()
                mock_exchange.assert_called_once()
                assert result is not None

    def test_get_exchange_no_key_file(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            result = instance.get_exchange()
            assert result is None


class TestHyperliquidUtilsSzDecimals:
    def test_get_sz_decimals(self):
        mock_meta = {
            "universe": [
                {"name": "BTC", "szDecimals": 5},
                {"name": "ETH", "szDecimals": 4},
            ]
        }
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance.info.meta = MagicMock(return_value=mock_meta)  # type: ignore[attr-defined]
            result = instance.get_sz_decimals()
            assert result == {"BTC": 5, "ETH": 4}

    def test_get_sz_decimals_with_dex(self):
        """szDecimals for an extra DEX queries meta(dex=...) instead of default."""
        mock_xyz_meta = {
            "universe": [
                {"name": "xyz:BABA", "szDecimals": 3},
                {"name": "xyz:AAPL", "szDecimals": 2},
            ]
        }
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance.info.meta = MagicMock(return_value=mock_xyz_meta)  # type: ignore[attr-defined]
            result = instance.get_sz_decimals(dex="xyz")
            # Should query meta(dex='xyz'), not just meta()
            instance.info.meta.assert_called_with(dex="xyz")
            assert result == {"xyz:BABA": 3, "xyz:AAPL": 2}


class TestHyperliquidUtilsPositions:
    def test_get_size_with_position(self, sample_user_state):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_size(sample_user_state, "BTC") == 1.5
            assert instance.get_size(sample_user_state, "ETH") == -10.0

    def test_get_size_no_position(self, sample_user_state):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_size(sample_user_state, "SOL") == 0.0

    def test_get_entry_px_str(self, sample_user_state):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_entry_px_str(sample_user_state, "BTC") == "40000"
            assert instance.get_entry_px_str(sample_user_state, "SOL") is None

    def test_get_liquidation_px_str(self, sample_user_state):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_liquidation_px_str(sample_user_state, "BTC") == "39000"
            assert instance.get_liquidation_px_str(sample_user_state, "SOL") is None

    def test_get_entry_px(self, sample_user_state):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_entry_px(sample_user_state, "BTC") == 40000.0
            assert instance.get_entry_px(sample_user_state, "SOL") == 0.0

    def test_get_unrealized_pnl(self):
        user_state = {
            "assetPositions": [{
                "position": {"coin": "BTC", "unrealizedPnl": "150.50"}
            }]
        }
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_unrealized_pnl(user_state, "BTC") == 150.50
            assert instance.get_unrealized_pnl(user_state, "SOL") == 0.0

    def test_get_return_on_equity(self, sample_user_state):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_return_on_equity(sample_user_state, "BTC") == 0.15
            assert instance.get_return_on_equity(sample_user_state, "SOL") == 0.0

    def test_get_leverage(self, sample_user_state):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_leverage(sample_user_state, "BTC") == 10
            assert instance.get_leverage(sample_user_state, "SOL") is None

    def test_get_coins_with_open_positions(self):
        mock_user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC"}},
                {"position": {"coin": "ETH"}},
            ]
        }
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance._extra_dexes = []
            instance.info.user_state = MagicMock(return_value=mock_user_state)  # type: ignore[attr-defined]
            result = instance.get_coins_with_open_positions()
            assert result == ["BTC", "ETH"]


class TestHyperliquidUtilsCoins:
    def test_extra_dexes_default_xyz(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch.dict(os.environ, {"HTB_USER_WALLET": "0x0", "HTB_EXTRA_DEXES": "xyz"}, clear=True):
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.extra_dexes() == ["xyz"]

    def test_extra_dexes_empty(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch.dict(os.environ, {"HTB_USER_WALLET": "0x0", "HTB_EXTRA_DEXES": ""}, clear=True):
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.extra_dexes() == []

    def test_extra_dexes_multiple(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch.dict(os.environ, {"HTB_USER_WALLET": "0x0", "HTB_EXTRA_DEXES": "xyz,flx,vntl"}, clear=True):
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.extra_dexes() == ["xyz", "flx", "vntl"]

    def test_get_coins_by_traded_volume(self):
        mock_response = (
            {"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "SOL"}]},
            [{"dayNtlVlm": "1000"}, {"dayNtlVlm": "3000"}, {"dayNtlVlm": "2000"}],
        )
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance._extra_dexes = []
            instance.info.meta_and_asset_ctxs = MagicMock(return_value=mock_response)  # type: ignore[attr-defined]
            result = instance.get_coins_by_traded_volume()
            assert result == ["ETH", "SOL", "BTC"]

    def test_get_coins_reply_markup(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch('hyperliquid_utils.utils.InlineKeyboardMarkup') as mock_markup:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            instance.get_coins_by_traded_volume = MagicMock(return_value=["BTC", "ETH"])  # type: ignore[method-assign]
            instance.get_coins_with_open_positions = MagicMock(return_value=["BTC"])  # type: ignore[method-assign]
            result = instance.get_coins_reply_markup()
            mock_markup.assert_called_once()
            assert result is not None

    def test_get_coins_reply_markup_strips_prefix(self):
        """When a dex is specified, coin labels should strip the dex_name: prefix."""
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils, InlineKeyboardMarkup
            instance = HyperliquidUtils()
            instance.get_coins_by_traded_volume = MagicMock(return_value=["xyz:BABA", "xyz:AAPL"])  # type: ignore[method-assign]
            instance.get_coins_with_open_positions = MagicMock(return_value=[])  # type: ignore[method-assign]

            markup = instance.get_coins_reply_markup(dex="xyz")
            assert isinstance(markup, InlineKeyboardMarkup)
            button_texts = [btn.text for row in markup.inline_keyboard for btn in row]
            assert "BABA" in button_texts
            assert "AAPL" in button_texts
            assert "xyz:BABA" not in button_texts
            button_data = [btn.callback_data for row in markup.inline_keyboard for btn in row]
            assert "xyz:BABA" in button_data
            assert "xyz:AAPL" in button_data


class TestHyperliquidUtilsSymbol:
    def test_get_hyperliquid_symbol_no_mapping(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_hyperliquid_symbol("BTC") == "BTC"

    def test_get_hyperliquid_symbol_mapping(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            assert instance.get_hyperliquid_symbol("SHIB") == "kSHIB"
            assert instance.get_hyperliquid_symbol("PEPE") == "kPEPE"
            assert instance.get_hyperliquid_symbol("FLOKI") == "kFLOKI"
            assert instance.get_hyperliquid_symbol("BONK") == "kBONK"


class TestHyperliquidUtilsFetchCryptos:
    def test_fetch_cryptos_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [{"symbol": "btc"}, {"symbol": "shib"}]
        mock_response.raise_for_status = MagicMock()

        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch('hyperliquid_utils.utils.requests.get', return_value=mock_response):
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            result = instance.fetch_cryptos({"vs_currency": "usd"})
            assert len(result) == 2
            assert result[0]["symbol"] == "BTC"
            assert result[1]["symbol"] == "kSHIB"

    def test_fetch_cryptos_request_exception(self):
        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch('hyperliquid_utils.utils.requests.get', side_effect=requests.RequestException("timeout")):
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            result = instance.fetch_cryptos({"vs_currency": "usd"})
            assert result == []

    def test_fetch_cryptos_pagination(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [{"symbol": "btc"}]
        mock_response.raise_for_status = MagicMock()

        with patch('hyperliquid_utils.utils.InfoProxy') as mock_info_proxy, \
                patch('hyperliquid_utils.utils.Info') as mock_info, \
                patch('hyperliquid_utils.utils.requests.get', return_value=mock_response) as mock_get:
            from hyperliquid_utils.utils import HyperliquidUtils
            instance = HyperliquidUtils()
            result = instance.fetch_cryptos({"vs_currency": "usd"}, page_count=3)
            assert len(result) == 3
            assert mock_get.call_count == 3
