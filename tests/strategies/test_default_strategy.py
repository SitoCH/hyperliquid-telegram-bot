from typing import Any
import pytest
from unittest.mock import MagicMock, patch
from strategies.default_strategy.default_strategy import DefaultStrategy


class TestDefaultStrategy:
    @pytest.mark.asyncio
    async def test_init_strategy_success(self):
        strategy = DefaultStrategy()
        context = MagicMock()

        with patch('strategies.default_strategy.default_strategy.logger') as mock_logger, \
                patch('strategies.default_strategy.default_strategy.telegram_utils') as mock_tg, \
                patch('strategies.default_strategy.default_strategy.CommandHandler') as mock_handler:
            await strategy.init_strategy(context)
            mock_logger.info.assert_called_with('Initializing default strategy')
            assert mock_tg.add_buttons.call_count == 2
            assert mock_tg.add_handler.call_count == 2
            assert mock_handler.call_count == 2
            assert mock_tg.run_repeating.call_count == 1

    @pytest.mark.asyncio
    async def test_init_strategy_error(self):
        strategy = DefaultStrategy()
        context = MagicMock()

        # Test error handling by making logger.info raise an exception
        with patch('strategies.default_strategy.default_strategy.logger') as mock_logger:
            mock_logger.info.side_effect = Exception("Test error")
            await strategy.init_strategy(context)
            mock_logger.error.assert_called()
            args, _ = mock_logger.error.call_args
            assert "Error executing default strategy: Test error" in args[0]

    def test_get_strategy_params(self):
        strategy = DefaultStrategy()
        with patch('strategies.default_strategy.default_strategy.hyperliquid_utils') as mock_hl:
            mock_hl.fetch_cryptos.return_value = [{"symbol": "BTC"}]
            mock_hl.info.all_mids.return_value = {"BTC": "50000"}
            mock_hl.info.meta.return_value = {"universe": []}

            cryptos, all_mids, meta = strategy.get_strategy_params()

            assert cryptos == [{"symbol": "BTC"}]
            assert all_mids == {"BTC": "50000"}
            assert meta == {"universe": []}
            mock_hl.fetch_cryptos.assert_called_once()

    def test_filter_top_cryptos(self):
        strategy = DefaultStrategy()
        cryptos = [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000},
            {"symbol": "ETH", "name": "Ethereum", "market_cap": 500},
            {"symbol": "XRP", "name": "Ripple", "market_cap": 100},
        ]
        all_mids = {"BTC": "50000", "ETH": "2500"}
        meta: dict[str, Any] = {}

        filtered = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(filtered) == 2
        assert filtered[0]["symbol"] == "BTC"
        assert filtered[1]["symbol"] == "ETH"
        assert "XRP" not in [f["symbol"] for f in filtered]
