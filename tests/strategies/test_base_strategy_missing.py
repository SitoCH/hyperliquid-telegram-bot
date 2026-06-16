import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Tuple
from strategies.base_strategy.base_strategy import (
    BaseStrategy,
    BaseStrategyConfig,
)

from telegram.constants import ParseMode


class _ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""
    def get_strategy_params(self) -> Tuple[List[Dict], Dict[str, str], Dict]:
        return [], {}, {}

    def filter_top_cryptos(self, cryptos, all_mids, meta) -> List[Dict]:
        return cryptos  # Return what we give it


class TestBaseStrategyMissing:
    @pytest.mark.asyncio
    async def test_check_allocation_drifts_exception(self):
        strategy = _ConcreteStrategy()
        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.info.user_state.side_effect = Exception("Drift error")
            with patch("strategies.base_strategy.base_strategy.logger") as mock_logger, \
                 patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg:
                mock_tg.send = AsyncMock()
                await strategy.check_position_allocation_drifts(MagicMock())
                mock_logger.error.assert_called()
                mock_tg.send.assert_called_with("Error checking allocation drifts: Drift error")

    @pytest.mark.asyncio
    async def test_display_crypto_info_exception(self):
        strategy = _ConcreteStrategy()
        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.info.user_state.side_effect = Exception("Display error")
            with patch("strategies.base_strategy.base_strategy.logger") as mock_logger:
                await strategy.display_crypto_info(MagicMock(), [], {}, {})
                mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_rebalance_exception(self):
        strategy = _ConcreteStrategy()
        update = MagicMock()
        context = MagicMock()
        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.get_exchange.side_effect = Exception("Rebalance error")
            with patch("strategies.base_strategy.base_strategy.logger") as mock_logger, \
                 patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg:
                mock_tg.reply = AsyncMock()
                await strategy.rebalance(update, context)
                mock_logger.critical.assert_called()
                mock_tg.reply.assert_any_call(
                    update, "Error during rebalancing: Rebalance error", parse_mode=ParseMode.HTML
                )

    @pytest.mark.asyncio
    async def test_rebalance_open_position_error(self):
        strategy = _ConcreteStrategy()
        update = MagicMock()
        context = MagicMock()
        mock_exchange = MagicMock()

        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl, \
             patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg, \
             patch("strategies.base_strategy.base_strategy.exit_all_positions", new_callable=AsyncMock):
            mock_hl.get_exchange.return_value = mock_exchange
            mock_tg.reply = AsyncMock()
            mock_hl.info.user_state.return_value = {
                "assetPositions": [],
                "crossMarginSummary": {"totalRawUsd": "1000.0"}
            }
            mock_hl.get_sz_decimals.return_value = {"BTC": 3}

            cryptos = [{"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000}]
            all_mids = {"BTC": "50000"}

            with patch.object(strategy, "get_strategy_params", return_value=(cryptos, all_mids, {})), \
                 patch("strategies.base_strategy.base_strategy.logger") as mock_logger:

                mock_exchange.market_open.side_effect = Exception("Open failed")
                await strategy.rebalance(update, context)
                mock_logger.error.assert_any_call("Unable to open position for BTC: Open failed")

    @pytest.mark.asyncio
    async def test_rebalance_low_target_value(self):
        strategy = _ConcreteStrategy(config=BaseStrategyConfig(leverage=3, min_position_size=10.0))
        update = MagicMock()
        context = MagicMock()
        mock_exchange = MagicMock()

        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl, \
             patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg, \
             patch("strategies.base_strategy.base_strategy.exit_all_positions", new_callable=AsyncMock):
            mock_hl.get_exchange.return_value = mock_exchange
            mock_tg.reply = AsyncMock()
            mock_hl.info.user_state.return_value = {
                "assetPositions": [],
                "crossMarginSummary": {"totalRawUsd": "1000.0"}
            }

            # Target value will be small if market cap is small relative to total
            # Total market cap will be 1,000,001. "SMALL" will get a tiny fraction.
            cryptos = [
                {"symbol": "BIG", "name": "Big", "market_cap": 1_000_000},
                {"symbol": "SMALL", "name": "Small", "market_cap": 1}
            ]
            all_mids = {"BIG": "100", "SMALL": "1.0"}

            with patch.object(strategy, "get_strategy_params", return_value=(cryptos, all_mids, {})), \
                 patch("strategies.base_strategy.base_strategy.logger") as mock_logger:

                await strategy.rebalance(update, context)
                # Check that logger.info was called for low target value in calculate_allocations
                info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                assert any("target value 0.00 USDC is below minimum" in str(msg) for msg in info_calls)
