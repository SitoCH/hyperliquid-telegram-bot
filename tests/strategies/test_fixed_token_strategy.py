import os
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any, Optional

from strategies.fixed_token_strategy.fixed_token_strategy import (
    FixedTokenStrategy,
    FixedTokenConfig,
)


class TestFixedTokenConfig:
    """Tests for FixedTokenConfig dataclass."""

    def test_fixed_token_config_creation(self):
        config = FixedTokenConfig(tokens={"BTC", "ETH"})
        assert config.tokens == {"BTC", "ETH"}

    def test_fixed_token_config_single_token(self):
        config = FixedTokenConfig(tokens={"BTC"})
        assert config.tokens == {"BTC"}

    def test_fixed_token_config_empty(self):
        config = FixedTokenConfig(tokens=set())
        assert config.tokens == set()


class TestFixedTokenStrategyInitialization:
    """Tests for FixedTokenStrategy.__init__."""

    def test_default_initialization(self):
        strategy = FixedTokenStrategy()
        assert strategy.config.leverage == 5
        assert strategy.config.min_yearly_performance == 15.0
        assert strategy._fixed_token_config.tokens == {"BTC", "ETH"}

    def test_custom_initialization(self):
        env_vars = {
            "HTB_FIXED_TOKEN_STRATEGY_LEVERAGE": "3",
            "HTB_FIXED_TOKEN_STRATEGY_MIN_YEARLY_PERFORMANCE": "10.0",
            "HTB_FIXED_TOKEN_STRATEGY_TOKENS": "BTC,ETH,SOL,ARB",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            strategy = FixedTokenStrategy()
            assert strategy.config.leverage == 3
            assert strategy.config.min_yearly_performance == 10.0
            assert strategy._fixed_token_config.tokens == {"BTC", "ETH", "SOL", "ARB"}

    def test_single_token_initialization(self):
        env_vars = {
            "HTB_FIXED_TOKEN_STRATEGY_TOKENS": "BTC",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            strategy = FixedTokenStrategy()
            assert strategy._fixed_token_config.tokens == {"BTC"}
            assert len(strategy._fixed_token_config.tokens) == 1

    def test_default_tokens_when_empty_env(self):
        env_vars = {
            "HTB_FIXED_TOKEN_STRATEGY_TOKENS": "",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            strategy = FixedTokenStrategy()
            assert strategy._fixed_token_config.tokens == set()


class TestGetStrategyParams:
    """Tests for get_strategy_params."""

    @pytest.fixture
    def strategy(self):
        return FixedTokenStrategy()

    def test_params_defaults(self, strategy):
        mock_cryptos = [{"symbol": "BTC", "market_cap": 1_200_000_000_000}]
        mock_mids = {"BTC": "85000"}
        mock_meta = {"universe": [{"name": "BTC", "maxLeverage": 10}]}

        with patch("strategies.fixed_token_strategy.fixed_token_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.fetch_cryptos.return_value = mock_cryptos
            mock_hl.info.all_mids.return_value = mock_mids
            mock_hl.info.meta.return_value = mock_meta

            cryptos, all_mids, meta = strategy.get_strategy_params()

            mock_hl.fetch_cryptos.assert_called_once()
            call_args, call_kwargs = mock_hl.fetch_cryptos.call_args
            params = call_args[0]
            assert params["vs_currency"] == "usd"
            assert params["order"] == "market_cap_desc"
            assert params["per_page"] == 250
            assert params["sparkline"] == "false"
            assert call_kwargs == {"page_count": 2}
            assert cryptos == mock_cryptos
            assert all_mids == mock_mids
            assert meta == mock_meta


class TestFilterTopCryptos:
    """Tests for filter_top_cryptos."""

    @pytest.fixture
    def strategy(self):
        return FixedTokenStrategy()

    def _make_coin(self, symbol: str, market_cap: float = 1_000_000_000_000, yearly_change: Optional[float] = 50.0) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "name": symbol,
            "market_cap": market_cap,
            "price_change_percentage_1y_in_currency": yearly_change,
        }

    def _make_meta(self, symbols_with_leverage: List[Tuple[str, int]]) -> Dict[str, Any]:
        return {
            "universe": [
                {"name": symbol, "maxLeverage": max_lev}
                for symbol, max_lev in symbols_with_leverage
            ]
        }

    def test_basic_filter_matching_tokens(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 80.0),
            self._make_coin("SOL", 80_000_000_000, 200.0),
        ]
        all_mids = {"BTC": "85000", "ETH": "2200", "SOL": "120"}
        meta = self._make_meta([("BTC", 10), ("ETH", 10), ("SOL", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 2
        symbols = {c["symbol"] for c in result}
        assert symbols == {"BTC", "ETH"}

    def test_token_not_in_fixed_list(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("SOL", 80_000_000_000, 200.0),
        ]
        all_mids = {"BTC": "85000", "SOL": "120"}
        meta = self._make_meta([("BTC", 10), ("SOL", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_token_not_available_on_hyperliquid(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 80.0),
        ]
        all_mids = {"BTC": "85000"}
        meta = self._make_meta([("BTC", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_yearly_change_below_minimum(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 5.0),
        ]
        all_mids = {"BTC": "85000", "ETH": "2200"}
        meta = self._make_meta([("BTC", 10), ("ETH", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_yearly_change_none_not_filtered(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, None),
        ]
        all_mids = {"BTC": "85000"}
        meta = self._make_meta([("BTC", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_leverage_exceeds_max(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
        ]
        all_mids = {"BTC": "85000"}
        meta = self._make_meta([("BTC", 3)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 0

    def test_leverage_missing_from_meta_not_filtered(self, strategy):
        cryptos = [self._make_coin("BTC", 1_200_000_000_000, 120.5)]
        all_mids = {"BTC": "85000"}
        meta: dict[str, Any] = {"universe": []}

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_empty_cryptos(self, strategy):
        result = strategy.filter_top_cryptos([], {}, {})
        assert result == []

    def test_all_filters_together(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 5.0),
            self._make_coin("SOL", 80_000_000_000, 200.0),
        ]
        all_mids = {"BTC": "85000", "ETH": "2200", "SOL": "120"}
        meta = self._make_meta([("BTC", 10), ("ETH", 10), ("SOL", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_result_sorted_by_market_cap(self, strategy):
        strategy._fixed_token_config.tokens = {"BTC", "SOL", "ETH"}
        cryptos = [
            self._make_coin("SOL", 80_000_000_000, 200.0),
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 80.0),
        ]
        all_mids = {"SOL": "120", "BTC": "85000", "ETH": "2200"}
        meta = self._make_meta([
            ("SOL", 10), ("BTC", 10), ("ETH", 10)
        ])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert result[0]["symbol"] == "BTC"
        assert result[1]["symbol"] == "ETH"
        assert result[2]["symbol"] == "SOL"

    def test_single_token(self, strategy):
        strategy._fixed_token_config.tokens = {"BTC"}
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 80.0),
        ]
        all_mids = {"BTC": "85000", "ETH": "2200"}
        meta = self._make_meta([("BTC", 10), ("ETH", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_empty_tokens_list_filters_everything(self, strategy):
        strategy._fixed_token_config.tokens = set()
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 80.0),
        ]
        all_mids = {"BTC": "85000", "ETH": "2200"}
        meta = self._make_meta([("BTC", 10), ("ETH", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)
        assert len(result) == 0

    def test_included_fields_in_result(self, strategy):
        cryptos = [self._make_coin("BTC", 1_200_000_000_000, 120.5)]
        all_mids = {"BTC": "85000"}
        meta = self._make_meta([("BTC", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        coin = result[0]
        assert "name" in coin
        assert "symbol" in coin
        assert "market_cap" in coin
        assert "price_change_percentage_1y_in_currency" in coin


class TestInitStrategy:
    """Tests for init_strategy."""

    @pytest.mark.asyncio
    async def test_sets_up_rebalance_button_and_handler(self):
        strategy = FixedTokenStrategy()
        context = MagicMock()

        with patch("strategies.fixed_token_strategy.fixed_token_strategy.telegram_utils") as mock_tg, \
                patch("strategies.fixed_token_strategy.fixed_token_strategy.CommandHandler") as mock_handler:
            mock_handler_instance = MagicMock()
            mock_handler.return_value = mock_handler_instance

            await strategy.init_strategy(context)

            mock_tg.add_buttons.assert_any_call(["/rebalance"], 1)
            mock_tg.add_handler.assert_any_call(mock_handler_instance)

    @pytest.mark.asyncio
    async def test_sets_up_analyze_button_and_handler(self):
        strategy = FixedTokenStrategy()
        context = MagicMock()

        with patch("strategies.fixed_token_strategy.fixed_token_strategy.telegram_utils") as mock_tg, \
                patch("strategies.fixed_token_strategy.fixed_token_strategy.CommandHandler") as mock_handler:
            mock_handler_instance = MagicMock()
            mock_handler.return_value = mock_handler_instance

            await strategy.init_strategy(context)

            mock_tg.add_buttons.assert_any_call(["/analyze"], 1)
            # analyze handler should also be added
            assert mock_tg.add_handler.call_count == 2

    @pytest.mark.asyncio
    async def test_no_repeating_check_for_fixed_token(self):
        strategy = FixedTokenStrategy()
        context = MagicMock()

        with patch("strategies.fixed_token_strategy.fixed_token_strategy.telegram_utils") as mock_tg, \
                patch("strategies.fixed_token_strategy.fixed_token_strategy.CommandHandler"):
            await strategy.init_strategy(context)

            assert not hasattr(mock_tg, "run_repeating") or mock_tg.run_repeating.call_count == 0
