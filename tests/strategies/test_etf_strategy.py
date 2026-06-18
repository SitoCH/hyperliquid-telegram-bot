import os
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any, Tuple

from strategies.etf_strategy.etf_strategy import EtfStrategy, EtfConfig


class TestEtfConfig:
    """Tests for EtfConfig dataclass."""

    def test_etf_config_creation(self):
        config = EtfConfig(
            coins_number=5,
            coins_offset=0,
            excluded_symbols={"USDT", "USDC"},
            category=None,
        )
        assert config.coins_number == 5
        assert config.coins_offset == 0
        assert config.excluded_symbols == {"USDT", "USDC"}
        assert config.category is None

    def test_etf_config_with_category(self):
        config = EtfConfig(
            coins_number=10,
            coins_offset=2,
            excluded_symbols=set(),
            category="defi",
        )
        assert config.coins_number == 10
        assert config.coins_offset == 2
        assert config.category == "defi"

    def test_etf_config_empty_excluded(self):
        config = EtfConfig(
            coins_number=3,
            coins_offset=0,
            excluded_symbols=set(),
            category=None,
        )
        assert config.excluded_symbols == set()


class TestEtfStrategyInitialization:
    """Tests for EtfStrategy.__init__."""

    def test_default_initialization(self):
        strategy = EtfStrategy()
        assert strategy.config.leverage == 5
        assert strategy.config.min_yearly_performance == 15.0
        assert strategy._etf_config.coins_number == 5
        assert strategy._etf_config.coins_offset == 0
        assert strategy._etf_config.excluded_symbols == set()
        assert strategy._etf_config.category is None

    def test_custom_initialization(self):
        env_vars = {
            "HTB_ETF_STRATEGY_LEVERAGE": "3",
            "HTB_ETF_STRATEGY_MIN_YEARLY_PERFORMANCE": "10.0",
            "HTB_ETF_STRATEGY_COINS_NUMBER": "10",
            "HTB_ETF_STRATEGY_COINS_OFFSET": "2",
            "HTB_ETF_STRATEGY_EXCLUDED_SYMBOLS": "USDT,USDC,DAI",
            "HTB_ETF_STRATEGY_CATEGORY": "defi",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            strategy = EtfStrategy()
            assert strategy.config.leverage == 3
            assert strategy.config.min_yearly_performance == 10.0
            assert strategy._etf_config.coins_number == 10
            assert strategy._etf_config.coins_offset == 2
            assert strategy._etf_config.excluded_symbols == {"USDT", "USDC", "DAI"}
            assert strategy._etf_config.category == "defi"


class TestGetStrategyParams:
    """Tests for get_strategy_params."""

    @pytest.fixture
    def strategy(self) -> EtfStrategy:
        return EtfStrategy()

    def test_default_params(self, strategy):
        mock_cryptos = [{"symbol": "BTC", "market_cap": 1_200_000_000_000}]
        mock_mids = {"BTC": "85000"}
        mock_meta = {"universe": [{"name": "BTC", "maxLeverage": 10}]}

        with patch("strategies.etf_strategy.etf_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.fetch_cryptos.return_value = mock_cryptos
            mock_hl.info.all_mids.return_value = mock_mids
            mock_hl.info.meta.return_value = mock_meta

            cryptos, all_mids, meta = strategy.get_strategy_params()

            mock_hl.fetch_cryptos.assert_called_once()
            call_args, call_kwargs = mock_hl.fetch_cryptos.call_args
            params = call_args[0]
            assert params["vs_currency"] == "usd"
            assert params["order"] == "market_cap_desc"
            assert params["per_page"] == 50
            assert "category" not in params
            assert cryptos == mock_cryptos
            assert all_mids == mock_mids
            assert meta == mock_meta

    def test_params_with_category(self, strategy):
        strategy._etf_config.category = "defi"

        with patch("strategies.etf_strategy.etf_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.fetch_cryptos.return_value = []
            mock_hl.info.all_mids.return_value = {}
            mock_hl.info.meta.return_value = {}

            strategy.get_strategy_params()

            call_args, call_kwargs = mock_hl.fetch_cryptos.call_args
            params = call_args[0]
            assert params["category"] == "defi"


class TestFilterTopCryptos:
    """Tests for filter_top_cryptos."""

    @pytest.fixture
    def strategy(self) -> EtfStrategy:
        return EtfStrategy()

    def _make_coin(self, symbol: str, market_cap: int = 1_000_000_000_000, yearly_change: float | None = 50.0) -> Dict[str, Any]:
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

    def test_basic_filter_top_cryptos(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 80.0),
        ]
        all_mids = {"BTC": "85000", "ETH": "2200"}
        meta = self._make_meta([("BTC", 10), ("ETH", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 2
        assert result[0]["symbol"] == "BTC"
        assert result[1]["symbol"] == "ETH"
        assert result[0]["market_cap"] == 1_200_000_000_000
        assert result[1]["market_cap"] == 400_000_000_000

    def test_excluded_symbols_filtered(self, strategy):
        strategy._etf_config.excluded_symbols = {"USDT"}
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("USDT", 100_000_000_000, 5.0),
        ]
        all_mids = {"BTC": "85000", "USDT": "1"}
        meta = self._make_meta([("BTC", 10), ("USDT", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_not_available_on_hyperliquid(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("FAKE", 500_000_000_000, 200.0),
        ]
        all_mids = {"BTC": "85000"}
        meta = self._make_meta([("BTC", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_yearly_change_below_minimum(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("LOW", 500_000_000_000, 5.0),
        ]
        all_mids = {"BTC": "85000", "LOW": "10"}
        meta = self._make_meta([("BTC", 10), ("LOW", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_yearly_change_none_not_filtered(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("NEW", 500_000_000_000, None),
        ]
        all_mids = {"BTC": "85000", "NEW": "10"}
        meta = self._make_meta([("BTC", 10), ("NEW", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 2

    def test_leverage_exceeds_max(self, strategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("LOWLEV", 500_000_000_000, 100.0),
        ]
        all_mids = {"BTC": "85000", "LOWLEV": "10"}
        meta = self._make_meta([("BTC", 10), ("LOWLEV", 3)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_leverage_missing_from_meta(self, strategy: EtfStrategy):
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
        ]
        all_mids = {"BTC": "85000"}
        meta: Dict[str, Any] = {"universe": []}

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_no_universe_in_meta(self, strategy: EtfStrategy):
        cryptos = [self._make_coin("BTC", 1_200_000_000_000, 120.5)]
        all_mids = {"BTC": "85000"}
        meta: Dict[str, Any] = {}

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_coins_offset_and_limit(self, strategy):
        strategy._etf_config.coins_number = 2
        strategy._etf_config.coins_offset = 1
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 80.0),
            self._make_coin("SOL", 80_000_000_000, 200.0),
        ]
        all_mids = {"BTC": "85000", "ETH": "2200", "SOL": "120"}
        meta = self._make_meta([
            ("BTC", 10), ("ETH", 10), ("SOL", 10)
        ])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 2
        assert result[0]["symbol"] == "ETH"
        assert result[1]["symbol"] == "SOL"

    def test_coins_offset_skips_first(self, strategy):
        strategy._etf_config.coins_number = 10
        strategy._etf_config.coins_offset = 1
        cryptos = [
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("ETH", 400_000_000_000, 80.0),
        ]
        all_mids = {"BTC": "85000", "ETH": "2200"}
        meta = self._make_meta([("BTC", 10), ("ETH", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "ETH"

    def test_coins_number_zero_returns_empty(self, strategy):
        strategy._etf_config.coins_number = 0
        cryptos = [self._make_coin("BTC", 1_200_000_000_000, 120.5)]
        all_mids = {"BTC": "85000"}
        meta = self._make_meta([("BTC", 10)])
        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)
        assert result == []

    def test_result_sorted_by_market_cap_descending(self, strategy):
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

    def test_empty_cryptos(self, strategy):
        result = strategy.filter_top_cryptos([], {}, {})
        assert result == []

    def test_all_coins_filtered_out(self, strategy):
        cryptos = [
            self._make_coin("FAKE1", 100_000_000_000, 5.0),
            self._make_coin("FAKE2", 100_000_000_000, 3.0),
        ]
        all_mids = {"FAKE1": "10", "FAKE2": "10"}
        meta = self._make_meta([("FAKE1", 10), ("FAKE2", 10)])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert result == []

    def test_multiple_filters_combined(self, strategy):
        strategy._etf_config.excluded_symbols = {"EXCLUDED"}
        strategy._etf_config.coins_number = 1
        cryptos = [
            self._make_coin("EXCLUDED", 2_000_000_000_000, 300.0),
            self._make_coin("BTC", 1_200_000_000_000, 120.5),
            self._make_coin("LOWPERF", 500_000_000_000, 5.0),
            self._make_coin("NOHL", 300_000_000_000, 100.0),
            self._make_coin("HIGHLEV", 200_000_000_000, 100.0),
        ]
        all_mids = {"EXCLUDED": "100", "BTC": "85000", "LOWPERF": "10", "NOHL": "20"}
        meta = self._make_meta([
            ("EXCLUDED", 10), ("BTC", 10), ("LOWPERF", 10), ("HIGHLEV", 3),
        ])

        result = strategy.filter_top_cryptos(cryptos, all_mids, meta)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

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
        strategy = EtfStrategy()
        context = MagicMock()

        with patch("strategies.etf_strategy.etf_strategy.telegram_utils") as mock_tg, \
                patch("strategies.etf_strategy.etf_strategy.CommandHandler") as mock_handler, \
                patch("strategies.etf_strategy.etf_strategy.datetime") as mock_dt, \
                patch("strategies.etf_strategy.etf_strategy.random") as mock_random:
            mock_random.randint.return_value = 720
            mock_handler_instance = MagicMock()
            mock_handler.return_value = mock_handler_instance
            mock_dt.timedelta.return_value = "timedelta_result"

            await strategy.init_strategy(context)

            mock_tg.add_buttons.assert_any_call(["/rebalance"], 1)
            mock_tg.add_handler.assert_any_call(mock_handler_instance)

    @pytest.mark.asyncio
    async def test_sets_up_analyze_button_and_handler(self):
        strategy = EtfStrategy()
        context = MagicMock()

        with patch("strategies.etf_strategy.etf_strategy.telegram_utils") as mock_tg, \
                patch("strategies.etf_strategy.etf_strategy.CommandHandler") as mock_handler, \
                patch("strategies.etf_strategy.etf_strategy.datetime") as mock_dt, \
                patch("strategies.etf_strategy.etf_strategy.random") as mock_random:
            mock_random.randint.return_value = 720
            mock_tg.add_buttons.return_value = None
            mock_tg.add_handler.return_value = None
            mock_handler_instance = MagicMock()
            mock_handler.return_value = mock_handler_instance
            mock_dt.timedelta.return_value = "timedelta_result"

            await strategy.init_strategy(context)

            mock_tg.add_buttons.assert_any_call(["/analyze"], 1)

    @pytest.mark.asyncio
    async def test_sets_up_repeating_check(self):
        strategy = EtfStrategy()
        context = MagicMock()

        with patch("strategies.etf_strategy.etf_strategy.telegram_utils") as mock_tg, \
                patch("strategies.etf_strategy.etf_strategy.CommandHandler"), \
                patch("strategies.etf_strategy.etf_strategy.datetime") as mock_dt, \
                patch("strategies.etf_strategy.etf_strategy.random") as mock_random:
            mock_random.randint.return_value = 720
            mock_dt.timedelta.return_value = "ANY_INTERVAL"

            await strategy.init_strategy(context)

            mock_tg.run_repeating.assert_called_once_with(
                strategy.check_position_allocation_drifts,
                interval="ANY_INTERVAL",
            )

    @pytest.mark.asyncio
    async def test_random_interval_around_12_hours(self):
        strategy = EtfStrategy()
        context = MagicMock()

        with patch("strategies.etf_strategy.etf_strategy.telegram_utils"), \
                patch("strategies.etf_strategy.etf_strategy.CommandHandler"), \
                patch("strategies.etf_strategy.etf_strategy.datetime"), \
                patch("strategies.etf_strategy.etf_strategy.random") as mock_random:
            mock_random.randint.return_value = 730

            await strategy.init_strategy(context)

            mock_random.randint.assert_called_once_with(12 * 60 - 10, 12 * 60 + 10)
