import pytest
from typing import List, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

from strategies.base_strategy.base_strategy import (
    BaseStrategy,
    BaseStrategyConfig,
    AllocationData,
    PositionData,
)


class TestBaseStrategyConfig:
    """Tests for BaseStrategyConfig dataclass and validation."""

    def test_valid_config(self):
        config = BaseStrategyConfig(leverage=3)
        assert config.leverage == 3
        assert config.min_yearly_performance == 15.0
        assert config.min_position_size == 10.0
        assert config.min_usdc_balance == 2.0
        assert config.usdc_balance_percent == 0.01
        assert config.min_rebalance_difference == 25.0
        assert config.drift_check_threshold == 25.0
        assert config.position_drift_threshold == 25.0

    def test_leverage_below_minimum(self):
        with pytest.raises(ValueError, match="Leverage must be between 1 and 10"):
            BaseStrategyConfig(leverage=0)

    def test_leverage_above_maximum(self):
        with pytest.raises(ValueError, match="Leverage must be between 1 and 10"):
            BaseStrategyConfig(leverage=11)

    def test_leverage_at_boundaries(self):
        BaseStrategyConfig(leverage=1)
        BaseStrategyConfig(leverage=10)

    def test_negative_min_yearly_performance(self):
        with pytest.raises(ValueError, match="Minimum yearly performance cannot be negative"):
            BaseStrategyConfig(leverage=3, min_yearly_performance=-1.0)

    def test_zero_min_yearly_performance(self):
        BaseStrategyConfig(leverage=3, min_yearly_performance=0)

    def test_min_position_size_below_minimum(self):
        with pytest.raises(ValueError, match="Minimum position size cannot be less than 10 USDC"):
            BaseStrategyConfig(leverage=3, min_position_size=9.99)

    def test_min_position_size_at_minimum(self):
        BaseStrategyConfig(leverage=3, min_position_size=10.0)

    def test_negative_min_usdc_balance(self):
        with pytest.raises(ValueError, match="Minimum USDC balance cannot be negative"):
            BaseStrategyConfig(leverage=3, min_usdc_balance=-1.0)

    def test_zero_min_usdc_balance(self):
        BaseStrategyConfig(leverage=3, min_usdc_balance=0.0)

    def test_usdc_balance_percent_zero(self):
        with pytest.raises(ValueError, match="USDC balance percentage must be between 0 and 1"):
            BaseStrategyConfig(leverage=3, usdc_balance_percent=0.0)

    def test_usdc_balance_percent_one(self):
        with pytest.raises(ValueError, match="USDC balance percentage must be between 0 and 1"):
            BaseStrategyConfig(leverage=3, usdc_balance_percent=1.0)

    def test_usdc_balance_percent_above_one(self):
        with pytest.raises(ValueError, match="USDC balance percentage must be between 0 and 1"):
            BaseStrategyConfig(leverage=3, usdc_balance_percent=1.5)

    def test_usdc_balance_percent_valid_thresholds(self):
        BaseStrategyConfig(leverage=3, usdc_balance_percent=0.001)
        BaseStrategyConfig(leverage=3, usdc_balance_percent=0.999)

    def test_default_creates_correct_config(self):
        config = BaseStrategyConfig.default()
        assert isinstance(config, BaseStrategyConfig)
        assert config.leverage == 3
        assert config.min_yearly_performance == 15.0
        assert config.min_position_size == 10.0
        assert config.min_usdc_balance == 2.0
        assert config.usdc_balance_percent == 0.01
        assert config.min_rebalance_difference == 25.0
        assert config.drift_check_threshold == 25.0
        assert config.position_drift_threshold == 25.0


class TestAllocationData:
    """Tests for AllocationData dataclass."""

    def test_allocation_data_creation(self):
        data = AllocationData(
            name="Bitcoin",
            symbol="BTC",
            target_value=5000.0,
            current_value=4500.0,
            difference=500.0,
            market_cap=1_200_000_000_000,
            price_change_1y=120.5,
        )
        assert data.name == "Bitcoin"
        assert data.symbol == "BTC"
        assert data.target_value == 5000.0
        assert data.current_value == 4500.0
        assert data.difference == 500.0
        assert data.market_cap == 1_200_000_000_000
        assert data.price_change_1y == 120.5

    def test_allocation_data_none_price_change(self):
        data = AllocationData(
            name="Bitcoin",
            symbol="BTC",
            target_value=5000.0,
            current_value=4500.0,
            difference=500.0,
            market_cap=1_200_000_000_000,
            price_change_1y=None,
        )
        assert data.price_change_1y is None


class TestPositionData:
    """Tests for PositionData dataclass."""

    def test_position_data_with_all_fields(self):
        data = PositionData(
            name="Bitcoin",
            symbol="BTC",
            current_value=5000.0,
            market_cap=1_200_000_000_000,
            price_change_1y=120.5,
        )
        assert data.name == "Bitcoin"
        assert data.current_value == 5000.0
        assert data.market_cap == 1_200_000_000_000
        assert data.price_change_1y == 120.5

    def test_position_data_minimal(self):
        data = PositionData(
            name="Bitcoin",
            symbol="BTC",
            current_value=5000.0,
        )
        assert data.market_cap is None
        assert data.price_change_1y is None


class _ConcreteStrategy(BaseStrategy):
    """Concrete strategy subclass for testing BaseStrategy."""

    def get_strategy_params(self) -> Tuple[List[Dict], Dict[str, str], Dict]:
        return [], {}, {}

    def filter_top_cryptos(
        self, cryptos: List[Dict], all_mids: Dict[str, str], meta: Dict
    ) -> List[Dict]:
        return cryptos


class TestBaseStrategyInstantiation:
    """Tests for BaseStrategy ABC and instantiation."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_concrete_subclass_no_config(self):
        strategy = _ConcreteStrategy()
        assert isinstance(strategy.config, BaseStrategyConfig)
        assert strategy.config.leverage == 3

    def test_concrete_subclass_with_config(self):
        config = BaseStrategyConfig(leverage=5)
        strategy = _ConcreteStrategy(config=config)
        assert strategy.config.leverage == 5
        assert strategy.config is config

    def test_concrete_subclass_with_default_config(self):
        config = BaseStrategyConfig.default()
        strategy = _ConcreteStrategy(config=config)
        assert strategy.config.leverage == 3

    @pytest.mark.asyncio
    async def test_abstract_methods_raise_not_implemented(self):
        class IncompleteStrategy(BaseStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()


class TestCalculateAccountValues:
    """Tests for calculate_account_values."""

    @pytest.fixture
    def strategy(self):
        return _ConcreteStrategy()

    def _make_user_state(self, positions=None, total_raw_usd="1000.0"):
        return {
            "assetPositions": positions or [],
            "crossMarginSummary": {
                "totalRawUsd": total_raw_usd,
            },
        }

    def test_no_positions_only_usdc(self, strategy):
        user_state = self._make_user_state(positions=[], total_raw_usd="5000.0")
        pos_values, total_value, usdc_target = strategy.calculate_account_values(user_state, 3)

        assert pos_values == {}
        assert total_value == 5000.0 * 3
        expected_target = max(5000.0 * 3 * 0.01 / 3, 2.0)
        assert usdc_target == expected_target

    def test_single_position(self, strategy):
        user_state = self._make_user_state(
            positions=[
                {"position": {"coin": "BTC", "positionValue": "8000.0"}}
            ],
            total_raw_usd="2000.0",
        )
        pos_values, total_value, usdc_target = strategy.calculate_account_values(user_state, 3)

        assert pos_values == {"BTC": 8000.0}
        assert total_value == 8000.0 + 2000.0 * 3
        expected_target = max((8000.0 + 2000.0 * 3) * 0.01 / 3, 2.0)
        assert usdc_target == expected_target

    def test_multiple_positions(self, strategy):
        user_state = self._make_user_state(
            positions=[
                {"position": {"coin": "BTC", "positionValue": "8000.0"}},
                {"position": {"coin": "ETH", "positionValue": "3000.0"}},
            ],
            total_raw_usd="1000.0",
        )
        pos_values, total_value, usdc_target = strategy.calculate_account_values(user_state, 3)

        assert pos_values == {"BTC": 8000.0, "ETH": 3000.0}
        assert total_value == 8000.0 + 3000.0 + 1000.0 * 3
        expected_target = max((8000.0 + 3000.0 + 1000.0 * 3) * 0.01 / 3, 2.0)
        assert usdc_target == expected_target

    def test_usdc_target_uses_min_when_percent_too_low(self, strategy):
        config = BaseStrategyConfig(leverage=3, usdc_balance_percent=0.001, min_usdc_balance=100.0)
        strategy = _ConcreteStrategy(config=config)
        user_state = self._make_user_state(positions=[], total_raw_usd="50.0")

        _, total_value, usdc_target = strategy.calculate_account_values(user_state, 3)
        percent_based = total_value * 0.001 / 3
        assert percent_based < 100.0
        assert usdc_target == 100.0

    def test_usdc_target_uses_percent_when_larger(self, strategy):
        config = BaseStrategyConfig(leverage=3, usdc_balance_percent=0.5, min_usdc_balance=2.0)
        strategy = _ConcreteStrategy(config=config)
        user_state = self._make_user_state(positions=[], total_raw_usd="10000.0")

        _, total_value, usdc_target = strategy.calculate_account_values(user_state, 3)
        percent_based = total_value * 0.5 / 3
        assert usdc_target > 2.0
        assert usdc_target == percent_based


class TestCalculateAllocations:
    """Tests for calculate_allocations."""

    @pytest.fixture
    def strategy(self):
        return _ConcreteStrategy()

    def test_basic_allocation(self, strategy):
        top_cryptos = [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1_200_000_000_000,
             "price_change_percentage_1y_in_currency": 120.5},
            {"symbol": "ETH", "name": "Ethereum", "market_cap": 400_000_000_000,
             "price_change_percentage_1y_in_currency": 80.0},
        ]
        position_values = {"BTC": 5000.0}
        tradeable_balance = 10000.0
        total_market_cap = 1_600_000_000_000
        cryptos = top_cryptos
        usdc_balance = 1000.0
        usdc_target_balance = 200.0

        allocation_data, other_positions, usdc_diff = strategy.calculate_allocations(
            top_cryptos, position_values, tradeable_balance,
            total_market_cap, cryptos, usdc_balance, usdc_target_balance,
        )

        assert len(allocation_data) == 2
        assert len(other_positions) == 0
        btc_allocation = allocation_data[0]
        assert btc_allocation.symbol == "BTC"
        assert btc_allocation.name == "Bitcoin"
        btc_rel_mcap = (1_200_000_000_000 / 1_600_000_000_000) * 100
        btc_expected_target = (btc_rel_mcap / 100) * 10000.0
        assert btc_allocation.target_value == btc_expected_target
        assert btc_allocation.current_value == 5000.0
        assert btc_allocation.difference == btc_expected_target - 5000.0
        assert btc_allocation.price_change_1y == 120.5
        assert usdc_diff == usdc_target_balance - usdc_balance

    def test_skipped_coins_below_min_position(self, strategy):
        config = BaseStrategyConfig(leverage=3, min_position_size=5000.0)
        strategy = _ConcreteStrategy(config=config)
        top_cryptos = [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1_200_000_000_000,
             "price_change_percentage_1y_in_currency": 120.5},
            {"symbol": "SOL", "name": "Solana", "market_cap": 80_000_000_000,
             "price_change_percentage_1y_in_currency": 200.0},
        ]
        position_values = {}
        tradeable_balance = 6000.0
        total_market_cap = 1_280_000_000_000
        cryptos = top_cryptos
        usdc_balance = 500.0
        usdc_target_balance = 100.0

        allocation_data, other_positions, usdc_diff = strategy.calculate_allocations(
            top_cryptos, position_values, tradeable_balance,
            total_market_cap, cryptos, usdc_balance, usdc_target_balance,
        )

        sol_rel_mcap = (80_000_000_000 / 1_280_000_000_000) * 100
        sol_target = (sol_rel_mcap / 100) * 6000.0
        if sol_target < 5000.0:
            assert len(allocation_data) == 1
            assert allocation_data[0].symbol == "BTC"

    def test_other_positions_not_in_top_cryptos(self, strategy):
        top_cryptos = [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1_200_000_000_000,
             "price_change_percentage_1y_in_currency": 120.5},
        ]
        position_values = {"BTC": 5000.0, "SOL": 1000.0}
        tradeable_balance = 10000.0
        total_market_cap = 1_200_000_000_000
        cryptos = top_cryptos + [
            {"symbol": "SOL", "name": "Solana", "market_cap": 80_000_000_000,
             "price_change_percentage_1y_in_currency": 200.0},
        ]
        usdc_balance = 1000.0
        usdc_target_balance = 200.0

        allocation_data, other_positions, usdc_diff = strategy.calculate_allocations(
            top_cryptos, position_values, tradeable_balance,
            total_market_cap, cryptos, usdc_balance, usdc_target_balance,
        )

        assert len(allocation_data) == 1
        assert len(other_positions) == 1
        assert other_positions[0].symbol == "SOL"
        assert other_positions[0].current_value == 1000.0
        assert other_positions[0].market_cap == 80_000_000_000
        assert other_positions[0].price_change_1y == 200.0

    def test_other_position_no_coin_data(self, strategy):
        top_cryptos = [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1_200_000_000_000,
             "price_change_percentage_1y_in_currency": 120.5},
        ]
        position_values = {"BTC": 5000.0, "UNKNOWN": 500.0}
        tradeable_balance = 10000.0
        total_market_cap = 1_200_000_000_000
        cryptos = top_cryptos
        usdc_balance = 1000.0
        usdc_target_balance = 200.0

        allocation_data, other_positions, usdc_diff = strategy.calculate_allocations(
            top_cryptos, position_values, tradeable_balance,
            total_market_cap, cryptos, usdc_balance, usdc_target_balance,
        )

        assert other_positions[0].symbol == "UNKNOWN"
        assert other_positions[0].name == "UNKNOWN"
        assert other_positions[0].market_cap is None
        assert other_positions[0].price_change_1y is None

    def test_redistributes_skipped_value(self, strategy):
        config = BaseStrategyConfig(leverage=3, min_position_size=30.0)
        strategy = _ConcreteStrategy(config=config)
        top_cryptos = [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1_000_000_000_000,
             "price_change_percentage_1y_in_currency": 100.0},
            {"symbol": "ETH", "name": "Ethereum", "market_cap": 500_000_000_000,
             "price_change_percentage_1y_in_currency": 80.0},
            {"symbol": "SOL", "name": "Solana", "market_cap": 100_000_000_000,
             "price_change_percentage_1y_in_currency": 200.0},
        ]
        total_market_cap = 1_600_000_000_000
        tradeable_balance = 100.0

        sol_rel_mcap = (100_000_000_000 / total_market_cap) * 100
        sol_target = (sol_rel_mcap / 100) * tradeable_balance

        allocation_data, _, _ = strategy.calculate_allocations(
            top_cryptos, {}, tradeable_balance, total_market_cap,
            top_cryptos, 10.0, 2.0,
        )

        if sol_target < 30.0:
            assert len(allocation_data) == 2
            btc_target_initial = (1000_000_000_000 / total_market_cap) * 100 / 100 * tradeable_balance
            eth_target_initial = (500_000_000_000 / total_market_cap) * 100 / 100 * tradeable_balance
            valid_total = btc_target_initial + eth_target_initial
            assert allocation_data[0].target_value > btc_target_initial

    def test_empty_top_cryptos(self, strategy):
        allocation_data, other_positions, usdc_diff = strategy.calculate_allocations(
            [], {}, 10000.0, 0, [], 1000.0, 200.0,
        )
        assert allocation_data == []
        assert other_positions == []
        assert usdc_diff == 200.0 - 1000.0

    def test_usdc_difference_positive(self, strategy):
        allocation_data, other_positions, usdc_diff = strategy.calculate_allocations(
            [], {}, 10000.0, 0, [], 50.0, 200.0,
        )
        assert usdc_diff == 150.0

    def test_usdc_difference_negative(self, strategy):
        allocation_data, other_positions, usdc_diff = strategy.calculate_allocations(
            [], {}, 10000.0, 0, [], 500.0, 200.0,
        )
        assert usdc_diff == -300.0


class TestGenerateTableRows:
    """Tests for generate_table_rows."""

    @pytest.fixture
    def strategy(self):
        return _ConcreteStrategy()

    def test_basic_rows(self, strategy):
        allocation = AllocationData(
            name="Bitcoin", symbol="BTC",
            target_value=5000.0, current_value=4500.0,
            difference=500.0, market_cap=1_200_000_000_000,
            price_change_1y=120.5,
        )
        rows = strategy.generate_table_rows(allocation, 10000.0, 1_600_000_000_000, idx=1)

        assert len(rows) == 7
        assert rows[0] == [1, "Bitcoin", "BTC"]
        assert "120.50%" in rows[1][2]

    def test_missing_price_change(self, strategy):
        allocation = AllocationData(
            name="Bitcoin", symbol="BTC",
            target_value=5000.0, current_value=4500.0,
            difference=500.0, market_cap=1_200_000_000_000,
            price_change_1y=None,
        )
        rows = strategy.generate_table_rows(allocation, 10000.0, 1_600_000_000_000, idx=1)

        assert rows[1][2] == "-%"

    def test_zero_tradeable_balance(self, strategy):
        allocation = AllocationData(
            name="Bitcoin", symbol="BTC",
            target_value=5000.0, current_value=4500.0,
            difference=500.0, market_cap=1_200_000_000_000,
            price_change_1y=120.5,
        )
        rows = strategy.generate_table_rows(allocation, 0.0, 1_600_000_000_000, idx=1)

        assert rows[4][3] == "0.00%"

    def test_low_market_cap_allocation(self, strategy):
        allocation = AllocationData(
            name="SmallCoin", symbol="SML",
            target_value=10.0, current_value=0.0,
            difference=10.0, market_cap=500_000_000,
            price_change_1y=50.0,
        )
        rows = strategy.generate_table_rows(allocation, 10000.0, 1_600_000_000_000, idx=2)

        assert rows[2][3] is not None

    def test_idx_none(self, strategy):
        allocation = AllocationData(
            name="Bitcoin", symbol="BTC",
            target_value=5000.0, current_value=4500.0,
            difference=500.0, market_cap=1_200_000_000_000,
            price_change_1y=120.5,
        )
        rows = strategy.generate_table_rows(allocation, 10000.0, 1_600_000_000_000, idx=None)
        assert rows[0][0] == ""


class TestGenerateTableData:
    """Tests for generate_table_data."""

    @pytest.fixture
    def strategy(self):
        return _ConcreteStrategy()

    def test_empty_data(self, strategy):
        table = strategy.generate_table_data(
            [], [], 10000.0, 1_600_000_000_000, 1000.0, 200.0, -800.0,
        )
        assert len(table) >= 4
        assert any("USDC" in str(row) for row in table)

    def test_with_allocations(self, strategy):
        allocations = [
            AllocationData(
                name="Bitcoin", symbol="BTC",
                target_value=5000.0, current_value=4500.0,
                difference=500.0, market_cap=1_200_000_000_000,
                price_change_1y=120.5,
            ),
        ]
        table = strategy.generate_table_data(
            allocations, [], 10000.0, 1_600_000_000_000, 1000.0, 200.0, -800.0,
        )

        assert len(table) >= 11
        assert any("Bitcoin" in str(row) for row in table)
        assert any("USDC" in str(row) for row in table)

    def test_other_positions_with_market_cap(self, strategy):
        allocations = [
            AllocationData(
                name="Bitcoin", symbol="BTC",
                target_value=5000.0, current_value=4500.0,
                difference=500.0, market_cap=1_200_000_000_000,
                price_change_1y=120.5,
            ),
        ]
        other = [
            PositionData(
                name="Solana", symbol="SOL",
                current_value=1000.0, market_cap=80_000_000_000,
                price_change_1y=200.0,
            ),
        ]
        table = strategy.generate_table_data(
            allocations, other, 10000.0, 1_600_000_000_000, 1000.0, 200.0, -800.0,
        )

        assert any("Solana" in str(row) for row in table)
        assert any("SOL" in str(row) for row in table)

    def test_other_positions_without_market_cap(self, strategy):
        other = [
            PositionData(
                name="Unknown", symbol="UNKN",
                current_value=500.0,
            ),
        ]
        table = strategy.generate_table_data(
            [], other, 10000.0, 1_600_000_000_000, 1000.0, 200.0, -800.0,
        )

        assert any("UNKN" in str(row) for row in table)

    def test_usdc_summary_rows(self, strategy):
        table = strategy.generate_table_data(
            [], [], 10000.0, 1_600_000_000_000, 1000.0, 200.0, -800.0,
        )

        usdc_rows = [row for row in table if any("USDC" in str(cell) for cell in row)]
        assert len(usdc_rows) >= 1
        assert any("200.00$" in str(row) for row in table)
        assert any("1,000.00$" in str(row) for row in table)


class TestDisplayCryptoInfo:
    """Tests for display_crypto_info."""

    @pytest.fixture
    def strategy(self):
        return _ConcreteStrategy()

    @pytest.mark.asyncio
    async def test_display_crypto_info_success(self, strategy):
        update = MagicMock()
        cryptos = [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1_200_000_000_000,
             "price_change_percentage_1y_in_currency": 120.5},
        ]
        all_mids = {"BTC": "85000"}
        meta = {"universe": [{"name": "BTC", "maxLeverage": 10}]}
        user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "positionValue": "5000.0"}},
            ],
            "crossMarginSummary": {"totalRawUsd": "2000.0"},
        }

        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl_utils, \
                patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg:
            mock_hl_utils.info.user_state.return_value = user_state
            mock_hl_utils.address = "0xtest"
            mock_tg.reply = AsyncMock()

            await strategy.display_crypto_info(update, cryptos, all_mids, meta)

            mock_tg.reply.assert_called_once()
            args, kwargs = mock_tg.reply.call_args
            assert update in args
            assert "Leveraged account value" in args[1]
            assert kwargs.get("parse_mode") is not None

    @pytest.mark.asyncio
    async def test_display_crypto_info_error(self, strategy):
        update = MagicMock()
        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl_utils:
            mock_hl_utils.info.user_state.side_effect = Exception("API error")

            with patch("strategies.base_strategy.base_strategy.logger") as mock_logger:
                await strategy.display_crypto_info(update, [], {}, {})
                mock_logger.error.assert_called_once()


class TestAnalyze:
    """Tests for analyze."""

    @pytest.mark.asyncio
    async def test_analyze_delegates_to_display(self):
        strategy = _ConcreteStrategy()
        update = MagicMock()
        context = MagicMock()

        with patch.object(strategy, "display_crypto_info", new_callable=AsyncMock) as mock_display, \
                patch.object(strategy, "get_strategy_params") as mock_params:
            mock_params.return_value = ([{"symbol": "BTC", "name": "Bitcoin", "market_cap": 1}], {"BTC": "100"}, {"universe": []})
            await strategy.analyze(update, context)
            mock_display.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_error_handling(self):
        strategy = _ConcreteStrategy()
        with patch.object(strategy, "get_strategy_params", side_effect=Exception("test error")), \
                patch("strategies.base_strategy.base_strategy.logger") as mock_logger:
            await strategy.analyze(MagicMock(), MagicMock())
            mock_logger.error.assert_called_once()


class TestCheckPositionAllocationDrifts:
    """Tests for check_position_allocation_drifts."""

    @pytest.mark.asyncio
    async def test_early_return_when_unrealized_pnl_below_threshold(self):
        strategy = _ConcreteStrategy()
        context = MagicMock()

        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl:
            user_state = {
                "assetPositions": [
                    {"position": {"coin": "BTC", "unrealizedPnl": "10.0"}},
                ],
                "crossMarginSummary": {"totalRawUsd": "1000.0"},
            }
            mock_hl.info.user_state.return_value = user_state
            mock_hl.address = "0xtest"

            await strategy.check_position_allocation_drifts(context)
            assert mock_hl.info.user_state.called

    @pytest.mark.asyncio
    async def test_drift_alert_sent_when_exceeds_threshold(self):
        strategy = _ConcreteStrategy()
        context = MagicMock()
        user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "unrealizedPnl": "100.0"}},
            ],
            "crossMarginSummary": {"totalRawUsd": "5000.0"},
        }

        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl, \
                patch.object(strategy, "get_strategy_params") as mock_params, \
                patch.object(strategy, "filter_top_cryptos") as mock_filter, \
                patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg:
            mock_hl.info.user_state.return_value = user_state
            mock_hl.address = "0xtest"
            cryptos = [
                {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1_200_000_000_000,
                 "price_change_percentage_1y_in_currency": 120.5},
            ]
            mock_params.return_value = (cryptos, {"BTC": "85000"}, {})
            mock_filter.return_value = cryptos
            mock_tg.send = AsyncMock()

            config = BaseStrategyConfig(leverage=3, position_drift_threshold=10.0)
            strategy = _ConcreteStrategy(config=config)

            await strategy.check_position_allocation_drifts(context)
            mock_tg.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_error_on_exception(self):
        strategy = _ConcreteStrategy()
        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.info.user_state.side_effect = Exception("Connection error")

            with patch("strategies.base_strategy.base_strategy.logger") as mock_logger, \
                    patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg:
                mock_tg.send = AsyncMock()
                await strategy.check_position_allocation_drifts(MagicMock())
                mock_logger.error.assert_called_once()
                mock_tg.send.assert_called_once()


class TestRebalance:
    """Tests for rebalance."""

    @pytest.mark.asyncio
    async def test_no_exchange_returns(self):
        strategy = _ConcreteStrategy()
        update = MagicMock()
        context = MagicMock()

        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl, \
                patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg:
            mock_hl.get_exchange.return_value = None
            mock_tg.reply = AsyncMock()

            await strategy.rebalance(update, context)
            mock_tg.reply.assert_called()

    @pytest.mark.asyncio
    async def test_insufficient_balance(self):
        strategy = _ConcreteStrategy()
        update = MagicMock()
        context = MagicMock()

        with patch("strategies.base_strategy.base_strategy.hyperliquid_utils") as mock_hl, \
                patch("strategies.base_strategy.base_strategy.telegram_utils") as mock_tg, \
                patch("strategies.base_strategy.base_strategy.exit_all_positions", new_callable=AsyncMock):
            mock_hl.get_exchange.return_value = MagicMock()
            mock_tg.reply = AsyncMock()
            user_state = {
                "assetPositions": [],
                "crossMarginSummary": {"totalRawUsd": "1.0"},
            }
            mock_hl.info.user_state.return_value = user_state
            mock_hl.address = "0xtest"

            with patch.object(strategy, "get_strategy_params") as mock_params, \
                    patch.object(strategy, "filter_top_cryptos") as mock_filter:
                mock_params.return_value = ([], {}, {})
                mock_filter.return_value = []

                await strategy.rebalance(update, context)
                assert mock_tg.reply.call_count >= 2
                first_call = mock_tg.reply.call_args_list[0]


