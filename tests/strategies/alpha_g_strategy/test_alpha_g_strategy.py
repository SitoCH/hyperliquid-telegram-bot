import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from strategies.alpha_g_strategy.alpha_g_strategy import AlphaGStrategy, ReversalSignal


def _make_candle(o: float, h: float, l: float, c: float, T: int) -> Dict[str, Any]:
    return {"o": str(o), "h": str(h), "l": str(l), "c": str(c), "T": T}


class TestReversalSignal:
    def test_default_not_confirmed(self):
        signal = ReversalSignal(symbol="BTC", name="Bitcoin", movement_type="surge",
                                full_candles_change_pct=15.0, current_change_pct=-2.0,
                                current_price=50000.0)
        assert not signal.confirmed
        assert signal.reasons == []

    def test_confirmed_with_reasons(self):
        signal = ReversalSignal(symbol="BTC", name="Bitcoin", movement_type="surge",
                                full_candles_change_pct=15.0, current_change_pct=-2.0,
                                current_price=50000.0, confirmed=True,
                                reasons=["Long upper wick"])
        assert signal.confirmed
        assert "Long upper wick" in signal.reasons


class TestCoinVolumeMapFromMeta:
    def test_basic_mapping(self):
        meta = (
            {"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "SOL"}]},
            [{"dayNtlVlm": "1000000"}, {"dayNtlVlm": "500000"}, {"dayNtlVlm": "750000"}]
        )
        result = AlphaGStrategy._coin_volume_map_from_meta(meta)
        assert result == {"BTC": 1000000.0, "ETH": 500000.0, "SOL": 750000.0}

    def test_missing_volume_defaults_zero(self):
        meta: Any = (
            {"universe": [{"name": "BTC"}]},
            [{}]
        )
        result = AlphaGStrategy._coin_volume_map_from_meta(meta)
        assert result == {"BTC": 0.0}


class TestComputeLowLiquidityPositions:
    def test_no_positions(self):
        strategy = AlphaGStrategy()
        result = strategy._compute_low_liquidity_positions({"assetPositions": []}, {"BTC": 5000000.0})
        assert result == []

    def test_all_positions_above_threshold(self):
        strategy = AlphaGStrategy()
        user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "1.0"}},
                {"position": {"coin": "ETH", "szi": "5.0"}},
            ]
        }
        coin_volume_map = {"BTC": 5000000.0, "ETH": 10000000.0}
        result = strategy._compute_low_liquidity_positions(user_state, coin_volume_map)
        assert result == []

    def test_position_below_threshold(self):
        strategy = AlphaGStrategy()
        user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "1.0"}},
            ]
        }
        coin_volume_map = {"BTC": 1000000.0}
        result = strategy._compute_low_liquidity_positions(user_state, coin_volume_map)
        assert len(result) == 1
        assert "BTC" in result[0]

    def test_mixed_positions(self):
        strategy = AlphaGStrategy()
        user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "1.0"}},
                {"position": {"coin": "SOL", "szi": "10.0"}},
            ]
        }
        coin_volume_map = {"BTC": 5000000.0, "SOL": 100000.0}
        result = strategy._compute_low_liquidity_positions(user_state, coin_volume_map)
        assert len(result) == 1
        assert "SOL" in result[0]
        assert "BTC" not in result[0]

    def test_missing_volume_defaults_zero(self):
        strategy = AlphaGStrategy()
        user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "1.0"}},
            ]
        }
        result = strategy._compute_low_liquidity_positions(user_state, {})
        assert len(result) == 1
        assert "BTC" in result[0]

    def test_k_prefix_stripped(self):
        strategy = AlphaGStrategy()
        user_state = {
            "assetPositions": [
                {"position": {"coin": "kBTC", "szi": "1.0"}},
            ]
        }
        coin_volume_map = {"BTC": 100000.0}
        result = strategy._compute_low_liquidity_positions(user_state, coin_volume_map)
        assert len(result) == 1


class TestBuildPortfolioSummaryMessage:
    def test_no_positions(self):
        result = AlphaGStrategy._build_portfolio_summary_message({"assetPositions": []})
        assert "Long Positions: 0" in result
        assert "Short Positions: 0" in result

    def test_only_longs(self):
        user_state = {
            "assetPositions": [
                {"position": {"szi": "1.0", "positionValue": "5000.0", "marginUsed": "1000.0"}},
                {"position": {"szi": "2.0", "positionValue": "8000.0", "marginUsed": "2000.0"}},
            ]
        }
        result = AlphaGStrategy._build_portfolio_summary_message(user_state)
        assert "Long Positions: 2" in result
        assert "Short Positions: 0" in result
        assert "3,000.00" in result

    def test_only_shorts(self):
        user_state = {
            "assetPositions": [
                {"position": {"szi": "-1.0", "positionValue": "5000.0", "marginUsed": "1000.0"}},
                {"position": {"szi": "-2.0", "positionValue": "8000.0", "marginUsed": "2000.0"}},
            ]
        }
        result = AlphaGStrategy._build_portfolio_summary_message(user_state)
        assert "Short Positions: 2" in result
        assert "Long Positions: 0" in result
        assert "3,000.00" in result

    def test_mixed_long_short(self):
        user_state = {
            "assetPositions": [
                {"position": {"szi": "1.0", "positionValue": "5000.0", "marginUsed": "1000.0"}},
                {"position": {"szi": "-0.5", "positionValue": "3000.0", "marginUsed": "500.0"}},
            ]
        }
        result = AlphaGStrategy._build_portfolio_summary_message(user_state)
        assert "Long Positions: 1" in result
        assert "Short Positions: 1" in result
        assert "1,000.00" in result
        assert "500.00" in result
        assert "3,000.00" in result

    def test_infinite_ratio_when_no_shorts(self):
        user_state = {
            "assetPositions": [
                {"position": {"szi": "1.0", "positionValue": "5000.0", "marginUsed": "1000.0"}},
            ]
        }
        result = AlphaGStrategy._build_portfolio_summary_message(user_state)
        assert "∞" in result


class TestExtractRecentCandles:
    def test_empty_candles(self):
        result = AlphaGStrategy._extract_recent_candles([], 1000, "BTC", 3)
        assert result == (None, None)

    def test_insufficient_complete_candles(self):
        candles = [_make_candle(100, 105, 95, 102, 100),
                   _make_candle(102, 108, 98, 105, 200)]
        result, partial = AlphaGStrategy._extract_recent_candles(candles, 500, "BTC", 3)
        assert result is None
        assert partial is None

    def test_normal_case_with_partial(self):
        candles = [
            _make_candle(100, 105, 95, 102, 100),
            _make_candle(102, 108, 98, 105, 200),
            _make_candle(105, 110, 100, 108, 300),
            _make_candle(108, 112, 104, 110, 400),
        ]
        result, partial = AlphaGStrategy._extract_recent_candles(candles, 350, "BTC", 3)
        assert result is not None
        assert len(result) == 3
        assert partial is not None
        assert partial['T'] >= 350

    def test_no_partial_candle(self):
        candles = [
            _make_candle(100, 105, 95, 102, 100),
            _make_candle(102, 108, 98, 105, 200),
            _make_candle(105, 110, 100, 108, 300),
        ]
        result, partial = AlphaGStrategy._extract_recent_candles(candles, 500, "BTC", 3)
        assert result is not None
        assert len(result) == 3
        assert partial is None


class TestComputeMeanATR:
    def test_standard_tr_formula(self):
        candles = [
            _make_candle(100, 110, 90, 105, 100),
            _make_candle(105, 115, 95, 110, 200),
            _make_candle(110, 120, 100, 115, 300),
        ]
        atr = AlphaGStrategy._compute_mean_atr(candles)
        assert atr > 0

    def test_single_candle(self):
        candles = [_make_candle(100, 110, 90, 105, 100)]
        atr = AlphaGStrategy._compute_mean_atr(candles)
        assert atr == 20.0

    def test_empty_candles(self):
        atr = AlphaGStrategy._compute_mean_atr([])
        assert atr == 0.0

    def test_tr_uses_max_of_three(self):
        candles = [
            _make_candle(100, 110, 90, 100, 100),
            _make_candle(100, 200, 50, 150, 200),
        ]
        atr = AlphaGStrategy._compute_mean_atr(candles)
        expected_first_tr = 20.0
        expected_second_tr = max(200 - 50, abs(200 - 100), abs(50 - 100))
        expected_atr = (expected_first_tr + expected_second_tr) / 2
        assert atr == expected_atr

    def test_clamping_negative_trs(self):
        candles = [
            _make_candle(100, 90, 80, 85, 100),
        ]
        atr = AlphaGStrategy._compute_mean_atr(candles)
        assert atr == 10.0


class TestClassifyMovement:
    def test_surge_both_thresholds_hit(self):
        strategy = AlphaGStrategy()
        strategy.ATR_MULT = 3.0
        candles = [_make_candle(100, 101, 99, 100, 100),
                   _make_candle(100, 101, 99, 101, 200),
                   _make_candle(101, 102, 100, 102, 300),
                   _make_candle(102, 103, 101, 125, 400)]
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._classify_movement(candles, {"symbol": "BTC", "name": "Bitcoin"}, 10.0)
        assert result is not None
        assert result[0] == "surge"
        assert result[1] > 0

    def test_crash_both_thresholds_hit(self):
        strategy = AlphaGStrategy()
        candles = [_make_candle(100, 101, 99, 100, 100),
                   _make_candle(100, 101, 99, 99, 200),
                   _make_candle(99, 100, 98, 98, 300),
                   _make_candle(98, 99, 97, 75, 400)]
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._classify_movement(candles, {"symbol": "BTC", "name": "Bitcoin"}, 10.0)
        assert result is not None
        assert result[0] == "crash"
        assert result[1] < 0

    def test_returns_none_when_atr_missed(self):
        strategy = AlphaGStrategy()
        strategy.ATR_MULT = 100.0
        candles = [_make_candle(100, 101, 99, 100.5, 100),
                   _make_candle(100.5, 102, 99.5, 101, 200)]
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._classify_movement(candles, {"symbol": "BTC", "name": "Bitcoin"}, 0.5)
        assert result is None

    def test_returns_none_when_fixed_threshold_missed(self):
        strategy = AlphaGStrategy()
        candles = [_make_candle(100, 101, 99, 100.5, 100),
                   _make_candle(100.5, 102, 99.5, 101, 200)]
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._classify_movement(candles, {"symbol": "BTC", "name": "Bitcoin"}, 50.0)
        assert result is None

    def test_returns_none_when_first_open_is_zero(self):
        strategy = AlphaGStrategy()
        candles = [_make_candle(0, 10, 0, 5, 100)]
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._classify_movement(candles, {"symbol": "BTC", "name": "Bitcoin"}, 10.0)
        assert result is None


class TestDetectPartialReversal:
    def test_surge_with_upper_wick(self):
        strategy = AlphaGStrategy()
        strategy.WICK_RATIO_MIN = 0.55
        full_candles = [_make_candle(100, 110, 90, 105, 100) for _ in range(22)]
        partial = _make_candle(120, 125, 110, 112, 500)
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._detect_partial_reversal("surge", partial, {"symbol": "BTC", "name": "Bitcoin"}, 15.0, full_candles)
        assert result is not None
        assert result.movement_type == "surge"

    def test_crash_with_lower_wick(self):
        strategy = AlphaGStrategy()
        strategy.WICK_RATIO_MIN = 0.55
        full_candles = [_make_candle(100, 110, 90, 105, 100) for _ in range(22)]
        partial = _make_candle(80, 85, 60, 82, 500)
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._detect_partial_reversal("crash", partial, {"symbol": "BTC", "name": "Bitcoin"}, -15.0, full_candles)
        assert result is not None
        assert result.movement_type == "crash"

    def test_no_reversal_same_direction(self):
        strategy = AlphaGStrategy()
        partial = _make_candle(100, 120, 95, 115, 500)
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._detect_partial_reversal("surge", partial, {"symbol": "BTC", "name": "Bitcoin"}, 15.0)
        assert result is None

    def test_returns_none_when_no_partial_candle(self):
        strategy = AlphaGStrategy()
        result = strategy._detect_partial_reversal("surge", None, {"symbol": "BTC", "name": "Bitcoin"}, 15.0)
        assert result is None

    def test_bb_upper_re_entry(self):
        strategy = AlphaGStrategy()
        strategy.WICK_RATIO_MIN = 0.55
        closes = [100.0] * 18 + [300.0, 350.0]
        candles = [_make_candle(c - 1, c + 2, c - 2, c, i * 100) for i, c in enumerate(closes)]
        partial = _make_candle(200, 205, 150, 180, 99999)
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._detect_partial_reversal("surge", partial, {"symbol": "BTC", "name": "Bitcoin"}, 15.0, candles)
        if result:
            assert any("BB" in r for r in result.reasons)

    def test_bb_lower_re_entry(self):
        strategy = AlphaGStrategy()
        strategy.WICK_RATIO_MIN = 0.55
        closes = [200.0] * 18 + [10.0, 5.0]
        candles = [_make_candle(c + 1, c + 3, c - 1, c, i * 100) for i, c in enumerate(closes)]
        partial = _make_candle(100, 120, 80, 110, 99999)
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = strategy._detect_partial_reversal("crash", partial, {"symbol": "BTC", "name": "Bitcoin"}, -15.0, candles)
        if result:
            assert any("BB" in r for r in result.reasons)


class TestDetectConfirmedReversal:
    def test_surge_then_reversal_then_confirmation(self):
        candles = []
        for i in range(5):
            candles.append(_make_candle(100 + i * 20, 100 + i * 20 + 5, 100 + i * 20 - 5, 100 + i * 20 + 3, i * 100))
        candles.append(_make_candle(180, 175, 165, 170, 800))
        candles.append(_make_candle(170, 172, 166, 168, 1000))

        coin_entry = {"symbol": "BTC", "name": "Bitcoin"}
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            result = AlphaGStrategy._detect_confirmed_reversal(candles, 950, coin_entry, 3, 15.0)
        assert result is not None
        assert result.confirmed
        assert result.movement_type == "surge"

    def test_insufficient_candles_returns_none(self):
        candles = [_make_candle(100, 105, 95, 102, 100)]
        result = AlphaGStrategy._detect_confirmed_reversal(candles, 500, {"symbol": "BTC", "name": "Bitcoin"}, 3, 15.0)
        assert result is None

    def test_no_trend_returns_none(self):
        candles = [
            _make_candle(100, 102, 98, 101, 100),
            _make_candle(101, 103, 99, 102, 200),
            _make_candle(102, 104, 100, 103, 300),
            _make_candle(103, 105, 101, 102, 400),
            _make_candle(102, 104, 100, 103, 500),
        ]
        result = AlphaGStrategy._detect_confirmed_reversal(candles, 450, {"symbol": "BTC", "name": "Bitcoin"}, 3, 15.0)
        assert result is None


class TestComputeBollinger:
    def test_normal_case(self):
        closes = [100.0, 102.0, 101.0, 103.0, 99.0, 98.0, 100.0, 101.0, 102.0, 100.0]
        mean, upper, lower, std = AlphaGStrategy._compute_bollinger(closes, 10)
        assert abs(mean - 100.6) < 0.01
        assert upper > mean
        assert lower < mean
        assert std > 0

    def test_empty_list(self):
        result = AlphaGStrategy._compute_bollinger([], 20)
        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_less_than_period(self):
        closes = [100.0, 101.0]
        mean, upper, lower, std = AlphaGStrategy._compute_bollinger(closes, 20)
        assert abs(mean - 100.5) < 0.01
        assert upper > mean
        assert lower < mean

    def test_single_value(self):
        closes = [100.0]
        mean, upper, lower, std = AlphaGStrategy._compute_bollinger(closes, 5)
        assert mean == 100.0
        assert std == 0.0
        assert upper == 100.0
        assert lower == 100.0


class TestBuildReversalLines:
    def test_empty_reversals(self):
        result = AlphaGStrategy._build_reversal_lines([])
        assert len(result) == 2
        assert "Reversal Signals" in result[0]

    def test_single_reversal(self):
        reversals = [
            ReversalSignal(symbol="BTC", name="Bitcoin", movement_type="surge",
                           full_candles_change_pct=15.0, current_change_pct=-2.0,
                           current_price=50000.0, reasons=["Long upper wick"]),
        ]
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.telegram_utils.get_link",
                   side_effect=lambda text, action: text):
            result = AlphaGStrategy._build_reversal_lines(reversals)
        assert any("BTC" in line for line in result)
        assert any("Bitcoin" in line for line in result)
        assert any("Surge" in line for line in result)
        assert any("Long upper wick" in line for line in result)

    def test_sorted_by_magnitude(self):
        reversals = [
            ReversalSignal(symbol="SOL", name="Solana", movement_type="crash",
                           full_candles_change_pct=-5.0, current_change_pct=1.0,
                           current_price=100.0),
            ReversalSignal(symbol="BTC", name="Bitcoin", movement_type="surge",
                           full_candles_change_pct=20.0, current_change_pct=-3.0,
                           current_price=50000.0),
        ]
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.telegram_utils.get_link",
                   side_effect=lambda link_text, link_action: link_text):
            result = AlphaGStrategy._build_reversal_lines(reversals)
        joined = "\n".join(result)
        btc_idx = joined.index("BTC")
        sol_idx = joined.index("SOL")
        assert btc_idx < sol_idx


class TestComputePremoveStatusFromCandles:
    def test_pump_status(self):
        strategy = AlphaGStrategy()
        strategy._lookback_days = 3
        user_state = {
            "assetPositions": [{"coin": "BTC", "position": {"szi": "1.0"}}]
        }
        candles = [
            _make_candle(100, 102, 98, 100, 100),
            _make_candle(100, 105, 99, 104, 200),
            _make_candle(104, 115, 102, 112, 300),
            _make_candle(112, 130, 110, 128, 400),
            _make_candle(128, 132, 125, 130, 500),
        ]
        candles_by_symbol = {"BTC": candles}
        result = strategy._compute_premove_status_from_candles(user_state, candles_by_symbol)
        assert len(result) >= 1
        assert any("BTC" in line for line in result)
        assert any("pump" in line for line in result)

    def test_no_eligible_position(self):
        strategy = AlphaGStrategy()
        user_state = {"assetPositions": [{"coin": "BTC", "position": {"szi": "1.0"}}]}
        result = strategy._compute_premove_status_from_candles(user_state, {})
        assert result == []

    def test_skips_small_moves(self):
        strategy = AlphaGStrategy()
        strategy._lookback_days = 1
        user_state = {
            "assetPositions": [{"coin": "BTC", "position": {"szi": "1.0"}}]
        }
        candles = [
            _make_candle(100, 101, 99, 100.5, 100),
            _make_candle(100.5, 102, 100, 101, 200),
            _make_candle(101, 102, 100, 101.5, 300),
        ]
        candles_by_symbol = {"BTC": candles}
        result = strategy._compute_premove_status_from_candles(user_state, candles_by_symbol)
        assert result == []


class TestFilterTopCoins:
    def test_filters_by_volume_and_availability(self):
        strategy = AlphaGStrategy()
        meta = (
            {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
            [{"dayNtlVlm": "100000000"}, {"dayNtlVlm": "50000000"}]
        )
        all_mids = {"BTC": "50000", "ETH": "3000"}

        mock_coins = [
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000000000000},
            {"symbol": "ETH", "name": "Ethereum", "market_cap": 500000000000},
            {"symbol": "SOL", "name": "Solana", "market_cap": 80000000000},
        ]

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.fetch_cryptos.return_value = mock_coins
            result = strategy.filter_top_coins(meta, all_mids)

        assert any(c["symbol"] == "BTC" for c in result)
        assert any(c["symbol"] == "ETH" for c in result)
        assert not any(c["symbol"] == "SOL" for c in result)

    def test_sorted_by_market_cap_desc(self):
        strategy = AlphaGStrategy()
        meta = (
            {"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "SOL"}]},
            [{"dayNtlVlm": "100000000"}, {"dayNtlVlm": "50000000"}, {"dayNtlVlm": "30000000"}]
        )
        all_mids = {"BTC": "50000", "ETH": "3000", "SOL": "100"}

        mock_coins = [
            {"symbol": "ETH", "name": "Ethereum", "market_cap": 500000000000},
            {"symbol": "SOL", "name": "Solana", "market_cap": 80000000000},
            {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000000000000},
        ]

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.fetch_cryptos.return_value = mock_coins
            result = strategy.filter_top_coins(meta, all_mids)

        assert result[0]["market_cap"] >= result[-1]["market_cap"]
        assert result[0]["symbol"] == "BTC"
        assert result[-1]["symbol"] == "SOL"

    def test_volume_below_minimum_excluded(self):
        strategy = AlphaGStrategy()
        strategy.COIN_MIN_VOLUME = 1000000
        meta = (
            {"universe": [{"name": "BTC"}, {"name": "MICRO"}]},
            [{"dayNtlVlm": "5000000"}, {"dayNtlVlm": "500"}]
        )
        all_mids = {"BTC": "50000", "MICRO": "1"}

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.fetch_cryptos.return_value = [
                {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000000000000},
                {"symbol": "MICRO", "name": "MicroCoin", "market_cap": 1000000},
            ]
            result = strategy.filter_top_coins(meta, all_mids)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"

    def test_coin_not_on_hyperliquid_excluded(self):
        strategy = AlphaGStrategy()
        meta = (
            {"universe": [{"name": "BTC"}]},
            [{"dayNtlVlm": "5000000"}]
        )
        all_mids = {"BTC": "50000"}

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.hyperliquid_utils") as mock_hl:
            mock_hl.fetch_cryptos.return_value = [
                {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000000000000},
                {"symbol": "FAKE", "name": "FakeCoin", "market_cap": 500000000},
            ]
            result = strategy.filter_top_coins(meta, all_mids)

        assert len(result) == 1
        assert result[0]["symbol"] == "BTC"


class TestDetectPriceMovements:
    @pytest.mark.asyncio
    async def test_detects_reversals(self):
        strategy = AlphaGStrategy()
        coins = [{"symbol": "BTC", "name": "Bitcoin"}]
        DAY_MS = 1440 * 60 * 1000
        midnight = 1781049600000
        candles = [
            _make_candle(100, 101, 99, 100, midnight - 5 * DAY_MS),
            _make_candle(100, 102, 99, 110, midnight - 4 * DAY_MS),
            _make_candle(110, 112, 109, 120, midnight - 3 * DAY_MS),
            _make_candle(120, 122, 119, 130, midnight - 2 * DAY_MS),
            _make_candle(130, 131, 128, 125, midnight - DAY_MS),
            _make_candle(125, 126, 120, 122, midnight),
        ]

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.get_candles_with_cache",
                   new_callable=AsyncMock) as mock_get_candles, \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"), \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.time.time",
                      return_value=midnight / 1000 + 3600):
            mock_get_candles.return_value = candles
            reversals, candles_by_symbol = await strategy.detect_price_movements(coins, 3, 20.0)

        assert len(reversals) >= 1
        assert reversals[0].symbol == "BTC"
        assert reversals[0].confirmed

    @pytest.mark.asyncio
    async def test_empty_coins_list(self):
        strategy = AlphaGStrategy()
        with patch("strategies.alpha_g_strategy.alpha_g_strategy.get_candles_with_cache",
                   new_callable=AsyncMock), \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            reversals, candles_by_symbol = await strategy.detect_price_movements([], 3, 20.0)
        assert reversals == []
        assert candles_by_symbol == {}

    @pytest.mark.asyncio
    async def test_exception_handling_per_coin(self):
        strategy = AlphaGStrategy()
        coins = [{"symbol": "BTC", "name": "Bitcoin"}]

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.get_candles_with_cache",
                   new_callable=AsyncMock) as mock_get_candles, \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            mock_get_candles.side_effect = Exception("API error")
            reversals, candles_by_symbol = await strategy.detect_price_movements(coins, 3, 20.0)

        assert reversals == []


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_no_positions(self):
        strategy = AlphaGStrategy()
        update = MagicMock()
        context = MagicMock()

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.hyperliquid_utils") as mock_hl, \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.telegram_utils") as mock_tg, \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            mock_hl.info.user_state.return_value = {"assetPositions": []}
            mock_hl.address = "0xtest"
            mock_tg.reply = AsyncMock()

            await strategy.analyze(update, context)

            mock_tg.reply.assert_called_once()
            args = mock_tg.reply.call_args[0]
            assert "No open positions found" in str(args[1])

    @pytest.mark.asyncio
    async def test_analyze_with_positions(self):
        strategy = AlphaGStrategy()
        update = MagicMock()
        context = MagicMock()

        user_state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "1.0", "positionValue": "5000.0", "marginUsed": "1000.0"}},
            ]
        }

        meta = (
            {"universe": [{"name": "BTC"}]},
            [{"dayNtlVlm": "5000000"}]
        )
        all_mids = {"BTC": "50000"}

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.hyperliquid_utils") as mock_hl, \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.telegram_utils") as mock_tg, \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.get_candles_with_cache",
                      new_callable=AsyncMock) as mock_get_candles, \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            mock_hl.info.user_state.return_value = user_state
            mock_hl.info.meta_and_asset_ctxs.return_value = meta
            mock_hl.info.all_mids.return_value = all_mids
            mock_hl.address = "0xtest"
            mock_hl.fetch_cryptos.return_value = [
                {"symbol": "BTC", "name": "Bitcoin", "market_cap": 1000000000000},
            ]
            mock_tg.reply = AsyncMock()
            mock_tg.send = AsyncMock()
            mock_get_candles.return_value = []

            await strategy.analyze(update, context)

            assert mock_tg.reply.call_count >= 1
            assert mock_tg.send.called


class TestInitStrategy:
    @pytest.mark.asyncio
    async def test_init_strategy(self):
        strategy = AlphaGStrategy()
        context = MagicMock()

        with patch("strategies.alpha_g_strategy.alpha_g_strategy.telegram_utils") as mock_tg, \
                patch("strategies.alpha_g_strategy.alpha_g_strategy.logger"):
            mock_tg.get_link = MagicMock(return_value="https://example.com")
            await strategy.init_strategy(context)

            assert mock_tg.add_buttons.called
            assert mock_tg.add_handler.called
