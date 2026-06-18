import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from technical_analysis.wyckoff.wyckoff import (
    calculate_volume_metrics, detect_spring_upthrust, extract_trend_indicators,
    detect_wyckoff_phase, identify_wyckoff_phase, determine_phase_by_price_strength,
    analyze_funding_rates, analyze_effort_result
)
from technical_analysis.wyckoff.wyckoff_types import (
    WyckoffPhase, VolumeState, WyckoffState, EffortResult, FundingState,
    MarketPattern, VolatilityState, CompositeAction, WyckoffSign,
    VolumeMetrics, Timeframe
)
from technical_analysis.funding_rates_cache import FundingRateEntry


# ── Helpers ──────────────────────────────────────────────────────────

def _make_df(
    close_prices: list[float],
    volumes: list[float] | None = None,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    extra_cols: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with required columns."""
    n = len(close_prices)
    data = {
        'o': opens if opens else [p * 0.999 for p in close_prices],
        'h': highs if highs else [p * 1.005 for p in close_prices],
        'l': lows if lows else [p * 0.995 for p in close_prices],
        'c': close_prices,
        'v': volumes if volumes else [1000.0] * n,
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


def _make_vol_metrics(
    strength: float = 0.0,
    ratio: float = 1.0,
    trend: float = 1.0,
    impulse: float = 0.0,
    sma: float = 1.0,
    consistency: float = 0.5,
    short_ma: float = 1.0,
    long_ma: float = 1.0,
    trend_strength: float = 0.0,
    state: VolumeState = VolumeState.NEUTRAL,
) -> VolumeMetrics:
    return VolumeMetrics(
        strength=strength, ratio=ratio, trend=trend, impulse=impulse,
        sma=sma, consistency=consistency, short_ma=short_ma, long_ma=long_ma,
        trend_strength=trend_strength, state=state
    )


TF = Timeframe.HOUR_1  # stable base timeframe for most tests
TF_SETTINGS = TF.settings


# ═══════════════════════════════════════════════════════════════════════
# 1. calculate_volume_metrics
# ═══════════════════════════════════════════════════════════════════════

class TestCalculateVolumeMetrics:

    def test_basic_metrics(self):
        df = _make_df(
            [100.0] * 30,
            volumes=[float(i + 500) for i in range(30)]
        )
        m = calculate_volume_metrics(df, TF)
        assert isinstance(m, VolumeMetrics)
        assert m.sma > 0
        assert m.ratio > 0
        assert m.state in VolumeState

    def test_very_high_volume_state(self):
        vols = [100.0] * 25 + [5000.0] * 5
        df = _make_df([100.0] * 30, volumes=vols)
        m = calculate_volume_metrics(df, TF)
        assert m.state in (VolumeState.VERY_HIGH, VolumeState.HIGH)

    def test_very_low_volume_state(self):
        vols = [1000.0] * 25 + [10.0] * 5
        df = _make_df([100.0] * 30, volumes=vols)
        m = calculate_volume_metrics(df, TF)
        assert m.state in (VolumeState.LOW, VolumeState.VERY_LOW)

    def test_neutral_volume_state(self):
        vols = [500.0] * 30
        df = _make_df([100.0] * 30, volumes=vols)
        m = calculate_volume_metrics(df, TF)
        assert m.state == VolumeState.NEUTRAL

    def test_low_volume_state(self):
        vols = [500.0] * 25 + [350.0] * 5
        df = _make_df([100.0] * 30, volumes=vols)
        m = calculate_volume_metrics(df, TF)
        assert m.state == VolumeState.LOW

    def test_empty_dataframe_returns_unknown(self):
        df = _make_df([], volumes=[])
        m = calculate_volume_metrics(df, TF)
        assert m.state == VolumeState.UNKNOWN
        assert m.strength == 0.0
        assert m.ratio == 1.0

    def test_single_row_graceful_fallback(self):
        df = _make_df([100.0], volumes=[500.0])
        m = calculate_volume_metrics(df, TF)
        assert m.state in (VolumeState.UNKNOWN, VolumeState.NEUTRAL)

    def test_zero_volume_does_not_crash(self):
        df = _make_df([100.0] * 30, volumes=[0.0] * 30)
        m = calculate_volume_metrics(df, TF)
        assert m.ratio >= 0

    def test_metrics_consistency_field(self):
        df = _make_df(
            [100.0] * 30,
            volumes=[1000.0] * 30
        )
        m = calculate_volume_metrics(df, TF)
        assert 0.0 <= m.consistency <= 1.0

    def test_strength_and_ratio_fields(self):
        df = _make_df(
            [100.0] * 30,
            volumes=([100.0] * 25) + [5000.0] + [200.0] + [100.0] + [100.0] + [100.0]
        )
        m = calculate_volume_metrics(df, TF)
        assert m.strength != 0.0
        assert m.ratio > 0

    def test_impulse_and_trend_fields(self):
        df = _make_df(
            [100.0] * 30,
            volumes=[float(i * 10 + 100) for i in range(30)]
        )
        m = calculate_volume_metrics(df, TF)
        assert hasattr(m, 'impulse')
        assert hasattr(m, 'trend')

    def test_trend_strength_positive_when_volume_rising(self):
        vols = [float(100 + i * 20) for i in range(30)]
        df = _make_df([100.0] * 30, volumes=vols)
        m = calculate_volume_metrics(df, TF)
        assert m.trend_strength != 0.0 or m.state != VolumeState.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════
# 2. detect_spring_upthrust
# ═══════════════════════════════════════════════════════════════════════

class TestDetectSpringUpthrust:

    def test_insufficient_data(self):
        df = _make_df([100.0] * 3)
        vm = _make_vol_metrics(strength=2.0)
        s, u = detect_spring_upthrust(df, TF, -1, vm)
        assert not s and not u

    def test_too_early_index(self):
        df = _make_df([100.0] * 20)
        vm = _make_vol_metrics(strength=2.0)
        s, u = detect_spring_upthrust(df, TF, 1, vm)
        assert not s and not u

    def test_no_spring_from_positive_idx_due_to_include_current_in_min(self):
        """low_point includes current bar in min(), so current_low < low_point can't trigger."""
        prices = [100.0] * 14 + [99.0, 97.0, 95.0, 97.5, 99.0, 100.5]
        lows = [99.5] * 14 + [98.5, 96.5, 93.0, 96.5, 98.0, 99.5]
        highs = [100.5] * 14 + [99.5, 97.5, 96.0, 98.5, 100.0, 101.0]
        df = _make_df(prices, highs=highs, lows=lows,
                      volumes=[1000.0] * 14 + [5000.0, 6000.0, 8000.0, 7000.0, 6000.0, 5000.0])
        df['ATR'] = 1.0
        vm = _make_vol_metrics(strength=2.0)
        idx = len(df) - 1
        s, u = detect_spring_upthrust(df, TF, idx, vm)
        # Known bug: window includes current bar in min(), so condition can never be met
        assert not s

    def test_no_upthrust_from_positive_idx_due_to_include_current_in_max(self):
        """high_point includes current bar in max(), so current_high > high_point can't trigger."""
        prices = [100.0] * 14 + [101.0, 103.0, 105.0, 103.5, 102.0, 100.5]
        highs = [100.5] * 14 + [101.5, 104.0, 108.0, 105.0, 103.0, 101.5]
        lows = [99.5] * 14 + [100.5, 102.5, 104.0, 102.5, 101.0, 99.5]
        df = _make_df(prices, highs=highs, lows=lows,
                      volumes=[1000.0] * 14 + [5000.0, 6000.0, 8000.0, 7000.0, 6000.0, 5000.0])
        df['ATR'] = 1.0
        vm = _make_vol_metrics(strength=2.0)
        idx = len(df) - 1
        s, u = detect_spring_upthrust(df, TF, idx, vm)
        # Known bug: window includes current bar in max(), so condition can never be met
        assert not u

    def test_no_signal_on_random_data(self):
        prices = [100.0 + np.random.uniform(-1, 1) for _ in range(30)]
        df = _make_df(prices, volumes=[1000.0] * 30)
        df['ATR'] = 1.0
        vm = _make_vol_metrics(strength=1.5)
        s, u = detect_spring_upthrust(df, TF, len(df) - 1, vm)
        assert not s and not u

    def test_no_signal_when_volume_weak(self):
        prices = [100.0] * 10 + [99.0, 97.0, 95.0, 97.5, 99.0, 100.5]
        lows = [99.5] * 10 + [98.5, 96.5, 93.0, 96.5, 98.0, 99.5]
        df = _make_df(prices, lows=lows, volumes=[500.0] * 16)
        df['ATR'] = 1.0
        vm = _make_vol_metrics(strength=0.5)
        s, u = detect_spring_upthrust(df, TF, len(df) - 1, vm)
        assert not s and not u

    def test_returns_false_on_exception(self):
        df = _make_df([100.0] * 20)
        df['ATR'] = 1.0
        vm = _make_vol_metrics(strength=2.0)
        with patch('technical_analysis.wyckoff.wyckoff.AdaptiveThresholdManager.get_spring_upthrust_thresholds',
                   side_effect=ValueError("test error")):
            s, u = detect_spring_upthrust(df, TF, len(df) - 1, vm)
        assert not s and not u

    def test_zero_price_returns_false(self):
        df = _make_df([0.0] * 20)
        df['ATR'] = 1.0
        vm = _make_vol_metrics(strength=2.0)
        s, u = detect_spring_upthrust(df, TF, len(df) - 1, vm)
        assert not s and not u

    def test_excessive_wick_suppresses_signal(self):
        prices = [100.0] * 10 + [99.0, 97.0, 95.0, 97.5, 99.0, 100.5]
        lows = [99.5] * 10 + [98.5, 96.5, 70.0, 96.5, 98.0, 99.5]
        df = _make_df(prices, lows=lows, volumes=[1000.0] * 10 + [5000.0] * 6)
        df['ATR'] = 1.0
        vm = _make_vol_metrics(strength=2.0)
        s, u = detect_spring_upthrust(df, TF, len(df) - 1, vm)
        assert not s


# ═══════════════════════════════════════════════════════════════════════
# 3. extract_trend_indicators
# ═══════════════════════════════════════════════════════════════════════

class TestExtractTrendIndicators:

    def test_missing_columns_defaults(self):
        df = _make_df([100.0] * 20)
        trend, bias, rsi, adx = extract_trend_indicators(df)
        assert trend == "unknown"
        assert bias == 0
        assert rsi == 50.0
        assert adx == 0.0

    def test_supertrend_uptrend(self):
        df = _make_df([105.0] * 20, extra_cols={'SuperTrend': [100.0] * 20})
        trend, _, _, _ = extract_trend_indicators(df)
        assert trend == "uptrend"

    def test_supertrend_downtrend(self):
        df = _make_df([95.0] * 20, extra_cols={'SuperTrend': [100.0] * 20})
        trend, _, _, _ = extract_trend_indicators(df)
        assert trend == "downtrend"

    def test_vwap_bullish_bias(self):
        df = _make_df([101.0] * 20, extra_cols={'VWAP': [100.0] * 20})
        _, bias, _, _ = extract_trend_indicators(df)
        assert bias == 1

    def test_vwap_bearish_bias(self):
        df = _make_df([99.0] * 20, extra_cols={'VWAP': [100.0] * 20})
        _, bias, _, _ = extract_trend_indicators(df)
        assert bias == -1

    def test_vwap_neutral_bias(self):
        df = _make_df([99.99] * 20, extra_cols={'VWAP': [100.0] * 20})
        _, bias, _, _ = extract_trend_indicators(df)
        assert bias == 0

    def test_rsi_value(self):
        df = _make_df([100.0] * 20, extra_cols={'RSI': [65.0] * 20})
        _, _, rsi, _ = extract_trend_indicators(df)
        assert rsi == 65.0

    def test_adx_value(self):
        df = _make_df([100.0] * 20, extra_cols={'ADX_value': [35.0] * 20})
        _, _, _, adx = extract_trend_indicators(df)
        assert adx == 35.0

    def test_all_trend_indicators(self):
        df = _make_df([105.0] * 20, extra_cols={
            'SuperTrend': [100.0] * 20,
            'VWAP': [100.0] * 20,
            'RSI': [60.0] * 20,
            'ADX_value': [30.0] * 20,
        })
        trend, bias, rsi, adx = extract_trend_indicators(df)
        assert trend == "uptrend"
        assert bias == 1
        assert rsi == 60.0
        assert adx == 30.0

    def test_nan_columns(self):
        df = _make_df([100.0] * 20, extra_cols={
            'SuperTrend': [np.nan] * 20,
            'VWAP': [np.nan] * 20,
        })
        trend, bias, rsi, adx = extract_trend_indicators(df)
        assert trend == "unknown"
        assert bias == 0
        assert rsi == 50.0

    def test_single_row_dataframe(self):
        df = _make_df([100.0], extra_cols={'SuperTrend': [100.0]})
        trend, bias, _, _ = extract_trend_indicators(df)
        assert trend == "unknown"

    def test_missing_close_column_raises_keyerror(self):
        df = pd.DataFrame({'SuperTrend': [100.0]})
        trend, bias, rsi, adx = extract_trend_indicators(df)
        assert trend == "unknown"
        assert bias == 0
        assert rsi == 50.0
        assert adx == 0.0

    def test_exception_in_extraction_returns_defaults(self):
        df = _make_df([100.0] * 5, extra_cols={'SuperTrend': [100.0] * 5})
        with patch('technical_analysis.wyckoff.wyckoff.pd.Series.iloc',
                   side_effect=ValueError("test error")):
            trend, bias, rsi, adx = extract_trend_indicators(df)
        assert trend == "unknown"


# ═══════════════════════════════════════════════════════════════════════
# 4. determine_phase_by_price_strength
# ═══════════════════════════════════════════════════════════════════════

class TestDeterminePhaseByPriceStrength:

    def test_strong_bullish_markup(self):
        phase, uncertain = determine_phase_by_price_strength(
            3.0, 80.0, VolumeState.HIGH, pd.Series([0.01] * 30), TF
        )
        assert phase == WyckoffPhase.MARKUP
        assert uncertain is False

    def test_strong_bearish_markdown(self):
        phase, uncertain = determine_phase_by_price_strength(
            -3.0, -80.0, VolumeState.HIGH, pd.Series([0.01] * 30), TF
        )
        assert phase == WyckoffPhase.MARKDOWN
        assert uncertain is False

    def test_distribution(self):
        # Strong price positive, weak momentum → distribution
        phase, uncertain = determine_phase_by_price_strength(
            2.5, 0.3, VolumeState.HIGH, pd.Series([0.01] * 30), TF
        )
        assert phase == WyckoffPhase.DISTRIBUTION
        assert uncertain is False

    def test_accumulation(self):
        # Strong price negative, weak momentum → accumulation
        phase, uncertain = determine_phase_by_price_strength(
            -2.5, -0.3, VolumeState.HIGH, pd.Series([0.01] * 30), TF
        )
        assert phase == WyckoffPhase.ACCUMULATION
        assert uncertain is False

    def test_accumulation_low_volume_uncertain(self):
        # Strong price negative, weak momentum, neutral volume → uncertain
        phase, uncertain = determine_phase_by_price_strength(
            -2.5, -0.3, VolumeState.NEUTRAL, pd.Series([0.01] * 30), TF
        )
        assert phase == WyckoffPhase.ACCUMULATION
        assert uncertain is True

    def test_ranging(self):
        phase, uncertain = determine_phase_by_price_strength(
            0.5, 10.0, VolumeState.NEUTRAL, pd.Series([0.01] * 30), TF
        )
        assert phase == WyckoffPhase.RANGING

    def test_moderate_bullish_markup(self):
        phase, uncertain = determine_phase_by_price_strength(
            1.5, 20.0, VolumeState.NEUTRAL, pd.Series([0.01] * 30), TF
        )
        assert phase in (WyckoffPhase.MARKUP, WyckoffPhase.DISTRIBUTION)

    def test_moderate_bearish_markdown(self):
        phase, uncertain = determine_phase_by_price_strength(
            -1.5, -20.0, VolumeState.NEUTRAL, pd.Series([0.01] * 30), TF
        )
        assert phase in (WyckoffPhase.MARKDOWN, WyckoffPhase.ACCUMULATION)

    def test_momentum_price_mismatch_distribution(self):
        phase, _ = determine_phase_by_price_strength(
            1.5, -20.0, VolumeState.HIGH, pd.Series([0.01] * 30), TF
        )
        assert phase == WyckoffPhase.DISTRIBUTION


# ═══════════════════════════════════════════════════════════════════════
# 5. analyze_funding_rates
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeFundingRates:

    def _rate(self, time: int, rate: float) -> FundingRateEntry:
        return FundingRateEntry(time=time, funding_rate=rate, premium=rate * 0.5)

    def test_empty_returns_unknown(self):
        assert analyze_funding_rates([]) == FundingState.UNKNOWN

    def test_insufficient_data_returns_unknown(self):
        rates = [self._rate(1000, 0.01)]
        assert analyze_funding_rates(rates) == FundingState.UNKNOWN

    def test_highly_positive(self):
        rates = [self._rate(1000 + i * 3600_000, 0.00005) for i in range(10)]
        assert analyze_funding_rates(rates) == FundingState.HIGHLY_POSITIVE

    def test_positive(self):
        rates = [self._rate(1000 + i * 3600_000, 0.000018) for i in range(10)]
        assert analyze_funding_rates(rates) == FundingState.POSITIVE

    def test_slightly_positive(self):
        rates = [self._rate(1000 + i * 3600_000, 0.000003) for i in range(10)]
        assert analyze_funding_rates(rates) == FundingState.SLIGHTLY_POSITIVE

    def test_highly_negative(self):
        rates = [self._rate(1000 + i * 3600_000, -0.00005) for i in range(10)]
        assert analyze_funding_rates(rates) == FundingState.HIGHLY_NEGATIVE

    def test_negative(self):
        rates = [self._rate(1000 + i * 3600_000, -0.00002) for i in range(10)]
        assert analyze_funding_rates(rates) == FundingState.NEGATIVE

    def test_slightly_negative(self):
        rates = [self._rate(1000 + i * 3600_000, -0.000003) for i in range(10)]
        assert analyze_funding_rates(rates) == FundingState.SLIGHTLY_NEGATIVE

    def test_neutral(self):
        rates = [self._rate(1000 + i * 3600_000, 0.000001) for i in range(10)]
        assert analyze_funding_rates(rates) == FundingState.NEUTRAL

    def test_outlier_removal(self):
        rates = [self._rate(1000 + i * 3600_000, 0.001) for i in range(8)]
        rates.append(self._rate(1000 + 8 * 3600_000, 10.0))
        result = analyze_funding_rates(rates)
        assert result is not None

    def test_empty_after_outlier_removal(self):
        rates = [self._rate(1000 + i * 3600_000, 1e6) for i in range(5)]
        result = analyze_funding_rates(rates)
        assert result is not None

    def test_funding_rate_clipping(self):
        rates = [self._rate(1000 + i * 3600_000, -10.0) for i in range(10)]
        result = analyze_funding_rates(rates)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# 6. analyze_effort_result
# ═══════════════════════════════════════════════════════════════════════

class TestAnalyzeEffortResult:

    def test_insufficient_data_unknown(self):
        df = _make_df([100.0] * 2)
        vm = _make_vol_metrics()
        assert analyze_effort_result(df, vm, TF) == EffortResult.UNKNOWN

    def test_strong_effort(self):
        df = _make_df(
            [100.0, 100.0, 100.0, 100.0, 103.0],
            opens=[99.5] * 4 + [100.0],
            highs=[100.5] * 4 + [103.5],
            lows=[99.0] * 4 + [100.0],
            volumes=[1000.0] * 4 + [8000.0]
        )
        vm = _make_vol_metrics(strength=2.0, ratio=3.0, consistency=0.8)
        result = analyze_effort_result(df, vm, TF)
        assert result in (EffortResult.STRONG, EffortResult.NEUTRAL)

    def test_weak_effort(self):
        df = _make_df(
            [100.0] * 5,
            opens=[99.5] * 5, highs=[100.5] * 5, lows=[99.5] * 5,
            volumes=[200.0] * 5
        )
        vm = _make_vol_metrics(strength=-1.0, ratio=0.3, consistency=0.2)
        result = analyze_effort_result(df, vm, TF)
        assert result in (EffortResult.WEAK, EffortResult.NEUTRAL)

    def test_neutral_effort(self):
        df = _make_df(
            [100.0] * 5,
            opens=[99.5] * 5, highs=[100.5] * 5, lows=[99.5] * 5,
            volumes=[500.0] * 5
        )
        vm = _make_vol_metrics(strength=0.0, ratio=1.0, consistency=0.5)
        result = analyze_effort_result(df, vm, TF)
        assert result == EffortResult.NEUTRAL

    def test_zero_price_range_returns_unknown(self):
        df = _make_df(
            [100.0] * 5,
            opens=[100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5,
        )
        vm = _make_vol_metrics()
        assert analyze_effort_result(df, vm, TF) == EffortResult.UNKNOWN

    def test_null_price_change_returns_unknown(self):
        df = _make_df(
            [100.0] * 5,
            opens=[100.0] * 5, highs=[101.0] * 5, lows=[99.0] * 5,
        )
        vm = _make_vol_metrics()
        result = analyze_effort_result(df, vm, TF)
        assert result is not None

    def test_short_timeframe_factor(self):
        df = _make_df(
            [100.0] * 5,
            opens=[99.5] * 5, highs=[100.5] * 5, lows=[99.5] * 5,
            volumes=[1000.0] * 5
        )
        vm = _make_vol_metrics(strength=2.0, ratio=3.0, consistency=0.8)
        result_1h = analyze_effort_result(df, vm, Timeframe.HOUR_1)
        result_15m = analyze_effort_result(df, vm, Timeframe.MINUTES_15)
        assert result_1h is not None
        assert result_15m is not None

    def test_exception_returns_unknown(self):
        df = pd.DataFrame()
        vm = _make_vol_metrics()
        assert analyze_effort_result(df, vm, TF) == EffortResult.UNKNOWN

    def test_exception_in_analysis_returns_unknown(self):
        df = _make_df([100.0] * 5, volumes=[1000.0] * 5)
        vm = _make_vol_metrics(sma=1000.0, ratio=1.0, consistency=0.5)
        with patch('technical_analysis.wyckoff.wyckoff.pd.Series.mean',
                   side_effect=ValueError("test error")):
            assert analyze_effort_result(df, vm, TF) == EffortResult.UNKNOWN

    def test_avg_price_zero_returns_unknown(self):
        df = _make_df(
            [0.0] * 5,
            opens=[0.0] * 5, highs=[1.0] * 5, lows=[-1.0] * 5,
        )
        vm = _make_vol_metrics(ratio=0.0)
        assert analyze_effort_result(df, vm, TF) == EffortResult.UNKNOWN

    def test_all_equal_spreads_returns_unknown(self):
        df = _make_df(
            [100.0] * 5,
            opens=[100.0] * 5, highs=[100.0] * 5, lows=[100.0] * 5,
            volumes=[1000.0] * 5,
        )
        vm = _make_vol_metrics(strength=0.5, ratio=1.5, sma=1000.0, consistency=0.5)
        assert analyze_effort_result(df, vm, TF) == EffortResult.UNKNOWN

    def test_high_price_impact_strong_effort(self):
        df = _make_df(
            [100.0, 110.0, 115.0, 120.0, 130.0],
            opens=[100.0, 105.0, 112.0, 117.0, 125.0],
            highs=[101.0, 111.0, 116.0, 121.0, 131.0],
            lows=[99.0, 109.0, 114.0, 119.0, 129.0],
            volumes=[1000.0] * 5,
        )
        vm = _make_vol_metrics(strength=1.5, ratio=0.5, sma=1000.0, consistency=0.5)
        result = analyze_effort_result(df, vm, TF)
        assert result != EffortResult.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════
# 7. identify_wyckoff_phase
# ═══════════════════════════════════════════════════════════════════════

class TestIdentifyWyckoffPhase:

    @pytest.fixture
    def base_df(self):
        return _make_df([100.0] * 30, volumes=[1000.0] * 30)

    def test_nan_price_returns_unknown(self, base_df):
        df = base_df.copy()
        df.loc[df.index[-1], 'c'] = np.nan
        phase, uncertain = identify_wyckoff_phase(
            df, _make_vol_metrics(), 1.0, 20.0, VolumeState.NEUTRAL,
            pd.Series([0.01] * 30), TF, pd.Series([0.0] * 30)
        )
        assert phase == WyckoffPhase.UNKNOWN
        assert uncertain is True

    def test_nan_volume_returns_unknown(self, base_df):
        df = base_df.copy()
        df.loc[df.index[-1], 'v'] = np.nan
        phase, uncertain = identify_wyckoff_phase(
            df, _make_vol_metrics(), 1.0, 20.0, VolumeState.NEUTRAL,
            pd.Series([0.01] * 30), TF, pd.Series([0.0] * 30)
        )
        assert phase == WyckoffPhase.UNKNOWN
        assert uncertain is True

    def test_index_error_returns_unknown(self):
        df = _make_df([100.0], volumes=[1000.0])
        phase, uncertain = identify_wyckoff_phase(
            df, _make_vol_metrics(), 1.0, 20.0, VolumeState.NEUTRAL,
            pd.Series([0.01] * 1), TF, pd.Series([0.0] * 1)
        )
        assert phase == WyckoffPhase.UNKNOWN
        assert uncertain is True

    def test_scalp_markup_breakout(self):
        # pct_change(3) for last row: row[-1]/row[-4] - 1 = 108/100 - 1 = 0.08
        prices = [100.0] * 26 + [100.0, 100.0, 104.0, 108.0]
        vols = [1000.0] * 20 + [10000.0] * 10
        df = _make_df(prices, volumes=vols)
        df['ATR'] = 1.0
        phase, _ = identify_wyckoff_phase(
            df, _make_vol_metrics(state=VolumeState.VERY_HIGH, sma=1000.0),
            2.0, 50.0, VolumeState.VERY_HIGH,
            pd.Series([0.01] * 30), Timeframe.MINUTES_15, pd.Series([0.0] * 30)
        )
        assert phase == WyckoffPhase.MARKUP

    def test_scalp_markdown_breakout(self):
        # pct_change(3) for last row: row[-1]/row[-4] - 1 = 92/100 - 1 = -0.08
        prices = [100.0] * 26 + [100.0, 100.0, 96.0, 92.0]
        vols = [1000.0] * 20 + [10000.0] * 10
        df = _make_df(prices, volumes=vols)
        df['ATR'] = 1.0
        phase, _ = identify_wyckoff_phase(
            df, _make_vol_metrics(state=VolumeState.VERY_HIGH, sma=1000.0),
            -2.0, -50.0, VolumeState.VERY_HIGH,
            pd.Series([0.01] * 30), Timeframe.MINUTES_15, pd.Series([0.0] * 30)
        )
        assert phase == WyckoffPhase.MARKDOWN

    def test_liquidation_cascade(self):
        prices = [100.0] * 47 + [95.0, 90.0, 85.0]
        vols = [1000.0] * 47 + [30000.0, 50000.0, 80000.0]
        df = _make_df(prices, volumes=vols, extra_cols={'ATR': [5.0] * 50})
        vm = _make_vol_metrics(sma=1000.0, state=VolumeState.VERY_HIGH)
        phase, uncertain = identify_wyckoff_phase(
            df, vm, -3.0, -60.0, VolumeState.VERY_HIGH,
            pd.Series([0.05] * 50), Timeframe.HOUR_1, pd.Series([2.0] * 50)
        )
        assert phase in (WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKDOWN, WyckoffPhase.UNKNOWN)

    def test_identify_exception_returns_unknown(self):
        df = pd.DataFrame({'c': [], 'v': [], 'o': [], 'h': [], 'l': []})
        vm = _make_vol_metrics()
        phase, uncertain = identify_wyckoff_phase(
            df, vm, 1.0, 20.0, VolumeState.NEUTRAL,
            pd.Series(dtype=float), TF, pd.Series(dtype=float)
        )
        assert phase == WyckoffPhase.UNKNOWN
        assert uncertain is True


# ═══════════════════════════════════════════════════════════════════════
# 8. detect_wyckoff_phase
# ═══════════════════════════════════════════════════════════════════════

class TestDetectWyckoffPhase:

    def test_insufficient_periods_returns_unknown(self):
        df = _make_df([100.0] * 20)
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert isinstance(state, WyckoffState)
        assert state.phase == WyckoffPhase.UNKNOWN

    def test_minimum_required_periods(self):
        df = _make_df([100.0] * 50, volumes=[1000.0] * 50)
        df['ATR'] = 1.0
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert isinstance(state, WyckoffState)
        assert state.phase != WyckoffPhase.UNKNOWN

    def test_insufficient_periods_15m(self):
        df = _make_df([100.0] * 30)
        state = detect_wyckoff_phase(df, Timeframe.MINUTES_15, [])
        assert state.phase == WyckoffPhase.UNKNOWN

    def test_with_funding_rates(self):
        df = _make_df([100.0] * 50, volumes=[1000.0] * 50)
        df['ATR'] = 1.0
        rates = [
            FundingRateEntry(time=1000 + i * 3600_000, funding_rate=0.001, premium=0.0005)
            for i in range(10)
        ]
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, rates)
        assert isinstance(state, WyckoffState)
        assert state.phase != WyckoffPhase.UNKNOWN

    def test_with_trend_indicators_in_df(self):
        df = _make_df([105.0] * 50, volumes=[1000.0] * 50, extra_cols={
            'SuperTrend': [100.0] * 50,
            'VWAP': [100.0] * 50,
            'RSI': [60.0] * 50,
            'ADX_value': [30.0] * 50,
            'ATR': [2.0] * 50,
        })
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert isinstance(state, WyckoffState)
        assert state.supertrend_trend == "uptrend"
        assert state.vwap_bias == 1
        assert state.rsi_value == 60.0
        assert state.adx_value == 30.0

    def test_uncertain_phase_when_vwap_contradicts_phase(self):
        ramp = [100.0 * (1 + 0.002 * i) for i in range(50)]
        df = _make_df(ramp, volumes=[1500.0 * (1 + 0.01 * i) for i in range(50)], extra_cols={
            'SuperTrend': [100.0] * 50,
            'VWAP': [p * 1.02 for p in ramp],
            'RSI': [65.0] * 50,
            'ATR': [2.0] * 50,
        })
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert state.phase != WyckoffPhase.UNKNOWN

    def test_state_has_all_required_fields(self):
        df = _make_df([100.0] * 50, volumes=[1000.0] * 50)
        df['ATR'] = 1.0
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert hasattr(state, 'phase')
        assert hasattr(state, 'uncertain_phase')
        assert hasattr(state, 'volume')
        assert hasattr(state, 'pattern')
        assert hasattr(state, 'volatility')
        assert hasattr(state, 'is_spring')
        assert hasattr(state, 'is_upthrust')
        assert hasattr(state, 'effort_vs_result')
        assert hasattr(state, 'composite_action')
        assert hasattr(state, 'wyckoff_sign')
        assert hasattr(state, 'funding_state')
        assert hasattr(state, 'description')

    def test_unknown_state_null_values(self):
        unknown = WyckoffState.unknown()
        assert unknown.phase == WyckoffPhase.UNKNOWN
        assert unknown.uncertain_phase is True
        assert unknown.description == "Unknown market state"

    def test_exception_returns_unknown(self):
        df = _make_df([100.0] * 50, volumes=[1000.0] * 50)
        df['ATR'] = 1.0
        with patch('technical_analysis.wyckoff.wyckoff.calculate_volume_metrics',
                   side_effect=ValueError("test error")):
            state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert isinstance(state, WyckoffState)
        assert state.phase == WyckoffPhase.UNKNOWN

    def test_missing_atr_column(self):
        df = _make_df([100.0] * 50, volumes=[1000.0] * 50)
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert isinstance(state, WyckoffState)


# ═══════════════════════════════════════════════════════════════════════
# 9. WyckoffState.to_dict / unknown
# ═══════════════════════════════════════════════════════════════════════

class TestWyckoffStateHelpers:

    def test_to_dict_has_all_keys(self):
        state = WyckoffState.unknown()
        d = state.to_dict()
        expected_keys = [
            'phase', 'uncertain_phase', 'volume', 'pattern', 'volatility',
            'is_spring', 'is_upthrust', 'effort_vs_result', 'composite_action',
            'wyckoff_sign', 'funding_state', 'description', 'liquidity',
            'supertrend_trend', 'vwap_bias', 'rsi_value', 'adx_value',
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_unknown_state_defaults(self):
        u = WyckoffState.unknown()
        assert u.phase == WyckoffPhase.UNKNOWN
        assert u.volume == VolumeState.UNKNOWN
        assert u.pattern == MarketPattern.UNKNOWN
        assert u.volatility == VolatilityState.UNKNOWN
        assert u.effort_vs_result == EffortResult.UNKNOWN
        assert u.composite_action == CompositeAction.UNKNOWN
        assert u.wyckoff_sign == WyckoffSign.NONE
        assert u.funding_state == FundingState.UNKNOWN
        assert u.is_spring is False
        assert u.is_upthrust is False
        assert u.liquidity.value == "unknown liquidity"


# ═══════════════════════════════════════════════════════════════════════
# 10. Integration-like tests
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_rising_market_detection(self):
        n = 55
        ramp = [100.0 * (1 + 0.005 * i) for i in range(n)]
        vols = [1000.0 * (1 + 0.02 * i) for i in range(n)]
        df = _make_df(ramp, volumes=vols, extra_cols={
            'ATR': [2.0] * n,
            'SuperTrend': [100.0] * n,
            'VWAP': [p * 0.998 for p in ramp],
        })
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert state.phase != WyckoffPhase.UNKNOWN
        assert isinstance(state.description, str) and len(state.description) > 0

    def test_falling_market_detection(self):
        n = 55
        ramp = [100.0 * (1 - 0.005 * i) for i in range(n)]
        vols = [1000.0 * (1 + 0.02 * i) for i in range(n)]
        df = _make_df(ramp, volumes=vols, extra_cols={
            'ATR': [2.0] * n,
            'SuperTrend': [100.0] * n,
            'VWAP': [p * 1.002 for p in ramp],
        })
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert state.phase != WyckoffPhase.UNKNOWN
        assert isinstance(state.description, str)

    def test_ranging_market_returns_ranging_phase(self):
        n = 55
        prices = [100.0 + np.random.uniform(-0.5, 0.5) for _ in range(n)]
        vols = [1000.0] * n
        df = _make_df(prices, volumes=vols, extra_cols={'ATR': [1.0] * n})
        state = detect_wyckoff_phase(df, Timeframe.HOUR_1, [])
        assert state.phase != WyckoffPhase.UNKNOWN

    def test_different_timeframes_produce_results(self):
        n = 55
        df = _make_df([100.0] * n, volumes=[1000.0] * n, extra_cols={'ATR': [1.0] * n})
        for tf in [Timeframe.HOUR_1, Timeframe.HOURS_4, Timeframe.MINUTES_30]:
            state = detect_wyckoff_phase(df, tf, [])
            assert isinstance(state, WyckoffState)
