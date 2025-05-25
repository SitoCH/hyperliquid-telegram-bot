import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final, Dict, List, Optional, Any, Tuple
from .wyckoff_types import (
    MarketPattern, VolatilityState, WyckoffState, WyckoffPhase, EffortResult, 
    CompositeAction, WyckoffSign, FundingState, VolumeState, Timeframe, VolumeMetrics,
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES
)
from .funding_rates_cache import FundingRateEntry
from .wyckoff_description import generate_wyckoff_description
from dataclasses import dataclass
from logging_utils import logger
from .adaptive_thresholds import AdaptiveThresholdManager
from .wyckoff_composite_action import detect_composite_action
from .wyckoff_signs import detect_wyckoff_signs


# Constants for Wyckoff analysis - optimized for crypto markets
VOLUME_THRESHOLD: Final[float] = 2.0
STRONG_DEV_THRESHOLD: Final[float] = 2.1
NEUTRAL_ZONE_THRESHOLD: Final[float] = 1.2
MIN_PERIODS: Final[int] = 50
VOLUME_MA_THRESHOLD: Final[float] = 1.8
VOLUME_SURGE_THRESHOLD: Final[float] = 2.8
VOLUME_TREND_SHORT: Final[int] = 10
VOLUME_TREND_LONG: Final[int] = 20

# Add new constants - more balanced values
EFFORT_VOLUME_THRESHOLD: Final[float] = 1.5  # Reduced from 1.8
RESULT_MIN_MOVE: Final[float] = 0.0018  # Fine-tuned from 0.002
HIGH_EFFICIENCY_THRESHOLD: Final[float] = 0.60  # Adjusted from 0.65
LOW_EFFICIENCY_THRESHOLD: Final[float] = 0.30   # Increased from 0.25 for better symmetry


def calculate_volume_metrics(df: pd.DataFrame, timeframe: Timeframe) -> VolumeMetrics:
    """Calculate normalized volume metrics and determine volume state."""
    try:
        # Use timeframe settings instead of hardcoded values
        volume_sma = df['v'].rolling(window=timeframe.settings.volume_ma_window).mean()
        volume_std = df['v'].rolling(window=timeframe.settings.volume_ma_window).std()
        volume_short_ma = df['v'].rolling(window=timeframe.settings.volume_short_ma_window).mean()
        volume_long_ma = df['v'].rolling(window=timeframe.settings.volume_long_ma_window).mean()
        
        # Avoid division by zero
        last_std = max(volume_std.iloc[-1], 1e-8)
        last_sma = max(volume_sma.iloc[-1], 1e-8)
        
        # Use timeframe settings for trend window
        volume_trend = ((volume_short_ma - volume_long_ma) / volume_long_ma).fillna(0)
        
        # Calculate volume consistency - more balanced approach
        recent_strong_volume = (df['v'].iloc[-3:] > volume_sma.iloc[-3:]).mean()
        
        # Calculate strength and ratio metrics
        strength = (df['v'].iloc[-1] - volume_sma.iloc[-1]) / last_std
        ratio = df['v'].iloc[-1] / last_sma
        
        # Optimized volume state determination with cleaner threshold logic
        volume_threshold = timeframe.settings.thresholds[0]
        
        # Define threshold boundaries - more balanced
        thresholds = {
            'very_high_ratio': volume_threshold * 1.5,
            'high_ratio': volume_threshold * 1.1,
            'low_ratio': 0.75,
            'very_low_ratio': 0.5,
            'very_high_strength': 2.5,
            'high_strength': 1.5,
            'low_strength': -0.8,
            'very_low_strength': -1.5
        }
        
        # Clean determination of volume state
        if ratio > thresholds['very_high_ratio'] or strength > thresholds['very_high_strength']:
            state = VolumeState.VERY_HIGH
        elif ratio > thresholds['high_ratio'] or strength > thresholds['high_strength']:
            state = VolumeState.HIGH
        elif ratio < thresholds['very_low_ratio'] or strength < thresholds['very_low_strength']:
            state = VolumeState.VERY_LOW
        elif ratio < thresholds['low_ratio'] or strength < thresholds['low_strength']:
            state = VolumeState.LOW
        else:
            state = VolumeState.NEUTRAL
        
        return VolumeMetrics(
            strength=strength,
            ratio=ratio,
            trend=df['v'].rolling(window=timeframe.settings.volume_trend_window).mean().iloc[-1] / last_sma,
            impulse=df['v'].pct_change().iloc[-1],
            sma=last_sma,
            consistency=recent_strong_volume,
            short_ma=volume_short_ma.iloc[-1],
            long_ma=volume_long_ma.iloc[-1],
            trend_strength=volume_trend.iloc[-1],
            state=state
        )
    except Exception as e:
        logger.warning(f"Error calculating volume metrics: {e}")
        return VolumeMetrics(
            strength=0.0, ratio=1.0, trend=1.0, impulse=0.0, 
            sma=1.0, consistency=0.0, short_ma=1.0, long_ma=1.0, 
            trend_strength=0.0, state=VolumeState.UNKNOWN
        )

def detect_spring_upthrust(df: pd.DataFrame, timeframe: Timeframe, idx: int, vol_metrics: VolumeMetrics) -> tuple[bool, bool]:
    """Enhanced spring/upthrust detection with improved validation and edge case handling."""
    if idx < 4 or len(df) <= idx:  # Added upper bound check
        return False, False
    
    try:
        # Use standard window for better reliability
        window = df.iloc[max(0, idx-4):idx+1]  # Ensure we don't go below 0
        
        if len(window) < 3:  # Need minimum window size
            return False, False
            
        low_point = window['l'].min()
        high_point = window['h'].max()
        close = window['c'].iloc[-1]
        current_low = window['l'].iloc[-1]
        current_high = window['h'].iloc[-1]
        
        # Calculate volatility context
        price = df['c'].iloc[-1]
        if price <= 0:  # Invalid price
            return False, False
        
        # Get adaptive thresholds based on market conditions
        thresholds = AdaptiveThresholdManager.get_spring_upthrust_thresholds(df, timeframe)
        spring_threshold = thresholds["spring"]
        upthrust_threshold = thresholds["upthrust"]
        
        # Check for extremely large candle wicks to avoid false signals
        max_wick = thresholds.get("max_wick", 0.2)
        total_range = window['h'].max() - window['l'].min()
        if total_range > price * max_wick:
            return False, False
            
        # Volume must be significant for both patterns
        if vol_metrics.strength <= 1.0:
            return False, False
        
        # Spring detection - more precise conditions
        spring_bounce = abs(close - current_low) if current_low < low_point else 0
        is_spring = (
            current_low < low_point and         # Makes new low
            close > low_point and               # Closes above the low
            spring_bounce > price * spring_threshold and  # Significant bounce
            close > current_low * 1.005        # Closes meaningfully above the low
        )
        
        # Upthrust detection - more precise conditions  
        upthrust_rejection = abs(current_high - close) if current_high > high_point else 0
        is_upthrust = (
            current_high > high_point and       # Makes new high
            close < high_point and              # Closes below the high
            upthrust_rejection > price * upthrust_threshold and  # Significant rejection
            close < current_high * 0.995       # Closes meaningfully below the high
        )
        
        return is_spring, is_upthrust
    except Exception as e:
        logger.error(f"Error in spring/upthrust detection: {e}")
        return False, False

def detect_wyckoff_phase(df: pd.DataFrame, timeframe: Timeframe, funding_rates: List[FundingRateEntry]) -> WyckoffState:
    """Analyze and store Wyckoff phase data incorporating funding rates."""
    # Safety check for minimum required periods
    if len(df) < MIN_PERIODS:
        return WyckoffState.unknown()

    try:
        # Calculate volume metrics first - now includes volume_state
        vol_metrics = calculate_volume_metrics(df, timeframe)
        
        # Process last period
        short_term_window = min(MIN_PERIODS, len(df) - 1)
        recent_df = df.iloc[-short_term_window:]
        
        # Rest of function remains the same
        price_sma = df['c'].rolling(window=MIN_PERIODS).mean()
        price_std = df['c'].rolling(window=MIN_PERIODS).std()
        
        # Volume analysis (VSA)
        effort_vs_result = pd.Series([0.0] * len(recent_df), index=recent_df.index)
        price_range_mask = (recent_df['h'] - recent_df['l']) > 0
        if price_range_mask.any():
            effort_vs_result[price_range_mask] = (
                (recent_df['c'] - recent_df['o']) / 
                (recent_df['h'] - recent_df['l'])
            )[price_range_mask]
        
        # Get current values
        avg_price = price_sma.iloc[-1]
        price_std_last = price_std.iloc[-1]
        volatility = price_std / avg_price
        
        # Enhanced momentum with Exponential ROC
        def exp_roc(series: pd.Series, periods: int) -> pd.Series:
            return (series / series.shift(periods)).pow(1/periods) - 1
        
        # Calculate momentum components
        fast_momentum = exp_roc(df['c'], 7).iloc[-1]
        medium_momentum = exp_roc(df['c'], 14).iloc[-1]
        slow_momentum = exp_roc(df['c'], 21).iloc[-1]
        
        # Calculate average momentum
        momentum_value = float(sum([fast_momentum, medium_momentum, slow_momentum]) / 3)

        # Calculate momentum standard deviation for scaling
        momentum_std = float(np.std([fast_momentum, medium_momentum, slow_momentum]))
        
        # Scale momentum
        normalized_momentum = momentum_value / (momentum_std * 2) if momentum_std != 0 else 0
        momentum_strength = max(min(normalized_momentum * 100, 100), -100)

        # Detect springs and upthrusts for the current period
        is_spring, is_upthrust = detect_spring_upthrust(df, timeframe, -1, vol_metrics)
        
        # Calculate price strength
        price_strength = (df['c'].iloc[-1] - avg_price) / (price_std_last + 1e-8)

        # Enhanced price strength calculation
        price_ma_short = df['c'].rolling(window=8).mean()
        price_ma_long = df['c'].rolling(window=21).mean()
        trend_strength = ((price_ma_short - price_ma_long) / price_ma_long).iloc[-1]
        price_strength = price_strength * 0.7 + trend_strength * 0.3

        funding_state = analyze_funding_rates(funding_rates or [])

        # Pass the entire df to identify_wyckoff_phase instead of individual parameters
        current_phase, uncertain_phase = identify_wyckoff_phase(
            df, vol_metrics, price_strength, 
            momentum_strength, vol_metrics.state, volatility, timeframe, 
            effort_vs_result
        )
        
        effort_result = analyze_effort_result(df, vol_metrics, timeframe)

        # Detect composite action and Wyckoff signs
        composite_action = detect_composite_action(df, price_strength, vol_metrics, effort_vs_result.iloc[-1])
        wyckoff_sign = detect_wyckoff_signs(df, price_strength, vol_metrics.trend_strength, is_spring, is_upthrust, timeframe)

        # Create WyckoffState instance
        wyckoff_state = WyckoffState(
            phase=current_phase,
            uncertain_phase=uncertain_phase,
            volume=vol_metrics.state,
            pattern=MarketPattern.TRENDING if abs(momentum_strength) > timeframe.settings.thresholds[3] else MarketPattern.RANGING,
            volatility=VolatilityState.HIGH if volatility.iloc[-1] > volatility.mean() else VolatilityState.NORMAL,
            is_spring=is_spring,
            is_upthrust=is_upthrust,
            effort_vs_result=effort_result,
            composite_action=composite_action,
            wyckoff_sign=wyckoff_sign,
            funding_state=funding_state,
            description=generate_wyckoff_description(
                current_phase, uncertain_phase, vol_metrics.state, 
                is_spring, is_upthrust, effort_result,
                composite_action, wyckoff_sign, funding_state
            )
        )

        return wyckoff_state # type: ignore
    except Exception as e:
        logger.error(f"Error in Wyckoff phase detection: {e}")
        return WyckoffState.unknown()

def identify_wyckoff_phase(
    df: pd.DataFrame,
    vol_metrics: VolumeMetrics,
    price_strength: float, 
    momentum_strength: float, 
    volume_state: VolumeState,
    volatility: pd.Series, 
    timeframe: Timeframe,
    effort_vs_result: pd.Series
) -> Tuple[WyckoffPhase, bool]:
    """Enhanced Wyckoff phase identification optimized for intraday crypto trading."""
    
    # Use volume state from vol_metrics to avoid redundancy
    volume_state = vol_metrics.state
      # Calculate values internally with proper error handling
    try:
        recent_change = df['c'].pct_change(3).iloc[-1]
        curr_volume = df['v'].iloc[-1]
        
        # Validate inputs
        if pd.isna(recent_change) or pd.isna(curr_volume):
            return WyckoffPhase.UNKNOWN, True
    except (IndexError, ValueError):
        return WyckoffPhase.UNKNOWN, True

    # Detect scalping opportunities based on timeframe
    is_scalp_timeframe = timeframe in SHORT_TERM_TIMEFRAMES
    if is_scalp_timeframe:
        # Use dynamic breakout threshold based on market conditions
        breakout_threshold = AdaptiveThresholdManager.get_breakout_threshold(df, timeframe)
        # Use higher volume states for more reliable breakout signals
        has_significant_volume = volume_state in (VolumeState.VERY_HIGH, VolumeState.HIGH)
        if abs(recent_change) > breakout_threshold and has_significant_volume:
            return (WyckoffPhase.MARKUP if recent_change > 0 else WyckoffPhase.MARKDOWN), False    # Enhanced liquidation cascade detection with proper safety checks
    volume_impulse = curr_volume / max(vol_metrics.sma, 1e-8)
    volatility_mean = max(volatility.mean(), 1e-8)
    price_velocity = abs(recent_change) / volatility_mean
    
    # Get adaptive liquidation detection thresholds
    liquidation_thresholds = AdaptiveThresholdManager.get_liquidation_thresholds(df, timeframe)
    vol_threshold = liquidation_thresholds["vol_threshold"]
    price_threshold = liquidation_thresholds["price_threshold"]
    velocity_threshold = liquidation_thresholds["velocity_threshold"]
    effort_threshold = liquidation_thresholds["effort_threshold"]

    # Improved liquidation detection with safety checks
    is_liquidation = (
        abs(recent_change) > price_threshold and  
        volume_impulse > vol_threshold and       
        price_velocity > velocity_threshold and   
        len(effort_vs_result) > 0 and
        abs(effort_vs_result.iloc[-1]) > effort_threshold 
    )

    if is_liquidation:
        # Calculate cascade signal strength with safety checks
        cascade_signal_strength = (
            (abs(recent_change) / max(price_threshold, 1e-8)) * 
            (volume_impulse / max(vol_threshold, 1e-8))
        )
          # Check for potential reversal after liquidation
        if cascade_signal_strength > 1.5:  # Strong liquidation event
            return (WyckoffPhase.ACCUMULATION if recent_change < 0 else WyckoffPhase.DISTRIBUTION), True
        return (WyckoffPhase.MARKUP if recent_change > 0 else WyckoffPhase.MARKDOWN), False

    # Rest of the existing phase determination logic
    return determine_phase_by_price_strength(
        price_strength, momentum_strength, volume_state, volatility, timeframe
    )

def determine_phase_by_price_strength(
    price_strength: float, momentum_strength: float, 
    volume_state: VolumeState, volatility: pd.Series,
    timeframe: Timeframe
) -> Tuple[WyckoffPhase, bool]:
    """Enhanced phase determination with cleaner logic and improved certainty assessment."""

    _, strong_dev_threshold, neutral_zone_threshold, \
    momentum_threshold, _, _ = timeframe.settings.thresholds

    # Calculate trend consistency metrics
    vol_ratio = volatility.iloc[-1] / max(volatility.mean(), 1e-8)
    has_high_volume = volume_state in (VolumeState.VERY_HIGH, VolumeState.HIGH)
    
    # Define strength indicators
    strong_price = abs(price_strength) > strong_dev_threshold
    strong_momentum = abs(momentum_strength) > momentum_threshold
    aligned_signals = (price_strength * momentum_strength) > 0
    
    # Very strong conditions - high certainty
    if strong_price and strong_momentum and aligned_signals:
        if price_strength > 0:
            return WyckoffPhase.MARKUP, False
        else:
            return WyckoffPhase.MARKDOWN, False
    
    # Distribution/Accumulation detection
    if strong_price and not strong_momentum:
        if price_strength > strong_dev_threshold:
            # High price, low momentum suggests distribution
            return WyckoffPhase.DISTRIBUTION, not has_high_volume
        elif price_strength < -strong_dev_threshold:
            # Low price, low momentum suggests accumulation
            return WyckoffPhase.ACCUMULATION, not has_high_volume
    
    # Ranging conditions
    if abs(price_strength) <= neutral_zone_threshold:
        uncertain = abs(momentum_strength) > momentum_threshold * 0.5
        return WyckoffPhase.RANGING, uncertain
    
    # Moderate strength conditions
    if price_strength > 0:
        phase = WyckoffPhase.MARKUP if momentum_strength > 0 else WyckoffPhase.DISTRIBUTION
        uncertain = not (strong_price or has_high_volume)
        return phase, uncertain
    else:
        phase = WyckoffPhase.MARKDOWN if momentum_strength < 0 else WyckoffPhase.ACCUMULATION
        uncertain = not (strong_price or has_high_volume)
        return phase, uncertain


def analyze_funding_rates(funding_rates: List[FundingRateEntry]) -> FundingState:
    """Analyze funding rates with enhanced crypto-specific features:
    - Non-linear time weighting for faster response to changes
    - Outlier detection to ignore manipulation spikes
    - Dynamic thresholds based on volatility
    Note: Funding rates are converted to EAR (Effective Annual Rate) using the formula:
    EAR = (1 + hourly_rate)^8760 - 1
    This gives the true annualized return accounting for compounding.
    """
    if not funding_rates or len(funding_rates) < 3:  # Need minimum samples
        return FundingState.UNKNOWN

    now = max(rate.time for rate in funding_rates)

    # Convert to numpy array with error handling
    rates = np.array([
        (1 + np.clip(rate.funding_rate, -0.5, 0.5)) ** 8760 - 1  # More reasonable clipping for extreme values
        for rate in funding_rates
    ])
    times = np.array([rate.time for rate in funding_rates])

    # Improve outlier detection for crypto's volatile funding rates
    # Use a wider IQR multiplier (2.0 instead of 1.5) to accommodate crypto's naturally higher volatility
    q1, q3 = np.percentile(rates, [25, 75])
    iqr = q3 - q1
    mask = (rates >= q1 - 2.0 * iqr) & (rates <= q3 + 2.0 * iqr)
    rates = rates[mask]
    times = times[mask]
    
    if len(rates) < 3:  # Check if we still have enough data after outlier removal
        return FundingState.UNKNOWN

    # Decay factor adjusted for crypto's faster-changing funding environment
    time_diff_hours = (now - times) / (1000 * 3600)
    weights = 1 / (1 + np.exp(0.75 * time_diff_hours - 2))  # Steeper curve (0.75 instead of 0.5)

    # Calculate weighted average with normalization
    avg_funding = np.sum(rates * weights) / np.sum(weights)

    # Use crypto-specific thresholds (more granularity in the slightly positive/negative range)
    if avg_funding > 0.25:
        return FundingState.HIGHLY_POSITIVE
    elif avg_funding > 0.1:
        return FundingState.POSITIVE
    elif avg_funding > 0.015:  # Reduced from 0.02 - more sensitive for crypto
        return FundingState.SLIGHTLY_POSITIVE
    elif avg_funding < -0.25:
        return FundingState.HIGHLY_NEGATIVE
    elif avg_funding < -0.1:
        return FundingState.NEGATIVE
    elif avg_funding < -0.015:  # Reduced from 0.02
        return FundingState.SLIGHTLY_NEGATIVE
    else:
        return FundingState.NEUTRAL

def analyze_effort_result(
    df: pd.DataFrame,
    vol_metrics: VolumeMetrics,
    timeframe: Timeframe
) -> EffortResult:
    """
    Balanced effort vs result analysis with improved validation and error handling.
    Args:
        df: Price and volume data
        vol_metrics: Volume metrics from calculate_volume_metrics
        timeframe: Current timeframe for context
    """
    try:
        # Get recent data - use timeframe settings
        lookback = timeframe.settings.effort_lookback
        
        if len(df) < lookback:
            return EffortResult.UNKNOWN
            
        recent_df = df.iloc[-lookback:]
        
        # Calculate normalized price movement with validation
        price_change = abs(recent_df['c'].iloc[-1] - recent_df['o'].iloc[-1])
        price_range = recent_df['h'].iloc[-1] - recent_df['l'].iloc[-1]
        
        # Validate price data
        if price_range <= 0 or pd.isna(price_change):
            return EffortResult.UNKNOWN
        
        # Apply minimum move threshold
        min_move = RESULT_MIN_MOVE * timeframe.settings.min_move_multiplier
        
        # Calculate volume quality with better validation
        volume_consistency = max(0.0, min(1.0, vol_metrics.consistency))
        
        # Improved spread quality calculation
        spread_ratio = (recent_df['h'] - recent_df['l'])
        close_open_diff = abs(recent_df['c'] - recent_df['o'])
        
        # Avoid division by zero
        valid_spreads = spread_ratio > 1e-8
        if valid_spreads.sum() == 0:
            return EffortResult.UNKNOWN
            
        spread_quality = 1.0 - (close_open_diff[valid_spreads] / spread_ratio[valid_spreads]).mean()
        spread_quality = max(0.0, min(1.0, spread_quality))
        
        # Balanced volume quality calculation
        volume_quality = (volume_consistency * 0.5 + spread_quality * 0.5)
        
        # Calculate price impact with improved validation
        avg_price = recent_df['c'].mean()
        if avg_price <= 0 or vol_metrics.ratio <= 0:
            return EffortResult.UNKNOWN
            
        price_impact = price_change / (avg_price * max(vol_metrics.ratio, 0.1))
        
        # Calculate efficiency score with validation
        base_efficiency = price_change / max(price_range, 1e-8)
        volume_weighted_efficiency = base_efficiency * (1 + vol_metrics.strength * timeframe.settings.volume_weighted_efficiency)
        
        # Balanced thresholds from settings
        high_threshold = HIGH_EFFICIENCY_THRESHOLD * timeframe.settings.high_threshold
        low_threshold = LOW_EFFICIENCY_THRESHOLD * timeframe.settings.low_threshold
        
        # Final efficiency score
        efficiency = min(1.0, max(0.0, volume_weighted_efficiency))
        
        # Calculate balanced volume factor
        is_short_timeframe = timeframe in SHORT_TERM_TIMEFRAMES
        volume_factor = vol_metrics.ratio / (EFFORT_VOLUME_THRESHOLD * 
                         (0.9 if is_short_timeframe else 1.0))
        
        # Clear classification logic
        if efficiency > high_threshold and volume_quality > 0.6:
            return EffortResult.STRONG
        elif efficiency < low_threshold and volume_quality < 0.4:
            return EffortResult.WEAK
        elif price_change < min_move * 0.5:
            return EffortResult.WEAK
        elif price_impact > 1.2 and volume_factor > 1.1:
            return EffortResult.STRONG
        elif price_impact < 0.4 and volume_factor < 0.7:
            return EffortResult.WEAK
        elif efficiency > 0.5:
            return EffortResult.STRONG
        else:
            return EffortResult.WEAK
            
    except Exception as e:
        logger.error(f"Error in effort vs result analysis: {e}")
        return EffortResult.UNKNOWN
