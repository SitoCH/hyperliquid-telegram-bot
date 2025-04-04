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
    """Enhanced spring/upthrust detection with more balanced signal generation."""
    if idx < 4:  # Maintain original lookback for better reliability
        return False, False
    
    try:
        # Use standard window for better reliability
        window = df.iloc[idx-4:idx+1]
        low_point = window['l'].min()
        high_point = window['h'].max()
        close = window['c'].iloc[-1]
        current_low = window['l'].iloc[-1]
        current_high = window['h'].iloc[-1]
        
        # Calculate volatility context
        price = df['c'].iloc[-1]
        
        # Get adaptive thresholds based on market conditions
        thresholds = AdaptiveThresholdManager.get_spring_upthrust_thresholds(df, timeframe)
        spring_threshold = thresholds["spring"]
        upthrust_threshold = thresholds["upthrust"]
        
        # Check for extremely large candle wicks to avoid false signals
        max_wick = thresholds.get("max_wick", 0.2)
        if (window['h'].max() - window['l'].min()) > price * max_wick:
            return False, False
            
        # Volume must be significant for both patterns
        if vol_metrics.strength <= 1.0:
            return False, False
        
        # Spring detection - optimized to separate conditions for clarity
        is_spring = (
            current_low < low_point and         # Makes new low
            close > low_point and               # Closes above the low
            abs(close - current_low) > price * spring_threshold  # Significant bounce
        )
        
        # Upthrust detection - optimized to separate conditions for clarity
        is_upthrust = (
            current_high > high_point and       # Makes new high
            close < high_point and              # Closes below the high
            abs(current_high - close) > price * upthrust_threshold  # Significant rejection
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

    # Calculate values internally that were previously passed as parameters
    recent_change = df['c'].pct_change(3).iloc[-1]
    curr_volume = df['v'].iloc[-1]

    # Calculate high volume price level
    decay_factor = -np.log(2) / 100
    weights = np.exp(np.linspace(decay_factor * len(df), 0, len(df)))
    weights = weights / weights.sum()

    # Detect scalping opportunities based on timeframe
    is_scalp_timeframe = timeframe in SHORT_TERM_TIMEFRAMES
    if is_scalp_timeframe:
        # Use dynamic breakout threshold based on market conditions
        breakout_threshold = AdaptiveThresholdManager.get_breakout_threshold(df, timeframe)
        # Use higher volume states for more reliable breakout signals
        has_significant_volume = volume_state in (VolumeState.VERY_HIGH, VolumeState.HIGH)
        if abs(recent_change) > breakout_threshold and has_significant_volume:
            return (WyckoffPhase.MARKUP if recent_change > 0 else WyckoffPhase.MARKDOWN), False

    # Enhanced liquidation cascade detection with adaptive thresholds
    volume_impulse = curr_volume / (vol_metrics.sma + 1e-8)
    price_velocity = abs(recent_change) / (volatility.mean() + 1e-8)
    
    # Get adaptive liquidation detection thresholds
    liquidation_thresholds = AdaptiveThresholdManager.get_liquidation_thresholds(df, timeframe)
    vol_threshold = liquidation_thresholds["vol_threshold"]
    price_threshold = liquidation_thresholds["price_threshold"]
    velocity_threshold = liquidation_thresholds["velocity_threshold"]
    effort_threshold = liquidation_thresholds["effort_threshold"]

    is_liquidation = (
        abs(recent_change) > price_threshold and  
        volume_impulse > vol_threshold and       
        price_velocity > velocity_threshold and   
        abs(effort_vs_result.iloc[-1]) > effort_threshold 
    )

    # Enhanced cascade detection using hourly execution awareness
    # Track if this signal is changing within a short period
    cascade_signal_strength = (abs(recent_change) / price_threshold) * (volume_impulse / vol_threshold)

    if is_liquidation:
        # Check for potential reversal after liquidation
        if cascade_signal_strength > 1.5:  # Strong liquidation event
            return (WyckoffPhase.ACCUMULATION if recent_change > 0 else WyckoffPhase.DISTRIBUTION), True
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
    """Enhanced phase determination with more decisive phase identification."""

    _, strong_dev_threshold, neutral_zone_threshold, \
    momentum_threshold, _, _ = timeframe.settings.thresholds

    # Calculate trend consistency
    vol_ratio = volatility.iloc[-1] / volatility.mean()
    
    # Define confirmation metrics
    price_trend_consistency = abs(price_strength) > strong_dev_threshold * 0.6
    momentum_consistency = abs(momentum_strength) > momentum_threshold * 0.7

    # Define extreme momentum conditions
    is_price_collapsing = momentum_strength < -momentum_threshold * 1.3 and price_strength < -strong_dev_threshold * 0.9
    is_price_surging = momentum_strength > momentum_threshold * 1.3 and price_strength > strong_dev_threshold * 0.9

    # Use volume state for more nuanced analysis
    has_high_volume = volume_state in (VolumeState.VERY_HIGH, VolumeState.HIGH)
    
    # Quickly mark phases as certain if strong indicators are present
    if abs(price_strength) > strong_dev_threshold * 1.5 and (price_strength * momentum_strength > 0):
        if price_strength > 0:
            return WyckoffPhase.MARKUP if momentum_strength > 0 else WyckoffPhase.DISTRIBUTION, False
        else:
            return WyckoffPhase.MARKDOWN if momentum_strength < 0 else WyckoffPhase.ACCUMULATION, False

    # Price above threshold - potential Distribution or Markup
    if price_strength > strong_dev_threshold:
        # Check for a price surge (strong positive momentum)
        if is_price_surging:
            # If price is surging rapidly, this is markup, not distribution
            return WyckoffPhase.MARKUP, False

        # Distribution requires momentum to be stabilizing or turning down
        if momentum_strength < momentum_threshold * 0.3:  # Neutral or negative momentum
            if volume_state in (VolumeState.VERY_HIGH, VolumeState.HIGH):
                return WyckoffPhase.DISTRIBUTION, False
            else:
                return WyckoffPhase.DISTRIBUTION, not (price_trend_consistency or momentum_consistency)
        else:
            # Strong positive momentum with positive price - still in Markup
            markup_certainty = price_trend_consistency or (
                volume_state in (VolumeState.VERY_HIGH, VolumeState.HIGH) or
                momentum_consistency > 0.8
            )
            return WyckoffPhase.MARKUP, not markup_certainty

    # Price below threshold - potential Accumulation or Markdown
    if price_strength < -strong_dev_threshold:
        # Check if we have a price collapse
        if is_price_collapsing:
            # If price is collapsing, this is markdown, not accumulation
            return WyckoffPhase.MARKDOWN, False

        # Accumulation requires momentum to be stabilizing or turning up
        if momentum_strength > -momentum_threshold * 0.3:  # Neutral or positive momentum
            if volume_state in (VolumeState.VERY_HIGH, VolumeState.HIGH):
                return WyckoffPhase.ACCUMULATION, False
            else:
                return WyckoffPhase.ACCUMULATION, not (price_trend_consistency or momentum_consistency)
        else:
            # Strong negative momentum with negative price - still in Markdown
            markdown_certainty = price_trend_consistency or (
                volume_state in (VolumeState.VERY_HIGH, VolumeState.HIGH) or
                momentum_consistency > 0.8
            )
            return WyckoffPhase.MARKDOWN, not markdown_certainty
    
    # Enhanced ranging detection with multi-factor certainty
    if abs(price_strength) <= neutral_zone_threshold:
        is_low_volatility = vol_ratio < 0.9
        
        # Simplify ranging certainty calculation
        conflicting_momentum = abs(momentum_strength) > momentum_threshold * 0.6
        
        return WyckoffPhase.RANGING, conflicting_momentum and not is_low_volatility

    # Improved trend confirmation
    trend_strength = abs(momentum_strength) / momentum_threshold
    is_strong_trend = trend_strength > 1.2 or has_high_volume

    # Check for trend phases (Markup/Markdown)
    if price_strength > 0:
        if momentum_strength > momentum_threshold * 0.7:
            return WyckoffPhase.MARKUP, not is_strong_trend
        return WyckoffPhase.MARKUP, not (price_trend_consistency or has_high_volume or momentum_strength > 0)
    elif price_strength < 0:
        if momentum_strength < -momentum_threshold * 0.7:
            return WyckoffPhase.MARKDOWN, not is_strong_trend
        return WyckoffPhase.MARKDOWN, not (price_trend_consistency or has_high_volume or momentum_strength < 0)
    
    # Fallback case is uncertain by default
    return WyckoffPhase.RANGING, True

def detect_wyckoff_signs(
    df: pd.DataFrame,
    price_strength: float,
    volume_trend: float,
    is_spring: bool,
    is_upthrust: bool,
    timeframe: Timeframe
) -> WyckoffSign:
    """
    Detect specific Wyckoff signs in market action optimized for crypto intraday trading.
    Uses adaptive settings based on timeframe for better hourly analysis.
    """
    if len(df) < 5:
        return WyckoffSign.NONE

    # Calculate key metrics with noise reduction
    price_change = df['c'].pct_change()
    volume_change = df['v'].pct_change()
    
    # Use timeframe-specific lookback periods instead of fixed values
    volatility_window = max(5, timeframe.settings.spring_upthrust_window)
    volume_ma_window = max(5, timeframe.settings.volume_ma_window // 3)  # Shorter for responsiveness
    price_ma_window = max(8, int(timeframe.settings.ema_length * 0.75))  # Derived from EMA setting
    
    # Calculate rolling metrics with adaptive windows
    price_volatility = df['c'].pct_change().rolling(volatility_window).std().iloc[-1]
    volume_ma = df['v'].rolling(volume_ma_window).mean()
    price_ma = df['c'].rolling(price_ma_window).mean()
    
    volatility_factor = 1.2
    if timeframe in SHORT_TERM_TIMEFRAMES:
        volatility_factor = 0.85
    elif timeframe in INTERMEDIATE_TIMEFRAMES:
        volatility_factor = 1.0

    min_price_move = max(0.03, price_volatility * 1.5 * volatility_factor)
    min_volume_surge = max(2.0, volume_change.rolling(volatility_window).std().iloc[-1] * 2 * volatility_factor)

    # Detect market context for better signal relevance
    is_high_volatility = price_volatility > df['c'].pct_change().rolling(volatility_window * 3).std().mean() * 1.5
    recency_factor = 3 if is_high_volatility else 2  # More emphasis on recent bars during high volatility

    # Check for strong confirmation across multiple candles with adaptive window
    def confirm_trend(window: int, threshold: float) -> bool:
        window = min(window, len(df) - 1)  # Safety check
        recent_changes = price_change.iloc[-window:]
        # Weighted confirmation - more recent changes have higher weight
        weights = np.linspace(1, recency_factor, len(recent_changes))
        weighted_confirms = (recent_changes > threshold) * weights
        return weighted_confirms.sum() >= np.sum(weights) * 0.4  # 40% threshold
        
    def confirm_volume(window: int, threshold: float) -> bool:
        window = min(window, len(df) - 1)  # Safety check
        recent_volume = volume_change.iloc[-window:]
        # Weighted confirmation for volume
        weights = np.linspace(1, recency_factor, len(recent_volume))
        weighted_confirms = (recent_volume > threshold) * weights
        return weighted_confirms.sum() >= np.sum(weights) * 0.4  # 40% threshold

    # Selling Climax (SC) - Optimized for crypto's sharper drops
    if (price_change.iloc[-1] < -min_price_move * 1.2 and  # More extreme for crypto
        volume_change.iloc[-1] > min_volume_surge * 1.1 and  # Higher volume spike
        price_strength < -STRONG_DEV_THRESHOLD and
        df['c'].iloc[-1] < price_ma.iloc[-1] * 0.95):
        return WyckoffSign.SELLING_CLIMAX
        
    # Automatic Rally (AR) - Must follow a selling climax
    # In crypto, these can be very sharp and quick
    if (price_change.iloc[-1] > min_price_move * 1.1 and
        confirm_trend(3, min_price_move * 0.6) and
        df['l'].iloc[-1] > df['l'].iloc[-5:].min() and  # Higher low
        price_strength < 0 and  # Still overall below average
        volume_change.iloc[-1] > 0):  # Some volume confirmation
        return WyckoffSign.AUTOMATIC_RALLY
        
    # Secondary Test (ST) - Lower volume test of support
    # Critical for crypto hourly analysis to identify accumulation
    if (abs(price_change.iloc[-1]) < price_volatility * 0.8 and  # Reduced volatility
        df['l'].iloc[-1] >= df['l'].iloc[-5:].min() * 0.99 and  # Test previous low
        df['l'].iloc[-1] <= df['l'].iloc[-5:].min() * 1.01 and  # Close to the low
        df['v'].iloc[-1] < volume_ma.iloc[-1] * 0.8 and  # Lower volume
        price_strength < 0):  # Still in overall downtrend
        return WyckoffSign.SECONDARY_TEST

    # Last Point of Support (LPS) - Spring with volume confirmation
    if (is_spring and
        volume_trend > 0.4 and  # Reduced from 0.5 for better sensitivity
        price_change.iloc[-1] > min_price_move * 0.8 and  # Lower threshold for crypto
        confirm_volume(3, 1.0)):  # Lower threshold for better detection
        return WyckoffSign.LAST_POINT_OF_SUPPORT
        
    # Sign of Strength (SOS) - Critical for identifying bullish continuation
    if (price_change.iloc[-1] > min_price_move * 1.5 and
        confirm_trend(3, min_price_move * 0.8) and  # Adjusted for crypto
        confirm_volume(3, 1.3) and  # Slightly reduced volume requirement
        price_strength > 0.5):
        return WyckoffSign.SIGN_OF_STRENGTH
        
    # Buying Climax (BC) - Extreme buying with high volume
    # Crypto can have extreme volume spikes during euphoria
    if (price_change.iloc[-1] > min_price_move * 2.0 and
        volume_change.iloc[-1] > min_volume_surge * 1.2 and 
        price_strength > STRONG_DEV_THRESHOLD and
        df['c'].iloc[-1] > price_ma.iloc[-1] * 1.05):  # Price well above MA
        return WyckoffSign.BUYING_CLIMAX
        
    # Upthrust (UT) - False breakout with rejection
    # Common in crypto due to stop hunting and liquidations
    if (is_upthrust and
        volume_change.iloc[-1] > min_volume_surge * 0.8 and
        price_change.iloc[-1] < -min_price_move * 0.5 and
        df['c'].iloc[-1] < df['h'].iloc[-1] * 0.985):  # Strong rejection
        return WyckoffSign.UPTHRUST

    # Secondary Test Resistance (STR) - Higher test with lower volume
    # Crypto often tests resistance multiple times before breakout/rejection
    if (abs(price_change.iloc[-1]) < price_volatility * 0.8 and
        df['h'].iloc[-1] <= df['h'].iloc[-5:].max() * 1.01 and  # Test previous high
        df['h'].iloc[-1] >= df['h'].iloc[-5:].max() * 0.99 and  # Close to the high
        df['v'].iloc[-1] < volume_ma.iloc[-1] * 0.8 and  # Lower volume
        price_strength > 0):  # Still in overall uptrend
        return WyckoffSign.SECONDARY_TEST_RESISTANCE

    # Last Point of Supply (LPSY) - Failed upthrust with volume
    # Important for distribution detection in crypto markets
    if (is_upthrust and
        volume_trend > 0.4 and  # Significant volume context
        price_change.iloc[-1] < -min_price_move * 0.9 and  # Sharp rejection
        confirm_volume(3, 1.0) and  # Volume confirmation
        price_strength > 0):  # Still above average price
        return WyckoffSign.LAST_POINT_OF_RESISTANCE

    # Sign of Weakness (SOW) - Strong down move with volume
    # Critical for identifying bearish continuation
    if (price_change.iloc[-1] < -min_price_move * 1.5 and
        confirm_trend(3, -min_price_move * 0.8) and  # Consistent weakness
        confirm_volume(3, 1.3) and  # Volume confirmation
        price_strength < -0.5):  # Below average price
        return WyckoffSign.SIGN_OF_WEAKNESS

    return WyckoffSign.NONE

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
    Balanced effort vs result analysis with more symmetric thresholds and a neutral zone.
    Args:
        df: Price and volume data
        vol_metrics: Volume metrics from calculate_volume_metrics
        timeframe: Current timeframe for context
    """
    try:
        # Get recent data - use timeframe settings
        lookback = timeframe.settings.effort_lookback
        
        recent_df = df.iloc[-lookback:]
        
        # Calculate normalized price movement
        price_change = abs(recent_df['c'].iloc[-1] - recent_df['o'].iloc[-1])
        price_range = recent_df['h'].iloc[-1] - recent_df['l'].iloc[-1]
        
        # Apply minimum move threshold - symmetrically applied
        min_move = RESULT_MIN_MOVE * timeframe.settings.min_move_multiplier
        
        # Calculate volume quality with timeframe context
        volume_consistency = vol_metrics.consistency
        spread_quality = 1.0 - (abs(recent_df['c'] - recent_df['o']) / (recent_df['h'] - recent_df['l'])).mean()
        
        # Use balanced timeframe-specific adjustments
        if timeframe in SHORT_TERM_TIMEFRAMES:
            volume_quality = (volume_consistency * 0.5 + spread_quality * 0.5)  # Equal weighting
        elif timeframe in INTERMEDIATE_TIMEFRAMES:
            volume_quality = (volume_consistency * 0.5 + spread_quality * 0.5)  # Equal weighting
        else:
            volume_quality = (volume_consistency * 0.5 + spread_quality * 0.5)  # Equal weighting
        
        # Calculate price impact with timeframe-adjusted volume ratio
        avg_price = recent_df['c'].mean()
        price_impact = price_change / (avg_price * vol_metrics.ratio)
        
        # Calculate efficiency score with balanced parameters
        base_efficiency = price_change / (price_range + 1e-8)
        volume_weighted_efficiency = base_efficiency * (1 + vol_metrics.strength * timeframe.settings.volume_weighted_efficiency)
        
        # Balanced thresholds from settings
        high_threshold = HIGH_EFFICIENCY_THRESHOLD * timeframe.settings.high_threshold
        low_threshold = LOW_EFFICIENCY_THRESHOLD * timeframe.settings.low_threshold
        
        # Final efficiency score
        efficiency = min(1.0, volume_weighted_efficiency)
        
        # Calculate a balanced volume factor
        is_short_timeframe = timeframe in SHORT_TERM_TIMEFRAMES
        volume_factor = vol_metrics.ratio / (EFFORT_VOLUME_THRESHOLD * 
                         (0.9 if is_short_timeframe else 1.0))
        
        # Define truly neutral zone conditions
        neutral_price_change = price_change < min_move * 1.5 and price_change > min_move * 0.5
        neutral_efficiency = efficiency > low_threshold * 1.1 and efficiency < high_threshold * 0.9
        neutral_volume = volume_quality > 0.4 and volume_quality < 0.6
        
        # More symmetric condition structure with UNKNOWN state for truly ambiguous cases
        if efficiency > high_threshold and volume_quality > 0.6:
            return EffortResult.STRONG
        elif efficiency < low_threshold and volume_quality < 0.4:
            return EffortResult.WEAK
        elif price_change < min_move * 0.5:
            # Very small moves are generally weak
            return EffortResult.WEAK
        elif price_impact > 1.2 and volume_factor > 1.1:
            # Strong price impact and enough volume
            return EffortResult.STRONG
        elif price_impact < 0.4 and volume_factor < 0.7:
            # Weak price impact despite volume
            return EffortResult.WEAK
        elif (neutral_price_change and neutral_efficiency) or neutral_volume:
            # Truly neutral case - neither clearly strong nor weak
            return EffortResult.UNKNOWN
        elif efficiency > 0.5:
            # Slightly favor strong when near boundary but not in neutral zone
            return EffortResult.STRONG
        else:
            # Default to weak for remaining cases
            return EffortResult.WEAK
            
    except Exception as e:
        logger.error(f"Error in effort vs result analysis: {e}")
        return EffortResult.UNKNOWN  # Changed from WEAK to UNKNOWN for errors
