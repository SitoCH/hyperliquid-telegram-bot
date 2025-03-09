import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final, Dict, List, Optional, Any, Tuple
from .wyckoff_types import (
    MarketPattern, VolatilityState, WyckoffState, WyckoffPhase, EffortResult, 
    CompositeAction, WyckoffSign, FundingState, VolumeState, Timeframe
)
from .funding_rates_cache import FundingRateEntry
from .wyckoff_description import generate_wyckoff_description
from dataclasses import dataclass
from logging_utils import logger
from .adaptive_thresholds import AdaptiveThresholdManager


# Constants for Wyckoff analysis
VOLUME_THRESHOLD: Final[float] = 1.7  # Increased from 1.5 for more significance with larger dataset
STRONG_DEV_THRESHOLD: Final[float] = 1.9  # Increased from 1.8 for wider historical context
NEUTRAL_ZONE_THRESHOLD: Final[float] = 1.0  # Increased from 0.8 for more stable neutral zone detection
EFFORT_THRESHOLD: Final[float] = 0.7  # Increased from 0.65 for clearer effort vs result signals
MIN_PERIODS: Final[int] = 40  # Increased from 30 to use more historical data
VOLUME_MA_THRESHOLD: Final[float] = 1.5  # Increased from 1.3 for stronger volume signals
VOLUME_SURGE_THRESHOLD: Final[float] = 2.2  # Increased from 2.0 for more significant volume events
VOLUME_TREND_SHORT: Final[int] = 7  # Increased from 5 for smoother short-term trends
VOLUME_TREND_LONG: Final[int] = 14  # Increased from 10 for longer-term trend context

# Add new constants after existing ones
EFFORT_VOLUME_THRESHOLD: Final[float] = 1.5  # Volume threshold for effort analysis
RESULT_MIN_MOVE: Final[float] = 0.001  # Minimum price move to consider (0.1%)
HIGH_EFFICIENCY_THRESHOLD: Final[float] = 0.8  # High efficiency threshold
LOW_EFFICIENCY_THRESHOLD: Final[float] = 0.4  # Low efficiency threshold

@dataclass
class VolumeMetrics:
    """Container for volume-related metrics"""
    strength: float      # Normalized volume (z-score)
    ratio: float        # Current volume / SMA ratio
    trend: float        # Short-term trend direction
    impulse: float      # Rate of change
    sma: float         # Simple moving average
    consistency: float  # Recent volume consistency
    short_ma: float    # Short-term moving average
    long_ma: float     # Long-term moving average
    trend_strength: float  # Trend strength indicator

@dataclass
class EffortAnalysis:
    """Container for effort vs result analysis"""
    efficiency: float          # How efficiently volume translates to price movement
    volume_quality: float     # Quality of the volume (considering spread and consistency)
    price_impact: float      # Actual price impact relative to volume
    result: EffortResult     # Final effort classification

def calculate_volume_metrics(df: pd.DataFrame, timeframe: Timeframe) -> VolumeMetrics:
    """Calculate normalized volume metrics."""
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
        
        # Calculate volume consistency
        recent_strong_volume = (df['v'].iloc[-3:] > volume_sma.iloc[-3:]).mean()
        
        return VolumeMetrics(
            strength=(df['v'].iloc[-1] - volume_sma.iloc[-1]) / last_std,
            ratio=df['v'].iloc[-1] / last_sma,
            trend=df['v'].rolling(window=5).mean().iloc[-1] / last_sma,
            impulse=df['v'].pct_change().iloc[-1],
            sma=last_sma,
            consistency=recent_strong_volume,
            short_ma=volume_short_ma.iloc[-1],
            long_ma=volume_long_ma.iloc[-1],
            trend_strength=volume_trend.iloc[-1]
        )
    except Exception as e:
        logger.warning(f"Error calculating volume metrics: {e}")
        return VolumeMetrics(0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0)

def detect_spring_upthrust(df: pd.DataFrame, timeframe: Timeframe, idx: int, vol_metrics: VolumeMetrics) -> tuple[bool, bool]:
    """Enhanced spring/upthrust detection using normalized volume metrics."""
    if idx < 4:
        return False, False
    
    try:
        window = df.iloc[idx-4:idx+1]
        low_point = window['l'].min()
        high_point = window['h'].max()
        close = window['c'].iloc[-1]
        
        # Calculate volatility context
        price = df['c'].iloc[-1]
        
        # Get adaptive thresholds based on market conditions
        thresholds = AdaptiveThresholdManager.get_spring_upthrust_thresholds(df, timeframe)
        spring_threshold = thresholds["spring"]
        upthrust_threshold = thresholds["upthrust"]
        
        # Check for extremely large candle wicks to avoid false signals
        if (window['h'].max() - window['l'].min()) > df['c'].iloc[-1] * 0.2:
            return False, False
        
        # Enhanced spring detection
        is_spring = (
            window['l'].iloc[-1] < low_point and  # Makes new low
            close > low_point and  # Closes above the low
            vol_metrics.strength > 1.0 and  # Using normalized volume
            abs(close - window['l'].iloc[-1]) > price * spring_threshold  # Significant bounce
        )
        
        # Enhanced upthrust detection
        is_upthrust = (
            window['h'].iloc[-1] > high_point and  # Makes new high
            close < high_point and  # Closes below the high
            vol_metrics.strength > 1.0 and  # Using normalized volume
            abs(window['h'].iloc[-1] - close) > price * upthrust_threshold  # Significant rejection
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
        # Calculate volume metrics first
        vol_metrics = calculate_volume_metrics(df, timeframe)
        
        # Define is_high_volume using normalized metrics
        is_high_volume = (
            (vol_metrics.ratio > timeframe.settings.thresholds[0] and vol_metrics.trend > 1.0) or
            (abs(vol_metrics.strength) > 2.0) or
            (vol_metrics.ratio > 1.2 and vol_metrics.trend > 1.2)
        )

        # Process last period
        short_term_window = min(MIN_PERIODS, len(df) - 1)
        recent_df = df.iloc[-short_term_window:]
        
        # Rest of function remains the same
        price_sma = df['c'].rolling(window=MIN_PERIODS).mean()
        price_std = df['c'].rolling(window=MIN_PERIODS).std()

        # Volume analysis (VSA)
        volume_spread = recent_df['v'] * (recent_df['h'] - recent_df['l'])
        volume_spread_ma = volume_spread.rolling(window=7).mean()
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
            momentum_strength, is_high_volume, volatility, timeframe, 
            effort_vs_result
        )
        
        # Replace simple effort_result calculation with new analysis
        effort_analysis = analyze_effort_result(df, vol_metrics, timeframe)
        effort_result = effort_analysis.result
        
        # Detect composite action and Wyckoff signs
        composite_action = detect_composite_action(df, price_strength, vol_metrics, effort_vs_result.iloc[-1])
        wyckoff_sign = detect_wyckoff_signs(df, price_strength, vol_metrics.trend_strength, is_spring, is_upthrust)
            
        # Create WyckoffState instance
        wyckoff_state = WyckoffState(
            phase=current_phase,
            uncertain_phase=uncertain_phase,
            volume=VolumeState.HIGH if is_high_volume else VolumeState.LOW,
            pattern=MarketPattern.TRENDING if abs(momentum_strength) > timeframe.settings.thresholds[3] else MarketPattern.RANGING,
            volatility=VolatilityState.HIGH if volatility.iloc[-1] > volatility.mean() else VolatilityState.NORMAL,
            is_spring=is_spring,
            is_upthrust=is_upthrust,
            volume_spread=VolumeState.HIGH if volume_spread.iloc[-1] > volume_spread_ma.iloc[-1] else VolumeState.LOW,
            effort_vs_result=effort_result,
            composite_action=composite_action,
            wyckoff_sign=wyckoff_sign,
            funding_state=funding_state,
            description=generate_wyckoff_description(
                current_phase, uncertain_phase, is_high_volume, momentum_strength, 
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
    is_high_volume: bool,
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
    is_scalp_timeframe = timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30]
    if is_scalp_timeframe:
        # Use dynamic breakout threshold based on market conditions
        breakout_threshold = AdaptiveThresholdManager.get_breakout_threshold(df, timeframe)
        if abs(recent_change) > breakout_threshold and is_high_volume:
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
            return (WyckoffPhase.DISTRIBUTION if recent_change > 0 else WyckoffPhase.ACCUMULATION), True
        return (WyckoffPhase.MARKUP if recent_change > 0 else WyckoffPhase.MARKDOWN), False

    # Rest of the existing phase determination logic
    return determine_phase_by_price_strength(
        price_strength, momentum_strength, is_high_volume, volatility, timeframe
    )

def determine_phase_by_price_strength(
    price_strength: float, momentum_strength: float, 
    is_high_volume: bool, volatility: pd.Series,
    timeframe: Timeframe
) -> Tuple[WyckoffPhase, bool]:
    """Enhanced phase determination with better trend confirmation."""
    # Get thresholds
    _, strong_dev_threshold, neutral_zone_threshold, \
    momentum_threshold, _, _ = timeframe.settings.thresholds

    # Calculate trend consistency
    vol_ratio = volatility.iloc[-1] / volatility.mean()
    
    # Adjust thresholds for intraday volatility
    if timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30]:
        strong_dev_threshold *= 0.85  # More sensitive for short timeframes
        neutral_zone_threshold *= 1.2  # Wider neutral zone for noise
        momentum_threshold *= 0.9  # More sensitive momentum

    # Modified reversal zone detection (less strict, more phases become certain)
    is_reversal_zone = abs(price_strength) > strong_dev_threshold * 1.3  # Reduced from 1.5
    
    # Calculate additional confirmation metrics
    price_trend_consistency = abs(price_strength) > strong_dev_threshold * 0.7  # Price consistency check
    momentum_consistency = abs(momentum_strength) > momentum_threshold * 0.8  # Momentum consistency
    
    # Check for extreme negative momentum (price collapse)
    is_price_collapsing = momentum_strength < -momentum_threshold * 1.5 and price_strength < -strong_dev_threshold
    
    # Price above threshold - potential Distribution or Markup
    if price_strength > strong_dev_threshold:
        if momentum_strength < -momentum_threshold and is_high_volume:
            return WyckoffPhase.DISTRIBUTION, not is_reversal_zone
        # Less uncertain if we have consistent price trend or decent momentum
        return WyckoffPhase.DISTRIBUTION, not (price_trend_consistency and (is_high_volume or momentum_consistency))
    
    # Price below threshold - potential Accumulation or Markdown
    if price_strength < -strong_dev_threshold:
        # Fix for the issue: Check if we have a price collapse (strong negative momentum)
        if is_price_collapsing:
            # If price is collapsing, this is markdown, not accumulation
            return WyckoffPhase.MARKDOWN, False
            
        # Accumulation requires momentum to be stabilizing or turning up
        if momentum_strength > -momentum_threshold * 0.3:  # Neutral or positive momentum
            if is_high_volume:  # Accumulation with supporting volume
                return WyckoffPhase.ACCUMULATION, not is_reversal_zone
            return WyckoffPhase.ACCUMULATION, not (price_trend_consistency and momentum_consistency)
        else:
            # Strong negative momentum with negative price - still in Markdown
            return WyckoffPhase.MARKDOWN, not (price_trend_consistency and (is_high_volume or momentum_consistency))

    # Enhanced ranging detection with multi-factor certainty
    if abs(price_strength) <= neutral_zone_threshold:
        # Evaluate ranging certainty based on multiple factors
        is_low_volatility = vol_ratio < 0.85
        is_momentum_neutral = abs(momentum_strength) < momentum_threshold * 0.7
        conflicting_signals = abs(momentum_strength) > momentum_threshold * 0.5
        
        # Calculate overall ranging certainty score (0-3 scale)
        certainty_factors = 0
        if is_low_volatility:
            certainty_factors += 1
        if is_momentum_neutral:
            certainty_factors += 1
        if abs(price_strength) < neutral_zone_threshold * 0.6:  # Very tight range
            certainty_factors += 1
            
        # More certain when multiple factors align, regardless of timeframe
        return WyckoffPhase.RANGING, certainty_factors < 2 or conflicting_signals

    # Improved trend confirmation with more phases considered certain
    trend_strength = abs(momentum_strength) / momentum_threshold
    # Lower threshold for strong trend - more trends become "strong"
    is_strong_trend = trend_strength > 1.3 and (is_high_volume or price_trend_consistency)

    # Check for trend phases (Markup/Markdown)
    if price_strength > 0:
        # Better distinguish markup from end of accumulation
        if momentum_strength > momentum_threshold * 0.8:  # Relaxed threshold
            return WyckoffPhase.MARKUP, not is_strong_trend
        # Less uncertain based on momentum alignment and strength
        return WyckoffPhase.MARKUP, not (price_trend_consistency or (abs(momentum_strength) > momentum_threshold * 0.6))
    elif price_strength < 0:
        # Better distinguish markdown from start of accumulation based on momentum
        if momentum_strength < -momentum_threshold * 0.8:  # Relaxed threshold
            return WyckoffPhase.MARKDOWN, not is_strong_trend
        # Less uncertain based on momentum alignment and strength
        return WyckoffPhase.MARKDOWN, not (price_trend_consistency or (abs(momentum_strength) > momentum_threshold * 0.6))
    
    # Fallback case is uncertain by default (this should rarely happen)
    return WyckoffPhase.RANGING, True

def detect_composite_action(
    df: pd.DataFrame,
    price_strength: float,
    vol_metrics: VolumeMetrics,
    effort_vs_result: float
) -> CompositeAction:
    """Enhanced composite action detection for crypto markets."""
    try:
        # Simplified liquidation cascade detection
        liquidation_cascade = (
            abs(df['c'].pct_change().iloc[-1]) > 0.05 and
            abs(vol_metrics.strength) > 2.5 and
            vol_metrics.ratio > 3.0 and
            abs(df['c'].iloc[-1] - df['o'].iloc[-1]) / 
            (df['h'].iloc[-1] - df['l'].iloc[-1]) > 0.8
        )
        
        if liquidation_cascade:
            return (CompositeAction.MARKING_UP if df['c'].pct_change().iloc[-1] > 0 
                    else CompositeAction.MARKING_DOWN)
        
        # Add whale wallet analysis patterns
        absorption_volume = (
            vol_metrics.ratio > 2.0 and                          # Double normal volume
            abs(vol_metrics.strength) > 1.5 and                  # Significant volume deviation
            abs(df['c'].iloc[-1] - df['o'].iloc[-1]) < 
            (df['h'].iloc[-1] - df['l'].iloc[-1]) * 0.3    # Small body
        )
        
        if absorption_volume:
            return (CompositeAction.ACCUMULATING if price_strength < 0 
                    else CompositeAction.DISTRIBUTING)
        
        # Check for whale manipulation patterns
        sudden_volume_spike = df['v'].iloc[-1] > df['v'].iloc[-5:].mean() * 3
        price_rejection = (abs(df['h'].iloc[-1] - df['c'].iloc[-1]) > 
                          abs(df['c'].iloc[-1] - df['o'].iloc[-1]) * 2)
        
        if sudden_volume_spike and price_rejection:
            if df['c'].iloc[-1] > df['o'].iloc[-1]:
                return CompositeAction.DISTRIBUTING
            return CompositeAction.ACCUMULATING
        
        # Check for absorption of supply/demand
        price_range = df['h'] - df['l']
        price_close = df['c'] - df['o']
        absorption = (price_range.iloc[-1] > price_range.mean()) and (abs(price_close.iloc[-1]) < price_close.std())
        
        if absorption and vol_metrics.trend > VOLUME_THRESHOLD:
            if price_strength < 0:
                return CompositeAction.ACCUMULATING
            return CompositeAction.DISTRIBUTING
            
        if effort_vs_result > EFFORT_THRESHOLD and vol_metrics.trend > 0:
            return CompositeAction.MARKING_UP
        if effort_vs_result < -EFFORT_THRESHOLD and vol_metrics.trend > 0:
            return CompositeAction.MARKING_DOWN
            
        # Add divergence analysis
        price_highs = df['h'].rolling(5).max()
        price_lows = df['l'].rolling(5).min()
        
        bullish_divergence = (
            price_lows.iloc[-1] < price_lows.iloc[-5] and
            df['v'].iloc[-1] > df['v'].iloc[-5] * 1.5 and
            effort_vs_result > 0
        )
        
        bearish_divergence = (
            price_highs.iloc[-1] > price_highs.iloc[-5] and
            df['v'].iloc[-1] > df['v'].iloc[-5] * 1.5 and
            effort_vs_result < 0
        )
        
        if bullish_divergence:
            return CompositeAction.ACCUMULATING
        if bearish_divergence:
            return CompositeAction.DISTRIBUTING
            
        return CompositeAction.NEUTRAL
    except Exception as e:
        logger.error(f"Error in composite action detection: {e}")
        return CompositeAction.UNKNOWN

def detect_wyckoff_signs(
    df: pd.DataFrame,
    price_strength: float,
    volume_trend: float,
    is_spring: bool,
    is_upthrust: bool
) -> WyckoffSign:
    """
    Detect specific Wyckoff signs in market action with stricter confirmation requirements
    to reduce noise and false signals.
    """
    if len(df) < 5:
        return WyckoffSign.NONE
        
    # Calculate key metrics with noise reduction
    price_change = df['c'].pct_change()
    volume_change = df['v'].pct_change()
    
    # Calculate rolling metrics for better context
    price_volatility = df['c'].pct_change().rolling(5).std().iloc[-1]
    volume_ma = df['v'].rolling(5).mean()
    price_ma = df['c'].rolling(8).mean()
    
    # Minimum thresholds scaled by volatility
    min_price_move = max(0.02, price_volatility * 1.5)
    min_volume_surge = max(2.0, volume_change.rolling(5).std().iloc[-1] * 2)
    
    # Check for strong confirmation across multiple candles
    def confirm_trend(window: int, threshold: float) -> bool:
        recent_changes = price_change.iloc[-window:]
        return (recent_changes > threshold).sum() >= window // 2

    def confirm_volume(window: int, threshold: float) -> bool:
        recent_volume = volume_change.iloc[-window:]
        return (recent_volume > threshold).sum() >= window // 2
        
    # Selling Climax (SC) - Requires panic selling with climactic volume
    if (price_change.iloc[-1] < -min_price_move and 
        volume_change.iloc[-1] > min_volume_surge and 
        price_strength < -STRONG_DEV_THRESHOLD and
        df['c'].iloc[-1] < price_ma.iloc[-1] * 0.95):  # Price well below MA
        return WyckoffSign.SELLING_CLIMAX
        
    # Automatic Rally (AR) - Must follow a selling climax
    if (price_change.iloc[-1] > min_price_move and
        confirm_trend(3, min_price_move * 0.5) and
        df['l'].iloc[-1] > df['l'].iloc[-5:].min() and  # Higher low
        price_strength < 0):
        return WyckoffSign.AUTOMATIC_RALLY
        
    # Secondary Test (ST) - Lower volume test of support
    if (abs(price_change.iloc[-1]) < price_volatility and
        df['l'].iloc[-1] >= df['l'].iloc[-5:].min() * 0.99 and  # Test previous low
        df['v'].iloc[-1] < volume_ma.iloc[-1] * 0.8 and  # Lower volume
        price_strength < 0):
        return WyckoffSign.SECONDARY_TEST
        
    # Last Point of Support (LPS) - Spring with volume confirmation
    if (is_spring and
        volume_trend > 0.5 and  # Strong volume
        price_change.iloc[-1] > min_price_move and
        confirm_volume(3, 1.2)):  # Sustained volume
        return WyckoffSign.LAST_POINT_OF_SUPPORT
        
    # Sign of Strength (SOS) - Strong move up with volume
    if (price_change.iloc[-1] > min_price_move * 1.5 and
        confirm_trend(3, min_price_move) and
        confirm_volume(3, 1.5) and
        price_strength > 0.5):  # Clear strength
        return WyckoffSign.SIGN_OF_STRENGTH
        
    # Buying Climax (BC) - Extreme buying with high volume
    if (price_change.iloc[-1] > min_price_move * 2 and 
        volume_change.iloc[-1] > min_volume_surge and 
        price_strength > STRONG_DEV_THRESHOLD and
        df['c'].iloc[-1] > price_ma.iloc[-1] * 1.05):  # Price well above MA
        return WyckoffSign.BUYING_CLIMAX
        
    # Upthrust (UT) - False breakout with rejection
    if (is_upthrust and
        volume_change.iloc[-1] > min_volume_surge * 0.8 and
        price_change.iloc[-1] < -min_price_move * 0.5 and
        df['c'].iloc[-1] < df['h'].iloc[-1] * 0.985):  # Strong rejection
        return WyckoffSign.UPTHRUST
        
    # Secondary Test Resistance (STR) - Higher test with lower volume
    if (abs(price_change.iloc[-1]) < price_volatility and
        df['h'].iloc[-1] <= df['h'].iloc[-5:].max() * 1.01 and
        df['v'].iloc[-1] < volume_ma.iloc[-1] * 0.8 and
        price_strength > 0):
        return WyckoffSign.SECONDARY_TEST_RESISTANCE
        
    # Last Point of Supply (LPSY) - Failed upthrust with volume
    if (is_upthrust and
        volume_trend > 0.5 and
        price_change.iloc[-1] < -min_price_move and
        confirm_volume(3, 1.2)):
        return WyckoffSign.LAST_POINT_OF_RESISTANCE
        
    # Sign of Weakness (SOW) - Strong down move with volume
    if (price_change.iloc[-1] < -min_price_move * 1.5 and
        confirm_trend(3, -min_price_move) and
        confirm_volume(3, 1.5) and
        price_strength < -0.5):
        return WyckoffSign.SIGN_OF_WEAKNESS
        
    return WyckoffSign.NONE

def analyze_funding_rates(funding_rates: List[FundingRateEntry]) -> FundingState:
    """
    Analyze funding rates with enhanced crypto-specific features:
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
) -> EffortAnalysis:
    """
    Enhanced effort vs result analysis with timeframe-specific optimization.
    
    Args:
        df: Price and volume data
        vol_metrics: Volume metrics from calculate_volume_metrics
        timeframe: Current timeframe for context
    """
    try:
        # Get recent data - adjust window based on timeframe
        lookback = {
            Timeframe.MINUTES_15: 3,  # Scalping needs faster response
            Timeframe.MINUTES_30: 4,  # Swing trades
            Timeframe.HOUR_1: 5,      # Main trend
            Timeframe.HOURS_2: 6,     # Main trend context
            Timeframe.HOURS_4: 8,     # Market structure
            Timeframe.HOURS_8: 10,    # Market context
        }.get(timeframe, 5)  # Default to 5 periods
        
        recent_df = df.iloc[-lookback:]
        
        # Calculate normalized price movement
        price_change = abs(recent_df['c'].iloc[-1] - recent_df['o'].iloc[-1])
        price_range = recent_df['h'].iloc[-1] - recent_df['l'].iloc[-1]
        
        # Skip tiny moves to avoid noise - adjust threshold by timeframe
        min_move = RESULT_MIN_MOVE * {
            Timeframe.MINUTES_15: 0.5,   # More sensitive
            Timeframe.MINUTES_30: 0.75,  # Still sensitive
            Timeframe.HOUR_1: 1.0,       # Base threshold
            Timeframe.HOURS_2: 1.25,     # Less sensitive
            Timeframe.HOURS_4: 1.5,      # Focus on larger moves
            Timeframe.HOURS_8: 2.0,      # Only significant moves
        }.get(timeframe, 1.0)
        
        if price_change < min_move:
            return EffortAnalysis(0.0, 0.0, 0.0, EffortResult.WEAK)
        
        # Calculate volume quality with timeframe context
        volume_consistency = vol_metrics.consistency
        spread_quality = 1.0 - (abs(recent_df['c'] - recent_df['o']) / (recent_df['h'] - recent_df['l'])).mean()
        
        # Adjust volume quality based on timeframe expectations
        if timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30]:
            # Short timeframes need stronger volume confirmation
            volume_quality = (volume_consistency * 0.7 + spread_quality * 0.3)
        elif timeframe in [Timeframe.HOUR_1, Timeframe.HOURS_2]:
            # Balanced for main trend
            volume_quality = (volume_consistency + spread_quality) / 2
        else:
            # Longer timeframes focus more on spread quality
            volume_quality = (volume_consistency * 0.3 + spread_quality * 0.7)
        
        # Calculate price impact with timeframe-adjusted volume ratio
        avg_price = recent_df['c'].mean()
        price_impact = price_change / (avg_price * vol_metrics.ratio)
        
        # Calculate efficiency score with timeframe optimization
        base_efficiency = price_change / (price_range + 1e-8)
        volume_weighted_efficiency = base_efficiency * (1 + vol_metrics.strength * {
            Timeframe.MINUTES_15: 0.3,  # More reactive
            Timeframe.MINUTES_30: 0.25,
            Timeframe.HOUR_1: 0.2,     # Base multiplier
            Timeframe.HOURS_2: 0.15,
            Timeframe.HOURS_4: 0.1,    # Less reactive
            Timeframe.HOURS_8: 0.05,   # Minimal adjustment
        }.get(timeframe, 0.2))
        
        # Adjust efficiency thresholds by timeframe
        high_threshold = HIGH_EFFICIENCY_THRESHOLD * {
            Timeframe.MINUTES_15: 0.8,   # Easier to achieve
            Timeframe.MINUTES_30: 0.85,
            Timeframe.HOUR_1: 1.0,       # Base threshold
            Timeframe.HOURS_2: 1.1,
            Timeframe.HOURS_4: 1.2,      # Harder to achieve
            Timeframe.HOURS_8: 1.3,
        }.get(timeframe, 1.0)
        
        low_threshold = LOW_EFFICIENCY_THRESHOLD * {
            Timeframe.MINUTES_15: 1.2,   # Higher floor
            Timeframe.MINUTES_30: 1.1,
            Timeframe.HOUR_1: 1.0,       # Base threshold
            Timeframe.HOURS_2: 0.9,
            Timeframe.HOURS_4: 0.8,      # Lower floor
            Timeframe.HOURS_8: 0.7,
        }.get(timeframe, 1.0)
        
        # Final efficiency score
        efficiency = min(1.0, volume_weighted_efficiency)
        
        # Determine effort result with timeframe context
        result = (
            EffortResult.STRONG if (
                efficiency > high_threshold and 
                volume_quality > (0.5 if timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30] else 0.6)
            )
            else EffortResult.WEAK if (
                efficiency < low_threshold or 
                volume_quality < (0.2 if timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30] else 0.3)
            )
            else EffortResult.STRONG if (
                price_impact > 1.2 and 
                vol_metrics.ratio > EFFORT_VOLUME_THRESHOLD * (
                    0.8 if timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30] else 1.0
                )
            )
            else EffortResult.WEAK
        )
        
        return EffortAnalysis(
            efficiency=efficiency,
            volume_quality=volume_quality,
            price_impact=price_impact,
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error in effort vs result analysis: {e}")
        return EffortAnalysis(0.0, 0.0, 0.0, EffortResult.WEAK)
