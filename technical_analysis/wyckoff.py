import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final, Dict, List, Optional, Any, Tuple
from .wyckoff_types import (
    MarketPattern, VolatilityState, WyckoffState, WyckoffPhase, EffortResult, 
    CompositeAction, WyckoffSign, FundingState, VolumeState, Timeframe
)
from .funding_rates_cache import FundingRateEntry
from statistics import mean
from .wyckoff_description import generate_wyckoff_description
from utils import log_execution_time
from dataclasses import dataclass
from logging_utils import logger

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

def detect_spring_upthrust(df: pd.DataFrame, idx: int, vol_metrics: VolumeMetrics) -> tuple[bool, bool]:
    """Enhanced spring/upthrust detection using normalized volume metrics."""
    if idx < 4:
        return False, False
    
    try:
        window = df.iloc[idx-4:idx+1]
        low_point = window['l'].min()
        high_point = window['h'].max()
        close = window['c'].iloc[-1]
        
        # Calculate volatility context
        atr = ta.atr(df['h'], df['l'], df['c'], length=14).iloc[-1]
        price = df['c'].iloc[-1]
        volatility_factor = atr / price  # Normalize ATR by price
        
        # Adaptive thresholds based on volatility
        spring_threshold = 0.001 * (1 + volatility_factor * 10)
        upthrust_threshold = 0.001 * (1 + volatility_factor * 10)
        
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
        
        # Get current values
        curr_price = df['c'].iloc[-1]
        curr_volume = df['v'].iloc[-1]
        
        # Use volume metrics instead of raw calculations
        volume_trend_value = vol_metrics.trend_strength
        relative_volume = vol_metrics.ratio
        recent_strong_volume = vol_metrics.consistency
        
        # Rest of function remains the same
        price_sma = df['c'].rolling(window=MIN_PERIODS).mean()
        price_std = df['c'].rolling(window=MIN_PERIODS).std()

        
        # Volume analysis (VSA)
        volume_spread = recent_df['v'] * (recent_df['h'] - recent_df['l'])
        volume_spread_ma = volume_spread.rolling(window=7).mean()
        effort_vs_result = (recent_df['c'] - recent_df['o']) / (recent_df['h'] - recent_df['l'])
        
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
        is_spring, is_upthrust = detect_spring_upthrust(df, -1, vol_metrics)
        
        # Market condition checks
        volume_sma_value = float(vol_metrics.sma) if not pd.isna(vol_metrics.sma) else 0.0
        
        # Calculate relative volume
        relative_volume = curr_volume / volume_sma_value if volume_sma_value > 0 else 1.0
        
        # Calculate volume consistency
        recent_strong_volume = vol_metrics.consistency
        
        # Get thresholds from timeframe settings
        volume_threshold, _, _, \
        momentum_threshold, effort_threshold, volume_surge_threshold = timeframe.settings.thresholds
        
        # Use thresholds directly
        is_high_volume = (
            (relative_volume > volume_threshold and volume_trend_value > 0) or
            (relative_volume > volume_surge_threshold) or
            (recent_strong_volume > 0.6 and relative_volume > 1.2)
        )

        price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)

        # Enhanced price strength calculation
        price_ma_short = df['c'].rolling(window=8).mean()
        price_ma_long = df['c'].rolling(window=21).mean()
        trend_strength = ((price_ma_short - price_ma_long) / price_ma_long).iloc[-1]
        price_strength = price_strength * 0.7 + trend_strength * 0.3

        # Calculate recent price change before calling identify_wyckoff_phase
        recent_change = df['c'].pct_change(3).iloc[-1]
        
        # Calculate time-weighted volume profile
        def calculate_weighted_volume_profile(df: pd.DataFrame) -> float:
            """
            Calculate volume profile using exponentially weighted volume across all available data.
            More recent volumes have higher weights but all data contributes to the analysis.
            
            Args:
                df: DataFrame with OHLCV data
            
            Returns:
                Price level with highest weighted volume
            """
            # Create exponential weights for all data points
            # Using exponential decay where most recent point has weight 1.0
            # and weight decays by half every ~100 periods
            decay_factor = -np.log(2) / 100  # Half-life of 100 periods
            weights = np.exp(np.linspace(decay_factor * len(df), 0, len(df)))
            weights = weights / weights.sum()  # Normalize weights
            
            # Calculate volume-weighted price levels
            weighted_volumes = df['v'] * weights
            volume_profile = weighted_volumes.groupby(df['c'].round(2)).sum()
            
            # Get price level with highest weighted volume
            return float(volume_profile.idxmax())

        high_vol_price = calculate_weighted_volume_profile(df)

        current_price = df['c'].iloc[-1]
        
        funding_state = analyze_funding_rates(funding_rates or [])

        # Phase identification for current period
        current_phase = identify_wyckoff_phase(
            is_spring, is_upthrust, curr_volume, vol_metrics.sma,
            effort_vs_result.iloc[-1], volume_spread.iloc[-1], volume_spread_ma.iloc[-1],
            price_strength, momentum_strength, is_high_volume, volatility,
            timeframe, recent_change, high_vol_price, current_price
        )
        
        # Replace simple effort_result calculation with new analysis
        effort_analysis = analyze_effort_result(df, vol_metrics, timeframe)
        effort_result = effort_analysis.result
        
        # Detect composite action and Wyckoff signs
        composite_action = detect_composite_action(df, price_strength, vol_metrics, effort_vs_result.iloc[-1])
        wyckoff_sign = detect_wyckoff_signs(df, price_strength, volume_trend_value, is_spring, is_upthrust)
            
        # Create WyckoffState instance
        wyckoff_state = WyckoffState(
            phase=current_phase,
            uncertain_phase=current_phase.value.startswith('~'),
            volume=VolumeState.HIGH if is_high_volume else VolumeState.LOW,
            pattern=MarketPattern.TRENDING if abs(momentum_strength) > momentum_threshold else MarketPattern.RANGING,
            volatility=VolatilityState.HIGH if volatility.iloc[-1] > volatility.mean() else VolatilityState.NORMAL,
            is_spring=is_spring,
            is_upthrust=is_upthrust,
            volume_spread=VolumeState.HIGH if volume_spread.iloc[-1] > volume_spread_ma.iloc[-1] else VolumeState.LOW,
            effort_vs_result=effort_result,
            composite_action=composite_action,
            wyckoff_sign=wyckoff_sign,
            funding_state=funding_state,
            description=generate_wyckoff_description(
                current_phase, current_phase.value.startswith('~'), is_high_volume, momentum_strength, 
                is_spring, is_upthrust, effort_result,
                composite_action, wyckoff_sign, funding_state
            )
        )

        return wyckoff_state # type: ignore
    except Exception as e:
        logger.error(f"Error in Wyckoff phase detection: {e}")
        return WyckoffState.unknown()

def identify_wyckoff_phase(
    is_spring: bool, is_upthrust: bool, curr_volume: float, volume_sma: float,
    effort_vs_result: float, volume_spread: float, volume_spread_ma: float,
    price_strength: float, momentum_strength: float, is_high_volume: bool,
    volatility: pd.Series, timeframe: Timeframe, recent_change: float,
    high_vol_price: float, current_price: float
) -> WyckoffPhase:
    """Enhanced Wyckoff phase identification optimized for intraday crypto trading."""

    # Detect scalping opportunities based on timeframe
    is_scalp_timeframe = timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30]
    if is_scalp_timeframe:
        # Short-term breakout detection
        breakout_threshold = 0.015  # 1.5% move for short timeframes
        if abs(recent_change) > breakout_threshold and is_high_volume:
            return (WyckoffPhase.MARKUP if recent_change > 0 
                   else WyckoffPhase.MARKDOWN)

    # Enhanced liquidation cascade detection optimized for hourly analysis
    volume_impulse = curr_volume / (volume_sma + 1e-8)
    price_velocity = abs(recent_change) / (volatility.mean() + 1e-8)
    
    # Use dynamic thresholds based on market conditions
    # More sensitive thresholds for shorter timeframes that update more frequently
    vol_threshold = 2.5
    price_threshold = 0.04  # 4% move
    velocity_threshold = 2.0
    effort_threshold = 0.7
    
    # Adjust thresholds by timeframe
    if timeframe == Timeframe.MINUTES_15:
        price_threshold = 0.035  # More sensitive on shorter timeframes
        vol_threshold = 2.2
    elif timeframe == Timeframe.HOURS_4 or timeframe == Timeframe.HOURS_8:
        price_threshold = 0.05  # Higher threshold for longer timeframes
        vol_threshold = 2.8
    
    is_liquidation = (
        abs(recent_change) > price_threshold and  
        volume_impulse > vol_threshold and       
        price_velocity > velocity_threshold and  
        abs(effort_vs_result) > effort_threshold 
    )

    # Enhanced cascade detection using hourly execution awareness
    # Track if this signal is changing within a short period
    cascade_signal_strength = (abs(recent_change) / price_threshold) * (volume_impulse / vol_threshold)
    
    if is_liquidation:
        # Check for potential reversal after liquidation
        if cascade_signal_strength > 1.5:  # Strong liquidation event
            return (WyckoffPhase.POSSIBLE_DISTRIBUTION if recent_change > 0
                   else WyckoffPhase.POSSIBLE_ACCUMULATION)
        return (WyckoffPhase.MARKUP if recent_change > 0
                else WyckoffPhase.MARKDOWN)

    # Rest of the existing phase determination logic
    return determine_phase_by_price_strength(
        price_strength, momentum_strength, is_high_volume, volatility, timeframe
    )

def _adjust_thresholds(
    is_very_strong_move: bool, is_strong_move: bool,
    volatility: pd.Series, avg_volatility: float,
    momentum_threshold: float, effort_threshold: float, volume_threshold: float
) -> Tuple[float, float, float]:
    """Adjust thresholds with enhanced intraday sensitivity."""
    
    # Calculate intraday volatility ratio
    recent_volatility = volatility.iloc[-12:].std()  # Last 12 periods
    volatility_ratio = recent_volatility / volatility.std()
    
    # Dynamic adjustments based on intraday volatility
    if volatility_ratio > 1.5:  # High intraday volatility
        momentum_threshold *= 0.7  # More sensitive
        effort_threshold *= 0.8
        volume_threshold *= 1.2  # Require more volume confirmation
    elif volatility_ratio < 0.7:  # Low intraday volatility
        momentum_threshold *= 1.2  # Less sensitive
        effort_threshold *= 1.1
        volume_threshold *= 0.9  # Accept lower volume

    # Existing price movement adjustments
    if is_very_strong_move:
        momentum_threshold *= 0.6
        effort_threshold *= 0.7
    elif is_strong_move:
        momentum_threshold *= 0.8
        effort_threshold *= 0.85

    # Enhanced volatility factor calculation
    volatility_factor = min(2.0, (1.0 + volatility.iloc[-1] / avg_volatility) * volatility_ratio)
    
    # Final adjustments with volatility factor
    return (
        momentum_threshold * (1.0 + volatility_factor * 0.2),
        effort_threshold * (1.0 / volatility_factor),
        volume_threshold * volatility_factor
    )

def determine_phase_by_price_strength(
    price_strength: float, momentum_strength: float, 
    is_high_volume: bool, volatility: pd.Series,
    timeframe: Timeframe
) -> WyckoffPhase:
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

    # Detect potential reversal zones
    is_reversal_zone = abs(price_strength) > strong_dev_threshold * 1.5

    if price_strength > strong_dev_threshold:
        if momentum_strength < -momentum_threshold and is_high_volume:
            return WyckoffPhase.DISTRIBUTION if is_reversal_zone else WyckoffPhase.POSSIBLE_DISTRIBUTION
        return WyckoffPhase.POSSIBLE_DISTRIBUTION
    
    if price_strength < -strong_dev_threshold:
        if momentum_strength > momentum_threshold and is_high_volume:
            return WyckoffPhase.ACCUMULATION if is_reversal_zone else WyckoffPhase.POSSIBLE_ACCUMULATION
        return WyckoffPhase.POSSIBLE_ACCUMULATION

    # Enhanced ranging detection
    if abs(price_strength) <= neutral_zone_threshold:
        is_low_volatility = vol_ratio < 0.8
        if abs(momentum_strength) < momentum_threshold and is_low_volatility:
            return WyckoffPhase.RANGING
        return WyckoffPhase.POSSIBLE_RANGING if vol_ratio < 1.2 else WyckoffPhase.POSSIBLE_MARKUP

    # Improved trend confirmation
    trend_strength = abs(momentum_strength) / momentum_threshold
    is_strong_trend = trend_strength > 1.5 and is_high_volume

    if price_strength > 0:
        if momentum_strength > momentum_threshold:
            return WyckoffPhase.MARKUP if is_strong_trend else WyckoffPhase.POSSIBLE_MARKUP
        return WyckoffPhase.POSSIBLE_MARKUP
    
    if momentum_strength < -momentum_threshold:
        return WyckoffPhase.MARKDOWN if is_strong_trend else WyckoffPhase.POSSIBLE_MARKDOWN
    return WyckoffPhase.POSSIBLE_MARKDOWN


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
        (1 + max(min(rate.funding_rate, 100), -100)) ** 8760 - 1  # Limit extreme values
        for rate in funding_rates
    ])
    times = np.array([rate.time for rate in funding_rates])
    
    # Remove outliers using IQR method
    q1, q3 = np.percentile(rates, [25, 75])
    iqr = q3 - q1
    mask = (rates >= q1 - 1.5 * iqr) & (rates <= q3 + 1.5 * iqr)
    rates = rates[mask]
    times = times[mask]
    
    if len(rates) < 3:  # Check if we still have enough data after outlier removal
        return FundingState.UNKNOWN
    
    # Non-linear time weighting (emphasizes recent values more)
    time_diff_hours = (now - times) / (1000 * 3600)
    weights = 1 / (1 + np.exp(0.5 * time_diff_hours - 2))  # Steeper sigmoid curve
    
    # Calculate weighted average with normalization
    avg_funding = np.sum(rates * weights) / np.sum(weights)

    # Determine state with granular thresholds
    if avg_funding > 0.25:
        return FundingState.HIGHLY_POSITIVE
    elif avg_funding > 0.1:
        return FundingState.POSITIVE
    elif avg_funding > 0.02:
        return FundingState.SLIGHTLY_POSITIVE
    elif avg_funding < -0.25:
        return FundingState.HIGHLY_NEGATIVE
    elif avg_funding < -0.1:
        return FundingState.NEGATIVE
    elif avg_funding < -0.02:
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
        # Get recent data - adjust window based on timeframe group
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
