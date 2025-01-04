import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final
from .wyckoff_types import *

# Constants for Wyckoff analysis
VOLUME_THRESHOLD: Final[float] = 1.2
STRONG_DEV_THRESHOLD: Final[float] = 1.5
NEUTRAL_ZONE_THRESHOLD: Final[float] = 0.5
MOMENTUM_THRESHOLD: Final[float] = 0.6
EFFORT_THRESHOLD: Final[float] = 0.7
MIN_PERIODS: Final[int] = 30
VOLUME_MA_THRESHOLD: Final[float] = 1.1
VOLUME_SURGE_THRESHOLD: Final[float] = 1.5
VOLUME_TREND_SHORT: Final[int] = 5
VOLUME_TREND_LONG: Final[int] = 10

def detect_spring_upthrust(df: pd.DataFrame, idx: int) -> tuple[bool, bool]:
    """Detect spring and upthrust patterns"""
    if idx < 4:
        return False, False
    
    window = df.iloc[idx-4:idx+1]
    low_point = window['l'].min()
    high_point = window['h'].max()
    close = window['c'].iloc[-1]
    
    is_spring = (window['l'].iloc[-1] < low_point) and (close > low_point) and (window['v'].iloc[-1] > window['v'].mean())
    is_upthrust = (window['h'].iloc[-1] > high_point) and (close < high_point) and (window['v'].iloc[-1] > window['v'].mean())
    
    return is_spring, is_upthrust

def detect_wyckoff_phase(df: pd.DataFrame) -> None:
    """
    Analyze and store Wyckoff phase data for the last two periods in the dataframe.
    """
    # Safety check for minimum required periods
    if len(df) < MIN_PERIODS:
        df.loc[df.index[-2:], 'wyckoff'] = [WyckoffState.unknown(), WyckoffState.unknown()]  # type: ignore[assignment]
        return

    # Get ATR with safety checks
    atr = df.get('ATR', pd.Series([0.0])).iloc[-1]
    if atr == 0 or pd.isna(atr):
        df.loc[df.index[-2:], 'wyckoff'] = [WyckoffState.unknown(), WyckoffState.unknown()]  # type: ignore[assignment]
        return

    # Process last three periods
    for i in [-3, -2, -1]:
        # Get the subset of data up to the current index
        data_subset = df.iloc[:i] if i != -1 else df
        short_term_window = min(MIN_PERIODS, len(data_subset) - 1)
        recent_df = data_subset.iloc[-short_term_window:]
        
        # Calculate technical indicators
        volume_sma = data_subset['v'].rolling(window=MIN_PERIODS).mean()
        price_sma = data_subset['c'].rolling(window=MIN_PERIODS).mean()
        price_std = data_subset['c'].rolling(window=MIN_PERIODS).std()
        momentum = data_subset['c'].diff(14) / data_subset['c'].shift(14)  # 14-period ROC
        
        # Improved volume trend calculation
        volume_short_ma = data_subset['v'].rolling(window=VOLUME_TREND_SHORT).mean()
        volume_long_ma = data_subset['v'].rolling(window=VOLUME_TREND_LONG).mean()
        volume_trend = ((volume_short_ma - volume_long_ma) / volume_long_ma).fillna(0)
        
        # Volume analysis (VSA)
        volume_spread = recent_df['v'] * (recent_df['h'] - recent_df['l'])
        volume_spread_ma = volume_spread.rolling(window=7).mean()
        effort_vs_result = (recent_df['c'] - recent_df['o']) / (recent_df['h'] - recent_df['l'])
        
        # Get current values relative to the position in the loop
        curr_price = data_subset['c'].iloc[-1]
        curr_volume = data_subset['v'].iloc[-1]
        avg_price = price_sma.iloc[-1]
        price_std_last = price_std.iloc[-1]
        volatility = price_std / avg_price
        
        # Detect springs and upthrusts for the current period
        is_spring, is_upthrust = detect_spring_upthrust(data_subset, -1)
        
        # Market condition checks
        volume_sma_value = float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else 0.0
        volume_trend_value = float(volume_trend.iloc[-1]) if not pd.isna(volume_trend.iloc[-1]) else 0.0
        
        # Calculate relative volume (current volume compared to recent average)
        relative_volume = curr_volume / volume_sma_value if volume_sma_value > 0 else 1.0
        
        # Calculate volume consistency (how many recent periods had above-average volume)
        recent_strong_volume = (data_subset['v'].iloc[-3:] > volume_sma.iloc[-3:]).mean()
        
        # Determine if volume is high based on multiple factors
        is_high_volume = (
            (relative_volume > VOLUME_MA_THRESHOLD and volume_trend_value > 0) or  # Trending up and above average
            (relative_volume > VOLUME_SURGE_THRESHOLD) or  # Significant volume spike
            (recent_strong_volume > 0.66 and relative_volume > 1.0)  # Consistent high volume
        )

        price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)
        momentum_strength = momentum.iloc[-1] * 100

        # Phase identification
        phase = identify_wyckoff_phase(
            is_spring, is_upthrust, curr_volume, volume_sma.iloc[-1],
            effort_vs_result.iloc[-1], volume_spread.iloc[-1], volume_spread_ma.iloc[-1],
            price_strength, momentum_strength, is_high_volume, volatility
        )
        
        # Determine if phase is uncertain based on the phase type
        uncertain_phase = phase.value.startswith('~')
        
        effort_result = EffortResult.STRONG if abs(effort_vs_result.iloc[-1]) > EFFORT_THRESHOLD else EffortResult.WEAK
        
        # Detect composite action and Wyckoff signs
        composite_action = detect_composite_action(data_subset, price_strength, volume_trend_value, effort_vs_result.iloc[-1])
        wyckoff_sign = detect_wyckoff_signs(data_subset, price_strength, volume_trend_value, is_spring, is_upthrust)
        
        # Create WyckoffState instance with correct parameters
        wyckoff_state = WyckoffState(
            phase=phase,
            uncertain_phase=uncertain_phase,
            volume=VolumeState.HIGH if is_high_volume else VolumeState.LOW,
            pattern=MarketPattern.TRENDING if abs(momentum_strength) > MOMENTUM_THRESHOLD else MarketPattern.RANGING,
            volatility=VolatilityState.HIGH if volatility.iloc[-1] > volatility.mean() else VolatilityState.NORMAL,
            is_spring=is_spring,
            is_upthrust=is_upthrust,
            volume_spread=VolumeState.HIGH if volume_spread.iloc[-1] > volume_spread_ma.iloc[-1] else VolumeState.LOW,
            effort_vs_result=effort_result,
            composite_action=composite_action,
            wyckoff_sign=wyckoff_sign,
            description=generate_wyckoff_description(
                phase, uncertain_phase, is_high_volume, momentum_strength, 
                is_spring, is_upthrust, effort_result,
                composite_action, wyckoff_sign
            )
        )

        df.loc[df.index[i], 'wyckoff'] = wyckoff_state  # type: ignore


def identify_wyckoff_phase(
    is_spring: bool, is_upthrust: bool, curr_volume: float, volume_sma: float,
    effort_vs_result: float, volume_spread: float, volume_spread_ma: float,
    price_strength: float, momentum_strength: float, is_high_volume: bool,
    volatility: pd.Series
) -> WyckoffPhase:
    """Identify the Wyckoff phase based on market conditions."""
    if is_spring and curr_volume > volume_sma * VOLUME_THRESHOLD:
        return WyckoffPhase.ACCUMULATION
    if is_upthrust and curr_volume > volume_sma * VOLUME_THRESHOLD:
        return WyckoffPhase.DISTRIBUTION
    if effort_vs_result > EFFORT_THRESHOLD and volume_spread > volume_spread_ma * 1.5:
        return WyckoffPhase.MARKUP
    if effort_vs_result < -EFFORT_THRESHOLD and volume_spread > volume_spread_ma * 1.5:
        return WyckoffPhase.MARKDOWN

    return determine_phase_by_price_strength(
        price_strength, momentum_strength, is_high_volume, volatility
    )

def determine_phase_by_price_strength(
    price_strength: float, momentum_strength: float, 
    is_high_volume: bool, volatility: pd.Series
) -> WyckoffPhase:
    """Determine the Wyckoff phase based on price strength and other indicators."""
    if price_strength > STRONG_DEV_THRESHOLD:
        if momentum_strength < -MOMENTUM_THRESHOLD and is_high_volume:
            return WyckoffPhase.DISTRIBUTION
        return WyckoffPhase.POSSIBLE_DISTRIBUTION
    
    if price_strength < -STRONG_DEV_THRESHOLD:
        if momentum_strength > MOMENTUM_THRESHOLD and is_high_volume:
            return WyckoffPhase.ACCUMULATION
        return WyckoffPhase.POSSIBLE_ACCUMULATION
    
    if abs(price_strength) <= NEUTRAL_ZONE_THRESHOLD:
        if abs(momentum_strength) < MOMENTUM_THRESHOLD and volatility.iloc[-1] < volatility.mean():
            return WyckoffPhase.RANGING
        return WyckoffPhase.POSSIBLE_RANGING
    
    # Transitional zones
    if price_strength > 0:
        if momentum_strength > MOMENTUM_THRESHOLD:
            return WyckoffPhase.MARKUP
        return WyckoffPhase.POSSIBLE_MARKUP
    
    if momentum_strength < -MOMENTUM_THRESHOLD:
        return WyckoffPhase.MARKDOWN
    return WyckoffPhase.POSSIBLE_MARKDOWN

def detect_composite_action(
    df: pd.DataFrame,
    price_strength: float,
    volume_trend: float,
    effort_vs_result: float
) -> CompositeAction:
    """Detect actions of composite operators based on Wyckoff principles."""
    # Check for absorption of supply/demand
    price_range = df['h'] - df['l']
    price_close = df['c'] - df['o']
    absorption = (price_range.iloc[-1] > price_range.mean()) and (abs(price_close.iloc[-1]) < price_close.std())
    
    if absorption and volume_trend > VOLUME_THRESHOLD:
        if price_strength < 0:
            return CompositeAction.ACCUMULATING
        return CompositeAction.DISTRIBUTING
        
    if effort_vs_result > EFFORT_THRESHOLD and volume_trend > 0:
        return CompositeAction.MARKING_UP
    if effort_vs_result < -EFFORT_THRESHOLD and volume_trend > 0:
        return CompositeAction.MARKING_DOWN
        
    return CompositeAction.NEUTRAL

def detect_wyckoff_signs(
    df: pd.DataFrame,
    price_strength: float,
    volume_trend: float,
    is_spring: bool,
    is_upthrust: bool
) -> WyckoffSign:
    """Detect specific Wyckoff signs in market action."""
    if len(df) < 5:
        return WyckoffSign.NONE
        
    price_change = df['c'].pct_change()
    volume_change = df['v'].pct_change()
    
    # Selling Climax detection
    if (price_change.iloc[-1] < -0.03 and 
        volume_change.iloc[-1] > 2.0 and 
        price_strength < -STRONG_DEV_THRESHOLD):
        return WyckoffSign.SELLING_CLIMAX
        
    # Buying Climax detection
    if (price_change.iloc[-1] > 0.03 and 
        volume_change.iloc[-1] > 2.0 and 
        price_strength > STRONG_DEV_THRESHOLD):
        return WyckoffSign.BUYING_CLIMAX
        
    # Other signs detection
    if is_spring:
        if volume_trend > 0:
            return WyckoffSign.LAST_POINT_OF_SUPPORT
        return WyckoffSign.SECONDARY_TEST
        
    if is_upthrust:
        if volume_trend > 0:
            return WyckoffSign.LAST_POINT_OF_RESISTANCE
        return WyckoffSign.SECONDARY_TEST_RESISTANCE
        
    return WyckoffSign.NONE

def generate_wyckoff_description(
    phase: WyckoffPhase,
    uncertain_phase: bool,
    is_high_volume: bool,
    momentum_strength: float,
    is_spring: bool,
    is_upthrust: bool,
    effort_vs_result: EffortResult,
    composite_action: CompositeAction,
    wyckoff_sign: WyckoffSign
) -> str:
    """Generate enhanced Wyckoff analysis description."""
    base_phase = phase.name.replace("_", " ").capitalize()
    
    # Start with composite operator action
    description_parts = [
        f"{base_phase} phase with composite operators {composite_action.value}"
    ]
    
    # Add Wyckoff sign if present
    if wyckoff_sign != WyckoffSign.NONE:
        description_parts.append(f"showing {wyckoff_sign.value}")
    
    # Add spring/upthrust patterns
    if is_spring:
        description_parts.append("with spring pattern indicating potential accumulation")
    elif is_upthrust:
        description_parts.append("with upthrust pattern indicating potential distribution")
    
    # Volume and effort analysis
    if is_high_volume and effort_vs_result == EffortResult.STRONG:
        description_parts.append("supported by institutional volume")
    elif not is_high_volume and effort_vs_result == EffortResult.STRONG:
        description_parts.append("showing efficient price movement despite low volume")
    
    # Join description
    main_description = ", ".join(description_parts)
    
    # Add trading suggestion
    suggestion = generate_trading_suggestion(
        phase,
        uncertain_phase,
        momentum_strength,
        is_spring,
        is_upthrust,
        effort_vs_result,
        composite_action,
        wyckoff_sign
    )
    
    return f"{main_description}.\nTrading suggestion: {suggestion}."

def generate_trading_suggestion(
    phase: WyckoffPhase,
    uncertain_phase: bool,
    momentum_strength: float,
    is_spring: bool,
    is_upthrust: bool,
    effort: EffortResult,
    composite_action: CompositeAction,
    wyckoff_sign: WyckoffSign
) -> str:
    """Generate trading suggestion based on enhanced Wyckoff analysis."""
    # Handle uncertain phase first
    if uncertain_phase:
        return "wait for confirmation of institutional activity"
    
    # Check for effort and patterns
    if effort == EffortResult.WEAK:
        return "wait for stronger market conviction"
    
    # Priority to specific Wyckoff signs with high-probability setups
    if wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.LAST_POINT_OF_SUPPORT] and is_spring:
        return "potential accumulation zone, consider preparing long positions"
    if wyckoff_sign in [WyckoffSign.BUYING_CLIMAX, WyckoffSign.LAST_POINT_OF_RESISTANCE] and is_upthrust:
        return "potential distribution zone, consider preparing short positions"
    
    # Consider composite operator actions with momentum confirmation
    if composite_action == CompositeAction.ACCUMULATING and momentum_strength > 0:
        return "institutional accumulation detected, consider joining with longs"
    if composite_action == CompositeAction.DISTRIBUTING and momentum_strength < 0:
        return "institutional distribution detected, consider joining with shorts"
    
    # Basic phase-based suggestions
    basic_suggestions = {
        WyckoffPhase.ACCUMULATION: "look for spring patterns and signs of absorption",
        WyckoffPhase.DISTRIBUTION: "look for upthrust patterns and signs of supply",
        WyckoffPhase.MARKUP: "follow the trend with stops under support",
        WyckoffPhase.MARKDOWN: "follow the trend with stops above resistance",
        WyckoffPhase.RANGING: "wait for clear institutional participation",
        WyckoffPhase.UNKNOWN: "wait for clear institutional activity"
    }
    
    return basic_suggestions.get(phase, "wait for clear institutional activity")
