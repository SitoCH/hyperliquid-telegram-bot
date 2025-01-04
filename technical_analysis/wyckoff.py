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
        
        # Create WyckoffState instance
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
            description=generate_wyckoff_description(
                phase, uncertain_phase, is_high_volume, momentum_strength, 
                volatility.iloc[-1], is_spring, is_upthrust, effort_result
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

def generate_trading_suggestion(
    phase: WyckoffPhase,
    uncertain_phase: bool,
    momentum_strength: float,
    is_spring: bool,
    is_upthrust: bool,
    effort: EffortResult
) -> str:
    """Generate trading suggestion based on Wyckoff analysis."""
    if uncertain_phase:
        return "wait for confirmation before trading"
    
    if is_spring and effort == EffortResult.STRONG:
        return "consider opening long position"
    if is_upthrust and effort == EffortResult.STRONG:
        return "consider opening short position"
    
    suggestions = {
        WyckoffPhase.ACCUMULATION: "accumulate long positions",
        WyckoffPhase.DISTRIBUTION: "accumulate short positions",
        WyckoffPhase.MARKUP: "hold / increase long positions" if momentum_strength > MOMENTUM_THRESHOLD else "hold long positions",
        WyckoffPhase.MARKDOWN: "hold / increase short positions" if momentum_strength < -MOMENTUM_THRESHOLD else "hold short positions",
        WyckoffPhase.POSSIBLE_ACCUMULATION: "cautiously accumulate long positions",
        WyckoffPhase.POSSIBLE_DISTRIBUTION: "cautiously accumulate short positions",
        WyckoffPhase.POSSIBLE_MARKUP: "cautiously hold long positions",
        WyckoffPhase.POSSIBLE_MARKDOWN: "cautiously hold short positions",
        WyckoffPhase.UNKNOWN: "wait for clear signal"
    }
    
    if effort == EffortResult.WEAK:
        suggestion = suggestions.get(phase, "wait for clear signal")
        return f"cautiously {suggestion}" if not suggestion.startswith(("wait", "cautiously")) else suggestion
        
    return suggestions.get(phase, "wait for clear signal")

def generate_wyckoff_description(
    phase: WyckoffPhase,
    uncertain_phase: bool,
    is_high_volume: bool,
    momentum_strength: float,
    volatility: float,
    is_spring: bool,
    is_upthrust: bool,
    effort_vs_result: EffortResult
) -> str:
    """Generate a descriptive text of the Wyckoff analysis results with trading suggestion."""
    base_phase = phase.name.replace("_", " ").capitalize()
    
    description_parts = [base_phase]
    
    # Pattern context first
    if is_spring:
        description_parts.append("showing Spring pattern")
    elif is_upthrust:
        description_parts.append("showing Upthrust pattern")
    
    # Volume and effort characteristics
    if is_high_volume and effort_vs_result == EffortResult.STRONG:
        description_parts.append("with strong volume and conviction")
    elif is_high_volume and effort_vs_result == EffortResult.WEAK:
        description_parts.append("with high volume but weak momentum")
    elif not is_high_volume and effort_vs_result == EffortResult.STRONG:
        description_parts.append("with efficient price movement despite low volume")
    
    # Market context
    context_parts = []
    if abs(momentum_strength) > MOMENTUM_THRESHOLD:
        context_parts.append("trending market")
    else:
        context_parts.append("ranging market")
    
    if volatility > 1.5:  # Use fixed threshold for high volatility
        context_parts.append("high volatility")
    
    if context_parts:
        description_parts.append(f"in a {' with '.join(context_parts)}")
    
    # Join main description
    main_description = ", ".join(description_parts)
    
    # Add trading suggestion on new line
    suggestion = generate_trading_suggestion(
        phase,
        uncertain_phase,
        momentum_strength,
        is_spring,
        is_upthrust,
        effort_vs_result
    )
    
    return f"{main_description}.\nTrading suggestion: {suggestion}."
