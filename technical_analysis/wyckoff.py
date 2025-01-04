import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final

# Constants for Wyckoff analysis
VOLUME_THRESHOLD: Final[float] = 1.2
STRONG_DEV_THRESHOLD: Final[float] = 1.5
NEUTRAL_ZONE_THRESHOLD: Final[float] = 0.5
MOMENTUM_THRESHOLD: Final[float] = 0.6
EFFORT_THRESHOLD: Final[float] = 0.7
MIN_PERIODS: Final[int] = 30

def detect_spring_upthrust(df: pd.DataFrame, idx: int, atr: float) -> tuple[bool, bool]:
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
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
    """
    # Safety check for minimum required periods
    if len(df) < MIN_PERIODS:
        df.loc[df.index[-2:], ['wyckoff_phase', 'uncertain_phase', 'wyckoff_volume', 
                              'wyckoff_pattern', 'wyckoff_volatility']] = "Unknown", True, "unknown", "unknown", "unknown"
        return

    # Get ATR with safety checks
    atr: float = df.get('ATR', pd.Series([0.0])).iloc[-1]
    if atr == 0 or pd.isna(atr):
        df.loc[df.index[-2:], ['wyckoff_phase', 'uncertain_phase', 'wyckoff_volume', 
                              'wyckoff_pattern', 'wyckoff_volatility']] = "Unknown", True, "unknown", "unknown", "unknown"
        return

    # Process last two periods
    for i in [-3, -2, -1]:
        end_idx = i if i == -1 else -2
        full_df = df[:end_idx]
        short_term_window = min(MIN_PERIODS, len(full_df) - 1)
        recent_df = full_df.iloc[-short_term_window:]
        
        # Calculate technical indicators
        volume_sma = full_df['v'].rolling(window=MIN_PERIODS).mean()
        price_sma = full_df['c'].rolling(window=MIN_PERIODS).mean()
        price_std = full_df['c'].rolling(window=MIN_PERIODS).std()
        momentum = full_df['c'].diff(14) / full_df['c'].shift(14)  # 14-period ROC
        volume_trend = full_df['v'].diff(5)  # 5-period volume trend
        
        # Volume analysis (VSA)
        volume_spread = recent_df['v'] * (recent_df['h'] - recent_df['l'])
        volume_spread_ma = volume_spread.rolling(window=7).mean()
        effort_vs_result = (recent_df['c'] - recent_df['o']) / (recent_df['h'] - recent_df['l'])
        
        # Detect springs and upthrusts
        is_spring, is_upthrust = detect_spring_upthrust(full_df, -1, atr)
        
        # Current market conditions
        curr_price = recent_df['c'].iloc[-1]
        curr_volume = recent_df['v'].iloc[-1]
        avg_price = price_sma.iloc[-1]
        price_std_last = price_std.iloc[-1]
        volatility = price_std / avg_price
        
        # Market condition checks
        is_high_volume = (curr_volume > volume_sma.iloc[-1] * 1.1) and (volume_trend.iloc[-1] > 0)
        price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)
        momentum_strength = momentum.iloc[-1] * 100

        # Phase identification
        phase, uncertain_phase = identify_wyckoff_phase(
            is_spring, is_upthrust, curr_volume, volume_sma.iloc[-1],
            effort_vs_result.iloc[-1], volume_spread.iloc[-1], volume_spread_ma.iloc[-1],
            price_strength, momentum_strength, is_high_volume, volatility
        )
        
        # Store results
        current_idx = df.index[i]
        store_wyckoff_results(
            df, current_idx, phase, uncertain_phase, is_high_volume,
            momentum_strength, volatility, is_spring, is_upthrust,
            volume_spread.iloc[-1], volume_spread_ma.iloc[-1],
            effort_vs_result.iloc[-1]
        )

def identify_wyckoff_phase(
    is_spring: bool, is_upthrust: bool, curr_volume: float, volume_sma: float,
    effort_vs_result: float, volume_spread: float, volume_spread_ma: float,
    price_strength: float, momentum_strength: float, is_high_volume: bool,
    volatility: pd.Series
) -> tuple[str, bool]:
    """Identify the Wyckoff phase based on market conditions."""
    if is_spring and curr_volume > volume_sma * VOLUME_THRESHOLD:
        return "acc.", False
    if is_upthrust and curr_volume > volume_sma * VOLUME_THRESHOLD:
        return "dist.", False
    if effort_vs_result > EFFORT_THRESHOLD and volume_spread > volume_spread_ma * 1.5:
        return "markup", False
    if effort_vs_result < -EFFORT_THRESHOLD and volume_spread > volume_spread_ma * 1.5:
        return "markdown", False

    return determine_phase_by_price_strength(
        price_strength, momentum_strength, is_high_volume, volatility
    )

def determine_phase_by_price_strength(
    price_strength: float, momentum_strength: float, 
    is_high_volume: bool, volatility: pd.Series
) -> tuple[str, bool]:
    """Determine the Wyckoff phase based on price strength and other indicators."""
    if price_strength > STRONG_DEV_THRESHOLD:
        if momentum_strength < -MOMENTUM_THRESHOLD and is_high_volume:
            return "dist.", False
        return "~ dist.", True
    
    if price_strength < -STRONG_DEV_THRESHOLD:
        if momentum_strength > MOMENTUM_THRESHOLD and is_high_volume:
            return "acc.", False
        return "~ acc.", True
    
    if abs(price_strength) <= NEUTRAL_ZONE_THRESHOLD:
        if abs(momentum_strength) < MOMENTUM_THRESHOLD and volatility.iloc[-1] < volatility.mean():
            return "rang.", False
        return "~ rang.", True
    
    # Transitional zones
    if price_strength > 0:
        if momentum_strength > MOMENTUM_THRESHOLD:
            return "markup", False
        return "~ markup", True
    
    if momentum_strength < -MOMENTUM_THRESHOLD:
        return "markdown", False
    return "~ markdown", True

def generate_trading_suggestion(
    phase: str,
    uncertain_phase: bool,
    momentum_strength: float,
    is_spring: bool,
    is_upthrust: bool,
    effort: str
) -> str:
    """Generate trading suggestion based on Wyckoff analysis."""
    if uncertain_phase:
        return "wait for confirmation before trading"
    
    if is_spring and effort == "strong":
        return "consider opening long position"
    if is_upthrust and effort == "strong":
        return "consider opening short position"
    
    suggestions = {
        "acc.": "accumulate long positions",
        "dist.": "accumulate short positions",
        "markup": "hold / increase long positions" if momentum_strength > MOMENTUM_THRESHOLD else "hold long positions",
        "markdown": "hold / increase short positions" if momentum_strength < -MOMENTUM_THRESHOLD else "hold short positions",
        "rang.": "trade range boundaries or wait for breakout"
    }
    
    if effort == "weak":
        return f"cautiously {suggestions.get(phase, 'wait for clear signal')}"
    return suggestions.get(phase, "wait for clear signal")

def generate_wyckoff_description(
    phase: str,
    uncertain_phase: bool,
    volume: str,
    pattern: str,
    volatility: str,
    is_spring: bool,
    is_upthrust: bool,
    volume_spread: str,
    effort: str,
    momentum_strength: float
) -> str:
    """Generate a descriptive text of the Wyckoff analysis results with trading suggestion."""
    base_phase = {
        "acc.": "Accumulation",
        "dist.": "Distribution",
        "markup": "Mark up",
        "markdown": "Mark down",
        "rang.": "Trading range",
        "~ acc.": "Possible accumulation",
        "~ dist.": "Possible distribution",
        "~ markup": "Possible mark up",
        "~ markdown": "Possible mark down",
        "~ rang.": "Possible trading range"
    }.get(phase, "Unknown")
    
    description_parts = [base_phase]
    
    # Pattern context first
    if is_spring:
        description_parts.append("showing Spring pattern")
    elif is_upthrust:
        description_parts.append("showing Upthrust pattern")
    
    # Volume and effort characteristics
    if volume == "high" and effort == "strong":
        description_parts.append("with strong volume and conviction")
    elif volume == "high" and effort == "weak":
        description_parts.append("with high volume but weak momentum")
    elif volume == "low" and effort == "strong":
        description_parts.append("with efficient price movement despite low volume")
    
    # Market context
    context_parts = []
    if pattern == "trending":
        context_parts.append("trending market")
    else:
        context_parts.append("ranging market")
    
    if volatility == "high":
        context_parts.append("high volatility")
    
    if context_parts:
        description_parts.append(f"in a {' with '.join(context_parts)}")
    
    # Join main description
    main_description = ", ".join(description_parts)
    
    # Add trading suggestion on new line
    suggestion = generate_trading_suggestion(
        phase, uncertain_phase, momentum_strength,
        is_spring, is_upthrust, effort
    )
    
    return f"{main_description}.\nTrading suggestion: {suggestion}."

def store_wyckoff_results(
    df: pd.DataFrame, idx: pd.Timestamp, phase: str, uncertain_phase: bool,
    is_high_volume: bool, momentum_strength: float, volatility: pd.Series,
    is_spring: bool, is_upthrust: bool, volume_spread: float,
    volume_spread_ma: float, effort_vs_result: float
) -> None:
    """Store Wyckoff analysis results in the DataFrame."""
    df.loc[idx, 'wyckoff_phase'] = phase
    df.loc[idx, 'uncertain_phase'] = uncertain_phase
    df.loc[idx, 'wyckoff_volume'] = "high" if is_high_volume else "low"
    df.loc[idx, 'wyckoff_pattern'] = "trending" if abs(momentum_strength) > MOMENTUM_THRESHOLD else "ranging"
    df.loc[idx, 'wyckoff_volatility'] = "high" if volatility.iloc[-1] > volatility.mean() else "normal"
    df.loc[idx, 'wyckoff_spring'] = is_spring
    df.loc[idx, 'wyckoff_upthrust'] = is_upthrust
    df.loc[idx, 'wyckoff_volume_spread'] = "high" if volume_spread > volume_spread_ma else "low"
    df.loc[idx, 'effort_vs_result'] = "strong" if abs(effort_vs_result) > EFFORT_THRESHOLD else "weak"
    
    # Generate and store description
    description = generate_wyckoff_description(
        phase=phase,
        uncertain_phase=uncertain_phase,
        volume="high" if is_high_volume else "low",
        pattern="trending" if abs(momentum_strength) > MOMENTUM_THRESHOLD else "ranging",
        volatility="high" if volatility.iloc[-1] > volatility.mean() else "normal",
        is_spring=is_spring,
        is_upthrust=is_upthrust,
        volume_spread="high" if volume_spread > volume_spread_ma else "low",
        effort="strong" if abs(effort_vs_result) > EFFORT_THRESHOLD else "weak",
        momentum_strength=momentum_strength  # Added momentum_strength parameter
    )
    df.loc[idx, 'wyckoff_description'] = description

