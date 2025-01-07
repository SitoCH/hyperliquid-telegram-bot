import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final
from .wyckoff_types import MarketPattern, VolatilityState, WyckoffState, WyckoffPhase, EffortResult, CompositeAction, WyckoffSign, FundingState, VolumeState
from .funding_rates_cache import FundingRateEntry
from statistics import mean
from typing import List, Optional, Dict, Any

# Constants for Wyckoff analysis
VOLUME_THRESHOLD: Final[float] = 1.5  # Increased from 1.2 for crypto's higher volatility
STRONG_DEV_THRESHOLD: Final[float] = 1.8  # Increased from 1.5 for wider price swings
NEUTRAL_ZONE_THRESHOLD: Final[float] = 0.8  # Increased from 0.5 for crypto's wider ranging periods
MOMENTUM_THRESHOLD: Final[float] = 0.5  # Decreased from 0.6 for faster momentum shifts
EFFORT_THRESHOLD: Final[float] = 0.65  # Decreased from 0.7 for crypto's quick moves
MIN_PERIODS: Final[int] = 30
VOLUME_MA_THRESHOLD: Final[float] = 1.3  # Increased from 1.1
VOLUME_SURGE_THRESHOLD: Final[float] = 2.0  # Increased from 1.5 for crypto's volume spikes
VOLUME_TREND_SHORT: Final[int] = 5
VOLUME_TREND_LONG: Final[int] = 10
FUNDING_EXTREME_THRESHOLD: Final[float] = 0.01  # 1% threshold for extreme funding
FUNDING_MODERATE_THRESHOLD: Final[float] = 0.005  # 0.5% threshold for moderate funding

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

def detect_wyckoff_phase(df: pd.DataFrame, funding_rates: Optional[List[FundingRateEntry]] = None) -> None:
    """
    Analyze and store Wyckoff phase data incorporating funding rates.
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

        # Enhanced volume analysis for crypto
        volume_short_ma = data_subset['v'].rolling(window=3).mean()  # Shorter window for faster response
        volume_long_ma = data_subset['v'].rolling(window=8).mean()  # Shorter window than stock markets
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
        
        # Enhanced momentum calculation for crypto
        fast_momentum = data_subset['c'].diff(7) / data_subset['c'].shift(7)  # Faster ROC
        slow_momentum = data_subset['c'].diff(14) / data_subset['c'].shift(14)  # Original ROC
        momentum = (fast_momentum + slow_momentum) / 2  # Combined momentum
        
        # Volume spike detection for crypto
        volume_std = data_subset['v'].rolling(window=12).std()
        volume_spike = (curr_volume - volume_sma_value) / (volume_std.iloc[-1] + 1e-8)
        
        # Enhanced relative volume calculation
        is_high_volume = (
            (relative_volume > VOLUME_MA_THRESHOLD and volume_trend_value > 0) or
            (volume_spike > 2.5) or  # More sensitive to extreme spikes
            (relative_volume > VOLUME_SURGE_THRESHOLD) or
            (recent_strong_volume > 0.6 and relative_volume > 1.2)  # More lenient conditions
        )

        price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)
        momentum_strength = momentum.iloc[-1] * 100

        # Enhanced price strength calculation
        price_ma_short = data_subset['c'].rolling(window=8).mean()
        price_ma_long = data_subset['c'].rolling(window=21).mean()
        trend_strength = ((price_ma_short - price_ma_long) / price_ma_long).iloc[-1]
        price_strength = price_strength * 0.7 + trend_strength * 0.3  # Combine both metrics

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
        
        funding_state = analyze_funding_rates(funding_rates or [])
        
        # Adjust phase confidence based on funding rates
        if not uncertain_phase and funding_state != FundingState.UNKNOWN:
            if phase in [WyckoffPhase.MARKUP, WyckoffPhase.ACCUMULATION]:
                if funding_state in [FundingState.HIGHLY_NEGATIVE, FundingState.NEGATIVE]:
                    uncertain_phase = True
            elif phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.DISTRIBUTION]:
                if funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.POSITIVE]:
                    uncertain_phase = True

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
            funding_state=funding_state,
            description=generate_wyckoff_description(
                phase, uncertain_phase, is_high_volume, momentum_strength, 
                is_spring, is_upthrust, effort_result,
                composite_action, wyckoff_sign, funding_state
            )
        )

        df.loc[df.index[i], 'wyckoff'] = wyckoff_state  # type: ignore


def identify_wyckoff_phase(
    is_spring: bool, is_upthrust: bool, curr_volume: float, volume_sma: float,
    effort_vs_result: float, volume_spread: float, volume_spread_ma: float,
    price_strength: float, momentum_strength: float, is_high_volume: bool,
    volatility: pd.Series
) -> WyckoffPhase:
    """Identify the Wyckoff phase with crypto-specific adjustments."""
    # Crypto markets are more volatile, adjust thresholds
    CRYPTO_VOLUME_THRESHOLD = VOLUME_THRESHOLD * 1.5  # Higher volume requirement
    CRYPTO_EFFORT_THRESHOLD = EFFORT_THRESHOLD * 0.8  # Lower effort requirement due to higher volatility
    
    if is_spring and curr_volume > volume_sma * CRYPTO_VOLUME_THRESHOLD:
        return WyckoffPhase.ACCUMULATION
    if is_upthrust and curr_volume > volume_sma * CRYPTO_VOLUME_THRESHOLD:
        return WyckoffPhase.DISTRIBUTION
    
    # Account for 24/7 trading and potential manipulation
    if effort_vs_result > CRYPTO_EFFORT_THRESHOLD:
        # Check for potential stop hunts in crypto
        if volume_spread > volume_spread_ma * 2.0:  # More aggressive volume spread threshold
            return WyckoffPhase.POSSIBLE_MARKUP
        return WyckoffPhase.MARKUP
        
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
    """Detect actions with crypto-specific patterns."""
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
    wyckoff_sign: WyckoffSign,
    funding_state: FundingState
) -> str:
    """Generate enhanced Wyckoff analysis description including funding rates."""
    base_phase = phase.name.replace("_", " ").lower()
    
    # Start with clear market structure
    description_parts = [
        f"Market is in {base_phase} phase",
        f"{composite_action.value.lower()}"
    ]
    
    # Add technical signs with more context
    if wyckoff_sign != WyckoffSign.NONE:
        description_parts.append(f"showing {wyckoff_sign.value} formation")
    
    # Enhanced spring/upthrust descriptions
    if is_spring:
        description_parts.append(
            "with bullish Spring pattern (test of support with rapid recovery) "
            "suggesting accumulation by institutions"
        )
    elif is_upthrust:
        description_parts.append(
            "with bearish Upthrust pattern (test of resistance with rapid rejection) "
            "suggesting distribution by institutions"
        )
    
    # Volume and effort analysis with more context
    if is_high_volume and effort_vs_result == EffortResult.STRONG:
        description_parts.append(
            "backed by significant institutional volume showing clear directional intent"
        )
    elif not is_high_volume and effort_vs_result == EffortResult.STRONG:
        description_parts.append(
            "showing efficient price movement despite low volume, suggesting smart money activity"
        )
    
    # Enhanced funding rate context
    if funding_state != FundingState.UNKNOWN:
        funding_context = {
            FundingState.HIGHLY_POSITIVE: "extremely high funding rates suggesting overheated longs",
            FundingState.POSITIVE: "positive funding rates showing long bias",
            FundingState.NEUTRAL: "neutral funding rates",
            FundingState.NEGATIVE: "negative funding rates showing short bias",
            FundingState.HIGHLY_NEGATIVE: "extremely negative funding rates suggesting overheated shorts"
        }
        description_parts.append(funding_context.get(funding_state, ""))

    # Join description with better flow
    main_description = ", ".join(description_parts)
    
    # Add trading suggestion
    suggestion = generate_trading_suggestion(
        phase, uncertain_phase, momentum_strength, is_spring, is_upthrust,
        effort_vs_result, composite_action, wyckoff_sign, funding_state
    )
    
    return f"{main_description}.\n\n<b>Trading Perspective:</b>\n{suggestion}"

def generate_trading_suggestion(
    phase: WyckoffPhase,
    uncertain_phase: bool,
    momentum_strength: float,
    is_spring: bool,
    is_upthrust: bool,
    effort: EffortResult,
    composite_action: CompositeAction,
    wyckoff_sign: WyckoffSign,
    funding_state: FundingState
) -> str:
    """Generate detailed crypto-specific trading suggestions."""
    
    # Handle uncertain market conditions with specific reasons
    if uncertain_phase:
        return (
            "Market structure is unclear. Consider reducing position sizes and "
            "waiting for institutional footprints (high volume with clear direction)"
        )
    
    if effort == EffortResult.WEAK:
        return (
            "Price movement lacks institutional backing. Watch for volume confirmation "
            "before taking positions to avoid potential stop hunts"
        )
    
    # Handle manipulation patterns with specific advice
    if wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.BUYING_CLIMAX]:
        return (
            "Possible climax formation detected. Wait for volume to stabilize and "
            "look for institutional absorption patterns before entering positions"
        )
    
    # High-probability Wyckoff setups with detailed guidance
    if wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.LAST_POINT_OF_SUPPORT] and is_spring:
        return (
            "High-probability Accumulation zone detected. Consider scaling into longs "
            "with stops below the Spring low. Look for volume confirmation on bounces"
        )
    if wyckoff_sign in [WyckoffSign.BUYING_CLIMAX, WyckoffSign.LAST_POINT_OF_RESISTANCE] and is_upthrust:
        return (
            "High-probability Distribution zone detected. Consider scaling into shorts "
            "with stops above the Upthrust high. Look for volume confirmation on drops"
        )
    
    # Composite operator actions with momentum confirmation
    if composite_action == CompositeAction.ACCUMULATING and momentum_strength > 0:
        return (
            "Clear institutional Accumulation detected. Consider joining longs after "
            "pullbacks with stops under recent support levels"
        )
    if composite_action == CompositeAction.DISTRIBUTING and momentum_strength < 0:
        return (
            "Clear institutional Distribution detected. Consider joining shorts after "
            "rallies with stops above recent resistance levels"
        )
    
    # Enhanced phase-based suggestions with specific guidance
    basic_suggestions = {
        WyckoffPhase.ACCUMULATION: (
            "Accumulation Phase: Watch for absorption of supply on tests of support. "
            "Prepare for longs after Spring patterns or signs of stopping action"
        ),
        WyckoffPhase.DISTRIBUTION: (
            "Distribution Phase: Watch for supply testing on rallies. "
            "Prepare for shorts after Upthrust patterns or signs of stopping action"
        ),
        WyckoffPhase.MARKUP: (
            "Markup Phase: Trail stops under developing support levels. "
            "Look to add on pullbacks with institutional volume"
        ),
        WyckoffPhase.MARKDOWN: (
            "Markdown Phase: Trail stops above developing resistance levels. "
            "Look to add on rallies with institutional volume"
        ),
        WyckoffPhase.RANGING: (
            "Ranging Phase: Focus on range extremes with volume confirmation. "
            "Watch for institutional activity at support and resistance"
        ),
        WyckoffPhase.UNKNOWN: (
            "Unclear market structure: Wait for price action to develop clear "
            "institutional patterns before taking positions"
        )
    }
    
    # Handle extreme funding rates with specific advice
    if funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.HIGHLY_NEGATIVE]:
        return (
            "Extreme funding rates suggest crowded positioning. "
            "Consider counter-trend trades on strong institutional signals, "
            "or reduce exposure until funding normalizes"
        )
    
    return basic_suggestions.get(phase, (
        "Market structure developing: Wait for clear institutional patterns "
        "before taking positions"
    ))

def analyze_funding_rates(funding_rates: List[FundingRateEntry]) -> FundingState:
    """
    Analyze funding rates with enhanced crypto-specific features:
    - Non-linear time weighting for faster response to changes
    - Outlier detection to ignore manipulation spikes
    - Dynamic thresholds based on volatility
    """
    if not funding_rates or len(funding_rates) < 3:  # Need minimum samples
        return FundingState.UNKNOWN
    
    now = max(rate['time'] for rate in funding_rates)
    
    # Convert to numpy array for efficient calculations
    rates = np.array([rate['fundingRate'] for rate in funding_rates])
    times = np.array([rate['time'] for rate in funding_rates])
    
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
    
    # Dynamic thresholds based on funding rate volatility
    volatility = np.std(rates)
    extreme_threshold = max(FUNDING_EXTREME_THRESHOLD, volatility * 2)
    moderate_threshold = max(FUNDING_MODERATE_THRESHOLD, volatility)
    
    # Determine state with dynamic thresholds
    if avg_funding > extreme_threshold:
        return FundingState.HIGHLY_POSITIVE
    elif avg_funding > moderate_threshold:
        return FundingState.POSITIVE
    elif avg_funding < -extreme_threshold:
        return FundingState.HIGHLY_NEGATIVE
    elif avg_funding < -moderate_threshold:
        return FundingState.NEGATIVE
    else:
        return FundingState.NEUTRAL
