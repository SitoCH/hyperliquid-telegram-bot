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
        funding_advice = {
            FundingState.HIGHLY_POSITIVE: (
                "Extreme positive funding indicates severely overleveraged longs. "
                "High risk of long liquidations. Consider waiting for funding reset or "
                "look for short opportunities with strong institutional signals"
            ),
            FundingState.POSITIVE: (
                "Positive funding shows aggressive long positioning. "
                "Consider reduced leverage for longs and tighter stops. "
                "Watch for potential long squeezes on strong distribution signals"
            ),
            FundingState.SLIGHTLY_POSITIVE: (
                "Slightly positive funding suggests mild long bias. "
                "Normal position sizing acceptable but monitor funding trend"
            ),
            FundingState.NEUTRAL: (
                "Neutral funding indicates balanced positioning. "
                "Focus on technical signals for trade direction"
            ),
            FundingState.SLIGHTLY_NEGATIVE: (
                "Slightly negative funding suggests mild short bias. "
                "Normal position sizing acceptable but monitor funding trend"
            ),
            FundingState.NEGATIVE: (
                "Negative funding shows aggressive short positioning. "
                "Consider reduced leverage for shorts and tighter stops. "
                "Watch for potential short squeezes on strong accumulation signals"
            ),
            FundingState.HIGHLY_NEGATIVE: (
                "Extreme negative funding indicates severely overleveraged shorts. "
                "High risk of short liquidations. Consider waiting for funding reset or "
                "look for long opportunities with strong institutional signals"
            ),
            FundingState.UNKNOWN: ""
        }
        description_parts.append(funding_advice.get(funding_state, ""))

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
    """Generate detailed crypto-specific trading suggestions with risk management."""
    
    # First, assess overall market risk level
    risk_level = "high" if uncertain_phase or effort == EffortResult.WEAK else "normal"
    if funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.HIGHLY_NEGATIVE]:
        risk_level = "extreme"
        
    # Determine position sizing based on risk
    position_guidance = {
        "extreme": "Consider very small positions (0.25x normal size) or staying flat",
        "high": "Reduce position sizes to 0.5x normal size",
        "normal": "Standard position sizing acceptable"
    }[risk_level]

    # Handle high-risk scenarios first
    if risk_level == "extreme":
        return (
            f"High-risk market conditions detected:\n"
            f"• Extreme funding rates suggesting potential liquidation cascade\n"
            f"• {position_guidance}\n"
            f"• Wait for funding reset or very strong institutional signals\n"
            f"• Use wider stops to account for increased volatility"
        )

    if uncertain_phase:
        return (
            f"Unclear market structure - Exercise caution:\n"
            f"• {position_guidance}\n"
            f"• Wait for clear institutional footprints\n"
            f"• Look for high volume with clear price direction\n"
            f"• Keep larger portion of portfolio in stable coins"
        )

    # Handle strong institutional patterns
    if wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.LAST_POINT_OF_SUPPORT] and is_spring:
        return (
            f"High-probability Accumulation setup detected:\n"
            f"• {position_guidance}\n"
            f"• Scale into longs gradually (3-4 entries)\n"
            f"• Initial stop under Spring low ({'-1.5' if risk_level == 'normal' else '-2.0'}x ATR)\n"
            f"• Look for volume confirmation on bounces\n"
            f"• Trail stops as price develops new support levels"
        )

    if wyckoff_sign in [WyckoffSign.BUYING_CLIMAX, WyckoffSign.LAST_POINT_OF_RESISTANCE] and is_upthrust:
        return (
            f"High-probability Distribution setup detected:\n"
            f"• {position_guidance}\n"
            f"• Scale into shorts gradually (3-4 entries)\n"
            f"• Initial stop above Upthrust high ({'+1.5' if risk_level == 'normal' else '+2.0'}x ATR)\n"
            f"• Look for volume confirmation on drops\n"
            f"• Trail stops as price develops new resistance levels"
        )

    # Handle strong composite operator actions
    if composite_action == CompositeAction.ACCUMULATING:
        direction = "bullish" if momentum_strength > 0 else "neutral to bullish"
        return (
            f"Institutional Accumulation detected - {direction} bias:\n"
            f"• {position_guidance}\n"
            f"• Enter longs on successful support tests\n"
            f"• Look for decreasing volume on pullbacks\n"
            f"• Watch for spring patterns near support\n"
            f"• Use shallow retracements for entries"
        )

    if composite_action == CompositeAction.DISTRIBUTING:
        direction = "bearish" if momentum_strength < 0 else "neutral to bearish"
        return (
            f"Institutional Distribution detected - {direction} bias:\n"
            f"• {position_guidance}\n"
            f"• Enter shorts on failed resistance tests\n"
            f"• Look for decreasing volume on rallies\n"
            f"• Watch for upthrust patterns near resistance\n"
            f"• Use shallow bounces for entries"
        )

    # Phase-specific suggestions with funding rate context
    phase_suggestions = {
        WyckoffPhase.ACCUMULATION: (
            f"Accumulation Phase - Building long positions:\n"
            f"• {position_guidance}\n"
            f"• Watch for absorption patterns at support\n"
            f"• Look for declining volume on dips\n"
            f"• Wait for spring patterns or stopping action\n"
            f"• Consider partial profits if funding turns highly positive"
        ),
        WyckoffPhase.DISTRIBUTION: (
            f"Distribution Phase - Building short positions:\n"
            f"• {position_guidance}\n"
            f"• Watch for supply testing on rallies\n"
            f"• Look for declining volume on bounces\n"
            f"• Wait for upthrust patterns or stopping action\n"
            f"• Consider partial profits if funding turns highly negative"
        ),
        WyckoffPhase.MARKUP: (
            f"Markup Phase - Long bias with protection:\n"
            f"• {position_guidance}\n"
            f"• Trail stops under developing support\n"
            f"• Add on high-volume pullbacks\n"
            f"• Watch for distribution signs at resistance\n"
            f"• Tighten stops if funding becomes extremely positive"
        ),
        WyckoffPhase.MARKDOWN: (
            f"Markdown Phase - Short bias with protection:\n"
            f"• {position_guidance}\n"
            f"• Trail stops above developing resistance\n"
            f"• Add on high-volume bounces\n"
            f"• Watch for accumulation signs at support\n"
            f"• Tighten stops if funding becomes extremely negative"
        ),
        WyckoffPhase.RANGING: (
            f"Ranging Phase - Two-way trading opportunities:\n"
            f"• {position_guidance}\n"
            f"• Trade range extremes with volume confirmation\n"
            f"• Use smaller position sizes for range trades\n"
            f"• Watch for range expansion signals\n"
            f"• Consider range breakout setups with funding confirmation"
        )
    }

    return phase_suggestions.get(phase, (
        f"Market structure developing:\n"
        f"• {position_guidance}\n"
        f"• Wait for clear institutional patterns\n"
        f"• Monitor volume and price action at key levels\n"
        f"• Keep larger cash position until clarity emerges"
    ))

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
    
    # Convert to numpy array using EAR calculation
    rates = np.array([(1 + rate.funding_rate) ** 8760 - 1 for rate in funding_rates])
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
