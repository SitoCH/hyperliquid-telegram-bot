import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final, Dict, List, Optional, Any
from .wyckoff_types import MarketPattern, VolatilityState, WyckoffState, WyckoffPhase, EffortResult, CompositeAction, WyckoffSign, FundingState, VolumeState, ThresholdConfig, Timeframe
from .funding_rates_cache import FundingRateEntry
from statistics import mean
from .wyckoff_description import generate_wyckoff_description
from utils import log_execution_time

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


def detect_wyckoff_phase(df: pd.DataFrame, timeframe: Timeframe, funding_rates: List[FundingRateEntry]) -> None:
    """Analyze and store Wyckoff phase data incorporating funding rates."""
    # Safety check for minimum required periods
    if len(df) < MIN_PERIODS:
        df.loc[df.index[-1:], 'wyckoff'] = WyckoffState.unknown()
        return
    
    thresholds = ThresholdConfig.for_timeframe(timeframe)

    # Process last period
    short_term_window = min(MIN_PERIODS, len(df) - 1)
    recent_df = df.iloc[-short_term_window:]
    
    # Calculate technical indicators
    volume_sma = df['v'].rolling(window=MIN_PERIODS).mean()
    price_sma = df['c'].rolling(window=MIN_PERIODS).mean()
    price_std = df['c'].rolling(window=MIN_PERIODS).std()

    # Enhanced volume analysis for crypto
    volume_short_ma = df['v'].rolling(window=3).mean()
    volume_long_ma = df['v'].rolling(window=8).mean()
    volume_trend = ((volume_short_ma - volume_long_ma) / volume_long_ma).fillna(0)
    
    # Volume analysis (VSA)
    volume_spread = recent_df['v'] * (recent_df['h'] - recent_df['l'])
    volume_spread_ma = volume_spread.rolling(window=7).mean()
    effort_vs_result = (recent_df['c'] - recent_df['o']) / (recent_df['h'] - recent_df['l'])
    
    # Get current values
    curr_price = df['c'].iloc[-1]
    curr_volume = df['v'].iloc[-1]
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
    is_spring, is_upthrust = detect_spring_upthrust(df, -1)
    
    # Market condition checks
    volume_sma_value = float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else 0.0
    volume_trend_value = float(volume_trend.iloc[-1]) if not pd.isna(volume_trend.iloc[-1]) else 0.0
    
    # Calculate relative volume
    relative_volume = curr_volume / volume_sma_value if volume_sma_value > 0 else 1.0
    
    # Calculate volume consistency
    recent_strong_volume = (df['v'].iloc[-3:] > volume_sma.iloc[-3:]).mean()
    
    # Use thresholds from config
    is_high_volume = (
        (relative_volume > thresholds.volume_threshold and volume_trend_value > 0) or
        (relative_volume > thresholds.volume_surge_threshold) or
        (recent_strong_volume > 0.6 and relative_volume > 1.2)
    )

    price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)

    # Enhanced price strength calculation
    price_ma_short = df['c'].rolling(window=8).mean()
    price_ma_long = df['c'].rolling(window=21).mean()
    trend_strength = ((price_ma_short - price_ma_long) / price_ma_long).iloc[-1]
    price_strength = price_strength * 0.7 + trend_strength * 0.3

    # Phase identification for current period
    current_phase = identify_wyckoff_phase(
        is_spring, is_upthrust, curr_volume, volume_sma.iloc[-1],
        effort_vs_result.iloc[-1], volume_spread.iloc[-1], volume_spread_ma.iloc[-1],
        price_strength, momentum_strength, is_high_volume, volatility,
        thresholds=thresholds
    )
    
    effort_result = EffortResult.STRONG if abs(effort_vs_result.iloc[-1]) > thresholds.effort_threshold else EffortResult.WEAK
    
    # Detect composite action and Wyckoff signs
    composite_action = detect_composite_action(df, price_strength, volume_trend_value, effort_vs_result.iloc[-1])
    wyckoff_sign = detect_wyckoff_signs(df, price_strength, volume_trend_value, is_spring, is_upthrust)
    
    funding_state = analyze_funding_rates(funding_rates or [])
    
    # Create WyckoffState instance
    wyckoff_state = WyckoffState(
        phase=current_phase,
        uncertain_phase=current_phase.value.startswith('~'),
        volume=VolumeState.HIGH if is_high_volume else VolumeState.LOW,
        pattern=MarketPattern.TRENDING if abs(momentum_strength) > thresholds.momentum_threshold else MarketPattern.RANGING,
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

    df.loc[df.index[-1], 'wyckoff'] = wyckoff_state # type: ignore


def identify_wyckoff_phase(
    is_spring: bool, is_upthrust: bool, curr_volume: float, volume_sma: float,
    effort_vs_result: float, volume_spread: float, volume_spread_ma: float,
    price_strength: float, momentum_strength: float, is_high_volume: bool,
    volatility: pd.Series, thresholds: ThresholdConfig
) -> WyckoffPhase:
    """Enhanced Wyckoff phase identification with timeframe-specific adjustments."""
    # Adaptive thresholds based on market volatility
    volatility_factor = min(2.0, 1.0 + volatility.iloc[-1] / volatility.mean())
    
    # Use thresholds directly from config, only adjust for volatility
    volume_threshold = thresholds.volume_threshold * volatility_factor
    effort_threshold = thresholds.effort_threshold * (1.0 / volatility_factor)
    
    # Detect potential manipulation
    potential_manipulation = (
        volume_spread > volume_spread_ma * 3.0 and  # Extreme volume spread
        abs(effort_vs_result) < effort_threshold * 0.5  # Low price result
    )
    
    if potential_manipulation:
        return (WyckoffPhase.POSSIBLE_DISTRIBUTION if price_strength > 0 
                else WyckoffPhase.POSSIBLE_ACCUMULATION)
    
    # Spring/Upthrust detection with volume confirmation
    if is_spring and curr_volume > volume_sma * volume_threshold:
        # Check for fake springs (common in crypto)
        if effort_vs_result < -effort_threshold:
            return WyckoffPhase.POSSIBLE_MARKDOWN
        return WyckoffPhase.ACCUMULATION
        
    if is_upthrust and curr_volume > volume_sma * volume_threshold:
        # Check for fake upthrusts
        if effort_vs_result > effort_threshold:
            return WyckoffPhase.POSSIBLE_MARKUP
        return WyckoffPhase.DISTRIBUTION
    
    # Add divergence detection
    volume_price_divergence = (
        abs(volume_spread - volume_spread_ma) / volume_spread_ma > 1.5 and
        abs(effort_vs_result) < effort_threshold * 0.5
    )
    
    if volume_price_divergence:
        # Potential reversal signal
        if price_strength > 0:
            return WyckoffPhase.POSSIBLE_DISTRIBUTION
        return WyckoffPhase.POSSIBLE_ACCUMULATION
    
    return determine_phase_by_price_strength(
        price_strength, momentum_strength, is_high_volume, volatility,
        thresholds=thresholds
    )

def determine_phase_by_price_strength(
    price_strength: float, momentum_strength: float, 
    is_high_volume: bool, volatility: pd.Series,
    thresholds: ThresholdConfig
) -> WyckoffPhase:
    """Determine the Wyckoff phase based on price strength and other indicators."""
    if price_strength > thresholds.strong_dev_threshold:
        if momentum_strength < -thresholds.momentum_threshold and is_high_volume:
            return WyckoffPhase.DISTRIBUTION
        return WyckoffPhase.POSSIBLE_DISTRIBUTION
    
    if price_strength < -thresholds.strong_dev_threshold:
        if momentum_strength > thresholds.momentum_threshold and is_high_volume:
            return WyckoffPhase.ACCUMULATION
        return WyckoffPhase.POSSIBLE_ACCUMULATION
    
    if abs(price_strength) <= thresholds.neutral_zone_threshold:
        if abs(momentum_strength) < thresholds.momentum_threshold and volatility.iloc[-1] < volatility.mean():
            return WyckoffPhase.RANGING
        return WyckoffPhase.POSSIBLE_RANGING
    
    # Transitional zones
    if price_strength > 0:
        if momentum_strength > thresholds.momentum_threshold:
            return WyckoffPhase.MARKUP
        return WyckoffPhase.POSSIBLE_MARKUP
    
    if momentum_strength < -thresholds.momentum_threshold:
        return WyckoffPhase.MARKDOWN
    return WyckoffPhase.POSSIBLE_MARKDOWN


def detect_composite_action(
    df: pd.DataFrame,
    price_strength: float,
    volume_trend: float,
    effort_vs_result: float
) -> CompositeAction:
    """Enhanced composite action detection for crypto markets."""
    # Add liquidation cascade detection
    price_moves = df['c'].pct_change()
    volume_surges = df['v'].pct_change()
    
    liquidation_cascade = (
        abs(price_moves.iloc[-1]) > 0.05 and  # Sharp price move
        volume_surges.iloc[-1] > 3.0 and  # Extreme volume
        abs(df['c'].iloc[-1] - df['o'].iloc[-1]) / (df['h'].iloc[-1] - df['l'].iloc[-1]) > 0.8  # Strong close
    )
    
    if liquidation_cascade:
        return (CompositeAction.MARKING_UP if price_moves.iloc[-1] > 0 
                else CompositeAction.MARKING_DOWN)
    
    # Add whale wallet analysis patterns
    absorption_volume = (
        df['v'].iloc[-1] > df['v'].iloc[-5:].mean() * 2 and
        abs(df['c'].iloc[-1] - df['o'].iloc[-1]) < (df['h'].iloc[-1] - df['l'].iloc[-1]) * 0.3
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
    
    if absorption and volume_trend > VOLUME_THRESHOLD:
        if price_strength < 0:
            return CompositeAction.ACCUMULATING
        return CompositeAction.DISTRIBUTING
        
    if effort_vs_result > EFFORT_THRESHOLD and volume_trend > 0:
        return CompositeAction.MARKING_UP
    if effort_vs_result < -EFFORT_THRESHOLD and volume_trend > 0:
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

def detect_wyckoff_signs(
    df: pd.DataFrame,
    price_strength: float,
    volume_trend: float,
    is_spring: bool,
    is_upthrust: bool
) -> WyckoffSign:
    """
    Detect specific Wyckoff signs in market action with comprehensive checks
    for all possible signs in the WyckoffSign enum.
    """
    if len(df) < 5:
        return WyckoffSign.NONE
        
    price_change = df['c'].pct_change()
    volume_change = df['v'].pct_change()
    
    # Selling Climax (SC)
    if (price_change.iloc[-1] < -0.03 and 
        volume_change.iloc[-1] > 2.0 and 
        price_strength < -STRONG_DEV_THRESHOLD):
        return WyckoffSign.SELLING_CLIMAX
        
    # Automatic Rally (AR)
    if (price_change.iloc[-1] > 0.02 and
        df['c'].iloc[-2:].pct_change().mean() > 0.015 and
        volume_change.iloc[-1] < volume_change.iloc[-2] and
        price_strength < 0):
        return WyckoffSign.AUTOMATIC_RALLY
        
    # Secondary Test (ST)
    if (abs(price_change.iloc[-1]) < 0.01 and
        df['l'].iloc[-1] >= df['l'].iloc[-5:].min() and
        volume_change.iloc[-1] < 1.0 and
        price_strength < 0):
        return WyckoffSign.SECONDARY_TEST
        
    # Last Point of Support (LPS)
    if (is_spring and
        volume_trend > 0 and
        price_change.iloc[-1] > 0):
        return WyckoffSign.LAST_POINT_OF_SUPPORT
        
    # Sign of Strength (SOS)
    if (price_change.iloc[-1] > 0.02 and
        volume_change.iloc[-1] > 1.5 and
        price_strength > 0 and
        volume_trend > 0):
        return WyckoffSign.SIGN_OF_STRENGTH
        
    # Buying Climax (BC)
    if (price_change.iloc[-1] > 0.03 and 
        volume_change.iloc[-1] > 2.0 and 
        price_strength > STRONG_DEV_THRESHOLD):
        return WyckoffSign.BUYING_CLIMAX
        
    # Upthrust (UT)
    if (is_upthrust and
        volume_change.iloc[-1] > 1.2 and
        price_change.iloc[-1] < 0):
        return WyckoffSign.UPTHRUST
        
    # Secondary Test Resistance (STR)
    if (abs(price_change.iloc[-1]) < 0.01 and
        df['h'].iloc[-1] <= df['h'].iloc[-5:].max() and
        volume_change.iloc[-1] < 1.0 and
        price_strength > 0):
        return WyckoffSign.SECONDARY_TEST_RESISTANCE
        
    # Last Point of Supply/Resistance (LPSY)
    if (is_upthrust and
        volume_trend > 0 and
        price_change.iloc[-1] < 0):
        return WyckoffSign.LAST_POINT_OF_RESISTANCE
        
    # Sign of Weakness (SOW)
    if (price_change.iloc[-1] < -0.02 and
        volume_change.iloc[-1] > 1.5 and
        price_strength < 0 and
        volume_trend > 0):
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
