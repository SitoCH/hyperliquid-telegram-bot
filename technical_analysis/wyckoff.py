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


def detect_wyckoff_phase(df: pd.DataFrame, funding_rates: Optional[List[FundingRateEntry]] = None) -> None:
    """Analyze and store Wyckoff phase data incorporating funding rates."""
    # Safety check for minimum required periods
    if len(df) < MIN_PERIODS:
        df.loc[df.index[-2:], 'wyckoff'] = [WyckoffState.unknown(), WyckoffState.unknown()]
        return

    # Determine timeframe and get appropriate thresholds
    avg_time_delta = pd.Timedelta(df.index.to_series().diff().mean())
    
    if avg_time_delta >= pd.Timedelta(hours=24):
        timeframe = Timeframe.DAY_1
    elif avg_time_delta >= pd.Timedelta(hours=4):
        timeframe = Timeframe.HOURS_4
    elif avg_time_delta >= pd.Timedelta(hours=1):
        timeframe = Timeframe.HOUR_1
    else:
        timeframe = Timeframe.MINUTES_15
    
    thresholds = ThresholdConfig.for_timeframe(timeframe)
    is_higher_timeframe = timeframe in [Timeframe.HOURS_4, Timeframe.DAY_1]

    # Process last three periods with phase smoothing
    phases_history = []
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
        
        # Add price volatility bands
        volatility_bands = {
            'upper': price_sma + price_std * 2,
            'lower': price_sma - price_std * 2
        }
        
        # Enhanced momentum with Exponential ROC - Improved calculation
        def exp_roc(series: pd.Series, periods: int) -> pd.Series:
            return (series / series.shift(periods)).pow(1/periods) - 1
            
        # Calculate momentum components
        fast_momentum = exp_roc(data_subset['c'], 7).iloc[-1]  # Get last value
        medium_momentum = exp_roc(data_subset['c'], 14).iloc[-1]  # Get last value
        slow_momentum = exp_roc(data_subset['c'], 21).iloc[-1]  # Get last value
        
        # Calculate average momentum
        momentum_value = float(sum([fast_momentum, medium_momentum, slow_momentum]) / 3)

        # Calculate momentum standard deviation for scaling
        momentum_std = float(np.std([fast_momentum, medium_momentum, slow_momentum]))

        # Scale momentum to a -100 to +100 range based on historical volatility
        normalized_momentum = momentum_value / (momentum_std * 2) if momentum_std != 0 else 0
        momentum_strength = max(min(normalized_momentum * 100, 100), -100)

        # Detect springs and upthrusts for the current period
        is_spring, is_upthrust = detect_spring_upthrust(data_subset, -1)
        
        # Market condition checks
        volume_sma_value = float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else 0.0
        volume_trend_value = float(volume_trend.iloc[-1]) if not pd.isna(volume_trend.iloc[-1]) else 0.0
        
        # Calculate relative volume (current volume compared to recent average)
        relative_volume = curr_volume / volume_sma_value if volume_sma_value > 0 else 1.0
        
        # Calculate volume consistency (how many recent periods had above-average volume)
        recent_strong_volume = (data_subset['v'].iloc[-3:] > volume_sma.iloc[-3:]).mean()
        
        # Volume spike detection for crypto
        volume_std = data_subset['v'].rolling(window=12).std()
        volume_spike = (curr_volume - volume_sma_value) / (volume_std.iloc[-1] + 1e-8)
        
        # Detect stop hunts (common in crypto)
        stop_hunt_up = (
            data_subset['h'].iloc[-1] > volatility_bands['upper'].iloc[-1] and
            data_subset['c'].iloc[-1] < price_sma.iloc[-1] and
            volume_spike > 2.0
        )
        
        stop_hunt_down = (
            data_subset['l'].iloc[-1] < volatility_bands['lower'].iloc[-1] and
            data_subset['c'].iloc[-1] > price_sma.iloc[-1] and
            volume_spike > 2.0
        )
        
        # Use thresholds from config
        is_high_volume = (
            (relative_volume > thresholds.volume_threshold and volume_trend_value > 0) or
            (volume_spike > 2.5) or
            (relative_volume > thresholds.volume_surge_threshold) or
            (recent_strong_volume > 0.6 and relative_volume > 1.2)
        )

        price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)

        # Enhanced price strength calculation
        price_ma_short = data_subset['c'].rolling(window=8).mean()
        price_ma_long = data_subset['c'].rolling(window=21).mean()
        trend_strength = ((price_ma_short - price_ma_long) / price_ma_long).iloc[-1]
        price_strength = price_strength * 0.7 + trend_strength * 0.3  # Combine both metrics

        # Enhanced volume clustering analysis
        volume_clusters = data_subset['v'].rolling(5).apply(
            lambda x: np.sum(np.diff(np.sort(x)) > np.std(x))
        )
        clustered_volume = volume_clusters.iloc[-1] >= 3  # At least 3 significant gaps
        
        # Detect potential manipulation
        manipulation_signs = (
            stop_hunt_up or stop_hunt_down or
            (volume_spike > 3.0 and abs(effort_vs_result.iloc[-1]) < 0.3) or
            (clustered_volume and volume_spike > 2.5)
        )
        
        # Phase identification with temporal smoothing
        current_phase = identify_wyckoff_phase(
            is_spring, is_upthrust, curr_volume, volume_sma.iloc[-1],
            effort_vs_result.iloc[-1], volume_spread.iloc[-1], volume_spread_ma.iloc[-1],
            price_strength, momentum_strength, is_high_volume, volatility,
            thresholds=thresholds
        )
        
        phases_history.append(current_phase)
        
        # Apply phase smoothing
        smoothed_phase = smooth_wyckoff_phase(phases_history, is_higher_timeframe)
        
        # Combine manipulation signs with phase uncertainty
        uncertain_phase = smoothed_phase.value.startswith('~') or manipulation_signs
        if manipulation_signs:
            smoothed_phase = WyckoffPhase.POSSIBLE_RANGING
        
        effort_result = EffortResult.STRONG if abs(effort_vs_result.iloc[-1]) > thresholds.effort_threshold else EffortResult.WEAK
        
        # Detect composite action and Wyckoff signs
        composite_action = detect_composite_action(data_subset, price_strength, volume_trend_value, effort_vs_result.iloc[-1])
        wyckoff_sign = detect_wyckoff_signs(data_subset, price_strength, volume_trend_value, is_spring, is_upthrust)
        
        funding_state = analyze_funding_rates(funding_rates or [])
        
        # Adjust phase confidence based on funding rates
        if not uncertain_phase and funding_state != FundingState.UNKNOWN:
            if smoothed_phase in [WyckoffPhase.MARKUP, WyckoffPhase.ACCUMULATION]:
                if funding_state in [FundingState.HIGHLY_NEGATIVE, FundingState.NEGATIVE]:
                    uncertain_phase = True
            elif smoothed_phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.DISTRIBUTION]:
                if funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.POSITIVE]:
                    uncertain_phase = True

        # Create WyckoffState instance with smoothed phase
        wyckoff_state = WyckoffState(
            phase=smoothed_phase,
            uncertain_phase=uncertain_phase,
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
                smoothed_phase, uncertain_phase, is_high_volume, momentum_strength, 
                is_spring, is_upthrust, effort_result,
                composite_action, wyckoff_sign, funding_state
            )
        )

        df.loc[df.index[i], 'wyckoff'] = wyckoff_state  # type: ignore


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

def smooth_wyckoff_phase(phases: List[WyckoffPhase], is_higher_timeframe: bool) -> WyckoffPhase:
    """Apply temporal smoothing to Wyckoff phases to reduce noise."""
    if len(phases) < 2:  # Changed from 3 to 2 since we now use 3 periods
        return phases[-1]
        
    # Count phase occurrences
    phase_counts: Dict[WyckoffPhase, float] = {}
    # Weight recent phases more heavily
    weights = [0.2, 0.3, 0.5][-len(phases):]  # Changed weights to sum to 1 with 3 periods
    
    for phase, weight in zip(phases, weights):
        phase_counts[phase] = phase_counts.get(phase, 0) + weight
    
    # Find the most common phase
    most_common_phase = max(phase_counts.items(), key=lambda x: x[1])[0]
    current_phase = phases[-1]
    
    # Higher timeframes require more consensus for phase changes
    if is_higher_timeframe:
        # Require strong consensus (>60%) to change phase
        if phase_counts[most_common_phase] > 0.6:
            return most_common_phase
        # Keep previous phase if no strong consensus
        if len(phases) > 1:
            return phases[-2]
    else:
        # Lower timeframes can change more readily but still need some consensus
        if phase_counts[most_common_phase] > 0.4:
            return most_common_phase
        
    # Default to current phase if no consensus
    return current_phase

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
    """Detect specific Wyckoff signs in market action."""
    if len(df) < 5:
        return WyckoffSign.NONE
        
    price_change = df['c'].pct_change()
    volume_change = df['v'].pct_change()
    
    # Selling Climax detection
    if (price_change.iloc[-1] < -0.03 and 
        volume_change.iloc[-1] > 2.0 and 
        price_strength < STRONG_DEV_THRESHOLD):
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
