from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import time
from scipy.stats import norm # type: ignore[import]
from scipy.special import expit # type: ignore[import]
from tzlocal import get_localzone
from .wyckoff_types import WyckoffState, WyckoffPhase, CompositeAction, FundingState, Timeframe
from .wyckoff import MIN_PERIODS, STRONG_DEV_THRESHOLD, detect_wyckoff_phase
from ..candles_cache import get_candles_with_cache
from ..funding_rates_cache import get_funding_with_cache
from ..data_processor import prepare_dataframe, apply_indicators, remove_partial_candle
from hyperliquid_utils.utils import hyperliquid_utils

def cluster_points(
    points: np.ndarray,
    volumes: np.ndarray, 
    timestamps: np.ndarray,
    min_price: float,
    max_price: float,
    tolerance: float,
    wyckoff_state: WyckoffState
) -> Dict[float, Dict[str, float]]:
    """
    Enhanced clustering that considers Wyckoff state
    """
    clusters: Dict[float, Dict[str, float]] = {}
    
    # Adjust weights based on Wyckoff phase
    volume_weight = 1.0
    if wyckoff_state:
        if wyckoff_state.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.DISTRIBUTION]:
            volume_weight = 1.25  # Give more weight to volume in accumulation/distribution
        if wyckoff_state.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING]:
            volume_weight = 1.35  # Even more weight when clear institutional action
    
    for idx, price in enumerate(points):
        if not min_price <= price <= max_price:
            continue
            
        # Add a decay factor to the recency weight
        time_decay = 0.95 ** (len(timestamps) - timestamps[idx])
        recency_weight = (1 / (1 + np.exp(-5 * (timestamps[idx] / len(timestamps) - 0.5)))) * time_decay
        nearby_price = next((p for p in clusters if abs(p - price) <= tolerance), None)
        
        if nearby_price:
            weight = volumes[idx] * recency_weight * volume_weight  # Apply Wyckoff-based weight
            clusters[nearby_price]['weight'] += weight
            clusters[nearby_price]['count'] += 1
            clusters[nearby_price]['vol_sum'] += volumes[idx]
            clusters[nearby_price]['price'] = clusters[nearby_price]['price'] * 0.7 + price * 0.3
        else:
            clusters[price] = {
                'weight': volumes[idx] * recency_weight * volume_weight,
                'count': 1,
                'vol_sum': volumes[idx],
                'price': price
            }
    
    return clusters

def score_level(
    cluster: Dict[str, float], 
    max_vol: float, 
    current_price: float,
    wyckoff_state: WyckoffState,
    total_periods: int,
    df: pd.DataFrame
) -> float:
    """
    Enhanced scoring that incorporates Wyckoff analysis and adjusts for available data
    """
    # Check if max_vol is zero to avoid division by zero
    if max_vol == 0:
        return 0.0
    
    # Improve volume score weighting based on relative size
    relative_volume = cluster['vol_sum'] / max_vol
    volume_score = expit(relative_volume * 4) * 0.45  # Increased weight from 0.4 to 0.45
    
    # Adjust touch thresholds based on available periods
    significant_touches = max(3, total_periods // 15)
    strong_touches = max(5, total_periods // 10)
    
    # Enhanced touch score with diminishing returns
    touch_ratio = cluster['count'] / total_periods
    touch_score = min(0.4, (1 - 1/(1 + np.log1p(touch_ratio * 120)))) * 0.35
    
    # Improved proximity scoring with steeper falloff
    price_deviation = abs(cluster['price'] - current_price) / current_price
    proximity_score = max(0, 1 - min(1, (price_deviation / 0.03)**1.5)) * 0.2  # Reduced from 0.25 to 0.2
    
    score = volume_score + touch_score + proximity_score
    
    # Adjust multipliers based on period-adjusted thresholds with diminishing returns
    if cluster['count'] >= significant_touches:
        score *= min(1.25, 1 + (cluster['count'] - significant_touches) * 0.02)
    if cluster['count'] >= strong_touches:
        score *= 1.1
        
    # More nuanced proximity effect
    if price_deviation < 0.005:  # Very close levels (0.5%)
        score *= 1.15
    elif price_deviation < 0.01:  # Close levels (1%)
        score *= 1.10
    elif price_deviation > 0.1:  # Far levels
        score *= 0.85
        
    # Adjust score based on Wyckoff phase
    if wyckoff_state:
        if wyckoff_state.phase == WyckoffPhase.ACCUMULATION:
            if cluster['price'] < current_price:  # Support more important in accumulation
                score *= 1.2
        elif wyckoff_state.phase == WyckoffPhase.DISTRIBUTION: 
            if cluster['price'] > current_price:  # Resistance more important in distribution
                score *= 1.2
                
        # Consider funding rates
        if wyckoff_state.funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.HIGHLY_NEGATIVE]:
            score *= 0.8
            
    # Add Bollinger Band context with importance based on recency
    if df is not None and all(col in df.columns for col in ['BB_upper', 'BB_lower']):
        price = cluster['price']
        
        # Weight recent BB touches more heavily with exponential decay
        bb_touch_score = 0
        for idx in range(-min(20, len(df)), 0):
            day_age = abs(idx)
            decay_factor = np.exp(-0.1 * day_age)  # Exponential decay based on recency
            
            if ((df['h'].iloc[idx] >= df['BB_upper'].iloc[idx] * 0.995 and 
                 abs(price - df['h'].iloc[idx]) / price < 0.005) or
                (df['l'].iloc[idx] <= df['BB_lower'].iloc[idx] * 1.005 and 
                 abs(price - df['l'].iloc[idx]) / price < 0.005)):
                bb_touch_score += decay_factor * 0.5
        
        # Apply BB touch boost
        if bb_touch_score > 0:
            score *= min(1.2, 1 + bb_touch_score)
            
        # More tolerant of levels outside BBs for crypto with adjustable thresholds
        bb_upper = df['BB_upper'].iloc[-1]
        bb_lower = df['BB_lower'].iloc[-1]
        bb_width_pct = (bb_upper - bb_lower) / current_price
        
        # Adjust thresholds based on current volatility
        volatility_factor = min(1.5, max(1.0, 1.2 * bb_width_pct / 0.05))
        if price > bb_upper * volatility_factor or price < bb_lower / volatility_factor:
            score *= 0.85
            
    return min(score, 1.0)

async def get_significant_levels_from_timeframe(coin: str, mid: float, timeframe: Timeframe, lookback_days: int) -> Tuple[List[float], List[float]]:
    now = int(time.time() * 1000)
    candles = await get_candles_with_cache(coin, timeframe, now, lookback_days, hyperliquid_utils.info.candles_snapshot)
    local_tz = get_localzone()
    df = prepare_dataframe(candles, local_tz)
    apply_indicators(df, timeframe)
    funding_rates = get_funding_with_cache(coin, now, 7)
    return find_significant_levels(df, detect_wyckoff_phase(remove_partial_candle(df, local_tz), timeframe, funding_rates), mid, timeframe)


def find_significant_levels(
    df: pd.DataFrame,
    wyckoff_state: WyckoffState,
    current_price: float,
    timeframe: Timeframe,
    n_levels: int = 4,
    min_score: float = 0.2
) -> Tuple[List[float], List[float]]:
    """
    Find significant price levels using timeframe-specific lookback and Bollinger Bands.
    """
    # Validation checks
    if df is None or df.empty or current_price <= 0:
        return [], []
        
    # Ensure required columns exist
    required_columns = ['h', 'l', 'c', 'v', 'ATR', 'BB_width', 'BB_upper', 'BB_lower']
    if not all(col in df.columns for col in required_columns):
        return [], []
        
    if len(df) < timeframe.settings.support_resistance_lookback:
        return [], []
    
    # Use BB width for dynamic volatility adjustment with improved sensitivity
    volatility_multiplier = 1.0
    
    # Add safety checks for BB calculations
    try:
        recent_bb_width = df['BB_width'].iloc[-1]
        bb_width_sma = df['BB_width'].rolling(window=min(5, len(df))).mean().iloc[-1]
        
        # Enhanced volatility detection with rate of change
        bb_width_change = recent_bb_width / bb_width_sma if bb_width_sma > 0 else 1.0
        if recent_bb_width > df['BB_width'].mean():
            # Scale multiplier based on BB width change rate
            volatility_multiplier = min(1.5, 1.2 + (bb_width_change - 1) * 0.3)
    except (IndexError, ZeroDivisionError):
        # Fallback to default
        volatility_multiplier = 1.0
    
    # Use specific lookback for price calculations based on timeframe
    lookback = min(timeframe.settings.support_resistance_lookback, len(df) - 1)
    price_sma = df['c'].rolling(window=lookback).mean()
    price_std = df['c'].rolling(window=lookback).std()
    
    # Avoid zero division with a minimum floor value
    last_price_sma = max(price_sma.iloc[-1], current_price * 0.001)
    volatility = (price_std.iloc[-1] / last_price_sma) * volatility_multiplier
    
    # Dynamic price range based on timeframe and current volatility
    # Use the significant_levels_factor from timeframe settings
    timeframe_factor = timeframe.settings.significant_levels_factor
    
    max_deviation = min(STRONG_DEV_THRESHOLD * volatility * timeframe_factor, 0.25)
    min_price = current_price * (1 - max_deviation)
    max_price = current_price * (1 + max_deviation)
    
    # Avoid unnecessary copy if we're not modifying the dataframe
    recent_df = df
    
    # More sensitive base tolerance for crypto's volatile price action
    # Use atr_multiplier from timeframe settings
    atr_multiplier = timeframe.settings.atr_multiplier
    
    volatility_component = volatility * 0.2  # Reduced from 0.25 for finer control
    # Safeguard against invalid ATR values
    default_atr = current_price * 0.002  # Default 0.2% ATR if missing
    base_tolerance = (recent_df['ATR'].iloc[-1] if recent_df['ATR'].iloc[-1] > 0 else default_atr) * (atr_multiplier + volatility_component)
    
    length_factor = np.log1p(len(recent_df) / timeframe.settings.support_resistance_lookback) / 2
    tolerance = base_tolerance * (1 + length_factor)
    
    data = dict(
        highs=recent_df['h'].values,
        lows=recent_df['l'].values,
        volumes=recent_df['v'].values,
        timestamps=np.arange(len(recent_df))
    )

    clusters = {
        'resistance': cluster_points(np.asarray(data['highs']), np.asarray(data['volumes']), np.asarray(data['timestamps']),
                                   min_price, max_price, tolerance, wyckoff_state),
        'support': cluster_points(np.asarray(data['lows']), np.asarray(data['volumes']), np.asarray(data['timestamps']),
                                min_price, max_price, tolerance, wyckoff_state)
    }
    
    max_vol = np.sum(np.asarray(data['volumes']))
    total_periods = len(recent_df)

    def filter_levels(clusters: Dict[float, Dict[str, float]], is_resistance: bool) -> List[float]:
        """Filters and scores clustered price levels to identify significant resistance or support."""
        if not clusters:
            return []

        scored_levels: List[Tuple[float, float]] = [
            (cluster['price'], score_level(cluster, max_vol, current_price, wyckoff_state, total_periods, df))
            for cluster in clusters.values()
        ]

        # Sort by score, then by price (high to low for resistance, low to high for support)
        sorted_levels: List[Tuple[float, float]] = sorted(
            scored_levels,
            key=lambda x: (x[1], x[0] if is_resistance else -x[0]),
            reverse=True
        )

        # Filter by position relative to current price and minimum score
        valid_levels: List[float] = [
            price for price, score in sorted_levels
            if (is_resistance and price > current_price or
                not is_resistance and price < current_price) and
            score > min_score
        ]

        # Check if valid_levels is empty
        if not valid_levels:
            return []

        return valid_levels[:n_levels]
    
    return (
        filter_levels(clusters['resistance'], True),
        filter_levels(clusters['support'], False)
    )
