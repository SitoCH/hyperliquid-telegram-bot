from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from scipy.stats import norm # type: ignore[import]
from scipy.special import expit # type: ignore[import]
from .wyckoff_types import WyckoffState, WyckoffPhase, CompositeAction, FundingState
from .wyckoff import MIN_PERIODS, STRONG_DEV_THRESHOLD

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
            
        recency_weight = 1 / (1 + np.exp(-5 * (timestamps[idx] / len(timestamps) - 0.5)))
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
    total_periods: int
) -> float:
    """
    Enhanced scoring that incorporates Wyckoff analysis and adjusts for available data
    """
    volume_score = expit(cluster['vol_sum'] / max_vol * 3) * 0.4
    
    # Adjust touch thresholds based on available periods
    significant_touches = max(3, total_periods // 15)  # More touches for longer periods
    strong_touches = max(5, total_periods // 10)  # Even more for strong confirmation
    
    # Scale touch score relative to available data
    touch_ratio = cluster['count'] / total_periods
    touch_score = (1 - 1/(1 + np.log1p(touch_ratio * 100))) * 0.35
    
    price_deviation = abs(cluster['price'] - current_price) / current_price
    proximity_score = norm.pdf(price_deviation, scale=0.05) * 0.25  # Decreased from 0.3 to 0.25
    
    score = volume_score + touch_score + proximity_score
    
    # Adjust multipliers based on period-adjusted thresholds
    if cluster['count'] >= significant_touches and cluster['vol_sum'] > max_vol * 0.15:
        score *= 1.25
    if cluster['count'] >= strong_touches:
        score *= 1.1
    if price_deviation < 0.01:
        score *= 1.10
    elif price_deviation > 0.1:
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
            score *= 0.8  # Reduce level confidence when funding is extreme
            
    return min(score, 1.0)

def find_significant_levels(
    df: pd.DataFrame,
    current_price: float,
    n_levels: int = 4, 
    min_score: float = 0.2
) -> Tuple[List[float], List[float]]:
    """Get significant price levels aligned with Wyckoff analysis"""
    if len(df) < MIN_PERIODS:
        return [], []
    
    price_sma = df['Close'].rolling(window=MIN_PERIODS).mean()
    price_std = df['Close'].rolling(window=MIN_PERIODS).std()
    volatility = (price_std / price_sma).iloc[-1]
    
    # Use full dataset instead of limited lookback
    recent_df = df.copy()
    
    max_deviation = min(STRONG_DEV_THRESHOLD * volatility, 0.25)
    min_price = current_price * (1 - max_deviation)
    max_price = current_price * (1 + max_deviation)
    
    # Scale tolerance with dataset length
    base_tolerance = recent_df['ATR'].iloc[-1] * (0.3 + volatility * 0.2)
    length_factor = np.log1p(len(recent_df) / MIN_PERIODS) / 2
    tolerance = base_tolerance * (1 + length_factor)
    
    data = dict(
        highs=recent_df['High'].values,
        lows=recent_df['Low'].values,
        volumes=recent_df['Volume'].values,
        timestamps=np.arange(len(recent_df))
    )

    wyckoff_state = df['wyckoff'].iloc[-1]

    clusters = {
        'resistance': cluster_points(np.asarray(data['highs']), np.asarray(data['volumes']), np.asarray(data['timestamps']),
                                   min_price, max_price, tolerance, wyckoff_state),
        'support': cluster_points(np.asarray(data['lows']), np.asarray(data['volumes']), np.asarray(data['timestamps']),
                                min_price, max_price, tolerance, wyckoff_state)
    }
    
    max_vol = np.sum(np.asarray(data['volumes']))
    total_periods = len(recent_df)

    def filter_levels(clusters: Dict[float, Dict[str, float]], is_resistance: bool) -> List[float]:
        # First, get all valid levels with their scores
        all_scored_levels = []
        for cluster in clusters.values():
            price = cluster['price']
            score = score_level(cluster, max_vol, current_price, wyckoff_state, total_periods)
            all_scored_levels.append((price, score))
        
        # Sort by price (high to low for resistance, low to high for support)
        sorted_levels = sorted(all_scored_levels, key=lambda x: x[0], reverse=is_resistance)
        
        # Filter by position relative to current price and minimum score
        valid_levels = [
            (price, score) for price, score in sorted_levels
            if ((is_resistance and price > current_price) or 
                (not is_resistance and price < current_price)) and
            score > min_score
        ]
        
        # Now take the top n_levels
        return [price for price, _ in valid_levels[:n_levels]]
    
    return (
        filter_levels(clusters['resistance'], True),
        filter_levels(clusters['support'], False)
    )
