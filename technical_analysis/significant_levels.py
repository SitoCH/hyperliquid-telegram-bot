from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import expit

def cluster_points(
    points: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
    min_price: float,
    max_price: float,
    tolerance: float
) -> Dict[float, Dict[str, float]]:
    clusters: Dict[float, Dict[str, float]] = {}
    
    for idx, price in enumerate(points):
        if not min_price <= price <= max_price:
            continue
            
        recency_weight = 1 / (1 + np.exp(-5 * (timestamps[idx] / len(timestamps) - 0.5)))
        nearby_price = next((p for p in clusters if abs(p - price) <= tolerance), None)
        
        if nearby_price:
            weight = volumes[idx] * recency_weight
            clusters[nearby_price]['weight'] += weight
            clusters[nearby_price]['count'] += 1
            clusters[nearby_price]['vol_sum'] += volumes[idx]
            clusters[nearby_price]['price'] = clusters[nearby_price]['price'] * 0.7 + price * 0.3
        else:
            clusters[price] = {
                'weight': volumes[idx] * recency_weight,
                'count': 1,
                'vol_sum': volumes[idx],
                'price': price
            }
    
    return clusters

def score_level(cluster: Dict[str, float], max_vol: float, current_price: float) -> float:
    volume_score = expit(cluster['vol_sum'] / max_vol * 3) * 0.4
    touch_score = (1 - 1/(1 + np.log1p(cluster['count']))) * 0.35  # Increased from 0.3 to 0.35
    
    price_deviation = abs(cluster['price'] - current_price) / current_price
    proximity_score = norm.pdf(price_deviation, scale=0.05) * 0.25  # Decreased from 0.3 to 0.25
    
    score = volume_score + touch_score + proximity_score
    
    # Enhanced multipliers for touches
    if cluster['count'] >= 3 and cluster['vol_sum'] > max_vol * 0.15:
        score *= 1.25  # Increased from 1.2 to 1.25
    if cluster['count'] >= 5:  # Additional bonus for many touches
        score *= 1.1
    if price_deviation < 0.01:
        score *= 1.10
    elif price_deviation > 0.1:
        score *= 0.85
        
    return min(score, 1.0)

def find_significant_levels(df: pd.DataFrame, n_levels: int = 4, min_score: float = 0.2) -> Tuple[List[float], List[float]]:
    if len(df) < 2:
        return [], []
        
    current_price = df['Close'].iloc[-1]
    
    # Safe volatility calculation with minimum lookback
    volatility = df['Close'].pct_change().std()
    if pd.isna(volatility):
        volatility = 0.01  # Default value if volatility can't be calculated
    
    # Ensure minimum lookback of 3 periods
    lookback = max(3, int(150 * (1 + volatility)))
    
    recent_df = df.iloc[-min(lookback, len(df)):]
    
    # Simplified price range
    max_deviation = min(0.15 * (1 + volatility), 0.25)
    min_price = current_price * (1 - max_deviation)
    max_price = current_price * (1 + max_deviation)
    
    # Simplified tolerance calculation
    tolerance = recent_df['ATR'].iloc[-1] * (0.3 + volatility * 0.2)
    
    data = dict(
        highs=recent_df['High'].values,
        lows=recent_df['Low'].values,
        volumes=recent_df['Volume'].values,
        timestamps=np.arange(len(recent_df))
    )
    
    clusters = {
        'resistance': cluster_points(data['highs'], data['volumes'], data['timestamps'],
                                   min_price, max_price, tolerance),
        'support': cluster_points(data['lows'], data['volumes'], data['timestamps'],
                                min_price, max_price, tolerance)
    }
    
    max_vol = data['volumes'].sum()
    
    def filter_levels(clusters: Dict[float, Dict[str, float]], is_resistance: bool) -> List[float]:
        scored_levels = [
            (cluster['price'], score_level(cluster, max_vol, current_price))
            for cluster in clusters.values()
        ]
        
        valid_levels = [
            price for price, score in scored_levels
            if score > min_score and (price > current_price) == is_resistance
        ]
        
        return sorted(valid_levels)[:n_levels]
    
    return (
        filter_levels(clusters['resistance'], True),
        filter_levels(clusters['support'], False)
    )
