from typing import List, Tuple, Dict
import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
from scipy import signal, stats  # type: ignore[import]
from scipy.special import expit  # type: ignore[import]
from scipy.stats import gaussian_kde  # type: ignore[import]
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore[import]
import numpy as np  # type: ignore[import]

def cluster_points(
    points: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
    min_price: float,
    max_price: float,
    base_tolerance: float
) -> Dict[float, Dict[str, float]]:
    clusters: Dict[float, Dict[str, float]] = {}
    
    # Use KDE to estimate price density
    kde = gaussian_kde(points, weights=volumes)
    density = kde(points)
    
    for idx, (price, dens) in enumerate(zip(points, density)):
        if not min_price <= price <= max_price:
            continue
            
        # Dynamic tolerance using local density estimation
        local_density = dens / np.max(density)
        local_tolerance = base_tolerance * (1 + np.exp(-local_density))
        
        # Use scipy's stats for weighted statistics
        nearby_prices = [p for p in clusters.keys() if abs(p - price) <= local_tolerance]
        
        # Calculate recency weight using sigmoid function
        recency_weight = 1 / (1 + np.exp(-5 * (timestamps[idx] / len(timestamps) - 0.5)))
        
        if nearby_prices:
            main_price = nearby_prices[0]
            weight = volumes[idx] * recency_weight
            clusters[main_price]['weight'] += weight
            clusters[main_price]['count'] += 1
            clusters[main_price]['vol_sum'] += volumes[idx]
            # Update price using exponential moving average
            alpha = 0.3
            clusters[main_price]['price'] = (
                (1 - alpha) * clusters[main_price]['price'] +
                alpha * price
            )
        else:
            clusters[price] = {
                'weight': volumes[idx] * recency_weight,
                'count': 1,
                'vol_sum': volumes[idx],
                'price': price
            }
    
    return clusters

def score_level(
    cluster: Dict[str, float],
    max_vol: float,
    current_price: float,
    resistance_clusters: Dict[float, Dict[str, float]]
) -> float:
    """Score a price level using scipy's statistical functions"""
    # Use expit (sigmoid) function for better score scaling
    volume_ratio = cluster['vol_sum'] / max_vol
    volume_score = expit(volume_ratio * 3) * 0.30
    
    # Log-modulated touch count score
    touch_count = cluster['count']
    touch_score = (1 - 1/(1 + stats.logistic.cdf(touch_count))) * 0.25
    
    # Gaussian-weighted recency score
    max_weight = max(c['weight'] for c in resistance_clusters.values())
    recency_score = stats.norm.cdf(cluster['weight'] / max_weight) * 0.20
    
    # Distance-based proximity score using Gaussian function
    price_deviation = abs(cluster['price'] - current_price) / current_price
    proximity_score = stats.norm.pdf(price_deviation, scale=0.05) * 0.15
    
    # Density score using KDE estimate
    if touch_count > 1:
        density_score = stats.gamma.cdf(cluster['vol_sum'] / (price_deviation + 1e-8), a=2) * 0.10
    else:
        density_score = 0
    
    # Calculate base score
    base_score = volume_score + touch_score + recency_score + proximity_score + density_score
    
    # Multipliers for special conditions
    multiplier = 1.0
    
    # Strong level bonus (multiple touches with high volume)
    if touch_count >= 3 and volume_ratio > 0.15:
        multiplier *= 1.25
    
    # Recent activity bonus
    if cluster['weight'] > max_weight * 0.8:
        multiplier *= 1.15
    
    # Near current price bonus
    if price_deviation < 0.01:  # Within 1% of current price
        multiplier *= 1.2
    
    # Penalize very distant levels
    if price_deviation > 0.1:  # More than 10% away
        multiplier *= 0.8
    
    return min(base_score * multiplier, 1.0)

def find_significant_levels(df: pd.DataFrame, n_levels: int = 3) -> Tuple[List[float], List[float]]:
    """Enhanced support and resistance detection using scipy for better precision"""
    # Return empty lists if there's insufficient data
    if len(df) < 2:
        return [], []
        
    current_price = df['Close'].iloc[-1]
    
    # Dynamic lookback based on timeframe
    timeframe_hours = (df.index[-1] - df.index[-2]).total_seconds() / 3600
    if timeframe_hours <= 1:  # 1h or less
        lookback = 200  # About 8 days for 1h
    elif timeframe_hours <= 4:  # 4h
        lookback = 180  # About 30 days for 4h
    else:  # daily
        lookback = 150  # About 5 months for daily
        
    # Use dynamic recent data points
    recent_df = df.iloc[-lookback:]
    highs = recent_df['High'].values
    lows = recent_df['Low'].values
    volumes = recent_df['Volume'].values
    closes = recent_df['Close'].values
    timestamps = np.arange(len(recent_df))
    
    # Dynamic tolerance based on recent volatility
    atr: float = recent_df['ATR'].iloc[-1]
    recent_volatility = recent_df['Close'].std() / recent_df['Close'].mean()
    base_tolerance = atr * np.log1p(recent_volatility) * 0.35
    
    # Price range limits based on current price
    max_deviation = 0.15  # 15% from current price
    min_price = current_price * (1 - max_deviation)
    max_price = current_price * (1 + max_deviation)
    
    # Price-volume profile with tighter range
    price_steps = 100
    volume_profile = np.zeros(price_steps)
    
    for price, vol in zip(closes, volumes):
        if min_price <= price <= max_price:
            idx = int((price - min_price) / (max_price - min_price) * (price_steps - 1))
            if 0 <= idx < price_steps:
                volume_profile[idx] += vol
    
    # Smooth volume profile using scipy's savgol filter instead of simple convolution
    volume_profile = signal.savgol_filter(volume_profile, window_length=5, polyorder=2)
    
    resistance_clusters = cluster_points(
        highs, volumes, timestamps,
        min_price, max_price, base_tolerance
    )
    support_clusters = cluster_points(
        lows, volumes, timestamps,
        min_price, max_price, base_tolerance
    )
    
    max_vol = max(volumes.sum(), 1)
    
    resistance_levels = sorted(
        [cluster['price'] for cluster in resistance_clusters.values()
         if score_level(cluster, max_vol, current_price, resistance_clusters) > 0.2],  # Increased threshold
        reverse=True
    )[:n_levels]
    
    support_levels = sorted(
        [cluster['price'] for cluster in support_clusters.values()
         if score_level(cluster, max_vol, current_price, resistance_clusters) > 0.2],  # Increased threshold
    )[:n_levels]
    
    return sorted(resistance_levels), sorted(support_levels)
