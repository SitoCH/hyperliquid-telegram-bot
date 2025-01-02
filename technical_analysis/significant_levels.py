from typing import List, Tuple, Dict
import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore[import]
import numpy as np  # type: ignore[import]

def find_significant_levels(df: pd.DataFrame, n_levels: int = 3) -> Tuple[List[float], List[float]]:
    """Enhanced support and resistance detection with dynamic lookback period"""
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
    
    # Smooth volume profile
    volume_profile = np.convolve(volume_profile, np.ones(5)/5, mode='same')
    
    def cluster_points(points: np.ndarray, volumes: np.ndarray, timestamps: np.ndarray) -> Dict[float, Dict[str, float]]:
        clusters: Dict[float, Dict[str, float]] = {}
        
        for idx, price in enumerate(points):
            # Skip prices outside our range of interest
            if not min_price <= price <= max_price:
                continue
                
            # Dynamic tolerance based on price level and distance from current price
            price_deviation = abs(price - current_price) / current_price
            local_tolerance = base_tolerance * (1 + price_deviation)
            
            nearby_prices = [p for p in clusters.keys() if abs(p - price) <= local_tolerance]
            
            # Exponential decay for older points
            recency_weight = np.exp(timestamps[idx] / len(timestamps) - 1)
            
            if nearby_prices:
                main_price = nearby_prices[0]
                weight = volumes[idx] * recency_weight
                clusters[main_price]['weight'] += weight
                clusters[main_price]['count'] += 1
                clusters[main_price]['vol_sum'] += volumes[idx]
                # Weighted average favoring recent prices
                clusters[main_price]['price'] = (
                    clusters[main_price]['price'] * 0.7 +
                    price * 0.3
                )
            else:
                clusters[price] = {
                    'weight': volumes[idx] * recency_weight,
                    'count': 1,
                    'vol_sum': volumes[idx],
                    'price': price
                }
        
        return clusters
    
    resistance_clusters = cluster_points(highs, volumes, timestamps)
    support_clusters = cluster_points(lows, volumes, timestamps)
    
    # Score and filter levels
    def score_level(cluster: Dict[str, float], max_vol: float) -> float:
        """
        Score a price level based on multiple weighted factors.
        Returns a score between 0 and 1.
        """
        current_price = df['Close'].iloc[-1]
        
        # Volume importance (30%)
        volume_ratio = cluster['vol_sum'] / max_vol
        volume_score = min(volume_ratio * 1.5, 1.0) * 0.30
        
        # Touch count importance (25%)
        # Logarithmic scaling with diminishing returns after 5 touches
        touch_count = cluster['count']
        touch_score = (1 - 1/(1 + np.log1p(touch_count))) * 0.25
        
        # Recency weight (20%)
        # Higher weight for more recent activity
        max_weight = max(c['weight'] for c in resistance_clusters.values())
        recency_score = (cluster['weight'] / max_weight) * 0.20
        
        # Price proximity (15%)
        # Exponential decay for prices further from current price
        price_deviation = abs(cluster['price'] - current_price) / current_price
        proximity_score = np.exp(-5 * price_deviation) * 0.15
        
        # Cluster density (10%)
        # Tighter clusters get higher scores
        if touch_count > 1:
            price_range = price_deviation * current_price
            density = cluster['vol_sum'] / (price_range * touch_count + 1e-8)
            density_score = min(density / (max_vol * 0.1), 1.0) * 0.10
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

    max_vol = max(volumes.sum(), 1)
    
    resistance_levels = sorted(
        [cluster['price'] for cluster in resistance_clusters.values()
         if score_level(cluster, max_vol) > 0.2],  # Increased threshold
        reverse=True
    )[:n_levels]
    
    support_levels = sorted(
        [cluster['price'] for cluster in support_clusters.values()
         if score_level(cluster, max_vol) > 0.2],  # Increased threshold
    )[:n_levels]
    
    return sorted(resistance_levels), sorted(support_levels)
