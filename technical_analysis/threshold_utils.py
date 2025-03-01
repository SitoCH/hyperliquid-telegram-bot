import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def calculate_volatility_adjusted_threshold(
    base_value: float, 
    volatility_factor: float,
    sensitivity: float = 1.0,
    min_multiplier: float = 0.6,
    max_multiplier: float = 2.0
) -> float:
    """
    Calculate a threshold value adjusted by market volatility
    
    Args:
        base_value: The standard threshold value
        volatility_factor: Current market volatility (1.0 = normal)
        sensitivity: How strongly volatility affects the threshold
        min_multiplier: Minimum adjustment multiplier
        max_multiplier: Maximum adjustment multiplier
        
    Returns:
        Adjusted threshold value
    """
    # Apply non-linear scaling for more stable results
    adjustment = np.tanh(volatility_factor * sensitivity - 1) + 1
    
    # Clamp the adjustment to reasonable bounds
    adjustment = np.clip(adjustment, min_multiplier, max_multiplier)
    
    return base_value * adjustment

def get_current_market_volatility(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """
    Measure current market volatility in different metrics
    
    Args:
        df: OHLCV dataframe
        lookback: Number of periods to analyze
        
    Returns:
        Dictionary of volatility metrics
    """
    if df.empty or len(df) < lookback:
        # Default values when data insufficient
        return {
            "price_volatility": 1.0,
            "volume_volatility": 1.0,
            "atr_ratio": 1.0
        }
        
    try:
        # Get recent data
        recent = df.iloc[-lookback:]
        
        # Price volatility: current vs historical
        if len(recent) >= lookback:
            current_std = recent['c'].pct_change().std()
            if len(df) > lookback * 2:
                historical_std = df['c'].pct_change().std()
                price_volatility = current_std / max(historical_std, 0.0001)
            else:
                price_volatility = 1.0
        else:
            price_volatility = 1.0
            
        # Volume volatility: current vs historical
        if len(recent) >= lookback and 'v' in recent.columns:
            current_vol_std = recent['v'].pct_change().std()
            if len(df) > lookback * 2:
                historical_vol_std = df['v'].pct_change().std()
                volume_volatility = current_vol_std / max(historical_vol_std, 0.0001)
            else:
                volume_volatility = 1.0
        else:
            volume_volatility = 1.0
            
        # ATR ratio: current vs average
        if 'ATR' in df.columns and not df['ATR'].isna().all():
            recent_atr = recent['ATR'].mean()
            if len(df) > lookback * 2:
                historical_atr = df['ATR'].mean()
                atr_ratio = recent_atr / max(historical_atr, 0.0001)
            else:
                atr_ratio = 1.0
        else:
            atr_ratio = 1.0
            
        return {
            "price_volatility": max(0.5, min(price_volatility, 3.0)),
            "volume_volatility": max(0.5, min(volume_volatility, 3.0)),
            "atr_ratio": max(0.5, min(atr_ratio, 3.0))
        }
    
    except Exception:
        # Fallback to neutral values
        return {
            "price_volatility": 1.0,
            "volume_volatility": 1.0, 
            "atr_ratio": 1.0
        }

def adjust_thresholds_for_timeframe(
    thresholds: Dict[str, float],
    timeframe_factor: float
) -> Dict[str, float]:
    """
    Adjust a set of thresholds based on timeframe factor
    
    Args:
        thresholds: Dictionary of threshold values
        timeframe_factor: Adjustment factor (1.0 = no change)
        
    Returns:
        Adjusted thresholds dictionary
    """
    return {k: v * timeframe_factor for k, v in thresholds.items()}
