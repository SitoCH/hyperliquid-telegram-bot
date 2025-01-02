import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]

def detect_wyckoff_phase(df: pd.DataFrame) -> None:
    """Analyze and store Wyckoff phase data directly in the dataframe"""
    # Get ATR with safety checks
    atr: float = df.get('ATR', pd.Series([0.0])).iloc[-1]
    if atr == 0 or pd.isna(atr):
        df['wyckoff_phase'] = "Unknown"
        df['uncertain_phase'] = True
        df['wyckoff_volume'] = "unknown"
        df['wyckoff_pattern'] = "unknown"
        df['wyckoff_volatility'] = "unknown"
        return
    
    # Use last 50 candles for calculation (more responsive to recent market conditions)
    analysis_window = min(50, len(df) - 1)
    recent_df = df.iloc[-analysis_window:]

    volume_sma = recent_df['v'].rolling(window=20).mean()
    price_sma = recent_df['c'].rolling(window=20).mean()
    price_std = recent_df['c'].rolling(window=20).std()

    momentum = recent_df['c'].pct_change(periods=5).rolling(window=10).mean()
    volume_trend = recent_df['v'].pct_change().rolling(window=10).mean()
    
    # Current market conditions with volatility
    curr_price = recent_df['c'].iloc[-1]
    curr_volume = recent_df['v'].iloc[-1]
    avg_price = price_sma.iloc[-1]
    price_std_last = price_std.iloc[-1]
    volatility = price_std / avg_price
    
    # Enhanced market condition checks
    is_high_volume = (curr_volume > volume_sma.iloc[-1] * 1.1) and (volume_trend.iloc[-1] > 0)
    price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)
    momentum_strength = momentum.iloc[-1] * 100

    strong_dev_threshold = 1.2 
    neutral_zone_threshold = 0.4
    momentum_threshold = 0.8

    uncertain_phase = False
    
    if price_strength > strong_dev_threshold:
        if momentum_strength < -momentum_threshold and is_high_volume:
            phase = "dist."
        else:
            uncertain_phase = True
            phase = "~ dist."
    
    elif price_strength < -strong_dev_threshold:
        if momentum_strength > momentum_threshold and is_high_volume:
            phase = "acc."
        else:
            uncertain_phase = True
            phase = "~ acc."
    
    elif -neutral_zone_threshold <= price_strength <= neutral_zone_threshold:
        if abs(momentum_strength) < momentum_threshold and volatility.iloc[-1] < volatility.mean():
            phase = "rang."
        else:
            uncertain_phase = True
            phase = "~ rang."
    
    else:  # Transitional zones
        if price_strength > 0:
            if is_high_volume and momentum_strength > momentum_threshold:
                phase = "markup"
            else:
                uncertain_phase = True
                phase = "~ mrkp"
        else:
            if is_high_volume and momentum_strength < -momentum_threshold:
                phase = "markdown"
            else:
                uncertain_phase = True
                phase = "~ mrkdwn"
    
    # Store results
    df.loc[:, 'wyckoff_phase'] = phase
    df.loc[:, 'uncertain_phase'] = uncertain_phase
    df.loc[:, 'wyckoff_volume'] = "high" if is_high_volume else "low"
    df.loc[:, 'wyckoff_pattern'] = "trending" if abs(momentum_strength) > momentum_threshold else "ranging"
    df.loc[:, 'wyckoff_volatility'] = "high" if volatility.iloc[-1] > volatility.mean() else "normal"

