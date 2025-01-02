import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]

def detect_wyckoff_phase(df: pd.DataFrame) -> None:
    """Analyze and store Wyckoff phase data for the last two periods in the dataframe"""
    # Get ATR with safety checks
    atr: float = df.get('ATR', pd.Series([0.0])).iloc[-1]
    if atr == 0 or pd.isna(atr):
        df.loc[df.index[-2:], 'wyckoff_phase'] = "Unknown"
        df.loc[df.index[-2:], 'uncertain_phase'] = True
        df.loc[df.index[-2:], 'wyckoff_volume'] = "unknown"
        df.loc[df.index[-2:], 'wyckoff_pattern'] = "unknown"
        df.loc[df.index[-2:], 'wyckoff_volatility'] = "unknown"
        return

    # Process last two periods
    for i in [-2, -1]:
        end_idx = i if i == -1 else -2
        analysis_window = min(75, len(df[:end_idx]) - 1)
        recent_df = df[:end_idx].iloc[-analysis_window:]
        
        volume_sma = recent_df['v'].rolling(window=30).mean()
        price_sma = recent_df['c'].rolling(window=30).mean()
        price_std = recent_df['c'].rolling(window=30).std()
        momentum = recent_df['c'].pct_change(periods=7).rolling(window=15).mean()
        volume_trend = recent_df['v'].pct_change().rolling(window=15).mean()
        
        # Current market conditions
        curr_price = recent_df['c'].iloc[-1]
        curr_volume = recent_df['v'].iloc[-1]
        avg_price = price_sma.iloc[-1]
        price_std_last = price_std.iloc[-1]
        volatility = price_std / avg_price
        
        # Market condition checks
        is_high_volume = (curr_volume > volume_sma.iloc[-1] * 1.1) and (volume_trend.iloc[-1] > 0)
        price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)
        momentum_strength = momentum.iloc[-1] * 100

        strong_dev_threshold = 1.5 
        neutral_zone_threshold = 0.5
        momentum_threshold = 0.6

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
                if momentum_strength > momentum_threshold:
                    phase = "markup"
                else:
                    uncertain_phase = True
                    phase = "~ markup"
            else:
                if momentum_strength < -momentum_threshold:
                    phase = "markdown"
                else:
                    uncertain_phase = True
                    phase = "~ markdown"
        
        # Store results for the current period
        current_idx = df.index[i]
        df.loc[current_idx, 'wyckoff_phase'] = phase
        df.loc[current_idx, 'uncertain_phase'] = uncertain_phase
        df.loc[current_idx, 'wyckoff_volume'] = "high" if is_high_volume else "low"
        df.loc[current_idx, 'wyckoff_pattern'] = "trending" if abs(momentum_strength) > momentum_threshold else "ranging"
        df.loc[current_idx, 'wyckoff_volatility'] = "high" if volatility.iloc[-1] > volatility.mean() else "normal"

