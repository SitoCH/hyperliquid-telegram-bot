"""
Data processing utilities for technical analysis.
Extracted from hyperliquid_candles.py for better modularity.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import pandas_ta_classic as ta # type: ignore[import]

from logging_utils import logger
from .wyckoff.wyckoff_types import Timeframe


def prepare_dataframe(candles: List[Dict[str, Any]], local_tz) -> pd.DataFrame:
    """Prepare DataFrame from candles data with error handling"""
    if not candles:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["T", "t", "c", "h", "l", "o", "v", "n"])
    
    try:
        df = pd.DataFrame(candles)
        required_columns = {"T", "t", "c", "h", "l", "o", "v", "n"}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            logger.warning(f"Missing columns in candles data: {missing_columns}")
            return pd.DataFrame(columns=["T", "t", "c", "h", "l", "o", "v", "n"])
        
        df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True).dt.tz_convert(local_tz)
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(local_tz)
        df[["c", "h", "l", "o", "v"]] = df[["c", "h", "l", "o", "v"]].astype(float)
        df["n"] = df["n"].astype(int)
        return remove_partial_candle(df, local_tz)
    except Exception as e:
        logger.warning(f"Error preparing DataFrame: {str(e)}")
        return pd.DataFrame(columns=["T", "t", "c", "h", "l", "o", "v", "n"])


def get_indicator_settings(timeframe: Timeframe, data_length: int) -> Tuple[int, int, int, int, int]:
    """Get optimized indicator settings based on timeframe and available data."""
    # Get base settings from timeframe
    settings = timeframe.settings
    atr_length, macd_fast, macd_slow, macd_signal, st_length = settings.atr_settings
    
    # Scale down if we don't have enough data
    if data_length < atr_length * 2:
        scale = data_length / (atr_length * 2)
        atr_length = max(int(atr_length * scale), 5)
        macd_fast = max(int(macd_fast * scale), 5)
        macd_slow = max(int(macd_slow * scale), macd_fast + 4)
        macd_signal = max(int(macd_signal * scale), 3)
        st_length = max(int(st_length * scale), 4)

    return atr_length, macd_fast, macd_slow, macd_signal, st_length


def remove_partial_candle(df: pd.DataFrame, local_tz: Any) -> pd.DataFrame:
    """
    Remove the last candle if it's partial (incomplete).
    
    Args:
        df: DataFrame containing candle data with "T" column as timestamp
        local_tz: Local timezone for timestamp comparison
    
    Returns:
        DataFrame with partial candle removed if present
    """
    if df.empty:
        return df
        
    try:
        now = pd.Timestamp.now(tz=local_tz)
        last_candle_time = df["T"].iloc[-1]
        
        # Calculate expected duration based on candle interval
        if len(df) >= 2:
            previous_time = df["T"].iloc[-2]
            candle_duration = last_candle_time - previous_time
        else:
            # For single candle, try to infer duration from timestamp patterns
            # Default to 15 minutes if unable to determine
            candle_duration = pd.Timedelta(minutes=15)
        
        # Check if the last candle is still "open" (partial)
        # If current time is less than the expected end time of the last candle, it's partial
        if last_candle_time + candle_duration > now:
            df = df.iloc[:-1]
            logger.debug(f"Removed partial candle with timestamp {last_candle_time}")
        
        return df
    except Exception as e:
        logger.error(f"Error removing partial candle: {str(e)}", exc_info=True)
        return df


def apply_indicators(df: pd.DataFrame, timeframe: Timeframe) -> None:
    """Apply technical indicators with Wyckoff-optimized settings"""
    df.set_index("T", inplace=True)
    df.sort_index(inplace=True)

    atr_length, macd_fast, macd_slow, macd_signal, st_length = get_indicator_settings(timeframe, len(df))

    _add_bollinger_bands(df, timeframe)
    _add_volume_indicators(df)
    _add_vwap_indicator(df)
    _add_atr_indicator(df, atr_length)
    _add_supertrend_indicator(df, timeframe, st_length)
    _add_macd_indicator(df, macd_fast, macd_slow, macd_signal)
    _add_ema_indicator(df, timeframe)
    _add_rsi_indicator(df, timeframe)
    _add_fibonacci_levels(df, timeframe)
    _add_pivot_points(df)


def _add_bollinger_bands(df: pd.DataFrame, timeframe: Optional[Timeframe] = None) -> None:
    """Add Bollinger Bands indicators with fixed per-timeframe periods for intraday trading."""
    bb_period = timeframe.settings.bb_period if timeframe else 20
    bb_period = min(bb_period, len(df) - 1)  # Ensure we have enough data
    bb_std = 2.0
    
    bb = ta.bbands(df["c"], length=bb_period, std=bb_std)
    df['BB_lower'] = bb[f'BBL_{bb_period}_{bb_std}'] # type: ignore[assignment]
    df['BB_middle'] = bb[f'BBM_{bb_period}_{bb_std}'] # type: ignore[assignment]
    df['BB_upper'] = bb[f'BBU_{bb_period}_{bb_std}'] # type: ignore[assignment]
    df['BB_width'] = bb[f'BBB_{bb_period}_{bb_std}'] # type: ignore[assignment]


def _add_volume_indicators(df: pd.DataFrame) -> None:
    """Add volume-based indicators with adaptive periods."""
    v_sma_period = max(5, len(df) // 6)  # Shorter for fast moves
    v_short_period = max(3, len(df) // 12)
    v_long_period = max(v_short_period + 3, len(df) // 6)
    
    # Ensure we have enough data for calculations
    if len(df) < v_short_period + 1:
        # Not enough data - set default values
        df['v_sma'] = df['v'].mean() if not df['v'].empty else 0
        df['v_ratio'] = 1.0
        df['v_trend'] = 1.0
        return
    
    # Volume SMA and ratio
    v_sma = ta.sma(df['v'], length=v_sma_period)
    df['v_sma'] = v_sma
    
    # Handle division by zero and null values
    df['v_ratio'] = df['v'] / v_sma.replace(0, df['v'].mean()) # type: ignore[assignment]
    df['v_ratio'] = df['v_ratio'].fillna(1.0)
    
    # Volume trend (short vs long SMA)
    v_short = ta.sma(df['v'], length=v_short_period)
    v_long = ta.sma(df['v'], length=v_long_period)
    df['v_trend'] = (v_short / v_long.replace(0, v_short.mean())).fillna(1.0) # type: ignore[assignment]


def _add_atr_indicator(df: pd.DataFrame, atr_length: int) -> None:
    """Add ATR (Average True Range) indicator."""
    atr_calc = ta.atr(df["h"], df["l"], df["c"], length=atr_length)
    if atr_calc is not None:
        df["ATR"] = atr_calc


def _add_supertrend_indicator(df: pd.DataFrame, timeframe: Timeframe, st_length: int) -> None:
    """Add SuperTrend indicator."""
    st_multiplier = timeframe.settings.supertrend_multiplier
    supertrend = ta.supertrend(df["h"], df["l"], df["c"], 
                              length=st_length, 
                              multiplier=st_multiplier)
    
    if (supertrend is not None) and (len(df) > st_length):
        df["SuperTrend"] = supertrend[f"SUPERT_{st_length}_{st_multiplier}"]


def _add_macd_indicator(df: pd.DataFrame, macd_fast: int, macd_slow: int, macd_signal: int) -> None:
    """Add MACD indicator."""
    macd = ta.macd(df["c"], 
                   fast=macd_fast, 
                   slow=macd_slow, 
                   signal=macd_signal)
    
    if macd is not None:
        df["MACD"] = macd[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"]
        df["MACD_Signal"] = macd[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"]
        df["MACD_Hist"] = macd[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"]


def _add_ema_indicator(df: pd.DataFrame, timeframe: Timeframe) -> None:
    """Add EMA indicator."""
    df["EMA"] = ta.ema(df["c"], length=timeframe.settings.ema_length)


def _add_vwap_indicator(df: pd.DataFrame) -> None:
    """Add VWAP (Volume Weighted Average Price) indicator."""
    typical_price = (df["h"] + df["l"] + df["c"]) / 3
    vwap = (typical_price * df["v"]).cumsum() / df["v"].cumsum()
    df["VWAP"] = vwap


def _add_rsi_indicator(df: pd.DataFrame, timeframe: Optional[Timeframe] = None) -> None:
    """Add RSI (Relative Strength Index) indicator with fixed per-timeframe periods."""
    rsi_period = timeframe.settings.rsi_period if timeframe else 14
    rsi_period = min(rsi_period, len(df) - 1)  # Ensure we have enough data
    rsi = ta.rsi(df["c"], length=rsi_period)
    if rsi is not None:
        df["RSI"] = rsi


def _add_fibonacci_levels(df: pd.DataFrame, timeframe: Optional[Timeframe] = None) -> None:
    """Add Fibonacci retracement levels with fixed per-timeframe lookback."""
    lookback_period = timeframe.settings.fib_lookback if timeframe else 32
    lookback_period = min(lookback_period, len(df) - 1)  # Ensure we have enough data
    if len(df) < lookback_period:
        df["FIB_23"] = df["FIB_38"] = df["FIB_50"] = df["FIB_61"] = df["FIB_78"] = df["c"]
        return
    
    # Calculate swing high and low for adaptive period
    swing_high = df["h"].rolling(window=lookback_period).max()
    swing_low = df["l"].rolling(window=lookback_period).min()
    
    # Determine trend direction for proper Fibonacci calculation
    recent_close = df["c"].iloc[-1]
    recent_high = swing_high.iloc[-1]
    recent_low = swing_low.iloc[-1]
    
    # If price is closer to high, we're in uptrend - calculate from low to high
    # If price is closer to low, we're in downtrend - calculate from high to low
    fib_range = swing_high - swing_low
    
    if (recent_close - recent_low) > (recent_high - recent_close):
        # Uptrend: retracements from swing low (support levels)
        df["FIB_23"] = swing_low + (fib_range * 0.236)
        df["FIB_38"] = swing_low + (fib_range * 0.382)
        df["FIB_50"] = swing_low + (fib_range * 0.5)
        df["FIB_61"] = swing_low + (fib_range * 0.618)
        df["FIB_78"] = swing_low + (fib_range * 0.786)
    else:
        # Downtrend: retracements from swing high (resistance levels)
        df["FIB_23"] = swing_high - (fib_range * 0.236)
        df["FIB_38"] = swing_high - (fib_range * 0.382)
        df["FIB_50"] = swing_high - (fib_range * 0.5)
        df["FIB_61"] = swing_high - (fib_range * 0.618)
        df["FIB_78"] = swing_high - (fib_range * 0.786)


def _add_pivot_points(df: pd.DataFrame) -> None:
    """Add pivot points (PP, R1, R2, S1, S2)."""
    # Use previous day's high, low, close for pivot calculation
    pivot = (df["h"].shift(1) + df["l"].shift(1) + df["c"].shift(1)) / 3
    df["PIVOT"] = pivot
    df["R1"] = 2 * pivot - df["l"].shift(1)
    df["R2"] = pivot + (df["h"].shift(1) - df["l"].shift(1))
    df["S1"] = 2 * pivot - df["h"].shift(1)
    df["S2"] = pivot - (df["h"].shift(1) - df["l"].shift(1))
