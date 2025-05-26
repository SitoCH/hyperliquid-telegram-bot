"""
Data processing utilities for technical analysis.
Extracted from hyperliquid_candles.py for better modularity.
"""

import time
from typing import List, Dict, Any, Tuple
import pandas as pd
import pandas_ta as ta # type: ignore[import]
from tzlocal import get_localzone

from logging_utils import logger
from technical_analysis.wyckoff_types import Timeframe


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
        return df
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
        df: DataFrame containing candle data
        local_tz: Local timezone for timestamp comparison
    
    Returns:
        DataFrame with partial candle removed if present
    """
    if df.empty:
        return df
        
    try:
        now = pd.Timestamp.now(tz=local_tz)
        last_candle_time = df.index[-1]
        
        # Calculate expected duration
        if len(df) >= 2:
            previous_time = df.index[-2]
            candle_duration = last_candle_time - previous_time
        else:
            candle_duration = pd.Timedelta(minutes=1)
        
        if last_candle_time + candle_duration > now:
            df = df.iloc[:-1]
        
        return df
    except Exception as e:
        logger.error(f"Error removing partial candle: {str(e)}", exc_info=True)
        return df


def apply_indicators(df: pd.DataFrame, timeframe: Timeframe) -> None:
    """Apply technical indicators with Wyckoff-optimized settings"""
    df.set_index("T", inplace=True)
    df.sort_index(inplace=True)

    # Get optimized settings based on timeframe and data length
    atr_length, macd_fast, macd_slow, macd_signal, st_length = get_indicator_settings(timeframe, len(df))

    # Add Bollinger Bands calculation
    _add_bollinger_bands(df)
    
    # Add volume analysis
    _add_volume_indicators(df)
    
    # Add technical indicators
    _add_atr_indicator(df, atr_length)
    _add_supertrend_indicator(df, timeframe, st_length)
    _add_macd_indicator(df, macd_fast, macd_slow, macd_signal)
    _add_ema_indicator(df, timeframe)


def _add_bollinger_bands(df: pd.DataFrame) -> None:
    """Add Bollinger Bands indicators."""
    bb_period = min(20, len(df) // 3)  # Adaptive period
    bb_std = 2.0
    df['BB_middle'] = df['c'].rolling(window=bb_period).mean()
    bb_std_dev = df['c'].rolling(window=bb_period).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std_dev * bb_std)
    df['BB_lower'] = df['BB_middle'] - (bb_std_dev * bb_std)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']


def _add_volume_indicators(df: pd.DataFrame) -> None:
    """Add volume-based indicators."""
    df['v_sma'] = df['v'].rolling(window=20).mean()
    df['v_std'] = df['v'].rolling(window=20).std()
    df['v_normalized'] = (df['v'] - df['v_sma']) / df['v_std']
    df['v_ratio'] = df['v'] / df['v_sma']
    
    # Volume trend strength
    df['v_trend'] = df['v'].rolling(window=5).mean() / df['v'].rolling(window=20).mean()


def _add_atr_indicator(df: pd.DataFrame, atr_length: int) -> None:
    """Add ATR (Average True Range) indicator."""
    atr_calc = ta.atr(df["h"], df["l"], df["c"], length=atr_length)
    if atr_calc is not None:
        df["ATR"] = atr_calc
    else:
        df["ATR"] = pd.Series([0.0] * len(df), index=df.index)


def _add_supertrend_indicator(df: pd.DataFrame, timeframe: Timeframe, st_length: int) -> None:
    """Add SuperTrend indicator."""
    st_multiplier = timeframe.settings.supertrend_multiplier
    supertrend = ta.supertrend(df["h"], df["l"], df["c"], 
                              length=st_length, 
                              multiplier=st_multiplier)
    
    if (supertrend is not None) and (len(df) > st_length):
        df["SuperTrend"] = supertrend[f"SUPERT_{st_length}_{st_multiplier}"]
        df["SuperTrend_Flip_Detected"] = (
            supertrend[f"SUPERTd_{st_length}_{st_multiplier}"].diff().abs() == 1
        )
    else:
        df["SuperTrend"] = df["c"]
        df["SuperTrend_Flip_Detected"] = False


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
    else:
        df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = 0.0


def _add_ema_indicator(df: pd.DataFrame, timeframe: Timeframe) -> None:
    """Add EMA indicator."""
    df["EMA"] = ta.ema(df["c"], length=timeframe.settings.ema_length)