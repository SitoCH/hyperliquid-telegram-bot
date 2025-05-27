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

    _add_bollinger_bands(df)
    _add_volume_indicators(df)
    _add_vwap_indicator(df)
    _add_atr_indicator(df, atr_length)
    _add_supertrend_indicator(df, timeframe, st_length)
    _add_macd_indicator(df, macd_fast, macd_slow, macd_signal)
    _add_ema_indicator(df, timeframe)
    _add_rsi_indicator(df)
    _add_stochastic_indicator(df)
    _add_fibonacci_levels(df)
    _add_pivot_points(df)
    _add_ichimoku_cloud(df)
    _add_momentum_indicators(df)


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


def _add_vwap_indicator(df: pd.DataFrame) -> None:
    """Add VWAP (Volume Weighted Average Price) indicator."""
    df["VWAP"] = (df["v"] * df["c"]).cumsum() / df["v"].cumsum()


def _add_rsi_indicator(df: pd.DataFrame) -> None:
    """Add RSI (Relative Strength Index) indicator."""
    rsi_period = min(14, len(df) // 2)
    rsi = ta.rsi(df["c"], length=rsi_period)
    if rsi is not None:
        df["RSI"] = rsi
    else:
        df["RSI"] = pd.Series([50.0] * len(df), index=df.index)


def _add_stochastic_indicator(df: pd.DataFrame) -> None:
    """Add Stochastic oscillator."""
    stoch_period = min(14, len(df) // 2)
    stoch = ta.stoch(df["h"], df["l"], df["c"], k=stoch_period)
    if stoch is not None:
        df["STOCH_K"] = stoch[f"STOCHk_{stoch_period}_3_3"]
        df["STOCH_D"] = stoch[f"STOCHd_{stoch_period}_3_3"]
    else:
        df["STOCH_K"] = df["STOCH_D"] = pd.Series([50.0] * len(df), index=df.index)


def _add_fibonacci_levels(df: pd.DataFrame) -> None:
    """Add Fibonacci retracement levels."""
    if len(df) < 20:
        df["FIB_23"] = df["FIB_38"] = df["FIB_50"] = df["FIB_61"] = df["FIB_78"] = df["c"]
        return
    
    # Calculate swing high and low for last 20 periods
    swing_high = df["h"].rolling(window=20).max()
    swing_low = df["l"].rolling(window=20).min()
    
    # Calculate Fibonacci levels
    fib_range = swing_high - swing_low
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


def _add_ichimoku_cloud(df: pd.DataFrame) -> None:
    """Add Ichimoku Cloud indicators."""
    # Adaptive periods based on data length
    period_9 = min(9, max(5, len(df) // 10))
    period_26 = min(26, max(10, len(df) // 5))
    period_52 = min(52, max(20, len(df) // 3))
    
    # Calculate Ichimoku lines
    tenkan_high = df["h"].rolling(window=period_9).max()
    tenkan_low = df["l"].rolling(window=period_9).min()
    df["TENKAN"] = (tenkan_high + tenkan_low) / 2
    
    kijun_high = df["h"].rolling(window=period_26).max()
    kijun_low = df["l"].rolling(window=period_26).min()
    df["KIJUN"] = (kijun_high + kijun_low) / 2
    
    df["SENKOU_A"] = ((df["TENKAN"] + df["KIJUN"]) / 2).shift(period_26)
    
    senkou_b_high = df["h"].rolling(window=period_52).max()
    senkou_b_low = df["l"].rolling(window=period_52).min()
    df["SENKOU_B"] = ((senkou_b_high + senkou_b_low) / 2).shift(period_26)
    
    df["CHIKOU"] = df["c"].shift(-period_26)


def _add_momentum_indicators(df: pd.DataFrame) -> None:
    """Add momentum-based indicators."""
    # Rate of Change
    roc_period = min(12, max(5, len(df) // 5))
    df["ROC"] = ta.roc(df["c"], length=roc_period)
    
    # Williams %R
    willr_period = min(14, max(7, len(df) // 4))
    df["WILLR"] = ta.willr(df["h"], df["l"], df["c"], length=willr_period)
    
    # Commodity Channel Index
    cci_period = min(20, max(10, len(df) // 3))
    df["CCI"] = ta.cci(df["h"], df["l"], df["c"], length=cci_period)
    
    # Money Flow Index
    mfi_period = min(14, max(7, len(df) // 4))
    df["MFI"] = ta.mfi(df["h"], df["l"], df["c"], df["v"], length=mfi_period)