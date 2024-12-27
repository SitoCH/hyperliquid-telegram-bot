import os
import io
import time
from tzlocal import get_localzone
from typing import Set, List, Dict, Any, Optional, cast, Tuple
import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore[import]
import numpy as np  # type: ignore[import]

from tabulate import tabulate, simple_separated_format
from telegram import Update, Message, CallbackQuery
from telegram.ext import CallbackContext, ContextTypes, ConversationHandler
from telegram.constants import ParseMode
from telegram.error import TelegramError

from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED, fmt, fmt_price

SELECTING_COIN_FOR_TA = range(1)


async def execute_ta(update: Update, context: CallbackContext) -> int:
    if not update.message:
        return ConversationHandler.END
    
    await update.message.reply_text("Choose a coin to analyze:", reply_markup=hyperliquid_utils.get_coins_reply_markup())
    return cast(int, SELECTING_COIN_FOR_TA)


async def selected_coin_for_ta(update: Update, context: CallbackContext) -> int:
    if not update.callback_query:
        return ConversationHandler.END

    query = update.callback_query
    await query.answer()

    if not query.data:
        return ConversationHandler.END

    coin = query.data
    if coin == "cancel":
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    await query.edit_message_text(text=f"Analyzing {coin}...")
    await analyze_candles_for_coin(context, coin, hyperliquid_utils.info.all_mids(), always_notify=True)
    await query.delete_message()
    return ConversationHandler.END


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    coins_to_analyze: Set[str] = set(coin for coin in os.getenv("HTB_COINS_TO_ANALYZE", "").split(",") if coin)

    if os.getenv('HTB_ANALYZE_COINS_WITH_OPEN_ORDERS', 'False') == 'True':
        coins_to_analyze = set(coins_to_analyze) | set(hyperliquid_utils.get_coins_with_open_positions())

    if len(coins_to_analyze) > 0:
        all_mids = hyperliquid_utils.info.all_mids()
        for coin in coins_to_analyze:
            await analyze_candles_for_coin(context, coin, all_mids, always_notify=False)


async def analyze_candles_for_coin(context: ContextTypes.DEFAULT_TYPE, coin: str, all_mids: Dict[str, Any], always_notify: bool) -> None:
    logger.info(f"Running TA for {coin}")
    try:
        now = int(time.time() * 1000)
        candles_5m = hyperliquid_utils.info.candles_snapshot(coin, "5m", now - 6 * 86400000, now)
        candles_1h = hyperliquid_utils.info.candles_snapshot(coin, "1h", now - 50 * 86400000, now)
        candles_4h = hyperliquid_utils.info.candles_snapshot(coin, "4h", now - 120 * 86400000, now)
        candles_1d = hyperliquid_utils.info.candles_snapshot(coin, "1d", now - 240 * 86400000, now)  # Last year

        local_tz = get_localzone()
        df_5m = prepare_dataframe(candles_5m, local_tz)
        df_1h = prepare_dataframe(candles_1h, local_tz)
        df_4h = prepare_dataframe(candles_4h, local_tz)
        df_1d = prepare_dataframe(candles_1d, local_tz)

        mid = float(all_mids[coin])
        apply_indicators(df_5m, mid)
        flip_on_1h = apply_indicators(df_1h, mid)
        flip_on_4h = apply_indicators(df_4h, mid)
        flip_on_1d = apply_indicators(df_1d, mid)
        
        flip_on_4h = flip_on_4h and 'T' in df_4h.columns and df_4h["T"].iloc[-1] >= pd.Timestamp.now(local_tz) - pd.Timedelta(hours=1)
        flip_on_1d = flip_on_1d and 'T' in df_1d.columns and df_1d["T"].iloc[-1] >= pd.Timestamp.now(local_tz) - pd.Timedelta(hours=4)

        if always_notify or flip_on_1h or flip_on_4h or flip_on_1d:
            await send_trend_change_message(context, mid, df_5m, df_1h, df_4h, df_1d, coin)
    except Exception as e:
        logger.critical(e, exc_info=True)
        await telegram_utils.send(f"Failed to analyze candles for {coin}: {str(e)}")


def prepare_dataframe(candles: List[Dict[str, Any]], local_tz) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True).dt.tz_convert(local_tz)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(local_tz)
    df[["c", "h", "l", "o", "v"]] = df[["c", "h", "l", "o", "v"]].astype(float)
    df["n"] = df["n"].astype(int)
    return df


def detect_wyckoff_distribution(df: pd.DataFrame) -> Dict[str, str]:
    # Get ATR from the dataframe
    atr: float = df['ATR'].iloc[-1]
    price_volatility = atr / df['c'].iloc[-1]  # Normalized volatility
    
    # Calculate patterns using ATR-adjusted thresholds
    volume_sma = df['v'].rolling(window=len(df)).mean()
    price_sma = df['c'].rolling(window=len(df)).mean()
    
    # Use ATR for trend detection
    price_trend = df['c'].diff().rolling(window=50).mean()
    is_trending = bool(abs(price_trend.iloc[-1]) > atr)
    
    # Current conditions with ATR context
    curr_price = df['c'].iloc[-1]
    curr_volume = df['v'].iloc[-1]
    is_high_volume = bool(curr_volume > volume_sma.iloc[-1])
    price_above_avg = bool(curr_price > price_sma.iloc[-1])
    
    # Get recent trends with ATR scaling
    volatility_adjusted_momentum = df['c'].pct_change().rolling(window=50).std() / price_volatility
    
    # Phase detection with ATR-adjusted conditions
    phase = "Unknown"
    
    # Distribution Phase - high prices with declining momentum relative to ATR
    if (price_above_avg and 
        volatility_adjusted_momentum.iloc[-1] < volatility_adjusted_momentum.mean() and
        curr_price > price_sma.iloc[-1] + atr):
        phase = "Distribution"
    
    # Markdown Phase - falling prices beyond ATR range
    elif (not price_above_avg and 
          price_trend.iloc[-1] < -atr and 
          is_high_volume):
        phase = "Markdown"
    
    # Accumulation Phase - low prices with increasing momentum
    elif (not price_above_avg and 
          volatility_adjusted_momentum.iloc[-1] > volatility_adjusted_momentum.mean() and
          curr_price < price_sma.iloc[-1] - atr):
        phase = "Accumulation"
    
    # Markup Phase - rising prices beyond ATR range
    elif (price_above_avg and 
          price_trend.iloc[-1] > atr and 
          is_high_volume):
        phase = "Markup"
    
    # Default phase detection with ATR context
    else:
        if price_above_avg and price_trend.iloc[-1] > atr/2:
            phase = "Markup?"
        elif price_above_avg and price_trend.iloc[-1] < -atr/2:
            phase = "Distribution?"
        elif not price_above_avg and price_trend.iloc[-1] < -atr/2:
            phase = "Markdown?"
        elif not price_above_avg and price_trend.iloc[-1] > atr/2:
            phase = "Accumulation?"
    
    return {
        "wyckoff_phase": phase,
        "volume_trend": "high" if is_high_volume else "low",
        "price_pattern": "trending" if is_trending else "ranging",
        "volatility": "high" if price_volatility > df['ATR'].mean() / df['c'].mean() else "normal"
    }


def apply_indicators(df: pd.DataFrame, mid: float) -> bool:
    # SuperTrend: shorter for faster response
    st_length = 10
    # ATR: standard setting
    atr_length = 14
    # EMA: longer for trend following
    ema_length = 21
    # Volume SMA: about one day in periods
    vol_length = 24

    df.set_index("T", inplace=True)
    df.sort_index(inplace=True)

    # ATR calculation
    df["ATR"] = ta.atr(df["h"], df["l"], df["c"], length=atr_length)

    # SuperTrend with optimized settings
    supertrend = ta.supertrend(df["h"], df["l"], df["c"], length=st_length, multiplier=3.5)
    df["SuperTrend"] = supertrend[f"SUPERT_{st_length}_3.5"]
    
    # Volume confirmation using longer period
    df["Volume_SMA"] = df["v"].rolling(window=vol_length).mean()
    df["Volume_Confirm"] = df["v"] > df["Volume_SMA"]
    
    df["SuperTrend_Flip_Detected"] = (
        supertrend[f"SUPERTd_{st_length}_3.5"].diff().abs() == 1) & df["Volume_Confirm"]

    # Use shorter length for VWAP
    df["VWAP"] = ta.vwap(df["h"], df["l"], df["c"], df["v"])
    df["VWAP_Flip_Detected"] = (mid > df["VWAP"].iloc[-2]) & (mid <= df["VWAP"].iloc[-1]) | (mid < df["VWAP"].iloc[-2]) & (mid >= df["VWAP"].iloc[-1])

    # MACD with standard settings
    macd = ta.macd(df["c"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_Signal"] = macd["MACDs_12_26_9"]
        df["MACD_Hist"] = macd["MACDh_12_26_9"]
    else:
        df["MACD"] = float('nan')
        df["MACD_Signal"] = float('nan')
        df["MACD_Hist"] = float('nan')

    # EMA with longer period for better trend following
    df["EMA"] = ta.ema(df["c"], length=ema_length)

    # Add Wyckoff analysis
    detect_wyckoff_distribution(df)
    
    return df["SuperTrend_Flip_Detected"].iloc[-1]


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    ha_df = pd.DataFrame(dict(Close=ha_close, Volume=df['Volume']))

    ha_df['Open'] = [0.0] * len(df)

    prekey = df.index[0]
    ha_df.at[prekey, 'Open'] = df.at[prekey, 'Open']

    for key in df.index[1:]:
        ha_df.at[key, 'Open'] = (ha_df.at[prekey, 'Open'] + ha_df.at[prekey, 'Close']) / 2.0
        prekey = key

    ha_df['High'] = pd.concat([ha_df.Open, df.High], axis=1).max(axis=1)
    ha_df['Low'] = pd.concat([ha_df.Open, df.Low], axis=1).min(axis=1)

    return ha_df


def generate_chart(df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, coin: str) -> List[io.BytesIO]:
    chart_buffers = []

    def find_significant_levels(df: pd.DataFrame, n_levels: int = 2) -> Tuple[List[float], List[float]]:
        """Find significant support and resistance levels using price action, volume, and ATR"""
        highs = df['High'].values
        lows = df['Low'].values
        volumes = df['Volume'].values
        timestamps = np.arange(len(df))  # For recency bias
        
        # Use ATR with adaptive scaling based on price volatility
        atr: float = df['ATR'].iloc[-1]
        volatility_scale = np.log1p(df['Close'].std() / df['Close'].mean())  # Normalized and dampened
        tolerance = atr * volatility_scale * 0.35  # Scale base tolerance by volatility
        
        # Find resistance levels (from highs)
        resistance_points: Dict[float, Dict[str, float]] = {}
        for idx, (price, vol) in enumerate(zip(highs, volumes)):
            nearby_prices = [p for p in resistance_points.keys() if abs(p - price) <= tolerance]
            if nearby_prices:
                # Add to existing cluster with volume and recency weighting
                main_price = nearby_prices[0]
                resistance_points[main_price]['count'] += 1
                resistance_points[main_price]['volume'] += vol
                resistance_points[main_price]['recency'] = max(resistance_points[main_price]['recency'], timestamps[idx])
            else:
                # Create new cluster
                resistance_points[price] = {
                    'count': 1,
                    'volume': vol,
                    'recency': timestamps[idx]
                }
        
        # Find support levels (from lows)
        support_points: Dict[float, Dict[str, float]] = {}
        for idx, (price, vol) in enumerate(zip(lows, volumes)):
            nearby_prices = [p for p in support_points.keys() if abs(p - price) <= tolerance]
            if nearby_prices:
                # Add to existing cluster with volume and recency weighting
                main_price = nearby_prices[0]
                support_points[main_price]['count'] += 1
                support_points[main_price]['volume'] += vol
                support_points[main_price]['recency'] = max(support_points[main_price]['recency'], timestamps[idx])
            else:
                # Create new cluster
                support_points[price] = {
                    'count': 1,
                    'volume': vol,
                    'recency': timestamps[idx]
                }
        
        # Score calculation with volume, touch count, and recency
        min_touches = 3  # Minimum number of touches to consider a level significant
        
        def score_level(price: float, data: Dict[str, float], max_time: float) -> float:
            if data['count'] < min_touches:
                return 0
            volume_score = data['volume'] / max(vol for vol in volumes)
            recency_score = data['recency'] / max_time
            touch_score = min(data['count'] / 10, 1.0)  # Cap at 10 touches
            return (volume_score * 0.4) + (recency_score * 0.3) + (touch_score * 0.3)
        
        # Get most significant levels based on scoring
        max_time = float(len(df) - 1)
        
        resistance_levels = sorted(
            [price for price, data in resistance_points.items() 
             if score_level(price, data, max_time) > 0],
            key=lambda p: score_level(p, resistance_points[p], max_time),
            reverse=True
        )[:n_levels]
        
        support_levels = sorted(
            [price for price, data in support_points.items() 
             if score_level(price, data, max_time) > 0],
            key=lambda p: score_level(p, support_points[p], max_time),
            reverse=True
        )[:n_levels]
        
        return sorted(resistance_levels), sorted(support_levels)

    def save_to_buffer(df_plot: pd.DataFrame, title: str) -> io.BytesIO:
        buf = io.BytesIO()
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Get current price from last close
        current_price = fmt_price(df_plot['Close'].iloc[-1])
        title_with_price = f"{title} ({current_price} USDC)"

        df_plot['SuperTrend_Green'] = df_plot.apply(lambda row: row['SuperTrend'] if row['Close'] > row['SuperTrend'] else float('nan'), axis=1)
        df_plot['SuperTrend_Red'] = df_plot.apply(lambda row: row['SuperTrend'] if row['Close'] <= row['SuperTrend'] else float('nan'), axis=1)

        ha_df = heikin_ashi(df_plot)

        strong_positive_threshold = df_plot['MACD_Hist'].max() * 0.5
        strong_negative_threshold = df_plot['MACD_Hist'].min() * 0.5

        def determine_color(value: float) -> str:
            if value >= strong_positive_threshold:
                return 'green'
            elif 0 < value < strong_positive_threshold:
                return 'lightgreen'
            elif strong_negative_threshold < value <= 0:
                return 'lightcoral'
            else:
                return 'red'

        macd_hist_colors = df_plot['MACD_Hist'].apply(determine_color).values

        # Find significant price levels
        resistance_levels, support_levels = find_significant_levels(df_plot)
        
        # Create horizontal lines for each level
        level_lines = []
        for level in resistance_levels:
            line = pd.Series([level] * len(df_plot), index=df_plot.index)
            level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=1, 
                                              label=f'R {fmt_price(level)}', linestyle='--'))
        
        for level in support_levels:
            line = pd.Series([level] * len(df_plot), index=df_plot.index)
            level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=1, 
                                              label=f'S {fmt_price(level)}', linestyle=':'))

        mpf.plot(ha_df,
                type='candle',
                ax=ax[0],
                volume=False,
                axtitle=title_with_price,  # Use updated title
                style='charles',
                addplot=[
                    mpf.make_addplot(df_plot['SuperTrend'], ax=ax[0], color='green', label='SuperTrend', width=0.75),
                    mpf.make_addplot(df_plot['SuperTrend_Red'], ax=ax[0], color='red', width=0.75),
                    mpf.make_addplot(df_plot['VWAP'], ax=ax[0], color='blue', label='VWAP', width=0.75),
                    mpf.make_addplot(df_plot['EMA'], ax=ax[0], color='orange', label='EMA', width=0.75),
                    *level_lines,
                    mpf.make_addplot(df_plot['MACD_Hist'], type='bar', width=0.7, color=macd_hist_colors, ax=ax[1], alpha=0.5, secondary_y=False)
                ])

        ax[0].legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    df_5m_plot = df_5m.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    # Show last 12 hours for 5m chart (144 candles)
    from_time = df_5m_plot['t'].max() - pd.Timedelta(hours=12)
    df_5m_plot = df_5m_plot.loc[df_5m_plot['t'] >= from_time]

    df_1h_plot = df_1h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    # Show last 3 days for 1h chart (72 candles)
    from_time = df_1h_plot['t'].max() - pd.Timedelta(days=3)
    df_1h_plot = df_1h_plot.loc[df_1h_plot['t'] >= from_time]

    df_4h_plot = df_4h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    # Show last 20 days for 4h chart (120 candles)
    from_time = df_4h_plot['t'].max() - pd.Timedelta(days=20)
    df_4h_plot = df_4h_plot.loc[df_4h_plot['t'] >= from_time]

    df_1d_plot = df_1d.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    # Show last 180 days for 1d chart
    from_time = df_1d_plot['t'].max() - pd.Timedelta(days=180)
    df_1d_plot = df_1d_plot.loc[df_1d_plot['t'] >= from_time]

    chart_buffers.append(save_to_buffer(df_5m_plot, f"{coin} - 5M Chart"))
    chart_buffers.append(save_to_buffer(df_1h_plot, f"{coin} - 1H Chart"))
    chart_buffers.append(save_to_buffer(df_4h_plot, f"{coin} - 4H Chart"))
    chart_buffers.append(save_to_buffer(df_1d_plot, f"{coin} - 1D Chart"))

    return chart_buffers


async def send_trend_change_message(context: ContextTypes.DEFAULT_TYPE, mid: float, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, coin: str) -> None:
    charts = generate_chart(df_5m, df_1h, df_4h, df_1d, coin)

    results_1h = get_ta_results(df_1h, mid)
    results_4h = get_ta_results(df_4h, mid)
    results_1d = get_ta_results(df_1d, mid)

    table_1h = format_table(results_1h)
    table_4h = format_table(results_4h)
    table_1d = format_table(results_1d)

    message_lines = [
        f"<b>Indicators for {coin}</b>",
        f"Market price: {fmt_price(mid)} USDC",
        "1h indicators:",
        f"<pre>{table_1h}</pre>",
        "4h indicators:",
        f"<pre>{table_4h}</pre>",
        "1d indicators:",
        f"<pre>{table_1d}</pre>",
    ]
    
    await telegram_utils.send("\n".join(message_lines), parse_mode=ParseMode.HTML)

    for buf in charts:
        await context.bot.send_photo(chat_id=telegram_utils.telegram_chat_id, photo=buf)


def get_ta_results(df: pd.DataFrame, mid: float) -> Dict[str, Any]:
    supertrend_prev, supertrend = df["SuperTrend"].iloc[-2], df["SuperTrend"].iloc[-1]
    vwap_prev, vwap = df["VWAP"].iloc[-2], df["VWAP"].iloc[-1]

    results = {
        "supertrend_prev": supertrend_prev,
        "supertrend": supertrend,
        "supertrend_trend_prev": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-2] else "downtrend",
        "supertrend_trend": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-1] else "downtrend",
        "vwap_prev": vwap_prev,
        "vwap": vwap,
        "vwap_trend_prev": "uptrend" if mid > vwap_prev else "downtrend",
        "vwap_trend": "uptrend" if mid > vwap else "downtrend",
    }
    
    # Add Wyckoff results
    wyckoff_data = detect_wyckoff_distribution(df)
    results.update({
        "wyckoff_phase": wyckoff_data['wyckoff_phase'],
        "wyckoff_volume": wyckoff_data['volume_trend'],
        "wyckoff_pattern": wyckoff_data['price_pattern']
    })
    
    return results


def format_table(results: Dict[str, Any]) -> str:
    table_data = [
        ["Supertrend: ", "", ""],
        ["Trend ", results["supertrend_trend_prev"], results["supertrend_trend"]],
        ["Value ", fmt_price(results["supertrend_prev"]), fmt_price(results["supertrend"])],
        ["", "", ""],
        ["VWAP: ", "", ""],
        ["Trend ", results["vwap_trend_prev"], results["vwap_trend"]],
        ["Value ", fmt_price(results["vwap_prev"]), fmt_price(results["vwap"])],
        ["", "", ""],
        ["Wyckoff: ", "", ""],
        ["Phase ", "", results["wyckoff_phase"]],
        ["Volume ", "", results["wyckoff_volume"]],
        ["Pattern ", "", results["wyckoff_pattern"]]
    ]
    
    return tabulate(
        table_data,
        headers=["", "Previous", "Current"],
        tablefmt=simple_separated_format(" "),
        colalign=("right", "right", "right"),
    )
