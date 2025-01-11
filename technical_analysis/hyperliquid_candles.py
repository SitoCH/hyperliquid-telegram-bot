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
from technical_analysis.significant_levels import find_significant_levels
from technical_analysis.wyckoff import detect_wyckoff_phase, WyckoffPhase, WyckoffState, VolumeState, MarketPattern
from technical_analysis.wyckoff_signal import detect_actionable_wyckoff_signal
from technical_analysis.candles_utils import get_coins_to_analyze
from technical_analysis.candles_cache import get_candles_with_cache
from technical_analysis.funding_rates_cache import get_funding_with_cache, FundingRateEntry


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


async def analyze_candles_for_coin_job(context: ContextTypes.DEFAULT_TYPE):
    await analyze_candles_for_coin(context, context.job.data['coin'], context.job.data['all_mids'], always_notify=False) # type: ignore


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze candles for configured coins and categories."""

    all_mids = hyperliquid_utils.info.all_mids()
    coins_to_analyze = await get_coins_to_analyze(all_mids)
    
    if not coins_to_analyze:
        return
        
    logger.info(f"Running TA for {len(coins_to_analyze)} coins")
    loop = 0
    for coin in coins_to_analyze:
        context.application.job_queue.run_once( # type: ignore
            analyze_candles_for_coin_job,
            when=loop * 10,
            data={"coin": coin, "all_mids": all_mids},
            job_kwargs={'misfire_grace_time': 60}
        )
        loop += 1

    logger.info(f"TA scheduled for {len(coins_to_analyze)} coins")


async def analyze_candles_for_coin(context: ContextTypes.DEFAULT_TYPE, coin: str, all_mids: Dict[str, Any], always_notify: bool) -> None:
    logger.info(f"Running TA for {coin}")
    try:
        now = int(time.time() * 1000)

        funding_rates = get_funding_with_cache(coin, now, 7)

        candles_15m = get_candles_with_cache(coin, "15m", now, 125, hyperliquid_utils.info.candles_snapshot)
        candles_1h = get_candles_with_cache(coin, "1h", now, 250, hyperliquid_utils.info.candles_snapshot)
        candles_4h = get_candles_with_cache(coin, "4h", now, 500, hyperliquid_utils.info.candles_snapshot)
        candles_1d = get_candles_with_cache(coin, "1d", now, 750, hyperliquid_utils.info.candles_snapshot)
        
        local_tz = get_localzone()
        df_15m = prepare_dataframe(candles_15m, local_tz)
        df_1h = prepare_dataframe(candles_1h, local_tz)
        df_4h = prepare_dataframe(candles_4h, local_tz)
        df_1d = prepare_dataframe(candles_1d, local_tz)

        mid = float(all_mids[coin])

        # Apply indicators
        apply_indicators(df_15m, mid, funding_rates)
        _, wyckoff_flip_1h = apply_indicators(df_1h, mid, funding_rates)
        _, wyckoff_flip_4h = apply_indicators(df_4h, mid, funding_rates)
        apply_indicators(df_1d, mid, funding_rates)
        
        # Check if 4h candle is recent enough
        is_4h_recent = 'T' in df_4h.columns and df_4h["T"].iloc[-1] >= pd.Timestamp.now(local_tz) - pd.Timedelta(hours=1)
        
        # Either strong 4h signal or coherent 1h and 4h signals
        should_notify = (
            always_notify or
            (is_4h_recent and (
                # Strong 4h signal alone is enough
                (wyckoff_flip_4h and df_4h['wyckoff_volume'].iloc[-1] == 'high' and not df_4h['uncertain_phase'].iloc[-1]) or
                # Or coherent 1h and 4h signals with less strict conditions
                (wyckoff_flip_1h and wyckoff_flip_4h and
                 df_1h['wyckoff_phase'].iloc[-1] == df_4h['wyckoff_phase'].iloc[-1])
            ))
        )

        if should_notify:
            await send_trend_change_message(context, mid, df_15m, df_1h, df_4h, df_1d, coin, always_notify)
    except Exception as e:
        logger.error(e, exc_info=True)
        await telegram_utils.send(f"Failed to analyze candles for {coin}: {str(e)}")


def prepare_dataframe(candles: List[Dict[str, Any]], local_tz) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True).dt.tz_convert(local_tz)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(local_tz)
    df[["c", "h", "l", "o", "v"]] = df[["c", "h", "l", "o", "v"]].astype(float)
    df["n"] = df["n"].astype(int)
    return df


def apply_indicators(df: pd.DataFrame, mid: float, funding_rates: Optional[List[FundingRateEntry]] = None) -> Tuple[bool, bool]:
    """Apply technical indicators with Wyckoff-optimized settings"""
    # SuperTrend: shorter for faster response to institutional activity
    st_length = 8  # Reduced from 10 to be more responsive
    # ATR: standard Wyckoff volatility measure
    atr_length = 14
    # Volume SMA: for effort vs result analysis
    vol_length = 20  # Changed to better match trading sessions
    # Price SMA: for trend context
    price_sma_length = 20

    df.set_index("T", inplace=True)
    df.sort_index(inplace=True)

    # Wyckoff Volume Analysis
    df["Volume_SMA"] = df["v"].rolling(window=vol_length).mean()
    df["Volume_Confirm"] = df["v"] > df["Volume_SMA"]
    
    # Volume Force: measures buying/selling pressure
    df["Volume_Force"] = ((df["c"] - df["o"]) / (df["h"] - df["l"])) * df["v"]
    df["Volume_Force_SMA"] = df["Volume_Force"].rolling(window=vol_length).mean()
    
    # Effort vs Result: key Wyckoff concept
    df["Price_Range"] = df["h"] - df["l"]
    df["Close_Range"] = df["c"] - df["o"]
    df["Effort_Result"] = (df["Close_Range"] / df["Price_Range"]) * (df["v"] / df["Volume_SMA"])

    # ATR for volatility analysis
    atr_calc = ta.atr(df["h"], df["l"], df["c"], length=atr_length)
    if atr_calc is not None:
        df["ATR"] = atr_calc
    else:
        df["ATR"] = pd.Series([0.0] * len(df), index=df.index)

    # SuperTrend with optimized multiplier
    supertrend = ta.supertrend(df["h"], df["l"], df["c"], length=st_length, multiplier=3.0)
    if supertrend is not None and len(df) > st_length:
        df["SuperTrend"] = supertrend[f"SUPERT_{st_length}_3.0"]
        df["SuperTrend_Flip_Detected"] = (
            supertrend[f"SUPERTd_{st_length}_3.0"].diff().abs() == 1
        ) & df["Volume_Confirm"]
    else:
        df["SuperTrend"] = df["c"]
        df["SuperTrend_Flip_Detected"] = False

    # VWAP for institutional reference
    df["VWAP"] = ta.vwap(df["h"], df["l"], df["c"], df["v"])
    
    # Wyckoff Momentum Analysis
    macd = ta.macd(df["c"], fast=8, slow=21, signal=5)  # Adjusted for institutional timeframes
    if macd is not None:
        df["MACD"] = macd["MACD_8_21_5"]
        df["MACD_Signal"] = macd["MACDs_8_21_5"]
        df["MACD_Hist"] = macd["MACDh_8_21_5"]
    else:
        df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = 0.0

    # Trend Analysis
    df["EMA"] = ta.ema(df["c"], length=21)  # Primary trend
    df["SMA"] = ta.sma(df["c"], length=price_sma_length)  # Secondary trend

    # Wyckoff Phase Detection
    detect_wyckoff_phase(df, funding_rates)
    wyckoff_flip = detect_actionable_wyckoff_signal(df, funding_rates)
    
    return df["SuperTrend_Flip_Detected"].iloc[-1], wyckoff_flip


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


def generate_chart(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, coin: str) -> List[io.BytesIO]:
    chart_buffers = []

    plt.switch_backend('Agg')

    def save_to_buffer(df: pd.DataFrame, title: str, chart_image_time_delta) -> io.BytesIO:
        from_time = df['t'].max() - chart_image_time_delta
        df_plot = df.loc[df['t'] >= from_time].copy()

        buf = io.BytesIO()
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

        df_plot['SuperTrend_Green'] = np.where(
            df_plot['Close'] > df_plot['SuperTrend'],
            df_plot['SuperTrend'],
            np.nan
        )
        df_plot['SuperTrend_Red'] = np.where(
            df_plot['Close'] <= df_plot['SuperTrend'],
            df_plot['SuperTrend'],
            np.nan
        )

        ha_df = heikin_ashi(df_plot)

        df_plot['MACD_Hist'] = df_plot['MACD_Hist'].fillna(0)
        strong_positive_threshold = max(df_plot['MACD_Hist'].max() * 0.4, 0.000001)
        strong_negative_threshold = min(df_plot['MACD_Hist'].min() * 0.4, -0.000001)

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

        resistance_levels, support_levels = find_significant_levels(df)
        
        level_lines = []
        for level in resistance_levels:
            line = pd.Series([level] * len(df_plot), index=df_plot.index)
            level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=0.5, 
                                              label=f'R {fmt_price(level)}', linestyle='--'))
        
        for level in support_levels:
            line = pd.Series([level] * len(df_plot), index=df_plot.index)
            level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=0.5, 
                                              label=f'S {fmt_price(level)}', linestyle=':'))

        current_price = df_plot['Close'].iloc[-1]
        is_ha_bullish = ha_df['Close'].iloc[-1] >= ha_df['Open'].iloc[-1]
        current_price_line = pd.Series([current_price] * len(df_plot), index=df_plot.index)
        level_lines.append(mpf.make_addplot(current_price_line, ax=ax[0], 
                                          color='green' if is_ha_bullish else 'red', 
                                          width=0.5, label=f'Current {fmt_price(current_price)}', 
                                          linestyle=':', alpha=0.6))

        mpf.plot(ha_df,
                type='candle',
                ax=ax[0],
                volume=False,
                axtitle=title,
                style='charles',
                addplot=[
                    mpf.make_addplot(df_plot['SuperTrend'], ax=ax[0], color='green', width=0.5),
                    mpf.make_addplot(df_plot['VWAP'], ax=ax[0], color='blue', width=0.5),
                    mpf.make_addplot(df_plot['EMA'], ax=ax[0], color='orange', width=0.5),
                    *level_lines,
                    mpf.make_addplot(df_plot['MACD_Hist'], type='bar', width=0.7, 
                                   color=macd_hist_colors, ax=ax[1], alpha=0.4)
                ])

        ax[0].legend(loc='upper left', fontsize='small')

        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        return buf

    try:
        df_15m_plot = df_15m.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        chart_buffers.append(save_to_buffer(df_15m_plot, f"{coin} - 15M Chart", pd.Timedelta(hours=36)))

        df_1h_plot = df_1h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        chart_buffers.append(save_to_buffer(df_1h_plot, f"{coin} - 1H Chart", pd.Timedelta(days=3)))

        df_4h_plot = df_4h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        chart_buffers.append(save_to_buffer(df_4h_plot, f"{coin} - 4H Chart", pd.Timedelta(days=20)))

        df_1d_plot = df_1d.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        chart_buffers.append(save_to_buffer(df_1d_plot, f"{coin} - 1D Chart", pd.Timedelta(days=180)))

    except Exception as e:
        # Clean up on error
        plt.close('all')
        for buf in chart_buffers:
            if buf and not buf.closed:
                buf.close()
        raise e

    plt.close('all')
    return chart_buffers

async def send_trend_change_message(context: ContextTypes.DEFAULT_TYPE, mid: float, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, coin: str, send_charts: bool) -> None:
    
    charts = []
    try:
        charts = generate_chart(df_15m, df_1h, df_4h, df_1d, coin) if send_charts else [None] * 4 # type: ignore
        
        results_15m = get_ta_results(df_15m, mid)
        results_1h = get_ta_results(df_1h, mid)
        results_4h = get_ta_results(df_4h, mid)
        results_1d = get_ta_results(df_1d, mid)

        # Send header
        await telegram_utils.send(
            f"<b>Indicators for {coin}</b>\n"
            f"Market price: {fmt_price(mid)} USDC",
            parse_mode=ParseMode.HTML
        )

        no_wyckoff_data_available = 'No Wyckoff data available'

        # Send all charts in sequence, using copies of the buffers
        for idx, (chart, period, results) in enumerate([
            (charts[0], "15m", results_15m),
            (charts[1], "1h", results_1h),
            (charts[2], "4h", results_4h),
            (charts[3], "1d", results_1d)
        ]):
            wyckoff_description = results['wyckoff'].description if results.get('wyckoff') else no_wyckoff_data_available
            caption = f"<b>{period} indicators:</b>\n{wyckoff_description}\n<pre>{format_table(results)}</pre>"
            
            if chart:
                # Create a copy of the buffer's contents
                chart_copy = io.BytesIO(chart.getvalue())
                
                try:
                    await context.bot.send_photo(
                        chat_id=telegram_utils.telegram_chat_id,
                        photo=chart_copy,
                        caption=caption,
                        parse_mode=ParseMode.HTML
                    )
                finally:
                    chart_copy.close()
            else:
                await telegram_utils.send(caption,parse_mode=ParseMode.HTML)

    finally:
        # Clean up the original buffers
        for chart in charts:
            if chart and not chart.closed:
                chart.close()


def get_ta_results(df: pd.DataFrame, mid: float) -> Dict[str, Any]:
    # Check if we have enough data points
    if len(df["SuperTrend"]) < 2 or len(df["VWAP"]) < 2:
        logger.warning("Insufficient data for technical analysis results")
        return {
            "supertrend_prev": 0,
            "supertrend": 0,
            "supertrend_trend_prev": "unknown",
            "supertrend_trend": "unknown",
            "vwap_prev": 0,
            "vwap": 0,
            "vwap_trend_prev": "unknown",
            "vwap_trend": "unknown",
            "wyckoff": None
        }

    supertrend_prev, supertrend = df["SuperTrend"].iloc[-2], df["SuperTrend"].iloc[-1]
    vwap_prev, vwap = df["VWAP"].iloc[-2], df["VWAP"].iloc[-1]

    wyckoff = df['wyckoff'].iloc[-1]

    return {
        "supertrend_prev": supertrend_prev,
        "supertrend": supertrend,
        "supertrend_trend_prev": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-2] else "downtrend",
        "supertrend_trend": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-1] else "downtrend",
        "vwap_prev": vwap_prev,
        "vwap": vwap,
        "vwap_trend_prev": "uptrend" if mid > vwap_prev else "downtrend",
        "vwap_trend": "uptrend" if mid > vwap else "downtrend",
        "wyckoff": wyckoff
    }


def format_table(results: Dict[str, Any]) -> str:
    wyckoff = results['wyckoff']
    
    table_data = [
        ["Supertrend:", ""],
        ["Trend", results["supertrend_trend"]],
        ["Value ", fmt_price(results["supertrend"])],
        ["VWAP:", ""],
        ["Trend", results["vwap_trend"]],
        ["Value ", fmt_price(results["vwap"])],
        ["Wyckoff:", ""],
        ["Phase", wyckoff.phase.value],
        ["Comp. Action", wyckoff.composite_action.name.lower()],
        ["Pattern", wyckoff.pattern.value],
        ["Volume", wyckoff.volume.value],
        ["Volatility", wyckoff.volatility.value],
        ["Funding", wyckoff.funding_state.value],
    ]
    
    return tabulate(
        table_data,
        headers=["", ""],
        tablefmt=simple_separated_format(" "),
        colalign=("right", "right"),
    )
