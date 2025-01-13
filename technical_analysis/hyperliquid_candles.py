import os
import io
import time
from tzlocal import get_localzone
from typing import List, Dict, Any, cast, Tuple
import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import mplfinance as mpf  # type: ignore[import]

from tabulate import tabulate, simple_separated_format
from telegram import Update
from telegram.ext import CallbackContext, ContextTypes, ConversationHandler
from telegram.constants import ParseMode

from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED, fmt_price, log_execution_time
from technical_analysis.wyckoff import detect_wyckoff_phase
from technical_analysis.wyckoff_signal import detect_actionable_wyckoff_signal
from technical_analysis.candles_utils import get_coins_to_analyze
from technical_analysis.candles_cache import get_candles_with_cache
from technical_analysis.wyckoff_types import Timeframe
from technical_analysis.funding_rates_cache import get_funding_with_cache, FundingRateEntry
from technical_analysis.wykcoff_chart import generate_chart
from technical_analysis.wyckoff_multi_timeframe import MultiTimeframeContext

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
    await analyze_candles_for_coin(context, coin, always_notify=True)
    await query.delete_message()
    return ConversationHandler.END


async def analyze_candles_for_coin_job(context: ContextTypes.DEFAULT_TYPE):
    await analyze_candles_for_coin(context, context.job.data['coin'], always_notify=False) # type: ignore


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
            when=loop * 8,
            data={"coin": coin},
            job_kwargs={'misfire_grace_time': 180}
        )
        loop += 1

    logger.info(f"TA scheduled for {len(coins_to_analyze)} coins")


async def analyze_candles_for_coin(context: ContextTypes.DEFAULT_TYPE, coin: str, always_notify: bool) -> None:
    start_time = time.time()
    logger.info(f"Running TA for {coin}")
    try:
        now = int(time.time() * 1000)

        funding_rates = get_funding_with_cache(coin, now, 7)

        candles_15m = get_candles_with_cache(coin, "15m", now, 50, hyperliquid_utils.info.candles_snapshot)
        candles_1h = get_candles_with_cache(coin, "1h", now, 100, hyperliquid_utils.info.candles_snapshot)
        candles_4h = get_candles_with_cache(coin, "4h", now, 300, hyperliquid_utils.info.candles_snapshot)
        candles_1d = get_candles_with_cache(coin, "1d", now, 750, hyperliquid_utils.info.candles_snapshot)
        
        local_tz = get_localzone()
        df_15m = prepare_dataframe(candles_15m, local_tz)
        df_1h = prepare_dataframe(candles_1h, local_tz)
        df_4h = prepare_dataframe(candles_4h, local_tz)
        df_1d = prepare_dataframe(candles_1d, local_tz)


        # Apply indicators with Wyckoff analysis
        states = {}
        apply_indicators(df_15m, Timeframe.MINUTES_15, funding_rates)
        states[Timeframe.MINUTES_15] = df_15m['wyckoff'].iloc[-1]
        
        _, wyckoff_flip_1h = apply_indicators(df_1h, Timeframe.HOUR_1, funding_rates)
        states[Timeframe.HOUR_1] = df_1h['wyckoff'].iloc[-1]
        
        _, wyckoff_flip_4h = apply_indicators(df_4h, Timeframe.HOURS_4, funding_rates)
        states[Timeframe.HOURS_4] = df_4h['wyckoff'].iloc[-1]
        
        apply_indicators(df_1d, Timeframe.DAY_1, funding_rates)
        states[Timeframe.DAY_1] = df_1d['wyckoff'].iloc[-1]

        # Add multi-timeframe analysis
        from technical_analysis.wyckoff_multi_timeframe import analyze_multi_timeframe
        mtf_context = analyze_multi_timeframe(states)
        
        # Update notification logic to include MTF analysis
        is_4h_recent = 'T' in df_4h.columns and df_4h["T"].iloc[-1] >= pd.Timestamp.now(local_tz) - pd.Timedelta(hours=1)
        
        should_notify = (
            always_notify or
            (is_4h_recent and (
                # Strong signal with multi-timeframe confirmation
                (mtf_context.confidence_level > 0.7 and mtf_context.alignment_score > 0.6) or
                # Original conditions
                (wyckoff_flip_4h and df_4h['wyckoff_volume'].iloc[-1] == 'high' and not df_4h['uncertain_phase'].iloc[-1]) or
                (wyckoff_flip_1h and wyckoff_flip_4h and
                 df_1h['wyckoff_phase'].iloc[-1] == df_4h['wyckoff_phase'].iloc[-1])
            ))
        )

        if should_notify:
            mid = float(hyperliquid_utils.info.all_mids()[coin])
            await send_trend_change_message(context, mid, df_15m, df_1h, df_4h, df_1d, coin, always_notify, mtf_context)

    except Exception as e:
        logger.error(e, exc_info=True)
        await telegram_utils.send(f"Failed to analyze candles for {coin}: {str(e)}")
    logger.info(f"TA for {coin} done in {(time.time() - start_time):.2f} seconds")


def prepare_dataframe(candles: List[Dict[str, Any]], local_tz) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True).dt.tz_convert(local_tz)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(local_tz)
    df[["c", "h", "l", "o", "v"]] = df[["c", "h", "l", "o", "v"]].astype(float)
    df["n"] = df["n"].astype(int)
    return df


def apply_indicators(df: pd.DataFrame, timeframe: Timeframe, funding_rates: List[FundingRateEntry]) -> Tuple[bool, bool]:
    """Apply technical indicators with Wyckoff-optimized settings"""
    # SuperTrend: shorter for faster response to institutional activity
    st_length = 8  # Reduced from 10 to be more responsive
    # ATR: standard Wyckoff volatility measure
    atr_length = 14

    df.set_index("T", inplace=True)
    df.sort_index(inplace=True)

    # ATR for volatility analysis
    atr_calc = ta.atr(df["h"], df["l"], df["c"], length=atr_length)
    if atr_calc is not None:
        df["ATR"] = atr_calc
    else:
        df["ATR"] = pd.Series([0.0] * len(df), index=df.index)

    # SuperTrend with optimized multiplier
    supertrend = ta.supertrend(df["h"], df["l"], df["c"], length=st_length, multiplier=3.0)
    if (supertrend is not None) and (len(df) > st_length):
        df["SuperTrend"] = supertrend[f"SUPERT_{st_length}_3.0"]
        df["SuperTrend_Flip_Detected"] = (
            supertrend[f"SUPERTd_{st_length}_3.0"].diff().abs() == 1
        )
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

    df["EMA"] = ta.ema(df["c"], length=21)
    
    # Wyckoff Phase Detection
    detect_wyckoff_phase(df, timeframe, funding_rates)
    wyckoff_flip = detect_actionable_wyckoff_signal(df)
    
    return df["SuperTrend_Flip_Detected"].iloc[-1], wyckoff_flip


async def send_trend_change_message(context: ContextTypes.DEFAULT_TYPE, mid: float, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, coin: str, send_charts: bool, mtf_context: MultiTimeframeContext) -> None:
    
    charts = []
    try:
        charts = generate_chart(df_15m, df_1h, df_4h, df_1d, coin) if send_charts else [None] * 4 # type: ignore
        
        results_15m = get_ta_results(df_15m, mid)
        results_1h = get_ta_results(df_1h, mid)
        results_4h = get_ta_results(df_4h, mid)
        results_1d = get_ta_results(df_1d, mid)

        # Add MTF analysis at the start of the message
        await telegram_utils.send(
            f"<b>Indicators for {coin}</b>\n"
            f"Market price: {fmt_price(mid)} USDC\n"
            f"Multi-Timeframe Analysis:\n{mtf_context.description}\n\n",
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
