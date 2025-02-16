import io
import os
import time
from tzlocal import get_localzone
from typing import List, Dict, Any, cast, Tuple, TypedDict
import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]

from telegram import Update
from telegram.ext import CallbackContext, ContextTypes, ConversationHandler
from telegram.constants import ParseMode

from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED, fmt_price, log_execution_time
from technical_analysis.significant_levels import find_significant_levels
from technical_analysis.wyckoff import detect_wyckoff_phase
from technical_analysis.candles_utils import get_coins_to_analyze
from technical_analysis.candles_cache import get_candles_with_cache
from technical_analysis.wyckoff_types import Timeframe, WyckoffState, SignificantLevelsData
from technical_analysis.funding_rates_cache import get_funding_with_cache, FundingRateEntry
from technical_analysis.wykcoff_chart import generate_chart
from technical_analysis.wyckoff_multi_timeframe import MultiTimeframeContext, analyze_multi_timeframe, MultiTimeframeDirection
from .hyperliquid_ratelimiter import hyperliquid_rate_limiter


SELECTING_COIN_FOR_TA = range(1)


def get_significant_levels(coin: str, mid: float, timeframe: Timeframe, lookback_days: int) -> Tuple[List[float], List[float]]:
    now = int(time.time() * 1000)
    candles = get_candles_with_cache(coin, timeframe, now, lookback_days, hyperliquid_utils.info.candles_snapshot)
    local_tz = get_localzone()
    df = prepare_dataframe(candles, local_tz)
    apply_indicators(df, timeframe)
    funding_rates = get_funding_with_cache(coin, now, 7)
    return find_significant_levels(df, detect_wyckoff_phase(remove_partial_candle(df, local_tz), timeframe, funding_rates), mid, timeframe)


async def execute_ta(update: Update, context: CallbackContext) -> int:
    if not update.message:
        return ConversationHandler.END

    if context.args and len(context.args) > 0:
        coin = context.args[0]
        await update.message.reply_text(text=f"Analyzing {coin}...")
        await analyze_candles_for_coin(context, coin, True)
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
    await analyze_candles_for_coin(context, coin, True)
    await query.delete_message()
    return ConversationHandler.END


async def analyze_candles_for_coin_job(context: ContextTypes.DEFAULT_TYPE):
    """Process coins one at a time with rate limiting."""

    coins_to_analyze = context.job.data['coins_to_analyze'] # type: ignore
    coin = coins_to_analyze.pop()
        
    await analyze_candles_for_coin(context, coin, False)
 
    # Schedule next coin if any remain
    if coins_to_analyze:
        weight_per_analysis = 175
        next_available = hyperliquid_rate_limiter.get_next_available_time(weight_per_analysis)
        delay = max(2, next_available)
        hyperliquid_rate_limiter.add_weight(weight_per_analysis)
        context.application.job_queue.run_once( # type: ignore
            analyze_candles_for_coin_job,
            when=delay,
            data={"coins_to_analyze": coins_to_analyze},
            job_kwargs={'misfire_grace_time': 180}
        )


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze candles for configured coins and categories."""

    all_mids = hyperliquid_utils.info.all_mids()
    coins_to_analyze = await get_coins_to_analyze(all_mids)
    
    if not coins_to_analyze:
        return

    logger.info(f"Running TA for {len(coins_to_analyze)} coins")
    context.application.job_queue.run_once( # type: ignore
        analyze_candles_for_coin_job,
        when=2,
        data={"coins_to_analyze": coins_to_analyze},
        job_kwargs={'misfire_grace_time': 180}
    )


async def analyze_candles_for_coin(context: ContextTypes.DEFAULT_TYPE, coin: str, interactive_analysis: bool) -> None:
    start_time = time.time()
    logger.info(f"Running TA for {coin}")
    try:
        now = int(time.time() * 1000)

        funding_rates = get_funding_with_cache(coin, now, 7)

        # Get candles for all timeframes
        candles_data = {
            Timeframe.MINUTES_15: get_candles_with_cache(coin, Timeframe.MINUTES_15, now, 40, hyperliquid_utils.info.candles_snapshot),
            Timeframe.MINUTES_30: get_candles_with_cache(coin, Timeframe.MINUTES_30, now, 70, hyperliquid_utils.info.candles_snapshot),
            Timeframe.HOUR_1: get_candles_with_cache(coin, Timeframe.HOUR_1, now, 90, hyperliquid_utils.info.candles_snapshot),
            Timeframe.HOURS_2: get_candles_with_cache(coin, Timeframe.HOURS_2, now, 100, hyperliquid_utils.info.candles_snapshot),
            Timeframe.HOURS_4: get_candles_with_cache(coin, Timeframe.HOURS_4, now, 110, hyperliquid_utils.info.candles_snapshot),
            Timeframe.HOURS_8: get_candles_with_cache(coin, Timeframe.HOURS_8, now, 150, hyperliquid_utils.info.candles_snapshot),
            Timeframe.DAY_1: get_candles_with_cache(coin, Timeframe.DAY_1, now, 200, hyperliquid_utils.info.candles_snapshot)
        }

        # Check if we have enough data for basic analysis
        if len(candles_data[Timeframe.MINUTES_15]) < 10:
            logger.warning(f"Insufficient candles for technical analysis on {coin}")
            return

        local_tz = get_localzone()
        dataframes = {
            tf: prepare_dataframe(candles, local_tz) 
            for tf, candles in candles_data.items()
        }

        # Apply indicators and get Wyckoff states
        states = {}
        for tf, df in dataframes.items():
            if not df.empty:
                apply_indicators(df, tf)
                states[tf] = detect_wyckoff_phase(remove_partial_candle(df, local_tz), tf, funding_rates)
            else:
                states[tf] = WyckoffState.unknown()

        # Add multi-timeframe analysis
        mid = float(hyperliquid_utils.info.all_mids()[coin])
        significant_levels: Dict[Timeframe, SignificantLevelsData] = { }
        
        resistance_30m, support_30m = find_significant_levels(dataframes[Timeframe.MINUTES_30], states[Timeframe.MINUTES_30], mid, Timeframe.MINUTES_30)
        significant_levels[Timeframe.MINUTES_30] = {
            'resistance': resistance_30m,
            'support': support_30m
        }

        resistance_1h, support_1h = find_significant_levels(dataframes[Timeframe.HOUR_1], states[Timeframe.HOUR_1], mid, Timeframe.HOUR_1)
        significant_levels[Timeframe.HOUR_1] = {
            'resistance': resistance_1h,
            'support': support_1h
        }

        resistance_4h, support_4h = find_significant_levels(dataframes[Timeframe.HOURS_4], states[Timeframe.HOURS_4], mid, Timeframe.HOURS_4)
        significant_levels[Timeframe.HOURS_4] = {
            'resistance': resistance_4h,
            'support': support_4h
        }
       
        mtf_context = analyze_multi_timeframe(
            states, 
            coin, 
            mid, 
            significant_levels,
            interactive_analysis
        )

        min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.75"))
        should_notify = (
            interactive_analysis or 
            (mtf_context.confidence_level >= min_confidence and mtf_context.direction != MultiTimeframeDirection.NEUTRAL)
        )

        if should_notify:
            await send_trend_change_message(
                context, 
                mid, 
                dataframes,
                states,
                coin, 
                interactive_analysis, 
                mtf_context
            )

    except Exception as e:
        logger.error(f"Failed to analyze candles for {coin}: {str(e)}", exc_info=True)
        if interactive_analysis:
            await telegram_utils.send(f"Failed to analyze candles for {coin}: {str(e)}")
    
    logger.info(f"TA for {coin} done in {(time.time() - start_time):.2f} seconds")


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


def get_indicator_settings(timeframe: Timeframe, data_length: int) -> tuple[int, int, int, int, int]:
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

    # Add volume normalization
    df['v_sma'] = df['v'].rolling(window=20).mean()
    df['v_std'] = df['v'].rolling(window=20).std()
    df['v_normalized'] = (df['v'] - df['v_sma']) / df['v_std']
    df['v_ratio'] = df['v'] / df['v_sma']
    
    # Volume trend strength
    df['v_trend'] = df['v'].rolling(window=5).mean() / df['v'].rolling(window=20).mean()

    # ATR for volatility analysis
    atr_calc = ta.atr(df["h"], df["l"], df["c"], length=atr_length)
    if atr_calc is not None:
        df["ATR"] = atr_calc
    else:
        df["ATR"] = pd.Series([0.0] * len(df), index=df.index)

    # SuperTrend with optimized parameters
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

    # VWAP for institutional reference
    df["VWAP"] = ta.vwap(df["h"], df["l"], df["c"], df["v"])
    
    # Wyckoff Momentum Analysis with timeframe-optimized MACD
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

    # EMA length based on timeframe
    df["EMA"] = ta.ema(df["c"], length=timeframe.settings.ema_length)


async def send_trend_change_message(context: ContextTypes.DEFAULT_TYPE, mid: float, dataframes: dict[Timeframe, pd.DataFrame], states: dict[Timeframe, WyckoffState], coin: str, send_charts: bool, mtf_context: MultiTimeframeContext) -> None:
    
    if send_charts:
        charts = []
        try:
            charts = generate_chart(dataframes, states, coin, mid)
            
            results_15m = get_ta_results(dataframes[Timeframe.MINUTES_15], states[Timeframe.MINUTES_15], mid)
            results_1h = get_ta_results(dataframes[Timeframe.HOUR_1], states[Timeframe.HOUR_1], mid)
            results_4h = get_ta_results(dataframes[Timeframe.HOURS_4], states[Timeframe.HOURS_4], mid)

            no_wyckoff_data_available = 'No Wyckoff data available'

            # Send all charts in sequence, using copies of the buffers
            for idx, (chart, period, results) in enumerate([
                (charts[0], "15m", results_15m),
                (charts[1], "1h", results_1h),
                (charts[2], "4h", results_4h)
            ]):
                wyckoff_description = results['wyckoff'].description if results.get('wyckoff') else no_wyckoff_data_available
                caption = f"<b>{period} indicators:</b>\n{wyckoff_description}"
                
                if chart:
                    # Create a copy of the buffer's contents
                    chart_copy = io.BytesIO(chart.getvalue())
                    
                    try:
                        if len(caption) >= 1024:
                            # Send chart and caption separately if caption is too long
                            await context.bot.send_photo(
                                chat_id=telegram_utils.telegram_chat_id,
                                photo=chart_copy,
                                caption=f"<b>{period} chart</b>",
                                parse_mode=ParseMode.HTML
                            )
                            await telegram_utils.send(caption, parse_mode=ParseMode.HTML)
                        else:
                            # Send together if caption is within limits
                            await context.bot.send_photo(
                                chat_id=telegram_utils.telegram_chat_id,
                                photo=chart_copy,
                                caption=caption,
                                parse_mode=ParseMode.HTML
                            )
                    finally:
                        chart_copy.close()
                else:
                    await telegram_utils.send(caption, parse_mode=ParseMode.HTML)

        finally:
            # Clean up the original buffers
            for chart in charts:
                if chart and not chart.closed:
                    chart.close()
    
    # Add MTF analysis
    await telegram_utils.send(
        f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n"
        f"Market price: {fmt_price(mid)} USDC\n"
        f"{mtf_context.description}",
        parse_mode=ParseMode.HTML
    )


def get_ta_results(df: pd.DataFrame, wyckoff: WyckoffState, mid: float) -> Dict[str, Any]:
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
