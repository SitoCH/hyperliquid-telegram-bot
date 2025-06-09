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

# Add analysis mode enum
from enum import Enum

class AnalysisMode(Enum):
    WYCKOFF = "wyckoff"
    LLM = "llm"

# Environment variable for analysis mode
ANALYSIS_MODE = AnalysisMode(os.getenv("HTB_ANALYSIS_MODE", "wyckoff").lower())

from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from utils import OPERATION_CANCELLED, fmt, fmt_price, log_execution_time
from .candles_utils import get_coins_to_analyze
from .candles_cache import get_candles_with_cache
from .funding_rates_cache import get_funding_with_cache, FundingRateEntry
from .data_processor import prepare_dataframe, apply_indicators, remove_partial_candle
from hyperliquid_utils.hyperliquid_ratelimiter import hyperliquid_rate_limiter
from .llm.llm_analyzer import LLMAnalyzer
from .wyckoff.wyckoff_analyzer import WyckoffAnalyzer


SELECTING_COIN_FOR_TA = range(1)


async def execute_ta(update: Update, context: CallbackContext) -> int:
    if not update.message:
        return ConversationHandler.END

    if context.args and len(context.args) > 0:
        coin = context.args[0]
        message = await update.message.reply_text(text=f"Analyzing {coin}...")
        await analyze_candles_for_coin(context, coin, True)
        await message.delete()
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
        weight_per_analysis = 145
        next_available = hyperliquid_rate_limiter.get_next_available_time(weight_per_analysis)
        delay = max(3, next_available)
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

    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    available = float(user_state['withdrawable'])
    minimum_balance = 5.0
    if available < minimum_balance:
        logger.info(f"Account balance ({fmt(available)}$) is below {fmt(minimum_balance)}$, skipping analysis")
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
    logger.info(f"Running TA for {coin} using {ANALYSIS_MODE.value} mode")
    try:
        if ANALYSIS_MODE == AnalysisMode.WYCKOFF:
            await analyze_candles_wyckoff(context, coin, interactive_analysis)
        elif ANALYSIS_MODE == AnalysisMode.LLM:
            await analyze_candles_llm(context, coin, interactive_analysis)
        else:
            logger.error(f"Unknown analysis mode: {ANALYSIS_MODE}")
            if interactive_analysis:
                await telegram_utils.send(f"Unknown analysis mode: {ANALYSIS_MODE}")
    except Exception as e:
        logger.error(f"Failed to analyze candles for {coin}: {str(e)}", exc_info=True)
        if interactive_analysis:
            await telegram_utils.send(f"Failed to analyze candles for {coin}: {str(e)}")
    
    logger.info(f"TA for {coin} done in {(time.time() - start_time):.2f} seconds")


async def analyze_candles_wyckoff(context: ContextTypes.DEFAULT_TYPE, coin: str, interactive_analysis: bool) -> None:
    """Original Wyckoff analysis implementation."""
    analyzer = WyckoffAnalyzer()
    await analyzer.analyze(context, coin, interactive_analysis)


async def analyze_candles_llm(context: ContextTypes.DEFAULT_TYPE, coin: str, interactive_analysis: bool) -> None:
    """New LLM-based analysis implementation."""
    analyzer = LLMAnalyzer()
    await analyzer.analyze(context, coin, interactive_analysis)
