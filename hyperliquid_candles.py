import os
import time
import pandas as pd
import pandas_ta as ta

from tabulate import simple_separated_format, tabulate

from logging_utils import logger

from telegram import Update
from telegram.ext import ContextTypes, CallbackContext, ConversationHandler
from telegram_utils import telegram_utils
from telegram.constants import ParseMode

from hyperliquid_utils import hyperliquid_utils

from utils import OPERATION_CANCELLED, fmt, fmt_price

SELECTING_COIN_FOR_TA = range(1)


async def execute_ta(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('Choose a coin to analyze:', reply_markup=hyperliquid_utils.get_coins_reply_markup())
    return SELECTING_COIN_FOR_TA


async def selected_coin_for_ta(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    coin = query.data
    if coin == 'cancel':
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    await query.edit_message_text(text=f"Analyzing {coin}...")
    await analyze_candles_for_coin(context, coin, True)
    await query.delete_message()
    return ConversationHandler.END


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    coins_to_analyze = []
    if len(coins_to_analyze) > 0:
        coins_to_analyze = set(os.getenv("HYPERLIQUID_TELEGRAM_BOT_ANALYZE_COINS", "").split(","))

    coins_with_open_positions = set(hyperliquid_utils.get_coins_with_open_positions())
    coins = coins_to_analyze | coins_with_open_positions
    for coin in coins:
        await analyze_candles_for_coin(context, coin, False)


async def analyze_candles_for_coin(context, coin: str, always_notify: bool) -> None:
    logger.info(f"Running TA for {coin}")
    try:
        now = int(time.time() * 1000)

        candles_1h = hyperliquid_utils.info.candles_snapshot(coin, "1h", now - 7 * 86400000, now)
        candles_4h = hyperliquid_utils.info.candles_snapshot(coin, "4h", now - 14 * 86400000, now)

        df_1h = prepare_dataframe(candles_1h)
        flip_on_1h = apply_indicators(df_1h)

        df_4h = prepare_dataframe(candles_4h)
        flip_on_4h = apply_indicators(df_4h)

        flip_on_4h = False if df_4h['T'].iloc[-1] < pd.Timestamp.now() - pd.Timedelta(hours=1) else flip_on_4h

        if always_notify or flip_on_1h or flip_on_4h:
            await send_trend_change_message(context, df_1h, df_4h, coin)
    except Exception as e:
        await context.bot.send_message(
            text=f"Failed to analyze candles for {coin}: {str(e)}",
            chat_id=telegram_utils.telegram_chat_id
        )


def prepare_dataframe(candles: list) -> pd.DataFrame:
    df = pd.DataFrame(candles)

    df['T'] = pd.to_datetime(df['T'], unit='ms')
    df['t'] = pd.to_datetime(df['t'], unit='ms')

    df[['c', 'h', 'l', 'o', 'v']] = df[['c', 'h', 'l', 'o', 'v']].astype(float)
    df['n'] = df['n'].astype(int)

    return df


def apply_indicators(df: pd.DataFrame) -> bool:

    length = 20

    # Aroon Indicator
    aroon = ta.aroon(df['h'], df['l'], length=length)
    df['Aroon_Up'] = aroon[f'AROONU_{length}']
    df['Aroon_Down'] = aroon[f'AROOND_{length}']
    df['Aroon_Up_Down_Flip'] = (df['Aroon_Up'] > df['Aroon_Down']).astype(int)
    df['Aroon_Flip_Detected'] = df['Aroon_Up_Down_Flip'].diff().abs() == 1

    # SuperTrend
    supertrend = ta.supertrend(df['h'], df['l'], df['c'], length=length, multiplier=3)
    df['SuperTrend'] = supertrend[f'SUPERT_{length}_3']
    df['InUptrend'] = supertrend[f'SUPERTd_{length}_3'] == 1
    df['SuperTrend_Flip_Detected'] = df['InUptrend'].diff().abs() == 1

    # Z-score
    zscore = ta.zscore(df['c'], length=length)
    df['Zscore'] = zscore
    df['Zscore_Flip_Detected'] = ((df['Zscore'] > 0) & (df['Zscore'].shift() <= 0)) | ((df['Zscore'] < 0) & (df['Zscore'].shift() >= 0))
    zscore_flip_detected = df['Zscore_Flip_Detected'].iloc[-1] and (df['Zscore_Flip_Detected'].iloc[-1] > 0.25 or df['Zscore_Flip_Detected'].iloc[-1] < 0.25)

    return df['Aroon_Flip_Detected'].iloc[-1] or df['SuperTrend_Flip_Detected'].iloc[-1] or zscore_flip_detected


async def send_trend_change_message(context, df_1h: pd.DataFrame, df_4h: pd.DataFrame, coin: str) -> None:
    results_1h = get_ta_results(df_1h)
    table_1h = tabulate(
        [
            ["Aroon: ", "", ""],
            ["Trend ", results_1h["aroon_trend_prev"], results_1h["aroon_trend"]],
            ["Up ", fmt(results_1h["aroon_up_prev"]), fmt(results_1h["aroon_up"])],
            ["Down ", fmt(results_1h["aroon_down_prev"]), fmt(results_1h["aroon_down"])],
            ["Supertrend: ", "", ""],
            ["Trend ", results_1h["supertrend_trend_prev"], results_1h["supertrend_trend"]],
            ["Value ", fmt_price(results_1h["supertrend_prev"]), fmt_price(results_1h["supertrend"])],
            ["Z-score: ", "", ""],
            ["Trend ", results_1h["zscore_trend_prev"], results_1h["zscore_trend"]],
            ["Value ", fmt(results_1h["zscore_prev"]), fmt(results_1h["zscore"])]
        ],
        headers=["", "Previous", "Current"],
        tablefmt=simple_separated_format(' '),
        colalign=("right", "right", "right")
    )

    results_4h = get_ta_results(df_4h)
    table_4h = tabulate(
        [
            ["Aroon: ", "", ""],
            ["Trend ", results_4h["aroon_trend_prev"], results_4h["aroon_trend"]],
            ["Up ", fmt(results_4h["aroon_up_prev"]), fmt(results_4h["aroon_up"])],
            ["Down ", fmt(results_4h["aroon_down_prev"]), fmt(results_4h["aroon_down"])],
            ["Supertrend: ", "", ""],
            ["Trend ", results_4h["supertrend_trend_prev"], results_4h["supertrend_trend"]],
            ["Value ", fmt_price(results_4h["supertrend_prev"]), fmt_price(results_4h["supertrend"])],
            ["Z-score: ", "", ""],
            ["Trend ", results_4h["zscore_trend_prev"], results_4h["zscore_trend"]],
            ["Value ", fmt(results_4h["zscore_prev"]), fmt(results_4h["zscore"])],
        ],
        headers=["", "Previous", "Current"],
        tablefmt=simple_separated_format(' '),
        colalign=("right", "right", "right")
    )

    message_lines = [
        f"<b>Indicators for {coin}</b>",
        "1h indicators:",
        f"<pre>{table_1h}</pre>",
        "4h indicators:",
        f"<pre>{table_4h}</pre>"
    ]

    await context.bot.send_message(
        text='\n'.join(message_lines),
        parse_mode=ParseMode.HTML,
        chat_id=telegram_utils.telegram_chat_id
    )


def get_ta_results(df):
    aroon_up_prev = df['Aroon_Up'].iloc[-2]
    aroon_down_prev = df['Aroon_Down'].iloc[-2]
    aroon_up = df['Aroon_Up'].iloc[-1]
    aroon_down = df['Aroon_Down'].iloc[-1]
    aroon_trend_prev = "uptrend" if aroon_up_prev > aroon_down_prev else "downtrend"
    aroon_trend = "uptrend" if aroon_up > aroon_down else "downtrend"

    supertrend_prev = df['SuperTrend'].iloc[-2]
    supertrend = df['SuperTrend'].iloc[-1]
    supertrend_trend_prev = "uptrend" if df['InUptrend'].iloc[-2] else "downtrend"
    supertrend_trend = "uptrend" if df['InUptrend'].iloc[-1] else "downtrend"

    zscore_prev = df['Zscore'].iloc[-2]
    zscore_trend_prev = "uptrend" if zscore_prev > 0.0 else "downtrend"
    zscore = df['Zscore'].iloc[-1]
    zscore_trend = "uptrend" if zscore > 0.0 else "downtrend"

    return {
        "aroon_up_prev": aroon_up_prev,
        "aroon_down_prev": aroon_down_prev,
        "aroon_up": aroon_up,
        "aroon_down": aroon_down,
        "aroon_trend_prev": aroon_trend_prev,
        "aroon_trend": aroon_trend,
        "supertrend_prev": supertrend_prev,
        "supertrend": supertrend,
        "supertrend_trend_prev": supertrend_trend_prev,
        "supertrend_trend": supertrend_trend,
        "zscore_prev": zscore_prev,
        "zscore": zscore,
        "zscore_trend_prev": zscore_trend_prev,
        "zscore_trend": zscore_trend
    }
