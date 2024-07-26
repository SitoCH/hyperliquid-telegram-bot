import os
import time
import pandas as pd
import pandas_ta as ta

from tabulate import simple_separated_format, tabulate

from logging_utils import logger

from telegram.ext import ContextTypes
from telegram_utils import telegram_utils
from telegram.constants import ParseMode

from hyperliquid_utils import hyperliquid_utils

from utils import fmt


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    coins = os.getenv("HYPERLIQUID_TELEGRAM_BOT_ANALYZE_COINS", "")
    if len(coins) > 0:
        for coin in coins.split(","):
            await analyze_candles_for_coin(context, coin)


async def analyze_candles_for_coin(context, coin: str) -> None:
    logger.info(f"Running TA for {coin}")
    try:
        now = int(time.time() * 1000)

        candles_1h = hyperliquid_utils.info.candles_snapshot(coin, "1h", now - 7 * 86400000, now)
        candles_4h = hyperliquid_utils.info.candles_snapshot(coin, "4h", now - 14 * 86400000, now)
        
        df_1h = prepare_dataframe(candles_1h)
        aroon_flip_1h, supertrend_flip_1h = apply_indicators(df_1h)

        df_4h = prepare_dataframe(candles_4h)
        aroon_flip_4h, supertrend_flip_4h = apply_indicators(df_4h)

        if aroon_flip_1h or supertrend_flip_1h or aroon_flip_4h or supertrend_flip_4h:
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


def apply_indicators(df: pd.DataFrame) -> tuple:
    # Aroon Indicator
    aroon = ta.aroon(df['h'], df['l'], length=14)
    df['Aroon_Up'] = aroon['AROONU_14']
    df['Aroon_Down'] = aroon['AROOND_14']
    df['Aroon_Up_Down_Flip'] = (df['Aroon_Up'] > df['Aroon_Down']).astype(int)
    df['Aroon_Flip_Detected'] = df['Aroon_Up_Down_Flip'].diff().abs() == 1

    # SuperTrend
    supertrend = ta.supertrend(df['h'], df['l'], df['c'], length=20, multiplier=3)
    df['SuperTrend'] = supertrend['SUPERT_20_3']
    df['InUptrend'] = supertrend['SUPERTd_20_3'] == 1
    df['SuperTrend_Flip_Detected'] = df['InUptrend'].diff().abs() == 1

    return df['Aroon_Flip_Detected'].iloc[-1], df['SuperTrend_Flip_Detected'].iloc[-1]


async def send_trend_change_message(context, df_1h: pd.DataFrame, df_4h: pd.DataFrame, coin: str) -> None:
    results_1h = get_ta_results(df_1h)
    results_4h = get_ta_results(df_4h)

    table = tabulate(
        [
            ["Aroon 1h: ", "", ""],
            ["Trend ", results_1h["aroon_trend_prev"], results_1h["aroon_trend"]],
            ["Up ", fmt(results_1h["aroon_up_prev"]), fmt(results_1h["aroon_up"])],
            ["Down ", fmt(results_1h["aroon_down_prev"]), fmt(results_1h["aroon_down"])],
            ["Supertrend 1h: ", "", ""],
            ["Trend ", results_1h["supertrend_trend_prev"], results_1h["supertrend_trend"]],
            ["Value ", round(results_1h["supertrend_prev"], 2 if results_1h["supertrend_prev"] > 1 else 4), round(results_1h["supertrend_prev"], 2 if results_1h["supertrend"] > 1 else 4)],
            ["Aroon 4h: ", "", ""],
            ["Trend ", results_4h["aroon_trend_prev"], results_4h["aroon_trend"]],
            ["Up ", fmt(results_4h["aroon_up_prev"]), fmt(results_4h["aroon_up"])],
            ["Down ", fmt(results_4h["aroon_down_prev"]), fmt(results_4h["aroon_down"])],
            ["Supertrend 4h: ", "", ""],
            ["Trend ", results_4h["supertrend_trend_prev"], results_4h["supertrend_trend"]],
            ["Value ", round(results_4h["supertrend_prev"], 2 if results_4h["supertrend_prev"] > 1 else 4), round(results_4h["supertrend_prev"], 2 if results_4h["supertrend"] > 1 else 4)],
        ],
        headers=["", "Previous", "Current"],
        tablefmt=simple_separated_format(' '),
        colalign=("right", "right", "right")
    )

    message_lines = [
        f"<b>A trend indicator changed for {coin}</b>",
        "Indicators:",
        f"<pre>{table}</pre>"
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
        "supertrend_trend": supertrend_trend
    }
