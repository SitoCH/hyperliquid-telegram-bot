import os
import time
import pandas as pd
import pandas_ta as ta
from tabulate import simple_separated_format, tabulate
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
    try:
        now = int(time.time() * 1000)
        five_days_ago = now - 7 * 86400000

        candles = hyperliquid_utils.info.candles_snapshot(coin, "1h", five_days_ago, now)
        df = prepare_dataframe(candles)

        aroon_flip_detected, supertrend_flip_detected = apply_indicators(df)

        if aroon_flip_detected or supertrend_flip_detected:
            await send_trend_change_message(context, df, coin)
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


async def send_trend_change_message(context, df: pd.DataFrame, coin: str) -> None:
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

    table = tabulate(
        [
            ["Aroon: ", "", ""],
            ["Trend: ", aroon_trend_prev, aroon_trend],
            ["Up: ", fmt(aroon_up_prev), fmt(aroon_up)],
            ["Down: ", fmt(aroon_down_prev), fmt(aroon_down)],
            ["Supertrend: ", "", ""],
            ["Trend: ", supertrend_trend_prev, supertrend_trend],
            ["Value: ", round(supertrend_prev, 2 if supertrend_prev > 1 else 4), round(supertrend_prev, 2 if supertrend > 1 else 4)],
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
