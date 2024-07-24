import pandas as pd
import ta
import time

from logging_utils import logger
from telegram.ext import ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    await analyze_candles_for_coin(context, "BTC")
    await analyze_candles_for_coin(context, "ETH")
    await analyze_candles_for_coin(context, "kPEPE")


async def analyze_candles_for_coin(context, coin):
    now = int(time.time() * 1000)
    yesterday = now - 5 * 86400000

    candles = hyperliquid_utils.info.candles_snapshot(coin, "1h", yesterday, now)

    df = pd.DataFrame(candles)

    df['T'] = pd.to_datetime(df['T'], unit='ms')
    df['t'] = pd.to_datetime(df['t'], unit='ms')

    df[['c', 'h', 'l', 'o', 'v']] = df[['c', 'h', 'l', 'o', 'v']].astype(float)
    df['n'] = df['n'].astype(int)

    aroon_indicator = ta.trend.AroonIndicator(high=df['h'], low=df['l'], window=14)
    df['Aroon_Up'] = aroon_indicator.aroon_up()
    df['Aroon_Down'] = aroon_indicator.aroon_down()
    df['Aroon_Up_Down_Flip'] = (df['Aroon_Up'] > df['Aroon_Down']).astype(int)
    df['Aaron_Flip_Detected'] = df['Aroon_Up_Down_Flip'].diff().abs() == 1

    logger.info(f"Aaron_Flip on ETH: {df['Aaron_Flip_Detected'].iloc[-1]}")
    if df['Aaron_Flip_Detected'].iloc[-1]:
        aroon_up_before = df['Aroon_Up'].iloc[-2]
        aroon_down_before = df['Aroon_Down'].iloc[-2]
        aroon_up = df['Aroon_Up'].iloc[-1]
        aroon_down = df['Aroon_Down'].iloc[-1]
        await context.bot.send_message(text=f"Aroon indicator flipped on {coin}, up {aroon_up_before} -> {aroon_up} down {aroon_down_before} -> {aroon_down}", chat_id=telegram_utils.telegram_chat_id)
