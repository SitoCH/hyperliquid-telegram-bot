import pandas as pd
import pandas_ta as ta
import time
import os


from telegram.ext import ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    coins = os.getenv("HYPERLIQUID_TELEGRAM_BOT_ANALYZE_COINS", "")
    for coin in coins.split(","):
        await analyze_candles_for_coin(context, coin)


async def analyze_candles_for_coin(context, coin):
    try:
        now = int(time.time() * 1000)
        yesterday = now - 5 * 86400000

        candles = hyperliquid_utils.info.candles_snapshot(coin, "1h", yesterday, now)

        df = pd.DataFrame(candles)

        df['T'] = pd.to_datetime(df['T'], unit='ms')
        df['t'] = pd.to_datetime(df['t'], unit='ms')

        df[['c', 'h', 'l', 'o', 'v']] = df[['c', 'h', 'l', 'o', 'v']].astype(float)
        df['n'] = df['n'].astype(int)

        # Aroon Indicator
        aroon_indicator = ta.aroon(df['h'], df['l'], length=14)
        df['Aroon_Up'] = aroon_indicator['AROOND_14']
        df['Aroon_Down'] = aroon_indicator['AROONU_14']
        df['Aroon_Up_Down_Flip'] = (df['Aroon_Up'] > df['Aroon_Down']).astype(int)
        df['Aaron_Flip_Detected'] = df['Aroon_Up_Down_Flip'].diff().abs() == 1

        # SuperTrend
        supertrend = ta.supertrend(df['h'], df['l'], df['c'], length=20, multiplier=3)

        df['SuperTrend'] = supertrend['SUPERT_20_3']
        df['InUptrend'] = supertrend['SUPERTd_20_3'] == 1
        df['SuperTrend_Flip_Detected'] = df['InUptrend'].diff().abs() == 1

        if df['Aaron_Flip_Detected'].iloc[-1] or df['SuperTrend_Flip_Detected'].iloc[-1]:
            aroon_up_before = df['Aroon_Up'].iloc[-2]
            aroon_down_before = df['Aroon_Down'].iloc[-2]
            aroon_up = df['Aroon_Up'].iloc[-1]
            aroon_down = df['Aroon_Down'].iloc[-1]
            aroon_trend_before = "uptrend" if aroon_up_before > aroon_down_before else "downtrend"
            aroon_trend = "uptrend" if aroon_up > aroon_down else "downtrend"

            supertrend_before = df['SuperTrend'].iloc[-2]
            supertrend = df['SuperTrend'].iloc[-1]
            supertrend_trend_before = "uptrend" if df['InUptrend'].iloc[-2] else "downtrend"
            supertrend_trend = "uptrend" if df['InUptrend'].iloc[-1] else "downtrend"

            message_lines = [
                f"<b>A trend indicator changed for {coin}:</b>",
                f"Aroon from {aroon_trend_before} to {aroon_trend}:",
                f"  Up: {aroon_up_before} -> {aroon_up}",
                f"  Down: {aroon_down_before} -> {aroon_down}",
                f"Supertrend from {supertrend_trend_before} to {supertrend_trend}",
                f"  Value: {supertrend_before} -> {supertrend}",
            ]

            await context.bot.send_message(
                text='\n'.join(message_lines),
                chat_id=telegram_utils.telegram_chat_id
            )
    except Exception as e:
        await context.bot.send_message(
            text=f"Failed to check orders: {str(e)}",
            chat_id=telegram_utils.telegram_chat_id
        )
