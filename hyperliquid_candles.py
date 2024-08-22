import os
import io
import time
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from tabulate import tabulate, simple_separated_format
from telegram import Update
from telegram.ext import CallbackContext, ContextTypes, ConversationHandler
from telegram.constants import ParseMode

from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED, fmt, fmt_price

SELECTING_COIN_FOR_TA = range(1)


async def execute_ta(update: Update, context: CallbackContext) -> int:
    reply_markup = hyperliquid_utils.get_coins_reply_markup()
    await update.message.reply_text("Choose a coin to analyze:", reply_markup=reply_markup)
    return SELECTING_COIN_FOR_TA


async def selected_coin_for_ta(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    coin = query.data
    if coin == "cancel":
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    await query.edit_message_text(text=f"Analyzing {coin}...")
    await analyze_candles_for_coin(context, coin, hyperliquid_utils.info.all_mids(), always_notify=True)
    await query.delete_message()
    return ConversationHandler.END


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    coins_to_analyze = []
    if len(coins_to_analyze) > 0:
        coins_to_analyze = os.getenv("HYPERLIQUID_TELEGRAM_BOT_ANALYZE_COINS", "").split(",")
    coins = set(coins_to_analyze) | set(hyperliquid_utils.get_coins_with_open_positions())
    all_mids = hyperliquid_utils.info.all_mids()

    for coin in coins:
        await analyze_candles_for_coin(context, coin, all_mids, always_notify=False)


async def analyze_candles_for_coin(context: ContextTypes.DEFAULT_TYPE, coin: str, all_mids, always_notify: bool) -> None:
    logger.info(f"Running TA for {coin}")
    try:
        now = int(time.time() * 1000)
        candles_5m = hyperliquid_utils.info.candles_snapshot(coin, "5m", now - 1 * 86400000, now)
        candles_1h = hyperliquid_utils.info.candles_snapshot(coin, "1h", now - 5 * 86400000, now)
        candles_4h = hyperliquid_utils.info.candles_snapshot(coin, "4h", now - 10 * 86400000, now)

        df_5m = prepare_dataframe(candles_5m)
        df_1h = prepare_dataframe(candles_1h)
        df_4h = prepare_dataframe(candles_4h)

        mid = float(all_mids[coin])
        apply_indicators(df_5m, mid)
        flip_on_1h = apply_indicators(df_1h, mid)
        flip_on_4h = apply_indicators(df_4h, mid)
        flip_on_4h = flip_on_4h and 'T' in df_4h.columns and df_4h["T"].iloc[-1] >= pd.Timestamp.now() - pd.Timedelta(hours=1)

        if always_notify or flip_on_1h or flip_on_4h:
            await send_trend_change_message(context, mid, df_5m, df_1h, df_4h, coin)
    except Exception as e:
        logger.critical(e, exc_info=True)
        await context.bot.send_message(text=f"Failed to analyze candles for {coin}: {str(e)}", chat_id=telegram_utils.telegram_chat_id)


def prepare_dataframe(candles: list) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["T"] = pd.to_datetime(df["T"], unit="ms")
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df[["c", "h", "l", "o", "v"]] = df[["c", "h", "l", "o", "v"]].astype(float)
    df["n"] = df["n"].astype(int)
    return df


def apply_indicators(df: pd.DataFrame, mid: float) -> bool:
    length = 20

    df.set_index("T", inplace=True)
    df.sort_index(inplace=True)

    # Aroon Indicator
    aroon = ta.aroon(df["h"], df["l"], length=length)
    df["Aroon_Up"] = aroon[f"AROONU_{length}"]
    df["Aroon_Down"] = aroon[f"AROOND_{length}"]
    df["Aroon_Flip_Detected"] = df["Aroon_Up"].gt(df["Aroon_Down"]).diff().abs() == 1

    # SuperTrend
    supertrend = ta.supertrend(df["h"], df["l"], df["c"], length=length, multiplier=3)
    df["SuperTrend"] = supertrend[f"SUPERT_{length}_3"]
    df["SuperTrend_Flip_Detected"] = supertrend[f"SUPERTd_{length}_3"].diff().abs() == 1

    # Z-score
    df["Zscore"] = ta.zscore(df["c"], length=length)
    df["Zscore_Flip_Detected"] = (df["Zscore"].gt(0) & df["Zscore"].shift().le(0)) | (df["Zscore"].lt(0) & df["Zscore"].shift().ge(0))

    # VWAP
    df["VWAP"] = ta.vwap(df["h"], df["l"], df["c"], df["v"])
    df["VWAP_Flip_Detected"] = (mid > df["VWAP"].iloc[-2]) & (mid <= df["VWAP"].iloc[-1]) | (mid < df["VWAP"].iloc[-2]) & (mid >= df["VWAP"].iloc[-1])

    return df[["Aroon_Flip_Detected", "SuperTrend_Flip_Detected", "Zscore_Flip_Detected", "VWAP_Flip_Detected"]].any(axis=1).iloc[-1]


def generate_chart(df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, coin: str) -> list:
    chart_buffers = []

    def save_to_buffer(df_plot: pd.DataFrame, title: str) -> io.BytesIO:
        buf = io.BytesIO()
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
        mpf.plot(df_plot.set_index("t"),
                 type='candle',
                 ax=ax[0],
                 volume=ax[1],
                 axtitle=title,
                 style='charles',
                 mav=(20),
                 addplot=[mpf.make_addplot(df_plot['SuperTrend'], ax=ax[0]),
                          mpf.make_addplot(df_plot['VWAP'], ax=ax[0])])
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

    df_5m_plot = df_5m.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df_1h_plot = df_1h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df_4h_plot = df_4h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})

    chart_buffers.append(save_to_buffer(df_5m_plot, f"{coin} - 5M Chart"))
    chart_buffers.append(save_to_buffer(df_1h_plot, f"{coin} - 1H Chart"))
    chart_buffers.append(save_to_buffer(df_4h_plot, f"{coin} - 4H Chart"))

    return chart_buffers


async def send_trend_change_message(context: ContextTypes.DEFAULT_TYPE, mid: float, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, coin: str) -> None:
    chart_buffers = generate_chart(df_5m, df_1h, df_4h, coin)

    for buf in chart_buffers:
        await context.bot.send_photo(chat_id=telegram_utils.telegram_chat_id, photo=buf)


    results_1h = get_ta_results(df_1h, mid)
    results_4h = get_ta_results(df_4h, mid)

    table_1h = format_table(results_1h)
    table_4h = format_table(results_4h)

    message_lines = [
        f"<b>Indicators for {coin}</b>",
        f"Market price: {fmt_price(mid)} USDC",
        "1h indicators:",
        f"<pre>{table_1h}</pre>",
        "4h indicators:",
        f"<pre>{table_4h}</pre>",
    ]

    await context.bot.send_message(chat_id=telegram_utils.telegram_chat_id, text="\n".join(message_lines), parse_mode=ParseMode.HTML)


def get_ta_results(df: pd.DataFrame, mid: float) -> dict:
    aroon_up_prev, aroon_down_prev = df["Aroon_Up"].iloc[-2], df["Aroon_Down"].iloc[-2]
    aroon_up, aroon_down = df["Aroon_Up"].iloc[-1], df["Aroon_Down"].iloc[-1]
    supertrend_prev, supertrend = df["SuperTrend"].iloc[-2], df["SuperTrend"].iloc[-1]
    zscore_prev, zscore = df["Zscore"].iloc[-2], df["Zscore"].iloc[-1]
    vwap_prev, vwap = df["VWAP"].iloc[-2], df["VWAP"].iloc[-1]

    return {
        "aroon_up_prev": aroon_up_prev,
        "aroon_down_prev": aroon_down_prev,
        "aroon_up": aroon_up,
        "aroon_down": aroon_down,
        "aroon_trend_prev": "uptrend" if aroon_up_prev > aroon_down_prev else "downtrend",
        "aroon_trend": "uptrend" if aroon_up > aroon_down else "downtrend",
        "supertrend_prev": supertrend_prev,
        "supertrend": supertrend,
        "supertrend_trend_prev": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-2] else "downtrend",
        "supertrend_trend": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-1] else "downtrend",
        "zscore_prev": zscore_prev,
        "zscore": zscore,
        "zscore_trend_prev": "uptrend" if zscore_prev > 0 else "downtrend",
        "zscore_trend": "uptrend" if zscore > 0 else "downtrend",
        "vwap_prev": vwap_prev,
        "vwap": vwap,
        "vwap_trend_prev": "uptrend" if mid > vwap_prev else "downtrend",
        "vwap_trend": "uptrend" if mid > vwap else "downtrend",
    }


def format_table(results: dict) -> str:
    return tabulate(
        [
            ["Aroon: ", "", ""],
            ["Trend ", results["aroon_trend_prev"], results["aroon_trend"]],
            ["Up ", fmt(results["aroon_up_prev"]), fmt(results["aroon_up"])],
            ["Down ", fmt(results["aroon_down_prev"]), fmt(results["aroon_down"])],
            ["", "", ""],
            ["Supertrend: ", "", ""],
            ["Trend ", results["supertrend_trend_prev"], results["supertrend_trend"]],
            ["Value ", fmt_price(results["supertrend_prev"]), fmt_price(results["supertrend"])],
            ["", "", ""],
            ["Z-score: ", "", ""],
            ["Trend ", results["zscore_trend_prev"], results["zscore_trend"]],
            ["Value ", fmt(results["zscore_prev"]), fmt(results["zscore"])],
            ["", "", ""],
            ["VWAP: ", "", ""],
            ["Trend ", results["vwap_trend_prev"], results["vwap_trend"]],
            ["Value ", fmt_price(results["vwap_prev"]), fmt_price(results["vwap"])],
        ],
        headers=["", "Previous", "Current"],
        tablefmt=simple_separated_format(" "),
        colalign=("right", "right", "right"),
    )
