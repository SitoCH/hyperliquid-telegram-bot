
from collections import defaultdict
from logging_utils import logger

from tabulate import simple_separated_format, tabulate

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram_utils import telegram_utils

from hyperliquid_utils import hyperliquid_utils


async def get_orders_from_hyperliquid():
    open_orders = hyperliquid_utils.info.frontend_open_orders(hyperliquid_utils.address)
    grouped_data = defaultdict(lambda: defaultdict(list))
    for order in open_orders:
        coin = order["coin"]
        order_type = order["orderType"]
        grouped_data[coin][order_type].append(order)

    sorted_grouped_data = dict(sorted(grouped_data.items()))

    return {coin: dict(order_types) for coin, order_types in sorted_grouped_data.items()}


async def update_open_orders(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:

    try:
        grouped_data = await get_orders_from_hyperliquid()

        all_mids = hyperliquid_utils.info.all_mids()

        message_lines = []

        exchange = hyperliquid_utils.get_exchange()
        if exchange is not None:

            for coin, order_types in grouped_data.items():

                mid = float(all_mids[coin])

                sz_decimals = get_sz_decimals()

                tp_raw_orders = order_types.get('Take Profit Market', [])
                tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=True)

                sl_raw_orders = order_types.get('Stop Market', [])
                sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=True)

                if len(tp_raw_orders) == len(sl_raw_orders):
                    first_sl_order = sl_raw_orders[0]
                    current_trigger_px = float(first_sl_order['triggerPx'])
                    first_sl_order_distance = ((1 - current_trigger_px / mid) * 100)
                    distance_limit = 7.5
                    if first_sl_order_distance > distance_limit:
                        message_lines.append(f"<b>{coin}:</b>")
                        new_sl_trigger_px = round(float(f"{(mid - (mid * (distance_limit - 0.5) / 100)):.5g}"), 6)
                        coin = first_sl_order['coin']
                        sz = round(float(first_sl_order['sz']), sz_decimals[coin])
                        modify_sl(message_lines, exchange, coin, first_sl_order, first_sl_order_distance, new_sl_trigger_px, sz)
                        matching_tp_order = next(
                            (order for order in tp_raw_orders if order['coin'] == coin and order['sz'] == first_sl_order['sz']),
                            None
                        )

                        if matching_tp_order:
                            new_tp_trigger_px = float(matching_tp_order['triggerPx']) + new_sl_trigger_px - current_trigger_px
                            modify_tp(message_lines, exchange, coin, matching_tp_order, new_tp_trigger_px, sz)
                        else:
                            message_lines.append("No matching TP order has been found")


        else:
            message_lines.append("Exchange is not enabled")

        message = "No orders to update"
        if len(message_lines) > 0:
            message = '\n'.join(message_lines)
      
        await update.message.reply_text(text=message, parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)

    except Exception as e:
        logger.critical(e, exc_info=True)
        await update.message.reply_text(text=f"Failed to update orders: {str(e)}")


def modify_sl(message_lines, exchange, coin, first_sl_order, first_sl_order_distance, new_trigger_px, sz):
    stop_order_type = {"trigger": {"triggerPx": new_trigger_px, "isMarket": True, "tpsl": "sl"}}
    exchange.modify_order(int(first_sl_order['oid']), coin, False, sz, float(first_sl_order['limitPx']), stop_order_type, True)
    message_lines.append(f"Adjusted SL from {first_sl_order['triggerPx']} ({first_sl_order_distance:.2f}%) to {new_trigger_px}")


def modify_tp(message_lines, exchange, coin, first_sl_order, new_trigger_px, sz):
    stop_order_type = {"trigger": {"triggerPx": new_trigger_px, "isMarket": True, "tpsl": "tp"}}
    res = exchange.modify_order(int(first_sl_order['oid']), coin, False, sz, float(first_sl_order['limitPx']), stop_order_type, True)
    print(res)
    message_lines.append(f"Adjusted TP from {first_sl_order['triggerPx']} to {new_trigger_px}")


def get_sz_decimals():
    meta = hyperliquid_utils.info.meta()
    sz_decimals = {}
    for asset_info in meta["universe"]:
        sz_decimals[asset_info["name"]] = asset_info["szDecimals"]
    return sz_decimals


async def get_open_orders(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:

    try:
        grouped_data = await get_orders_from_hyperliquid()

        all_mids = hyperliquid_utils.info.all_mids()

        for coin, order_types in grouped_data.items():

            message_lines = [
                f"<b>{coin}:</b>"
            ]
            mid = float(all_mids[coin])

            tp_raw_orders = order_types.get('Take Profit Market', [])
            tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=True)
            tp_orders = [
                        [
                            f"{order['sz']}",
                            f"{order['triggerPx']}",
                            f"{((float(order['triggerPx']) / mid - 1) * 100):.2f} %"
                        ]
                for order in tp_raw_orders
            ]

            sl_raw_orders = order_types.get('Stop Market', [])
            sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=True)
            sl_orders = [
                        [
                            f"{order['sz']}",
                            f"{order['triggerPx']}",
                            f"{((1 - float(order['triggerPx']) / mid) * 100):.2f} %"
                        ]
                for order in sl_raw_orders
            ]

            tablefmt = simple_separated_format(' ')
            table = tabulate(
                tp_orders + [["Current", all_mids[coin], ""]] + sl_orders,
                headers=[
                    "Size",
                    "Trigger price",
                    "Distance"
                ],
                tablefmt=tablefmt
            )

            message_lines.append(f"<pre>{table}</pre>")

            message = '\n'.join(message_lines)
            await update.message.reply_text(text=message, parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)

    except Exception as e:
        await update.message.reply_text(text=f"Failed to check orders: {str(e)}")
