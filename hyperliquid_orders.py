
from collections import defaultdict
from logging_utils import logger

from tabulate import simple_separated_format, tabulate

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram_utils import telegram_utils

from hyperliquid_utils import hyperliquid_utils

SL_DISTANCE_LIMIT = 6.5


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

        exchange = hyperliquid_utils.get_exchange()

        updated_orders = False
        if exchange is not None:

            sz_decimals = hyperliquid_utils.get_sz_decimals()

            for coin, order_types in grouped_data.items():

                mid = float(all_mids[coin])

                tp_raw_orders = order_types.get('Take Profit Market', [])
                tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=True)

                sl_raw_orders = order_types.get('Stop Market', [])
                sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=True)

                for index, sl_order in enumerate(sl_raw_orders):
                    current_trigger_px = float(sl_order['triggerPx'])
                    sl_order_distance = ((1 - current_trigger_px / mid) * 100)
                    current_sl_distance_limit = SL_DISTANCE_LIMIT + index / 3
                    if sl_order_distance > current_sl_distance_limit:
                        await increase_sl_trigger(update, exchange, coin, mid, sz_decimals, tp_raw_orders, sl_order, current_trigger_px, sl_order_distance, current_sl_distance_limit)
                        updated_orders = True

        else:
            await update.message.reply_text(text="Exchange is not enabled", parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)

        if not updated_orders:
            await update.message.reply_text(text="No orders to update", parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)

    except Exception as e:
        logger.critical(e, exc_info=True)
        await update.message.reply_text(text=f"Failed to update orders: {str(e)}")


async def increase_sl_trigger(update, exchange, coin, mid, sz_decimals, tp_raw_orders, sl_order, current_trigger_px, sl_order_distance, distance_limit):
    message_lines = [f"<b>{coin}:</b>"]
    new_sl_trigger_px = round(float(f"{(mid - (mid * (distance_limit - 0.5) / 100)):.5g}"), 6)
    sz = round(float(sl_order['sz']), sz_decimals[coin])
    modify_sl(message_lines, exchange, coin, sl_order, sl_order_distance, new_sl_trigger_px, sz)
    matching_tp_order = next(
                            (order for order in tp_raw_orders if order['coin'] == coin and order['sz'] == sl_order['sz']), None
    )

    if matching_tp_order:
        new_tp_trigger_px = float(matching_tp_order['triggerPx']) + (new_sl_trigger_px - current_trigger_px) / 2
        modify_tp(message_lines, exchange, coin, matching_tp_order, new_tp_trigger_px, sz)
    else:
        message_lines.append("No matching TP order has been found")

    await update.message.reply_text(text='\n'.join(message_lines), parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)


def modify_sl(message_lines, exchange, coin, sl_order, order_distance, new_trigger_px, sz):
    stop_order_type = {"trigger": {"triggerPx": new_trigger_px, "isMarket": True, "tpsl": "sl"}}
    exchange.modify_order(int(sl_order['oid']), coin, False, sz, float(sl_order['limitPx']), stop_order_type, True)
    message_lines.append(f"Modified SL trigger from {sl_order['triggerPx']} ({order_distance:.2f}%) to {new_trigger_px}")


def modify_tp(message_lines, exchange, coin, order, new_trigger_px, sz):
    stop_order_type = {"trigger": {"triggerPx": new_trigger_px, "isMarket": True, "tpsl": "tp"}}
    exchange.modify_order(int(order['oid']), coin, False, sz, float(order['limitPx']), stop_order_type, True)
    message_lines.append(f"Modified TP trigger from {order['triggerPx']} to {new_trigger_px}")


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
