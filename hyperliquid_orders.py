
from collections import defaultdict
from logging_utils import logger

from tabulate import simple_separated_format, tabulate

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram_utils import telegram_utils

from hyperliquid_utils import hyperliquid_utils

SL_DISTANCE_LIMIT = 3.00


async def get_orders_from_hyperliquid():
    open_orders = hyperliquid_utils.info.frontend_open_orders(hyperliquid_utils.address)
    grouped_data = defaultdict(lambda: defaultdict(list))
    for order in open_orders:
        coin = order["coin"]
        order_type = order["orderType"]
        grouped_data[coin][order_type].append(order)

    sorted_grouped_data = dict(sorted(grouped_data.items()))

    return {coin: dict(order_types) for coin, order_types in sorted_grouped_data.items()}


def get_adjusted_sl_distance_limit(user_state, coin):
    leverage = hyperliquid_utils.get_leverage(user_state, coin)
    if leverage >= 30:
        return max(SL_DISTANCE_LIMIT - 1.0, 1.5)
    if leverage >= 20:
        return max(SL_DISTANCE_LIMIT - 0.75, 1.5)
    if leverage >= 10:
        return max(SL_DISTANCE_LIMIT - 0.50, 1.5)
    return SL_DISTANCE_LIMIT


async def update_orders_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update_open_orders(context, True)


async def update_open_orders(context: ContextTypes.DEFAULT_TYPE, send_message_on_no_updates=False) -> None:

    try:
        grouped_data = await get_orders_from_hyperliquid()

        all_mids = hyperliquid_utils.info.all_mids()
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)

        exchange = hyperliquid_utils.get_exchange()

        updated_orders = False
        if exchange is not None:

            sz_decimals = hyperliquid_utils.get_sz_decimals()

            for coin, order_types in grouped_data.items():

                mid = float(all_mids[coin])
                is_long, sl_raw_orders, tp_raw_orders = get_sl_tp_orders(order_types, mid)

                for index, sl_order in enumerate(sl_raw_orders):
                    current_trigger_px = float(sl_order['triggerPx'])
                    sl_order_distance = abs((1 - current_trigger_px / mid) * 100)
                    current_sl_distance_limit = get_adjusted_sl_distance_limit(user_state, coin) + index * 0.5
                    if sl_order_distance > current_sl_distance_limit:
                        await adjust_sl_trigger(context, exchange, coin, mid, sz_decimals, tp_raw_orders, is_long, sl_order, current_trigger_px, sl_order_distance, current_sl_distance_limit)
                        updated_orders = True

        else:
            await context.bot.send_message(text="Exchange is not enabled", chat_id=telegram_utils.telegram_chat_id, parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)

        if not updated_orders and send_message_on_no_updates:
            await context.bot.send_message(text="No orders to update", chat_id=telegram_utils.telegram_chat_id, parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)

    except Exception as e:
        logger.critical(e, exc_info=True)
        await context.bot.send_message(text=f"Failed to update orders: {str(e)}", chat_id=telegram_utils.telegram_chat_id)


def get_sl_tp_orders(order_types, mid):
    sl_raw_orders = order_types.get('Stop Market', [])
    is_long = True
    if len(sl_raw_orders) > 0 and mid < float(sl_raw_orders[0]['triggerPx']):
        is_long = False

    sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)

    tp_raw_orders = order_types.get('Take Profit Market', [])
    tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    return is_long, sl_raw_orders, tp_raw_orders


async def adjust_sl_trigger(context, exchange, coin, mid, sz_decimals, tp_raw_orders, is_long, sl_order, current_trigger_px, sl_order_distance, distance_limit):
    message_lines = [f"<b>{coin}:</b>"]
    px = mid - mid * (distance_limit - 0.25) / 100.0 if is_long else mid + mid * (distance_limit - 0.25) / 100.0
    new_sl_trigger_px = round(float(f"{px:.5g}"), 6)
    sz = round(float(sl_order['sz']), sz_decimals[coin])
    modify_sl_order(message_lines, exchange, coin, is_long, sl_order, sl_order_distance, new_sl_trigger_px, sz)

    matching_tp_order = next(
                            (order for order in tp_raw_orders if order['coin'] == coin and order['sz'] == sl_order['sz']), None
    )

    if matching_tp_order:
        old_tp_trigger_px = float(matching_tp_order['triggerPx'])
        new_delta = abs(new_sl_trigger_px - current_trigger_px) / 2.0
        new_tp_trigger_px = old_tp_trigger_px + new_delta if is_long else old_tp_trigger_px - new_delta
        modify_tp_order(message_lines, exchange, coin, is_long, matching_tp_order, new_tp_trigger_px, sz)
    else:
        message_lines.append("No matching TP order has been found")

    await context.bot.send_message(text='\n'.join(message_lines), chat_id=telegram_utils.telegram_chat_id, parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)


def modify_sl_order(message_lines, exchange, coin, is_long, sl_order, order_distance, new_trigger_px, sz):
    stop_order_type = {"trigger": {"triggerPx": new_trigger_px, "isMarket": True, "tpsl": "sl"}}
    order_result = exchange.modify_order(int(sl_order['oid']), coin, not is_long, sz, float(sl_order['limitPx']), stop_order_type, True)
    logger.info(order_result)
    message_lines.append(f"Modified SL trigger from {sl_order['triggerPx']} ({order_distance:.2f}%) to {new_trigger_px}")


def modify_tp_order(message_lines, exchange, coin, is_long, order, new_trigger_px, sz):
    stop_order_type = {"trigger": {"triggerPx": new_trigger_px, "isMarket": True, "tpsl": "tp"}}
    order_result = exchange.modify_order(int(order['oid']), coin, not is_long, sz, float(order['limitPx']), stop_order_type, True)
    logger.info(order_result)
    message_lines.append(f"Modified TP trigger from {order['triggerPx']} to {new_trigger_px}")


async def get_open_orders(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:

    try:
        grouped_data = await get_orders_from_hyperliquid()

        all_mids = hyperliquid_utils.info.all_mids()

        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)

        message_lines = []

        for coin, order_types in grouped_data.items():

            message_lines.append(f"<b>{coin}</b>")
            mid = float(all_mids[coin])
            is_long, sl_raw_orders, tp_raw_orders = get_sl_tp_orders(order_types, mid)
            message_lines.append(f"Mode: {'long' if is_long else 'short'}")
            message_lines.append(f"Leverage: {hyperliquid_utils.get_leverage(user_state, coin)}x")

            tp_orders = [
                        [
                            order['sz'],
                            order['triggerPx'],
                            f"{abs((float(order['triggerPx']) / mid - 1) * 100):.2f}%"
                        ]
                for order in tp_raw_orders
            ]

            sl_orders = [
                        [
                            order['sz'],
                            order['triggerPx'],
                            f"{abs(((1 - float(order['triggerPx']) / mid) * 100)):.2f}%"
                        ]
                for order in sl_raw_orders
            ]

            tablefmt = simple_separated_format(' ')

            table_orders = tp_orders + [["Current", all_mids[coin], ""]] + sl_orders
            if not is_long:
                table_orders.reverse()

            table = tabulate(
                table_orders,
                headers=[
                    "Size",
                    "Trigger price",
                    "Distance"
                ],
                tablefmt=tablefmt,
                colalign=("right", "right", "right")
            )

            message_lines.append(f"<pre>{table}</pre>")

        if len(message_lines) == 0:
            message_lines.append("No open orders")

        message = '\n'.join(message_lines)
        await update.message.reply_text(text=message, parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)

    except Exception as e:
        await update.message.reply_text(text=f"Failed to check orders: {str(e)}")
