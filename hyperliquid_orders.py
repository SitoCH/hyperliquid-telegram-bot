from collections import defaultdict
from logging_utils import logger
from tabulate import simple_separated_format, tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils

SL_DISTANCE_LIMIT = 2.00


async def get_orders_from_hyperliquid():
    open_orders = hyperliquid_utils.info.frontend_open_orders(hyperliquid_utils.address)
    grouped_data = defaultdict(lambda: defaultdict(list))
    for order in open_orders:
        grouped_data[order["coin"]][order["orderType"]].append(order)
    return {coin: dict(order_types) for coin, order_types in sorted(grouped_data.items())}


def get_unrealized_pnl_limit(leverage):
    if leverage >= 30:
        return 15.0
    elif leverage >= 20:
        return 10.0
    elif leverage >= 10:
        return 5.0
    else:
        return 2.5


def get_adjusted_sl_distance_limit(leverage):
    if leverage >= 30:
        return max(SL_DISTANCE_LIMIT - 1.0, 1.0)
    elif leverage >= 20:
        return max(SL_DISTANCE_LIMIT - 0.75, 1.0)
    elif leverage >= 10:
        return max(SL_DISTANCE_LIMIT - 0.50, 1.0)
    else:
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
                    if await adjust_sl_trigger(context, exchange, user_state, coin, mid, sz_decimals, tp_raw_orders, is_long, sl_order, index):
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
    is_long = len(sl_raw_orders) > 0 and mid < float(sl_raw_orders[0]['triggerPx'])
    sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    tp_raw_orders = order_types.get('Take Profit Market', [])
    tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    return is_long, sl_raw_orders, tp_raw_orders


async def adjust_sl_trigger(context, exchange, user_state, coin, current_price, sz_decimals, tp_raw_orders, is_long, sl_order, order_index):
    current_trigger_px = float(sl_order['triggerPx'])
    unrealized_pnl = hyperliquid_utils.get_unrealized_pnl(user_state, coin)

    if unrealized_pnl <= 0.0:
        return False

    entry_px = hyperliquid_utils.get_entry_px(user_state, coin)
    leverage = hyperliquid_utils.get_leverage(user_state, coin)
    new_sl_trigger_px = None

    if unrealized_pnl > get_unrealized_pnl_limit(leverage):
        new_sl_trigger_px = determine_new_sl_trigger(is_long, entry_px, current_trigger_px, current_price)

    if new_sl_trigger_px is not None:
        logger.info(f"Updating order due to sufficient PnL on {coin}, stop-loss at {current_trigger_px}, current price at {current_price}")
        await update_sl_and_tp_orders(context, exchange, coin, is_long, sl_order, new_sl_trigger_px, current_trigger_px, sz_decimals, tp_raw_orders, current_price, unrealized_pnl)
        return True

    if current_trigger_px < entry_px:
        return False

    sl_order_distance = calculate_sl_order_distance(current_trigger_px, current_price)
    distance_limit = get_adjusted_sl_distance_limit(leverage) + order_index * 0.25

    if sl_order_distance > distance_limit:
        new_sl_trigger_px = calculate_new_trigger_price(is_long, current_price, distance_limit)
        logger.info(f"Updating order due to sufficient SL distance on {coin}, stop-loss at {current_trigger_px}, current price at {current_price}")
        await update_sl_and_tp_orders(context, exchange, coin, is_long, sl_order, new_sl_trigger_px, current_trigger_px, sz_decimals, tp_raw_orders, current_price, unrealized_pnl)
        return True

    return False


def determine_new_sl_trigger(is_long, entry_px, current_trigger_px, current_price):
    if is_long and entry_px > current_trigger_px:
        new_px = entry_px + (current_price - entry_px) / 4.0
    elif not is_long and entry_px < current_trigger_px:
        new_px = entry_px - (entry_px - current_price) / 4.0
    else:
        return None
    return round(new_px, 6)


def calculate_sl_order_distance(current_trigger_px, current_price):
    return abs((1 - current_trigger_px / current_price) * 100)


def calculate_new_trigger_price(is_long, current_price, distance_limit):
    if is_long:
        return round(current_price - current_price * (distance_limit - 0.10) / 100.0, 6)
    else:
        return round(current_price + current_price * (distance_limit - 0.10) / 100.0, 6)


async def update_sl_and_tp_orders(context, exchange, coin, is_long, sl_order, new_sl_trigger_px, current_trigger_px, sz_decimals, tp_raw_orders, current_price, unrealized_pnl):
    message_lines = [
        f"<b>{coin}:</b>",
        f"Current price: {current_price}",
        f"Unrealized PnL: {unrealized_pnl:,.2f} USDC",
        f"Size: {sl_order['sz']}"
    ]
    sz = round(float(sl_order['sz']), sz_decimals[coin])
    modify_sl_order(message_lines, exchange, coin, is_long, sl_order, new_sl_trigger_px, sz)
    sl_delta = abs(new_sl_trigger_px - current_trigger_px)
    modify_matching_tp_order(exchange, coin, tp_raw_orders, is_long, sl_order, sl_delta, message_lines, sz)
    await context.bot.send_message(text='\n'.join(message_lines), chat_id=telegram_utils.telegram_chat_id, parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)


def modify_matching_tp_order(exchange, coin, tp_raw_orders, is_long, sl_order, sl_delta, message_lines, sz):
    matching_tp_order = next((order for order in tp_raw_orders if order['coin'] == coin and order['sz'] == sl_order['sz']), None)
    if matching_tp_order:
        modify_tp_order(message_lines, exchange, coin, is_long, matching_tp_order, sz, sl_delta)
    else:
        message_lines.append("No matching TP order has been found")


def modify_sl_order(message_lines, exchange, coin, is_long, sl_order, new_trigger_px, sz):
    stop_order_type = {"trigger": {"triggerPx": new_trigger_px, "isMarket": True, "tpsl": "sl"}}
    order_result = exchange.modify_order(int(sl_order['oid']), coin, not is_long, sz, float(sl_order['limitPx']), stop_order_type, True)
    logger.info(order_result)
    message_lines.append(f"Modified stop-loss trigger from {sl_order['triggerPx']} to {new_trigger_px}")


def modify_tp_order(message_lines, exchange, coin, is_long, order, sz, sl_delta):
    new_delta = sl_delta / 5.0
    new_trigger_px = round(float(order['triggerPx']) + (new_delta if is_long else -new_delta), 6)
    new_limit_px = round(float(order['limitPx']) + (new_delta if is_long else -new_delta), 6)
    stop_order_type = {"trigger": {"triggerPx": new_trigger_px, "isMarket": True, "tpsl": "tp"}}
    order_result = exchange.modify_order(int(order['oid']), coin, not is_long, sz, new_limit_px, stop_order_type, True)
    logger.info(order_result)
    message_lines.append(f"Modified take-profit trigger from {order['triggerPx']} to {new_trigger_px}")


async def get_open_orders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
                [order['sz'], order['triggerPx'], f"{abs((float(order['triggerPx']) / mid - 1) * 100):.2f}%"]
                for order in tp_raw_orders
            ]
            sl_orders = [
                [order['sz'], order['triggerPx'], f"{abs(((1 - float(order['triggerPx']) / mid) * 100)):.2f}%"]
                for order in sl_raw_orders
            ]

            table_orders = tp_orders + [["Current", all_mids[coin], ""]] + sl_orders
            if not is_long:
                table_orders.reverse()

            table = tabulate(
                table_orders,
                headers=["Size", "Trigger price", "Distance"],
                tablefmt=simple_separated_format(' '),
                colalign=("right", "right", "right")
            )
            message_lines.append(f"<pre>{table}</pre>")

        if not message_lines:
            message_lines.append("No open orders")

        await update.message.reply_text(text='\n'.join(message_lines), parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)

    except Exception as e:
        await update.message.reply_text(text=f"Failed to check orders: {str(e)}")
