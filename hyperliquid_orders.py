from collections import defaultdict
from logging_utils import logger
from tabulate import simple_separated_format, tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import fmt, px_round

SL_DISTANCE_LIMIT = 1.50
SL_MINIMUM_DISTANCE_LIMIT = 0.50


async def get_orders_from_hyperliquid():
    open_orders = hyperliquid_utils.info.frontend_open_orders(hyperliquid_utils.address)
    grouped_data = defaultdict(lambda: defaultdict(list))
    for order in open_orders:
        grouped_data[order["coin"]][order["orderType"]].append(order)
    return {coin: dict(order_types) for coin, order_types in sorted(grouped_data.items())}


def get_return_on_equity_limit(leverage):
    if leverage >= 30:
        return 15.0
    elif leverage >= 20:
        return 10.0
    elif leverage >= 10:
        return 7.5
    else:
        return 5.0


def get_adjusted_sl_distance_limit(leverage):
    if leverage >= 30:
        return max(SL_DISTANCE_LIMIT - 1.0, SL_MINIMUM_DISTANCE_LIMIT)
    elif leverage >= 20:
        return max(SL_DISTANCE_LIMIT - 0.75, SL_MINIMUM_DISTANCE_LIMIT)
    elif leverage >= 10:
        return max(SL_DISTANCE_LIMIT - 0.50, SL_MINIMUM_DISTANCE_LIMIT)
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
                logger.info(f"Verifying orders on {coin}")
                mid = float(all_mids[coin])
                is_long = hyperliquid_utils.get_size(user_state, coin) > 0.0
                sl_raw_orders, tp_raw_orders = get_sl_tp_orders(order_types, is_long)

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


def get_sl_tp_orders(order_types, is_long: bool):
    sl_raw_orders = order_types.get('Stop Market', [])
    sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    tp_raw_orders = order_types.get('Take Profit Market', [])
    tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    return sl_raw_orders, tp_raw_orders


async def adjust_sl_trigger(context, exchange, user_state, coin, current_price, sz_decimals, tp_raw_orders, is_long, sl_order, order_index):
    current_trigger_px = float(sl_order['triggerPx'])
    return_on_equity = hyperliquid_utils.get_return_on_equity(user_state, coin) * 100.0
    unrealized_pnl = hyperliquid_utils.get_unrealized_pnl(user_state, coin)
    logger.info(f"Verifying order {sl_order['oid']} on {coin}, unrealized PnL {fmt(unrealized_pnl)}$, RoE {fmt(return_on_equity)}%")

    if return_on_equity <= 0.0:
        logger.info("RoE too low to update order")
        return False

    if unrealized_pnl <= 2.5:
        logger.info("Unrealized PnL too low to update order")
        return False

    entry_px = hyperliquid_utils.get_entry_px(user_state, coin)
    leverage = hyperliquid_utils.get_leverage(user_state, coin)

    new_sl_trigger_px = None

    return_on_equity_limit = get_return_on_equity_limit(leverage)
    if return_on_equity > return_on_equity_limit:
        new_sl_trigger_px = determine_new_sl_trigger(is_long, entry_px, current_trigger_px, current_price)

    if new_sl_trigger_px is not None:
        logger.info(f"Updating order due to sufficient PnL on {coin}, stop-loss at {current_trigger_px}, entry price at {entry_px}, current price at {current_price}")
        await update_sl_and_tp_orders(context, exchange, coin, is_long, sl_order, new_sl_trigger_px, current_trigger_px, sz_decimals, tp_raw_orders, current_price,
                                      return_on_equity, unrealized_pnl)
        return True

    sl_already_updated_by_pnl = entry_px < current_trigger_px if is_long else entry_px > current_trigger_px
    if not sl_already_updated_by_pnl:
        logger.info(f"Distance limit adjustment ignored due to missing previous PnL adjustment: entry price {entry_px}, current trigger price {current_trigger_px}")
        return False

    sl_order_distance = calculate_sl_order_distance(current_trigger_px, current_price)
    distance_limit = get_adjusted_sl_distance_limit(leverage) + order_index * 0.1

    if sl_order_distance > distance_limit:
        new_sl_trigger_px = calculate_new_trigger_price(is_long, current_price, distance_limit)
        logger.info(f"Updating order due to sufficient SL distance on {coin}, stop-loss at {current_trigger_px}, current price at {current_price}")
        await update_sl_and_tp_orders(context, exchange, coin, is_long, sl_order, new_sl_trigger_px, current_trigger_px, sz_decimals, tp_raw_orders, current_price, 
                                      return_on_equity, unrealized_pnl)
        return True

    logger.info("No valid conditions have been found to update the order")
    return False


def determine_new_sl_trigger(is_long, entry_px, current_trigger_px, current_price):
    if is_long and entry_px > current_trigger_px:
        new_px = entry_px + (current_price - entry_px) / 4.0
    elif not is_long and entry_px < current_trigger_px:
        new_px = entry_px - (entry_px - current_price) / 4.0
    else:
        return None
    return new_px


def calculate_sl_order_distance(current_trigger_px, current_price):
    return abs((1 - current_trigger_px / current_price) * 100)


def calculate_new_trigger_price(is_long, current_price, distance_limit):
    if is_long:
        return round(current_price - current_price * (distance_limit - 0.05) / 100.0, 6)
    else:
        return round(current_price + current_price * (distance_limit - 0.05) / 100.0, 6)


async def update_sl_and_tp_orders(context, exchange, coin, is_long, sl_order, new_sl_trigger_px, current_trigger_px, sz_decimals, tp_raw_orders, current_price,
                                  return_on_equity, unrealized_pnl):
    message_lines = [
        f"<b>{coin}:</b>",
        f"Current price: {current_price}",
        f"Unrealized PnL: {fmt(unrealized_pnl)} USDC",
        f"Return on equity: {fmt(return_on_equity)}%",
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
    stop_order_type = {"trigger": {"triggerPx": px_round(new_trigger_px), "isMarket": True, "tpsl": "sl"}}
    order_result = exchange.modify_order(int(sl_order['oid']), coin, not is_long, sz, float(sl_order['limitPx']), stop_order_type, True)
    logger.info(order_result)
    message_lines.append(f"Modified stop-loss trigger from {sl_order['triggerPx']} to {new_trigger_px}")


def modify_tp_order(message_lines, exchange, coin, is_long, order, sz, sl_delta):
    new_delta = sl_delta / 4.0
    new_trigger_px = round(float(order['triggerPx']) + (new_delta if is_long else -new_delta), 6)
    new_limit_px = round(float(order['limitPx']) + (new_delta if is_long else -new_delta), 6)
    stop_order_type = {"trigger": {"triggerPx": px_round(new_trigger_px), "isMarket": True, "tpsl": "tp"}}
    order_result = exchange.modify_order(int(order['oid']), coin, not is_long, sz, px_round(new_limit_px), stop_order_type, True)
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
            is_long = hyperliquid_utils.get_size(user_state, coin) > 0.0
            sl_raw_orders, tp_raw_orders = get_sl_tp_orders(order_types, is_long)
            message_lines.append(f"Mode: {'long' if is_long else 'short'}")
            message_lines.append(f"Leverage: {hyperliquid_utils.get_leverage(user_state, coin)}x")

            tp_orders = format_orders(tp_raw_orders, mid, percentage_format=lambda triggerPx, mid: abs((triggerPx / mid - 1) * 100))
            sl_orders = format_orders(sl_raw_orders, mid, percentage_format=lambda triggerPx, mid: abs(((1 - triggerPx / mid) * 100)))

            table_orders = tp_orders + [["Current", all_mids[coin], ""]] + sl_orders

            entry_px = hyperliquid_utils.get_entry_px_str(user_state, coin)
            liquidation_px = hyperliquid_utils.get_liquidation_px_str(user_state, coin)

            table_orders = insert_order(table_orders, ["Entry", entry_px, ""], is_long)
            table_orders = insert_order(table_orders, ["Liq.", liquidation_px, ""], is_long)

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

        await update.message.reply_text(
            text='\n'.join(message_lines),
            parse_mode=ParseMode.HTML,
            reply_markup=telegram_utils.reply_markup
        )

    except Exception as e:
        await update.message.reply_text(text=f"Failed to check orders: {str(e)}")


def format_orders(raw_orders, mid, percentage_format):
    return [
        [
            order['sz'],
            order['triggerPx'],
            f"{fmt(percentage_format(float(order['triggerPx']), mid ))}%"
        ]
        for order in raw_orders
    ]


def insert_order(orders, new_order, is_long):
    order_px = float(new_order[1])
    inserted = False

    for i in range(len(orders)):
        if (order_px > float(orders[i][1]) if is_long else order_px < float(orders[i][1])):
            orders.insert(i, new_order)
            inserted = True
            break

    if not inserted:
        orders.append(new_order)

    return orders
