from collections import defaultdict
from logging_utils import logger
from tabulate import simple_separated_format, tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import fmt, px_round


async def get_orders_from_hyperliquid():
    open_orders = hyperliquid_utils.info.frontend_open_orders(hyperliquid_utils.address)
    grouped_data = defaultdict(lambda: defaultdict(list))
    for order in open_orders:
        grouped_data[order["coin"]][order["orderType"]].append(order)
    return {coin: dict(order_types) for coin, order_types in sorted(grouped_data.items())}


def get_sl_tp_orders(order_types, is_long: bool):
    sl_raw_orders = order_types.get('Stop Market', [])
    sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    tp_raw_orders = order_types.get('Take Profit Market', [])
    tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    return sl_raw_orders, tp_raw_orders


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
